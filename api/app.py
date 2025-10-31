import asyncio
import io
import logging
import sqlite3
import uuid
import tempfile
from sqlalchemy import create_engine

import pandas as pd
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from link_service.linking_service import link_rows
from nlp.process_texts import process_texts
from quality_checks.quality_check import quality_check
import os
import json
import datetime as _dt
from typing import Dict
import multiprocessing as mp
import signal
import time as _time
import queue as _queue
import traceback
import psutil
import mimetypes

# Keep status/logs separate from large result tables to avoid locks
STATUS_DB = "pipeline_status.db"
RESULTS_DB = "pipeline_results.db"

# Initialize FastAPI app
app = FastAPI()

# Add CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- cancellation support ---


class CancelledError(Exception):
    pass


CANCEL_EVENTS: Dict[str, asyncio.Event] = {}
# Store both PID and the Process object for proper cleanup
PROCESS_PIDS: Dict[str, tuple[int, mp.Process]] = {}


def create_cancel_event(task_id: str):
    CANCEL_EVENTS[task_id] = asyncio.Event()    


def request_cancel(task_id: str) -> bool:
    ev = CANCEL_EVENTS.get(task_id)
    if not ev:
        return False
    ev.set()
    return True


def is_cancelled(task_id: str) -> bool:
    ev = CANCEL_EVENTS.get(task_id)
    return bool(ev and ev.is_set())


def check_cancelled_or_raise(task_id: str):
    if is_cancelled(task_id):
        raise CancelledError(f"Task {task_id} cancelled by user")


def _register_process(task_id: str, pid: int, process: mp.Process):
    """Register the subprocess PID and Process object for cleanup."""
    PROCESS_PIDS[task_id] = (pid, process)


def _clear_process(task_id: str):
    """Remove and ensure the process is properly joined/reaped."""
    info = PROCESS_PIDS.pop(task_id, None)
    if info:
        _, proc = info
        # Give it a chance to exit cleanly
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
        # Force kill if still alive
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=0.5)
        # Final join to reap zombie
        try:
            proc.join(timeout=0.1)
        except Exception:
            pass


def _kill_task_process(task_id: str, force: bool = False, timeout: float = 2.0) -> bool:
    """
    Kill only the task's subprocess and its descendants, not the API server.
    Uses psutil to find child processes without killing the parent worker.
    """
    info = PROCESS_PIDS.get(task_id)
    if not info:
        return False

    pid, proc = info

    try:
        parent = psutil.Process(pid)
        # Collect all descendants (children of the subprocess)
        children = parent.children(recursive=True)
        processes_to_kill = [parent] + children
    except psutil.NoSuchProcess:
        _clear_process(task_id)
        return True
    except Exception as e:
        print(f"[WARN] Error collecting process tree for task {task_id}: {e}")
        _clear_process(task_id)
        return False

    # Send SIGTERM (or SIGKILL if force=True)
    sig = signal.SIGKILL if force else signal.SIGTERM
    for p in processes_to_kill:
        try:
            p.send_signal(sig)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not force:
        # Wait for graceful shutdown
        t0 = _time.time()
        while _time.time() - t0 < timeout:
            try:
                parent = psutil.Process(pid)
                if not parent.is_running():
                    break
                _time.sleep(0.1)
            except psutil.NoSuchProcess:
                break

        # Escalate to SIGKILL if still alive
        try:
            parent = psutil.Process(pid)
            if parent.is_running():
                for p in [parent] + parent.children(recursive=True):
                    try:
                        p.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except psutil.NoSuchProcess:
            pass

    # Wait for the multiprocessing.Process to actually exit and be reaped
    proc.join(timeout=1.0)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=0.5)

    _clear_process(task_id)
    return True


# NEW: top-level child target so it's picklable under spawn
def _subproc_child(q: mp.Queue, target, args, kwargs):
    try:
        # Create a new session so we can cleanly kill this subtree later
        if hasattr(os, "setsid"):
            os.setsid()
    except Exception:
        pass
    try:
        res = target(*args, **kwargs)
        q.put(("ok", res))
    except Exception:
        q.put(("err", traceback.format_exc()))


def _run_in_subprocess(task_id: str, target, *args, **kwargs):
    """
    Run target(*args, **kwargs) in a separate process so we can kill it without affecting the API.
    Returns target's result (pickleable) or raises CancelledError/Exception.
    """
    q: mp.Queue = mp.Queue()
    p = mp.Process(
        target=_subproc_child,
        args=(q, target, args, kwargs),
        daemon=False,  # Change to False so we control cleanup
    )
    p.start()
    _register_process(task_id, p.pid, p)  # Store both PID and Process

    result = None
    try:
        while p.is_alive():
            if is_cancelled(task_id):
                _kill_task_process(task_id, force=False)
                raise CancelledError(f"Task {task_id} cancelled")

            try:
                status, payload = q.get(timeout=0.2)
                if status == "ok":
                    result = payload
                else:
                    raise RuntimeError(payload)
                break
            except _queue.Empty:
                pass

        if result is None and not q.empty():
            status, payload = q.get_nowait()
            if status == "ok":
                result = payload
            else:
                raise RuntimeError(payload)

        # Wait for process to finish normally
        p.join(timeout=1.0)

    finally:
        # Ensure cleanup even on exception
        _clear_process(task_id)

    return result


async def run_quality_check_task(task_id: str, data_file: bytes, disease_type: str = "sarcoma"):
    """
    Background job: run quality_check on a single uploaded spreadsheet.
    Stores output so it can be fetched later via /results/{task_id}/quality_check.
    """
    task_logger = get_task_logger(task_id)
    try:
        update_status(task_id, "Initializing", 0)
        task_logger.info(
            "Quality-check task initialised for disease_type=%s.", disease_type)
        check_cancelled_or_raise(task_id)

        update_status(task_id, "Loading data", 10)
        # Support both Excel and CSV
        df: pd.DataFrame = await asyncio.to_thread(_read_uploaded_file, data_file)
        task_logger.info("File read into DataFrame (shape=%s).", df.shape)
        check_cancelled_or_raise(task_id)

        update_status(task_id, "Running quality check", 60)
        task_logger.info("Invoking quality_check() in isolated process.")
        # Run in separate process group
        qc_result = await asyncio.to_thread(_run_in_subprocess, task_id, quality_check, df, disease_type)
        check_cancelled_or_raise(task_id)

        final_df: pd.DataFrame | None = None
        if isinstance(qc_result, pd.DataFrame):
            final_df = qc_result
        elif isinstance(qc_result, tuple):
            for item in qc_result:
                if isinstance(item, pd.DataFrame):
                    final_df = item
                    break
        if final_df is None:
            task_logger.info(
                "quality_check returned no DataFrame. Using input as output.")
            final_df = df

        update_status(task_id, "Saving results", 90)
        await asyncio.to_thread(store_step_output, task_id, "quality_check", final_df)

        update_status(task_id, "Completed", 100, "Quality-check finished.")
        task_logger.info(
            "Quality-check task completed successfully (shape=%s).", final_df.shape)
    except CancelledError as exc:
        update_status(task_id, "Cancelled", 100, str(exc))
        task_logger.warning("Quality-check task cancelled.")
        return
    except Exception as exc:
        update_status(task_id, "Failed", 100, str(exc))
        task_logger.error("Quality-check task failed: %s", exc)
        raise
    finally:
        CANCEL_EVENTS.pop(task_id, None)


def _read_uploaded_file(file_bytes: bytes) -> pd.DataFrame:
    """
    Read uploaded file (Excel or CSV) into a DataFrame.
    Tries Excel first, falls back to CSV if that fails.
    """
    try:
        # Try Excel first
        return pd.read_excel(io.BytesIO(file_bytes))
    except Exception:
        # Fall back to CSV
        return pd.read_csv(io.BytesIO(file_bytes))


@app.post("/run/quality_check")
async def quality_check_call(
    file: UploadFile = File(...),
    disease_type: str = "sarcoma",
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    task_id = str(uuid.uuid4())
    create_cancel_event(task_id)
    data_content = await file.read()

    # Start the task in the background
    background_tasks.add_task(run_quality_check_task,
                              task_id, data_content, disease_type)

    return {
        "task_id": task_id,
        "message": "quality_check started – poll /status/{task_id} for progress.",
    }


@app.get("/results/{task_id}/{step_name}")
def get_step_results_as_csv(task_id: str, step_name: str):
    """
    Fetch pipeline step data from the database and return as a CSV file.
    """
    conn = None
    try:
        conn = sqlite3.connect(RESULTS_DB, timeout=5)
        conn.execute("PRAGMA busy_timeout=5000;")
        query = f'SELECT * FROM "{task_id}_{step_name}"'
        df: pd.DataFrame = pd.read_sql(query, conn)  # type: ignore

        if df.empty:
            raise HTTPException(
                status_code=404, detail="No data found for the specified task and step."
            )

        # Write the DataFrame to CSV in memory
        csv_data = io.BytesIO()
        df.to_csv(csv_data, index=False, encoding='utf-8')
        csv_data.seek(0)

        # Return the CSV file as a streaming response
        return StreamingResponse(
            csv_data,
            media_type="text/csv",
            headers={

                "Content-Disposition": f'attachment; filename="{task_id}_{step_name}.csv"'
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if conn is not None:
            conn.close()


async def run_link_rows_task(task_id: str, data_file: bytes, disease_type: str = "sarcoma"):
    """
    Background job: run link_rows on a single uploaded spreadsheet.
    The output is stored with store_step_output so it can be fetched later via
    /results/{task_id}/linked_data.
    """
    task_logger = get_task_logger(task_id)
    try:
        update_status(task_id, "Initializing", 0)
        task_logger.info(
            "Link-row task initialised for disease_type=%s.", disease_type)
        check_cancelled_or_raise(task_id)

        update_status(task_id, "Loading data", 10)
        df: pd.DataFrame = await asyncio.to_thread(_read_uploaded_file, data_file)
        task_logger.info("File read into DataFrame (shape=%s).", df.shape)
        check_cancelled_or_raise(task_id)

        update_status(task_id, "Linking rows", 60)
        task_logger.info("Invoking link_rows() in isolated process.")
        linked_df: pd.DataFrame = await asyncio.to_thread(_run_in_subprocess, task_id, link_rows, df, disease_type)
        task_logger.info("link_rows complete (shape=%s).", linked_df.shape)
        check_cancelled_or_raise(task_id)

        await asyncio.to_thread(store_step_output, task_id, "linked_data", linked_df)

        update_status(task_id, "Completed", 100, "Link-rows finished.")
        task_logger.info("Link-row task completed successfully.")
    except CancelledError as exc:
        update_status(task_id, "Cancelled", 100, str(exc))
        task_logger.warning("Link-row task cancelled.")
        return
    except Exception as exc:
        update_status(task_id, "Failed", 100, str(exc))
        task_logger.error("Link-row task failed: %s", exc)
        raise
    finally:
        CANCEL_EVENTS.pop(task_id, None)


@app.post("/run/link_rows")
async def link_rows_call(
    file: UploadFile = File(...),
    disease_type: str = "sarcoma",
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Accepts a single Excel/CSV, runs only link_rows, and returns a task_id.
    Use /status/{task_id} to track progress and
    /results/{task_id}/linked_data to download the output CSV.
    """
    task_id = str(uuid.uuid4())
    create_cancel_event(task_id)
    data_content = await file.read()

    background_tasks.add_task(
        run_link_rows_task, task_id, data_content, disease_type)

    return {
        "task_id": task_id,
        "message": "link_rows started – poll /status/{task_id} for progress.",
    }


async def run_pipeline_task(task_id: str, data_file: bytes, text_file: bytes, disease_type: str = "sarcoma"):
    """Function to run the pipeline task asynchronously.
    """
    task_logger = get_task_logger(task_id)
    try:
        update_status(task_id, "Initializing", 0)
        task_logger.info(
            "Pipeline initialization started for disease_type=%s.", disease_type)
        check_cancelled_or_raise(task_id)

        update_status(task_id, "Loading data", 10)
        task_logger.info("Loading data from uploaded files.")
        excel_data: pd.DataFrame = await asyncio.to_thread(_read_uploaded_file, data_file)
        free_texts: pd.DataFrame = await asyncio.to_thread(_read_uploaded_file, text_file)
        check_cancelled_or_raise(task_id)

        update_status(task_id, "Processing free texts", 30)
        task_logger.info(
            "Processing free texts and structuring data in isolated process.")
        # Use top-level wrapper instead of a local function (picklable under spawn)
        # Create a wrapper that includes disease_type

        def _call_process_texts_with_disease(ft, ex):
            return _call_process_texts(ft, ex, disease_type)

        process_result = await asyncio.to_thread(
            _run_in_subprocess, task_id, _call_process_texts_with_disease, free_texts, excel_data
        )

        # Unpack the result (excel_data, llm_results)
        if isinstance(process_result, tuple) and len(process_result) == 2:
            structured_data, llm_results = process_result
            # Store LLM results as a separate downloadable step
            if llm_results is not None and not llm_results.empty:
                await asyncio.to_thread(store_step_output, task_id, "llm_annotations", llm_results)
                task_logger.info(
                    "LLM annotations saved (shape=%s).", llm_results.shape)
        else:
            # Backward compatibility
            structured_data = process_result

        await asyncio.to_thread(store_step_output, task_id, "processed_texts", structured_data)
        check_cancelled_or_raise(task_id)

        update_status(task_id, "Linking rows", 60)
        task_logger.info("Linking rows based on criteria in isolated process.")
        linked_data = await asyncio.to_thread(_run_in_subprocess, task_id, link_rows, structured_data, disease_type)
        await asyncio.to_thread(store_step_output, task_id, "linked_data", linked_data)
        check_cancelled_or_raise(task_id)

        update_status(task_id, "Performing data quality checks", 90)
        task_logger.info("Performing data quality checks in isolated process.")
        await asyncio.to_thread(_run_in_subprocess, task_id, quality_check, linked_data, disease_type)
        final_data = linked_data
        await asyncio.to_thread(store_step_output, task_id, "quality_check", final_data)

        update_status(task_id, "Completed", 100,
                      result="Pipeline completed successfully!")
        task_logger.info("Pipeline completed successfully.")
    except CancelledError as e:
        update_status(task_id, "Cancelled", 100, result=str(e))
        task_logger.warning("Pipeline cancelled.")
        return
    except Exception as e:
        update_status(task_id, "Failed", 100, result=str(e))
        task_logger.error("Pipeline failed with error: %s", e)
        raise
    finally:
        CANCEL_EVENTS.pop(task_id, None)


@app.post("/pipeline")
async def start_pipeline(
    data_file: UploadFile = File(...),
    text_file: UploadFile = File(...),
    disease_type: str = "sarcoma",
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Starts the pipeline - accepts Excel or CSV files"""
    task_id = str(uuid.uuid4())
    create_cancel_event(task_id)
    data_content = await data_file.read()
    text_content = await text_file.read()

    # Add the task to the background
    background_tasks.add_task(
        run_pipeline_task, task_id, data_content, text_content, disease_type)

    return {
        "task_id": task_id,
        "message": "Pipeline started. Use /status/{task_id} to track progress.",
    }


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Provides the status of a task given its id"""
    status = get_status_from_db(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    # Always return a quick, non-blocking status with running flag
    is_running = status["step"] not in ("Completed", "Failed") and (
        status["progress"] is None or status["progress"] < 100)
    return {**status, "is_running": bool(is_running)}


@app.get("/logs/{task_id}")
async def get_logs(task_id: str):
    """
    Retrieve logs for a specific task.
    """
    conn = sqlite3.connect(STATUS_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp, log_level, message
        FROM pipeline_logs
        WHERE task_id = ?
        ORDER BY timestamp
    """,
        (task_id,),
    )
    logs = [{"timestamp": row[0], "level": row[1], "message": row[2]}
            for row in cursor.fetchall()]
    conn.close()
    if not logs:
        raise HTTPException(
            status_code=404, detail="No logs found for the specified task.")
    return {"task_id": task_id, "logs": logs}


# Initialize SQLite database for logs
def init_logs_db():
    conn = sqlite3.connect(STATUS_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_logs (
            task_id TEXT,
            timestamp TEXT,
            log_level TEXT,
            message TEXT,
            FOREIGN KEY (task_id) REFERENCES pipeline_status (task_id)
        )
    """
    )
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA busy_timeout=5000;")
    conn.commit()
    conn.close()


init_logs_db()


def log_message(task_id: str, log_level: str, message: str):
    """
    Save a log message to the database.
    """
    conn = sqlite3.connect(STATUS_DB, timeout=5)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO pipeline_logs (task_id, timestamp, log_level, message)
        VALUES (?, datetime('now'), ?, ?)
    """,
        (task_id, log_level, message),
    )
    conn.commit()
    conn.close()


# Configure standard Python logger
logger = logging.getLogger("pipeline")
logger.setLevel(logging.DEBUG)


# Log handler to save logs to the database
class DBLogHandler(logging.Handler):
    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id

    def emit(self, record):
        # Use formatted message to include args, avoid raw record.msg
        log_message(self.task_id, record.levelname, record.getMessage())


# Example: Add DBLogHandler to logger dynamically
def get_task_logger(task_id: str):
    task_logger = logging.getLogger(f"pipeline_{task_id}")
    task_logger.setLevel(logging.DEBUG)
    # Avoid adding duplicate handlers for the same task_id
    if not any(isinstance(h, DBLogHandler) and getattr(h, "task_id", None) == task_id for h in task_logger.handlers):
        task_logger.addHandler(DBLogHandler(task_id))
    # Prevent duplicate propagation to root loggers
    task_logger.propagate = False
    return task_logger


@app.post("/cancel/{task_id}")
def cancel_task(task_id: str):
    """
    Request cancellation of a running task by ID. Sends SIGTERM to the task subprocess only.
    """
    current = get_status_from_db(task_id)
    if not current:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    update_status(task_id, "Cancelling", int(
        current.get("progress") or 0), current.get("result") or "")
    requested = request_cancel(task_id)
    _kill_task_process(task_id, force=False)
    return {"ok": bool(requested), "message": "Cancellation requested."}


@app.post("/kill/{task_id}")
def force_kill_task(task_id: str):
    """
    Force kill the task's subprocess (SIGKILL). Use if normal cancel doesn't work.
    """
    current = get_status_from_db(task_id)
    if not current:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    request_cancel(task_id)
    killed = _kill_task_process(task_id, force=True)
    update_status(task_id, "Cancelled", 100,
                  "Force-killed by user" if killed else "Kill requested")
    return {"ok": bool(killed), "message": "Force kill sent."}


# Helper: top-level wrapper for process_texts (avoid local nested func)
def _call_process_texts(ft: pd.DataFrame, ex: pd.DataFrame, disease_type: str = "sarcoma"):
    try:
        # Updated to handle tuple return (excel_data, llm_results)
        result = process_texts(ft, ex, disease_type=disease_type)
        # If process_texts returns a tuple, return it; otherwise wrap in tuple
        if isinstance(result, tuple):
            return result
        else:
            # Backward compatibility if it returns only excel_data
            return (result, None)
    except TypeError:
        # Legacy signature fallback
        result = process_texts(ft, ex, None, None, None,
                               disease_type=disease_type)
        if isinstance(result, tuple):
            return result
        else:
            return (result, None)


def init_db():
    """Initialize SQLite database for pipeline status."""
    conn = sqlite3.connect(STATUS_DB)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_status (
            task_id TEXT PRIMARY KEY,
            step TEXT,
            progress INTEGER,
            result TEXT
        )
    """
    )
    # Enable WAL so readers don't block on writers
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA busy_timeout=5000;")
    conn.commit()
    conn.close()


init_db()


def _cell_to_sql_scalar(x):
    # None stays None
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None

    # Python datetime → ISO string
    if isinstance(x, (_dt.datetime, _dt.date, _dt.time)):
        # for dates with time keep full precision
        if isinstance(x, _dt.datetime):
            return x.strftime("%Y-%m-%d %H:%M:%S.%f")
        if isinstance(x, _dt.date):
            return x.strftime("%Y-%m-%d")
        return x.strftime("%H:%M:%S")

    # Pandas / NumPy datetimes
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        ts = pd.to_datetime(x, errors="coerce")
        return None if pd.isna(ts) else ts.strftime("%Y-%m-%d %H:%M:%S.%f")

    # NumPy scalars → Python scalars
    if isinstance(x, np.generic):
        return x.item()

    # Lists / tuples / sets / dicts → JSON string (or join if you prefer)
    if isinstance(x, (list, tuple, set, dict)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)

    # bytes → utf8
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return x.decode("latin-1", errors="ignore")

    return x


def sanitize_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize dtypes that often cause trouble
    df = df.copy()

    # Convert datetime64 columns to strings
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S.%f")

    # Map every cell to a SQL-safe scalar
    for c in df.columns:
        df[c] = df[c].map(_cell_to_sql_scalar)

    return df


def update_status(task_id: str, step: str, progress: int, result: str = ""):
    """
    Update the pipeline status in the database.
    """
    conn = sqlite3.connect(STATUS_DB, timeout=5)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO pipeline_status (task_id, step, progress, result)
        VALUES (?, ?, ?, ?)
    """,
        (task_id, step, progress, result),
    )
    conn.commit()
    conn.close()


def get_status_from_db(task_id: str):
    """
    Retrieve the pipeline status from the database.
    """
    conn = sqlite3.connect(STATUS_DB, timeout=5)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT step, progress, result FROM pipeline_status WHERE task_id = ?",
        (task_id,),
    )
    status = cursor.fetchone()
    conn.close()
    return (
        {"step": status[0], "progress": status[1], "result": status[2]}
        if status
        else None
    )


# Example: define a placeholder for DB interactions
def store_step_output(task_id: str, step_name: str, data: pd.DataFrame):
    """
    Save data to the database after each pipeline step.
    """
    # Use a separate DB for bulky results to reduce contention with STATUS_DB
    engine = create_engine(
        f"sqlite:///{RESULTS_DB}",
        echo=False,
        future=True,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    # Put results DB in WAL as well
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        conn.exec_driver_sql("PRAGMA busy_timeout=5000;")

    data = sanitize_for_sqlite(data)
    data.to_sql(f"{task_id}_{step_name}", con=engine,
                if_exists="replace", index=False)  # type: ignore
    engine.dispose()


@app.post("/cancel/{task_id}")
def cancel_task(task_id: str):
    """
    Request cancellation of a running task by ID. Sends SIGTERM to the task subprocess only.
    """
    current = get_status_from_db(task_id)
    if not current:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    update_status(task_id, "Cancelling", int(
        current.get("progress") or 0), current.get("result") or "")
    requested = request_cancel(task_id)
    _kill_task_process(task_id, force=False)
    return {"ok": bool(requested), "message": "Cancellation requested."}


@app.post("/kill/{task_id}")
def force_kill_task(task_id: str):
    """
    Force kill the task's subprocess (SIGKILL). Use if normal cancel doesn't work.
    """
    current = get_status_from_db(task_id)
    if not current:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    request_cancel(task_id)
    killed = _kill_task_process(task_id, force=True)
    update_status(task_id, "Cancelled", 100,
                  "Force-killed by user" if killed else "Kill requested")
    return {"ok": bool(killed), "message": "Force kill sent."}
