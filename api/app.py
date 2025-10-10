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

# Keep status/logs separate from large result tables to avoid locks
STATUS_DB = "pipeline_status.db"
RESULTS_DB = "pipeline_results.db"


app = FastAPI()

# Add CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


async def run_quality_check_task(task_id: str, data_file: bytes):
    """
    Background job: run quality_check on a single uploaded spreadsheet.
    Stores output so it can be fetched later via /results/{task_id}/quality_check.
    """
    task_logger = get_task_logger(task_id)
    try:
        update_status(task_id, "Initializing", 0)
        task_logger.info("Quality-check task initialised.")

        # Load the uploaded file (off main loop)
        update_status(task_id, "Loading data", 10)
        # type: ignore
        df: pd.DataFrame = await asyncio.to_thread(pd.read_excel, io.BytesIO(data_file))
        task_logger.info("File read into DataFrame (shape=%s).", df.shape)

        # Run quality check (off main loop)
        update_status(task_id, "Running quality check", 60)
        task_logger.info("Invoking quality_check() on DataFrame.")
        final_df: pd.DataFrame | None = None

        try:
            # prefer DF input
            qc_result = await asyncio.to_thread(quality_check, df)
        except TypeError:
            task_logger.info(
                "quality_check(df) incompatible. Falling back to file path.")

            def _run_qc_on_tmp(_df: pd.DataFrame):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, "input.xlsx")
                    _df.to_excel(tmp_path, index=False)
                    return quality_check(tmp_path)
            qc_result = await asyncio.to_thread(_run_qc_on_tmp, df)

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
    except Exception as exc:
        update_status(task_id, "Failed", 100, str(exc))
        task_logger.error("Quality-check task failed: %s", exc)
        raise


@app.post("/run/quality_check")
async def quality_check_call(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    task_id = str(uuid.uuid4())
    data_content = await file.read()

    # Start the task in the background
    background_tasks.add_task(run_quality_check_task, task_id, data_content)

    return {
        "task_id": task_id,
        "message": "quality_check started – poll /status/{task_id} for progress.",
    }


@app.get("/results/{task_id}/{step_name}")
def get_step_results_as_excel(task_id: str, step_name: str):
    """
    Fetch pipeline step data from the database and return as an Excel file.
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

        # Write the DataFrame to an Excel file in memory
        excel_data = io.BytesIO()
        with pd.ExcelWriter(excel_data, engine="openpyxl") as writer:
            df.to_excel(writer, index=False,
                        sheet_name="Results")  # type: ignore
        excel_data.seek(0)

        # Return the Excel file as a streaming response
        return StreamingResponse(
            excel_data,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{task_id}_{step_name}.xlsx"'
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if conn is not None:
            conn.close()


async def run_link_rows_task(task_id: str, data_file: bytes):
    """
    Background job: run link_rows on a single uploaded spreadsheet.
    The output is stored with store_step_output so it can be fetched later via
    /results/{task_id}/linked_data.
    """
    task_logger = get_task_logger(task_id)

    try:
        update_status(task_id, "Initializing", 0)
        task_logger.info("Link-row task initialised.")

        update_status(task_id, "Loading data", 10)
        df: pd.DataFrame = await asyncio.to_thread(pd.read_excel, io.BytesIO(data_file))  # noqa
        task_logger.info("File read into DataFrame (shape=%s).", df.shape)

        update_status(task_id, "Linking rows", 60)
        linked_df: pd.DataFrame = await asyncio.to_thread(link_rows, df)  # noqa
        task_logger.info("link_rows complete (shape=%s).", linked_df.shape)

        await asyncio.to_thread(store_step_output, task_id, "linked_data", linked_df)

        update_status(task_id, "Completed", 100, "Link-rows finished.")
        task_logger.info("Link-row task completed successfully.")
    except Exception as exc:
        update_status(task_id, "Failed", 100, str(exc))
        task_logger.error("Link-row task failed: %s", exc)
        raise


@app.post("/run/link_rows")
async def link_rows_call(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Accepts a single Excel/CSV, runs only link_rows, and returns a task_id.
    Use /status/{task_id} to track progress and
    /results/{task_id}/linked_data to download the output Excel.
    """
    task_id = str(uuid.uuid4())
    data_content = await file.read()

    background_tasks.add_task(run_link_rows_task, task_id, data_content)

    return {
        "task_id": task_id,
        "message": "link_rows started – poll /status/{task_id} for progress.",
    }


async def run_pipeline_task(task_id: str, data_file: bytes, text_file: bytes):
    """Function to run the pipeline task asynchronously.

    Args:
        task_id (str): _description_
        data_file (bytes): _description_
        text_file (bytes): _description_
    """
    task_logger = get_task_logger(task_id)

    try:
        update_status(task_id, "Initializing", 0)
        task_logger.info("Pipeline initialization started.")

        update_status(task_id, "Loading data", 10)
        task_logger.info("Loading data from uploaded files.")
        # type: ignore
        excel_data: pd.DataFrame = await asyncio.to_thread(pd.read_excel, io.BytesIO(data_file))
        # type: ignore
        free_texts: pd.DataFrame = await asyncio.to_thread(pd.read_excel, io.BytesIO(text_file))

        # Step 2: Process free texts (off main loop)
        update_status(task_id, "Processing free texts", 30)
        task_logger.info("Processing free texts and structuring data.")
        structured_data = await asyncio.to_thread(process_texts, free_texts, excel_data)
        await asyncio.to_thread(store_step_output, task_id, "processed_texts", structured_data)

        # Step 3: Link rows
        update_status(task_id, "Linking rows", 60)
        task_logger.info("Linking rows based on criteria.")
        linked_data = await asyncio.to_thread(link_rows, structured_data)
        await asyncio.to_thread(store_step_output, task_id, "linked_data", linked_data)

        # Step 4: Data quality check
        update_status(task_id, "Performing data quality checks", 90)
        task_logger.info("Performing data quality checks.")
        await asyncio.to_thread(quality_check, linked_data)
        final_data = linked_data
        await asyncio.to_thread(store_step_output, task_id, "quality_check", final_data)

        update_status(task_id, "Completed", 100,
                      result="Pipeline completed successfully!")
        task_logger.info("Pipeline completed successfully.")
    except Exception as e:
        update_status(task_id, "Failed", 100, result=str(e))
        task_logger.error("Pipeline failed with error: %s", e)
        raise


@app.post("/pipeline")
async def start_pipeline(
    data_file: UploadFile = File(...),
    text_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Starts the pipeline"""
    task_id = str(uuid.uuid4())
    data_content = await data_file.read()
    text_content = await text_file.read()

    # Start the pipeline in the background
    background_tasks.add_task(
        run_pipeline_task, task_id, data_content, text_content)

    return {
        "task_id": task_id,
        "message": "Pipeline started! Use this task_id to check the status later.",
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
    conn = sqlite3.connect("pipeline_status.db")
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
    logs = [
        {"timestamp": row[0], "level": row[1], "message": row[2]}
        for row in cursor.fetchall()
    ]
    conn.close()

    if not logs:
        raise HTTPException(
            status_code=404, detail="No logs found for the specified task."
        )
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
