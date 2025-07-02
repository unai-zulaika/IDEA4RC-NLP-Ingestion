import asyncio
import io
import logging
import sqlite3
import uuid
import tempfile

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from link_service.linking_service import link_rows
from nlp.process_texts import process_texts
from quality_checks.quality_check import quality_check
import os

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
    conn = sqlite3.connect("pipeline_status.db")
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
    conn.commit()
    conn.close()


init_db()


def update_status(task_id: str, step: str, progress: int, result: str = ""):
    """
    Update the pipeline status in the database.
    """
    conn = sqlite3.connect("pipeline_status.db")
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
    conn = sqlite3.connect("pipeline_status.db")
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
    # Insert into DB here. For example:
    # db.insert(table="pipeline_results", dict={"task_id": task_id, "step_name": step_name, "data": data_json})
    conn: sqlite3.Connection = sqlite3.connect("pipeline_status.db")
    data.to_sql(f"{task_id}_{step_name}", con=conn, if_exists="replace", index=False)  # type: ignore
    conn.close()

async def run_quality_check(data_file: bytes):
    """Function to run the quality check task asynchronously.

    Args:
        data_file (bytes): _description_
    """
    # 1) read into DF
    df = pd.read_excel(io.BytesIO(data_file))

    # 2) write to a temporary Excel file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "input.xlsx")
        df.to_excel(tmp_path, index=False)

        # 3) call your core function, passing the path
        quality_check(
            tmp_path,
        )


@app.post("/run/quality_check")
async def quality_check_call(
    file: UploadFile = File(...),
):
    task_id = str(uuid.uuid4())
    data_content = await file.read()

    # Start the pipeline in the background
    await run_quality_check(data_content)

    return {
        "task_id": task_id,
        "message": "Pipeline started! Use this task_id to check the status later.",
    }

@app.get("/results/{task_id}/{step_name}")
def get_step_results_as_excel(task_id: str, step_name: str):
    """
    Fetch pipeline step data from the database and return as an Excel file.
    """
    conn = None
    try:
        conn = sqlite3.connect("pipeline_status.db")
        query = f'SELECT * FROM "{task_id}_{step_name}"'
        df: pd.DataFrame = pd.read_sql(query, conn)  # type: ignore

        if df.empty:
            raise HTTPException(
                status_code=404, detail="No data found for the specified task and step."
            )

        # Write the DataFrame to an Excel file in memory
        excel_data = io.BytesIO()
        with pd.ExcelWriter(excel_data, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")  # type: ignore
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

        # Load the uploaded file
        update_status(task_id, "Loading data", 10)
        df: pd.DataFrame = pd.read_excel(io.BytesIO(data_file))          # noqa
        task_logger.info("File read into DataFrame (shape=%s).", df.shape)

        # Run linking
        update_status(task_id, "Linking rows", 60)
        linked_df: pd.DataFrame = link_rows(df)                          # noqa
        task_logger.info("link_rows complete (shape=%s).", linked_df.shape)

        # Persist result so the existing /results route can serve it
        store_step_output(task_id, "linked_data", linked_df)

        # Mark done
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

        # Step 1: Load data
        update_status(task_id, "Loading data", 10)
        task_logger.info("Loading data from uploaded files.")
        excel_data: pd.DataFrame = pd.read_excel(io.BytesIO(data_file))  # type: ignore
        free_texts: pd.DataFrame = pd.read_excel(io.BytesIO(text_file))  # type: ignore

        await asyncio.sleep(1)  # Simulate processing

        # Step 2: Process free texts
        update_status(task_id, "Processing free texts", 30)
        task_logger.info("Processing free texts and structuring data.")
        structured_data = process_texts(free_texts, excel_data)
        store_step_output(task_id, "processed_texts", structured_data)

        await asyncio.sleep(1)

        # Step 3: Link rows
        update_status(task_id, "Linking rows", 60)
        task_logger.info("Linking rows based on criteria.")
        # linked_data = link_rows(structured_data, linking_criteria={"by_date": True})
        linked_data = link_rows(structured_data)

        store_step_output(task_id, "linked_data", linked_data)

        await asyncio.sleep(1)

        # Step 4: Data quality check
        update_status(task_id, "Performing data quality checks", 90)
        task_logger.info("Performing data quality checks.")
        # final_data, quality_report = quality_check(linked_data)
        quality_check(linked_data)
        final_data = linked_data  
        store_step_output(task_id, "quality_check", final_data)

        # Save result (simulate)
        update_status(
            task_id, "Completed", 100, result="Pipeline completed successfully!"
        )
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
    background_tasks.add_task(run_pipeline_task, task_id, data_content, text_content)

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
    return status


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
    conn = sqlite3.connect("pipeline_status.db")
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
    conn.commit()
    conn.close()


init_logs_db()


def log_message(task_id: str, log_level: str, message: str):
    """
    Save a log message to the database.
    """
    conn = sqlite3.connect("pipeline_status.db")
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
        log_message(self.task_id, record.levelname, record.msg)


# Example: Add DBLogHandler to logger dynamically
def get_task_logger(task_id: str):
    task_logger = logging.getLogger(f"pipeline_{task_id}")
    task_logger.setLevel(logging.DEBUG)
    task_logger.addHandler(DBLogHandler(task_id))
    return task_logger
