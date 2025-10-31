import streamlit as st
import requests
import time
import os

st.set_page_config(page_title='IDEA4RC data ingestion',
                   page_icon="LogoIDEA4RC-300x200.jpg", layout="wide")

# Persistent store for last task across reruns/refreshes (server-side)


@st.cache_resource
def _persistent_store():
    return {"last_task_id": None}


_store = _persistent_store()

st.title("IDEA4RC Data Ingestion")

ETL_HOST = os.getenv("ETL_HOST", "localhost:4001")
RESULTS_UI_HOST = os.getenv("RESULTS_UI_HOST", "localhost:5173")

# Define API host early so it's available to the "Last run" block
mode = os.getenv("API_HOST", "localhost:8000")


def _api(path: str) -> str:
    return f"http://{mode}{path}"


st.write(
    """
    In this application you can find different options to run the data ingestion pipeline for the IDEA4RC project.\n
    The pipeline consists of three main steps:\n
    1. **Text Processing**: Extracts and processes free text data from patient records.\n
    2. **Data Linking**: Links the processed text data with structured data from the database.\n
    3. **Quality Checks**: Performs quality checks on the linked data to ensure accuracy and completeness.\n
    You can run the full pipeline or individual steps as needed.\n
    The pipeline is designed to be flexible and can handle various data formats and structures.\n
    Please select one of the options below to start the process.

    If you are in Point of Injection 1, please use the _OPTION 1_ below.\n
    If you are in Point of Injection 2, please use the _OPTION 2_ or _OPTION 3_ below (depending on your needs).\n

    You can also run the ETL process directly using _OPTION 4_ if you have already processed the data and just need to upload it.\n
    """
)

# --- Last run quick controls (always visible) ---
st.markdown("### Last run")
_last_task_id = st.session_state.get("task_id") or _store.get("last_task_id")

if _last_task_id:
    cols = st.columns([3, 2, 2, 2])
    with cols[0]:
        st.write(f"Task ID: {_last_task_id}")
        try:
            resp = requests.get(_api(f"/status/{_last_task_id}"), timeout=5)
            if resp.status_code == 200:
                s = resp.json()
                st.write(f"Step: {s['step']} | Progress: {s['progress']}%")
            else:
                st.warning(resp.json().get("detail", "Status unavailable"))
        except Exception as e:
            st.warning(f"Status error. Backend {mode} not reachable: {e}")
    with cols[1]:
        if st.button("Refresh Status", key="refresh_last_status"):
            pass  # triggers a rerun
    with cols[2]:
        if st.button("Cancel", key="cancel_last"):
            try:
                r = requests.post(_api(f"/cancel/{_last_task_id}"), timeout=5)
                if r.status_code == 200:
                    st.success(r.json().get(
                        "message", "Cancellation requested."))
                else:
                    st.error(r.json().get("detail", "Cancel failed."))
            except Exception as e:
                st.error(f"Cancel failed. Backend {mode} not reachable: {e}")
    with cols[3]:
        if st.button("Force Kill", key="kill_last"):
            try:
                r = requests.post(_api(f"/kill/{_last_task_id}"), timeout=5)
                if r.status_code == 200:
                    st.success(r.json().get("message", "Force kill sent."))
                else:
                    st.error(r.json().get("detail", "Force kill failed."))
            except Exception as e:
                st.error(
                    f"Force kill failed. Backend {mode} not reachable: {e}")
else:
    st.info("No previous run found. Start a task to see it here.")

st.title("_OPTION 1_ :blue[Full process]")

st.write(
    """
    _Upload the normalized table and the free texts, then click on 'Start pipeline' to start the process._\n
    This option will run the full pipeline, including text processing, data linking, and quality checks.\n
    You can check the status of the pipeline and download the results at any time.\n
    If you want to run only the quality checks, please use the option below.\n
    If you want to run only the ETL, please use the option below.\n
    THIS IS THE RECOMMENDED OPTION FOR Point of Injection 1.
    """
)

# Initialize session state
if "task_id" not in st.session_state:
    st.session_state.task_id = None

# Disease type selector for full pipeline
disease_type_full = st.selectbox(
    "Select disease type:",
    options=["sarcoma", "head_and_neck"],
    key="disease_type_full"
)

# File upload section
uploaded_excel = st.file_uploader(
    "Upload a file with the structured data (Excel or CSV)", type=["xlsx", "csv"], key="excel_uploader"
)
uploaded_text = st.file_uploader(
    "Upload a file with the patients text (Excel or CSV)", type=["xlsx", "csv"], key="text_uploader"
)

# Always render the button; disable until files are present
start_clicked = st.button(
    "Start Pipeline",
    key="btn_start_pipeline",
    disabled=not (uploaded_excel and uploaded_text),
)
if start_clicked:
    # Prepare files for API
    data_mime = "text/csv" if uploaded_excel.name.endswith(
        '.csv') else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    text_mime = "text/csv" if uploaded_text.name.endswith(
        '.csv') else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    files = {
        "data_file": (
            uploaded_excel.name,
            uploaded_excel.getvalue(),
            data_mime,
        ),
        "text_file": (
            uploaded_text.name,
            uploaded_text.getvalue(),
            text_mime,
        ),
    }
    response = requests.post(
        _api("/pipeline"),
        files=files,
        params={"disease_type": disease_type_full},
        timeout=30
    )
    if response.status_code == 200:
        st.session_state.task_id = response.json()["task_id"]
        _store["last_task_id"] = st.session_state.task_id  # persist last
        st.success(f"Pipeline started! Task ID: {st.session_state.task_id}")
        st.info("Save this Task ID to check the status later.")
    else:
        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

# Show a cancel button for the current (most recently started) task if any
if st.session_state.task_id:
    with st.expander("Current Task Controls", expanded=True):
        st.write(f"Current Task ID: {st.session_state.task_id}")
        cols = st.columns(2)
        with cols[0]:
            if st.button("Cancel This Task", key="cancel_current_pipeline"):
                try:
                    resp = requests.post(
                        _api(f"/cancel/{st.session_state.task_id}"), timeout=5)
                    if resp.status_code == 200:
                        st.success(resp.json().get(
                            "message", "Cancellation requested."))
                    else:
                        st.error(resp.json().get(
                            "detail", "Failed to request cancellation."))
                except Exception as e:
                    st.error(
                        f"Cancellation call failed. Backend {mode} not reachable: {e}")
        with cols[1]:
            if st.button("Force Kill Task", key="kill_current_pipeline"):
                try:
                    resp = requests.post(
                        _api(f"/kill/{st.session_state.task_id}"), timeout=5)
                    if resp.status_code == 200:
                        st.success(resp.json().get(
                            "message", "Force kill sent."))
                    else:
                        st.error(resp.json().get(
                            "detail", "Failed to force kill."))
                except Exception as e:
                    st.error(
                        f"Force kill call failed. Backend {mode} not reachable: {e}")

# ─── OPTION 2 : linking service only ───────────────────────────────────────────
st.divider()
st.title("_OPTION 2_ :blue[Run the linking service]")

# Disease type selector for linking
disease_type_link = st.selectbox(
    "Select disease type:",
    options=["sarcoma", "head_and_neck"],
    key="disease_type_link"
)

uploaded_linking_file = st.file_uploader(
    "Upload a file to run the linking service (Excel or CSV)",
    type=["xlsx", "csv"],
    key="linking_file_uploader",
)

link_btn = st.button(
    "Execute Linking Service",
    key="btn_linking",
    disabled=uploaded_linking_file is None,
)
if link_btn:
    link_mime = "text/csv" if uploaded_linking_file.name.endswith(
        '.csv') else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    files = {
        "file": (
            uploaded_linking_file.name,
            uploaded_linking_file.getvalue(),
            link_mime,
        )
    }
    resp = requests.post(
        _api("/run/link_rows"),
        files=files,
        params={"disease_type": disease_type_link},
        timeout=30
    )

    if resp.status_code != 200:
        st.error(resp.json().get("detail", "Failed to start linking service."))
        st.stop()

    task_id = resp.json()["task_id"]
    st.session_state.task_id = task_id  # remember for cancel controls
    _store["last_task_id"] = task_id    # persist last
    st.info(f"Task started (ID {task_id}). This usually finishes in 5-30 s…")

    # simple polling loop with a progress bar
    bar = st.progress(0, text="Linking rows…")
    step = ""
    while step not in ("Completed", "Failed", "Cancelled"):
        time.sleep(1)
        status = requests.get(_api(f"/status/{task_id}"), timeout=5).json()
        bar.progress(status["progress"], text=status["step"])
        step = status["step"]

    bar.empty()

    if step == "Failed":
        st.error(f"Linking failed: {status.get('result', 'no message')}")
        st.stop()

    result = requests.get(_api(f"/results/{task_id}/linked_data"), timeout=30)
    if result.status_code != 200:
        st.error(
            f"Finished but couldn't fetch result: "
            f"{result.json().get('detail', 'Unknown error')}"
        )
        st.stop()

    st.success("Linking service completed – download the CSV below:")
    st.download_button(
        label="Download Linking Service Result (CSV)",
        data=result.content,
        file_name=f"{task_id}_linked_data.csv",
        mime="text/csv",
    )

# ─── OPTION 3 : just quality checks ────────────────────────────────────────────
st.divider()
st.title("_OPTION 3_ :blue[Just quality checks]")

st.write("### Run Quality Check on a File")

# Disease type selector for quality check
disease_type_qc = st.selectbox(
    "Select disease type:",
    options=["sarcoma", "head_and_neck"],
    key="disease_type_qc"
)

uploaded_qc_file = st.file_uploader(
    "Upload a file to run quality check (Excel or CSV)", type=["xlsx", "csv"], key="qc_file_uploader"
)

qc_btn = st.button(
    "Execute Quality Check",
    key="btn_qc",
    disabled=uploaded_qc_file is None,
)
if qc_btn:
    qc_mime = "text/csv" if uploaded_qc_file.name.endswith(
        '.csv') else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    files = {
        "file": (
            uploaded_qc_file.name,
            uploaded_qc_file.getvalue(),
            qc_mime,
        )
    }
    resp = requests.post(
        _api("/run/quality_check"),
        files=files,
        params={"disease_type": disease_type_qc},
        timeout=30
    )

    if resp.status_code != 200:
        st.error(resp.json().get(
            "detail", "Failed to start quality check."))
        st.stop()

    task_id = resp.json()["task_id"]
    st.session_state.task_id = task_id
    _store["last_task_id"] = task_id
    st.info(
        f"Quality check started (ID {task_id}). This usually finishes in 30–60 s…")

    bar = st.progress(0, text="Running quality checks…")
    step = ""
    while step not in ("Completed", "Failed", "Cancelled"):
        time.sleep(1)
        status = requests.get(_api(f"/status/{task_id}"), timeout=5).json()
        bar.progress(status["progress"], text=status["step"])
        step = status["step"]

    bar.empty()

    if step == "Failed":
        st.error(f"Quality check failed: {status.get('result', 'no message')}")
        st.stop()

    result = requests.get(
        _api(f"/results/{task_id}/quality_check"), timeout=30)
    if result.status_code != 200:
        st.error(
            f"Finished but couldn't fetch result: "
            f"{result.json().get('detail', 'Unknown error')}"
        )
        st.stop()

    st.success("Quality check completed – download the CSV below:")
    st.download_button(
        label="Download Quality Check Result (CSV)",
        data=result.content,
        file_name=f"{task_id}_quality_check.csv",
        mime="text/csv",
    )

# SEE data quality

# Initialize state only once
if "data_quality_report" not in st.session_state:
    st.session_state.data_quality_report = False

# Button to toggle visibility
if st.button("See Data Quality Report"):
    st.session_state.data_quality_report = not st.session_state.data_quality_report

# Show report if toggled
if st.session_state.data_quality_report:
    st.markdown("## Data Report")
    st.components.v1.html(
        f"""
        <iframe src="http://{RESULTS_UI_HOST}/" width="100%" height="1000" style="border:none;"></iframe>
        """,
        height=1000
    )

# ─── OPTION 4 : ETL ───────────────────────────────────────────────────────────
st.divider()
st.title("_OPTION 4_ :blue[Run ETL]")

st.write("### Run ETL on a File")

# Disease type selector for ETL
disease_type_etl = st.selectbox(
    "Select disease type:",
    options=["sarcoma", "head_and_neck"],
    key="disease_type_etl"
)

uploaded_etl_file = st.file_uploader(
    "Upload a file to run ETL", type=["xlsx", "csv"], key="etl_file_uploader"
)

etl_btn = st.button(
    "Execute ETL",
    key="btn_etl",
    disabled=uploaded_etl_file is None,
)
if etl_btn:
    files = {
        "dataFile": (
            uploaded_etl_file.name,
            uploaded_etl_file.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    try:
        # Add disease_type parameter to ETL upload
        upload_response = requests.post(
            f"http://{ETL_HOST}/etl/upload",
            files=files,
            params={"disease_type": disease_type_etl}
        )
        if upload_response.status_code == 200:
            st.success("Final file successfully uploaded!")
        else:
            st.error(
                f"Upload failed with status code {upload_response.status_code}")
    except Exception as e:
        st.error(f"Upload failed: {e}")

    response = requests.post(
        f"http://{mode}/results/quality_check",
        files=files,
        params={"disease_type": disease_type_etl}
    )

    if response.status_code == 200:
        st.success("Quality check completed successfully. Download result below:")
        st.download_button(
            label="Download Quality Check Result",
            data=response.content,
            file_name=f"quality_check_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")


# PIPELINE STATUS
st.divider()

# Check pipeline status section
st.write("### Check Pipeline Status")
task_id_input = st.text_input("Enter Task ID to check status:")

if st.button("Check Status"):
    if task_id_input:
        response = requests.get(f"http://{mode}/status/{task_id_input}")
        if response.status_code == 200:
            status = response.json()
            st.write(f"Step: {status['step']}")
            st.write(f"Progress: {status['progress']}%")
            if status["result"]:
                st.write(f"Result: {status['result']}")

            # New: stop-running button
            if status.get("is_running", False):
                if st.button("Stop this task"):
                    stop_resp = requests.post(
                        f"http://{mode}/cancel/{task_id_input}")
                    if stop_resp.status_code == 200:
                        st.info(stop_resp.json().get(
                            "message", "Cancellation requested."))
                    else:
                        try:
                            st.error(stop_resp.json().get(
                                "detail", "Unable to cancel task"))
                        except Exception:
                            st.error("Unable to cancel task")

            # Step 0: LLM annotations (raw)
            response_data_0 = requests.get(
                _api(f"/results/{task_id_input}/llm_annotations"), timeout=30)
            if response_data_0.status_code == 200:
                st.download_button(
                    label="Download LLM Annotations (Raw CSV)",
                    data=response_data_0.content,
                    file_name=f"{task_id_input}_llm_annotations.csv",
                    mime="text/csv",
                )
            else:
                st.info("LLM annotations not available yet (only for full pipeline)")

            # Step 1: Processed Texts (after regex extraction)
            response_data_1 = requests.get(
                _api(f"/results/{task_id_input}/processed_texts"), timeout=30)
            if response_data_1.status_code == 200:
                st.download_button(
                    label="Download CSV step 1 (Processed Texts)",
                    data=response_data_1.content,
                    file_name=f"{task_id_input}_processed_texts.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"Error fetching file 1: {response_data_1.text}")

            response_data_2 = requests.get(
                _api(f"/results/{task_id_input}/linked_data"), timeout=30)
            if response_data_2.status_code == 200:
                st.download_button(
                    label="Download CSV step 2 (Linked Data)",
                    data=response_data_2.content,
                    file_name=f"{task_id_input}_linked_data.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"Error fetching file 2: {response_data_2.text}")

            response_data_3 = requests.get(
                _api(f"/results/{task_id_input}/quality_check"), timeout=30)
            if response_data_3.status_code == 200:
                st.download_button(
                    label="Download CSV step 3 (Quality Check)",
                    data=response_data_3.content,
                    file_name=f"{task_id_input}_quality_check.csv",
                    mime="text/csv",
                )

                # Upload option if pipeline is completed
                if status["step"] == "Completed":
                    st.write(
                        "✅ Pipeline finished. You can now upload the final file.")
                    if st.button("Send Final File to Upload Endpoint"):
                        try:
                            files = {
                                'dataFile': (
                                    f'{task_id_input}_quality_check.csv',
                                    response_data_3.content,
                                    'text/csv'
                                )
                            }
                            upload_response = requests.post(
                                f"http://{ETL_HOST}/etl/upload", files=files)
                            if upload_response.status_code == 200:
                                st.success("Final file successfully uploaded!")
                            else:
                                st.error(
                                    f"Upload failed with status code {upload_response.status_code}")
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
            else:
                st.error(f"Error fetching file 3: {response_data_3.text}")
        else:
            st.error(
                f"Error: {response.json().get('detail', 'Unknown error')}")
    else:
        st.warning("Please enter a valid Task ID.")

st.divider()

# Fetch logs section
st.write("### View Logs")
task_id_logs = st.text_input("Enter Task ID to view logs:")

if st.button("Fetch Logs"):
    if task_id_logs:
        try:
            response = requests.get(_api(f"/logs/{task_id_logs}"), timeout=10)
            if response.status_code == 200:
                logs = response.json()["logs"]
                st.write("### Logs:")
                for log in logs:
                    st.write(
                        f"[{log['timestamp']}] {log['level']}: {log['message']}")
            else:
                st.error(
                    f"Error: {response.json().get('detail', 'Task not found')}")
        except Exception as e:
            st.error(f"Logs fetch failed. Backend {mode} not reachable: {e}")
    else:
        st.warning("Please enter a valid Task ID.")
