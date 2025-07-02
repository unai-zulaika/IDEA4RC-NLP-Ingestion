import streamlit as st
import requests
import time

st.set_page_config(page_title='IDEA4RC data ingestion', page_icon="LogoIDEA4RC-300x200.jpg", layout="wide")
# st.set_page_config(page_title='IDEA4RC data ingestion', page_icon=favicon) # , page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')

st.title("IDEA4RC Data Ingestion")

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

# mode = "localhost"
mode = "backend"

# Initialize session state
if "task_id" not in st.session_state:
    st.session_state.task_id = None

# File upload section
uploaded_excel = st.file_uploader(
    "Upload an Excel file with the structured data", type=["xlsx"], key="excel_uploader"
)
uploaded_text = st.file_uploader(
    "Upload an Excel file with the patients text", type=["xlsx"], key="text_uploader"
)

if uploaded_excel and uploaded_text:
    if st.button("Start Pipeline"):
        # Prepare files for API
        files = {
            "data_file": (
                "data.xlsx",
                uploaded_excel.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            "text_file": ("text.xlsx",
                uploaded_text.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",),
        }
        # Start the pipeline
        response = requests.post(f"http://{mode}:8000/pipeline", files=files)
        if response.status_code == 200:
            st.session_state.task_id = response.json()["task_id"]
            st.success(f"Pipeline started! Task ID: {st.session_state.task_id}")
            st.info("Save this Task ID to check the status later.")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")


#### ONLY FROM LINKING ONWARDS

# ─── OPTION 2 : linking service only ───────────────────────────────────────────

st.divider()
st.title("_OPTION 2_ :blue[Run the linking service]")

uploaded_linking_file = st.file_uploader(
    "Upload a file to run the linking service",
    type=["xlsx"],
    key="linking_file_uploader",
)

if uploaded_linking_file and st.button("Execute Linking Service"):
    # kick off the background job
    files = {
        "file": (
            uploaded_linking_file.name,
            uploaded_linking_file.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    resp = requests.post(f"http://{mode}:8000/run/link_rows", files=files)

    if resp.status_code != 200:
        st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
        st.stop()

    task_id = resp.json()["task_id"]
    st.info(f"Task started (ID {task_id}). This usually finishes in 5-30 s…")

    # simple polling loop with a progress bar
    bar = st.progress(0, text="Linking rows…")
    step = ""
    while step not in ("Completed", "Failed"):
        time.sleep(2)
        status = requests.get(f"http://{mode}:8000/status/{task_id}").json()
        bar.progress(status["progress"], text=status["step"])
        step = status["step"]

    bar.empty()  # remove the progress bar

    if step == "Failed":
        st.error(f"Link-rows job failed: {status.get('result', 'no message')}")
        st.stop()

    # job is done → fetch the real workbook
    result = requests.get(
        f"http://{mode}:8000/results/{task_id}/linked_data"
    )
    if result.status_code != 200:
        st.error(
            f"Finished but couldn’t fetch result: "
            f"{result.json().get('detail', 'Unknown error')}"
        )
        st.stop()

    st.success("Linking service completed – download the workbook below:")
    st.download_button(
        label="Download Linking Service Result",
        data=result.content,
        file_name=f"{task_id}_linked_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )




#### ONLY QUALITY CHECKS

st.divider()
st.title("_OPTION 3_ :blue[Just quality checks]")

st.write("### Run Quality Check on a File")

uploaded_qc_file = st.file_uploader(
    "Upload a file to run quality check", type=["xlsx"], key="qc_file_uploader"
)

if uploaded_qc_file and st.button("Execute Quality Check"):
    files = {
        "file": (
            uploaded_qc_file.name,
            uploaded_qc_file.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    response = requests.post(f"http://{mode}:8000/run/quality_check", files=files)

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

#### SEE data quality

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
        """
        <iframe src="http://localhost:5173/" width="100%" height="1000" style="border:none;"></iframe>
        """,
        height=1000
    )

#### SKIP ALL STEPS AND RUN ETL

st.divider()
st.title("_OPTION 4_ :blue[Run ETL]")

st.write("### Run ETL on a File")

uploaded_etl_file = st.file_uploader(
    "Upload a file to run ETL", type=["xlsx"], key="etl_file_uploader"
)

if uploaded_etl_file and st.button("Execute ETL"):
    files = {
        "dataFile": (
            uploaded_etl_file.name,
            uploaded_etl_file.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    try:
        upload_response = requests.post("http://localhost:4001/etl/upload", files=files)
        if upload_response.status_code == 200:
            st.success("Final file successfully uploaded!")
        else:
            st.error(f"Upload failed with status code {upload_response.status_code}")
    except Exception as e:
        st.error(f"Upload failed: {e}")

    response = requests.post(f"http://{mode}:8000/results/quality_check", files=files)

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

#### PIPELINE STATUS

st.divider()

# Check pipeline status section
st.write("### Check Pipeline Status")
task_id_input = st.text_input("Enter Task ID to check status:")

if st.button("Check Status"):
    if task_id_input:
        response = requests.get(f"http://{mode}:8000/status/{task_id_input}")
        if response.status_code == 200:
            status = response.json()
            st.write(f"Step: {status['step']}")
            st.write(f"Progress: {status['progress']}%")
            if status["result"]:
                st.write(f"Result: {status['result']}")

            response_data_1 = requests.get(
                f"http://{mode}:8000/results/{task_id_input}/processed_texts"
            )
            if response_data_1.status_code == 200:
                # Provide the file content for download
                st.download_button(
                    label="Download Excel step 1",
                    data=response_data_1.content,
                    file_name=f"{task_id_input}_processed_texts.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.error(
                    f"Error fetching file 1: {response.json().get('detail', 'Unknown error')}"
                )

            response_data_2 = requests.get(
                f"http://{mode}:8000/results/{task_id_input}/linked_data"
            )
            if response_data_2.status_code == 200:
                # Provide the file content for download
                st.download_button(
                    label="Download Excel step 2",
                    data=response_data_2.content,
                    file_name=f"{task_id_input}_linked_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.error(
                    f"Error fetching file 2: {response.json().get('detail', 'Unknown error')}"
                )

            response_data_3 = requests.get(
                f"http://{mode}:8000/results/{task_id_input}/quality_check"
            )
            if response_data_3.status_code == 200:
                st.download_button(
                    label="Download Excel step 3",
                    data=response_data_3.content,
                    file_name=f"{task_id_input}_quality_check.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # Upload option if pipeline is completed
                if status["step"] == "Completed":
                    st.write("✅ Pipeline finished. You can now upload the final file.")
                    if st.button("Send Final File to Upload Endpoint"):
                        try:
                            # Use an in-memory bytes object from response content
                            files = {
                                'dataFile': (
                                    f'{task_id_input}_quality_check.xlsx',
                                    response_data_3.content,
                                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                )
                            }
                            upload_response = requests.post("http://localhost:4001/etl/upload", files=files)
                            if upload_response.status_code == 200:
                                st.success("Final file successfully uploaded!")
                            else:
                                st.error(f"Upload failed with status code {upload_response.status_code}")
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
            else:
                st.error(
                    f"Error fetching file 3: {response.json().get('detail', 'Unknown error')}"
                )

    else:
        st.warning("Please enter a valid Task ID.")

st.divider()

# Fetch logs section
st.write("### View Logs")
task_id_logs = st.text_input("Enter Task ID to view logs:")

if st.button("Fetch Logs"):
    if task_id_logs:
        response = requests.get(f"http://{mode}:8000/logs/{task_id_logs}")
        if response.status_code == 200:
            logs = response.json()["logs"]
            st.write("### Logs:")
            for log in logs:
                st.write(f"[{log['timestamp']}] {log['level']}: {log['message']}")
        else:
            st.error(f"Error: {response.json().get('detail', 'Task not found')}")
    else:
        st.warning("Please enter a valid Task ID.")
