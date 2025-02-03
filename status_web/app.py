import streamlit as st
import requests

st.title("IDEA4RC Data Ingestion")

st.write(
    "*Upload the normalized table and the free texts, then click on 'Start pipeline' to start the process.*"
)

mode = "localhost"

# Initialize session state
if "task_id" not in st.session_state:
    st.session_state.task_id = None

# File upload section
uploaded_excel = st.file_uploader(
    "Upload an Excel file", type=["xlsx"], key="excel_uploader"
)
uploaded_text = st.file_uploader(
    "Upload a text file", type=["txt"], key="text_uploader"
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
            "text_file": ("data.txt", uploaded_text.getvalue(), "text/plain"),
        }
        # Start the pipeline
        response = requests.post(f"http://{mode}:8000/pipeline", files=files)
        if response.status_code == 200:
            st.session_state.task_id = response.json()["task_id"]
            st.success(f"Pipeline started! Task ID: {st.session_state.task_id}")
            st.info("Save this Task ID to check the status later.")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

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
                # Provide the file content for download
                st.download_button(
                    label="Download Excel step 3",
                    data=response_data_3.content,
                    file_name=f"{task_id_input}_quality_check.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.error(
                    f"Error fetching file 3: {response.json().get('detail', 'Unknown error')}"
                )
        else:
            st.error(
                f"Error fetching file: {response.json().get('detail', 'Unknown error')}"
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
