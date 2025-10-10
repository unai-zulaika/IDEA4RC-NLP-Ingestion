""" 
NLP Text Processing Module

This module provides functions for processing medical texts using LLMs and regex-based extraction.

MAIN WORKFLOW (process_texts):
================================
Input: Clinical notes DataFrame (text, date, p_id, note_id, report_type)
  ↓
Step 1: LLM Annotation (process_texts_with_llm)
  - Run LLM with multiple prompts on each text
  - Generate structured annotations
  - Parse multiple values from single annotation
  ↓
Step 2: Regex Pattern Matching
  - Apply regex patterns to LLM annotations
  - Extract structured values using sarcoma_dictionary patterns
  ↓
Step 3: Data Integration
  - Map to core_variables
  - Handle patient_id propagation
  - Use extracted dates or fall back to note dates
  - Manage record_ids for entity grouping
  ↓
Output: Structured excel_data DataFrame with extracted entities

Key Features:
- process_texts: Main pipeline function (LLM → Regex → Structured Data)
- process_texts_with_llm: LLM annotation with multiple value extraction
- llm_annotations_to_excel_format: Direct LLM-to-Excel conversion (alternative path)
- parse_annotation_values: Smart parsing of multiple values from annotations

The LLM processor can parse multiple values from annotations in various formats:
- JSON arrays: ["value1", "value2", "value3"]
- Comma-separated: value1, value2, value3
- Semicolon-separated: value1; value2; value3
- Newline-separated lists (with or without bullets/numbers)

Date Handling:
- Primary: Dates extracted by LLM from annotation text
- Fallback: Original note date if extraction fails

Patient ID: Carried through entire pipeline from input to output

See PROCESSING_WORKFLOW.md for detailed documentation.
"""

from transformers import pipeline  # type: ignore  # Example NLP library
import pandas
import re
from pathlib import Path
import json
from typing import List, Dict
from nlp.model_runner import load_prompts_from_json, get_prompt, run_model_with_prompt, init_model


types_map = {
    "ENUM": "CodeableConcept",
    "NUMBER": "float",
    "DATE": "date in the ISO format ISO8601  https://en.wikipedia.org/wiki/ISO_8601 ",
    "TEXT": "string",
    "DEFAULT": "CodeableConcept",
}


def process_texts_with_llm(
    texts: pandas.DataFrame,
    excel_data: pandas.DataFrame,
    model_path: str = None,
    prompts_json_path: str = None,
    fewshots_dict: Dict[str, List[tuple]] = None,
    parse_multiple_values: bool = True,
    task_id: str = None,
    is_cancelled_callback=None
) -> pandas.DataFrame:
    """
    Process texts row by row using the LLM for each prompt type.

    Args:
        texts (pandas.DataFrame): DataFrame with columns: text, date, p_id, note_id, report_type
        excel_data (pandas.DataFrame): Existing Excel data (not used but kept for compatibility)
        model_path (str): Path to the LLM model file. If None, uses default path.
        prompts_json_path (str): Path to prompts.json. If None, uses default path.
        fewshots_dict (Dict[str, List[tuple]]): Dictionary mapping prompt_type to list of (text, annotation) tuples.
                                                 If None, uses empty fewshots.
        parse_multiple_values (bool): If True, attempts to parse multiple values from a single annotation.
                                      Supports JSON arrays, comma-separated, semicolon-separated, and newline-separated values.
                                      Each value will create a separate row in the output.
        task_id (str): Optional task ID for cancellation checking
        is_cancelled_callback (callable): Optional callback function to check if task is cancelled

    Returns:
        pandas.DataFrame: DataFrame with columns: prompt_type, annotation, p_id, date, note_id, report_type, raw_output

    Raises:
        RuntimeError: If task is cancelled during processing
    """

    # Set default paths
    script_dir = Path(__file__).resolve().parent

    if prompts_json_path is None:
        prompts_json_path = script_dir / "prompts.json"

    if model_path is None:
        model_path = script_dir / "meta-llama-3.1-8b-instruct-q4_k_m.gguf"

    # Initialize model and load prompts
    print("[INFO] Loading prompts...")
    load_prompts_from_json(prompts_json_path)

    print("[INFO] Initializing model...")
    init_model(str(model_path))

    # Get all available prompt types
    import json
    with open(prompts_json_path, "r", encoding="utf-8") as f:
        prompts_config = json.load(f)

    prompt_types = list(prompts_config.keys())
    print(f"[INFO] Found {len(prompt_types)} prompt types: {prompt_types}")

    # Initialize fewshots if not provided
    if fewshots_dict is None:
        fewshots_dict = {pt: [] for pt in prompt_types}

    # Prepare results list
    results = []

    # Process each row in texts
    total_rows = len(texts)
    for idx, row in texts.iterrows():
        # Check for cancellation at the start of each text processing
        if is_cancelled_callback and task_id and is_cancelled_callback(task_id):
            print(
                f"[WARN] Task {task_id} cancelled during text processing at row {idx + 1}/{total_rows}")
            raise RuntimeError(f"Task {task_id} was cancelled by user")

        text = row.get("text", "")
        date = row.get("date", "")
        p_id = row.get("p_id", "")
        note_id = row.get("note_id", "")
        report_type = row.get("report_type", "")

        print(
            f"[INFO] Processing row {idx + 1}/{total_rows} - Patient: {p_id}, Note: {note_id}")

        # Run model for each prompt type
        for prompt_type in prompt_types:
            print(f"  - Running prompt type: {prompt_type}")

            # Get fewshots for this prompt type
            fewshots = fewshots_dict.get(prompt_type, [])

            # Build the prompt
            prompt = get_prompt(
                task_key=prompt_type,
                fewshots=fewshots,
                note_text=text
            )

            # Run the model
            try:
                output = run_model_with_prompt(
                    prompt=prompt,
                    max_new_tokens=256,
                    temperature=0.3
                )

                annotation = output["normalized"]
                raw_output = output["raw"]

                # Parse annotation to extract multiple values if enabled
                if parse_multiple_values:
                    annotation_values = parse_annotation_values(annotation)
                else:
                    annotation_values = [annotation]

                # Create a row for each extracted value
                for annotation_value in annotation_values:
                    # Extract dates from annotation if present (optional enhancement)
                    extracted_dates = extract_dates_from_annotation(
                        annotation_value)

                    # Append result
                    results.append({
                        "prompt_type": prompt_type,
                        "annotation": annotation_value,
                        "p_id": p_id,
                        "date": date,
                        "note_id": note_id,
                        "report_type": report_type,
                        "raw_output": raw_output,
                        "extracted_dates": extracted_dates
                    })

            except Exception as e:
                print(
                    f"    [ERROR] Failed to process prompt type {prompt_type}: {e}")
                results.append({
                    "prompt_type": prompt_type,
                    "annotation": f"ERROR: {str(e)}",
                    "p_id": p_id,
                    "date": date,
                    "note_id": note_id,
                    "report_type": report_type,
                    "raw_output": "",
                    "extracted_dates": []
                })

    # Create DataFrame from results
    results_df = pandas.DataFrame(results)

    print(
        f"[INFO] Processing complete. Generated {len(results_df)} annotations.")
    return results_df


def parse_annotation_values(annotation: str) -> List[str]:
    """
    Parse annotation to extract multiple values if present.
    Supports multiple formats:
    - JSON arrays: ["value1", "value2"]
    - Comma-separated: value1, value2, value3
    - Semicolon-separated: value1; value2; value3
    - Newline-separated: value1\nvalue2\nvalue3
    - Bullet points: - value1\n- value2
    - Numbered lists: 1. value1\n2. value2

    Args:
        annotation (str): The annotation text to parse

    Returns:
        List[str]: List of extracted values (single value if no multiple values detected)
    """
    if not annotation or annotation.startswith("ERROR:"):
        return [annotation]

    # Remove leading/trailing whitespace
    annotation = annotation.strip()

    # Try to parse as JSON array
    if annotation.startswith("[") and annotation.endswith("]"):
        try:
            import json
            values = json.loads(annotation)
            if isinstance(values, list):
                # Filter out empty values and convert to strings
                return [str(v).strip() for v in values if v and str(v).strip()]
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to detect and parse delimited values
    # Check for semicolon-separated (higher priority than comma for medical text)
    if ";" in annotation and annotation.count(";") >= 1:
        values = [v.strip() for v in annotation.split(";") if v.strip()]
        if len(values) > 1:
            return values

    # Check for comma-separated (but avoid single sentences with commas)
    if "," in annotation:
        values = [v.strip() for v in annotation.split(",") if v.strip()]
        # Only treat as list if we have 2+ values and none are too long (likely sentences)
        if len(values) > 1 and all(len(v) < 100 for v in values):
            return values

    # Check for newline-separated with bullet points or numbers
    lines = annotation.split("\n")
    if len(lines) > 1:
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove bullet points and numbering
            line = re.sub(r"^[-*•]\s+", "", line)  # Bullets
            line = re.sub(r"^\d+[\.)]\s+", "", line)  # Numbers
            if line:
                cleaned_lines.append(line)

        if len(cleaned_lines) > 1:
            return cleaned_lines

    # Check for pipe-separated
    if "|" in annotation:
        values = [v.strip() for v in annotation.split("|") if v.strip()]
        if len(values) > 1:
            return values

    # No multiple values detected, return as single value
    return [annotation]


def extract_dates_from_annotation(annotation: str) -> List[str]:
    """
    Extract dates from annotation text.
    Supports formats: DD/MM/YYYY, YYYY-MM-DD, and other common date patterns.

    Args:
        annotation (str): The annotation text to extract dates from

    Returns:
        List[str]: List of extracted date strings
    """
    date_patterns = [
        r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{1,2}/\d{1,2}/\d{4}',  # D/M/YYYY or DD/M/YYYY
    ]

    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, annotation)
        dates.extend(matches)

    return dates


def process_texts(
    texts: pandas.DataFrame,
    excel_data: pandas.DataFrame,
    model_path: str = None,
    prompts_json_path: str = None,
    fewshots_dict: Dict[str, List[tuple]] = None,
    task_id: str = None,
    is_cancelled_callback=None
) -> pandas.DataFrame:
    """
    Main function to process texts using LLM and extract structured data.

    Workflow:
    1. Call process_texts_with_llm to get LLM annotations for each text
    2. Use regex patterns to parse the structured annotations
    3. Extract values and insert into excel_data DataFrame

    Args:
        texts (pandas.DataFrame): DataFrame with columns: text, date, p_id, note_id, report_type
        excel_data (pandas.DataFrame): Existing Excel data to append to
        model_path (str): Path to the LLM model file. If None, uses default path.
        prompts_json_path (str): Path to prompts.json. If None, uses default path.
        fewshots_dict (Dict[str, List[tuple]]): Dictionary mapping prompt_type to list of (text, annotation) tuples.
        task_id (str): Optional task ID for cancellation checking
        is_cancelled_callback (callable): Optional callback function to check if task is cancelled

    Returns:
        pandas.DataFrame: Excel data with structured data integrated

    Raises:
        RuntimeError: If task is cancelled during processing
    """

    print("[INFO] Starting text processing pipeline...")

    # Step 1: Get LLM annotations for all texts
    print("[INFO] Step 1: Running LLM to annotate texts...")
    llm_results = process_texts_with_llm(
        texts=texts,
        excel_data=excel_data,
        model_path=model_path,
        prompts_json_path=prompts_json_path,
        fewshots_dict=fewshots_dict,
        parse_multiple_values=True,  # Enable multi-value extraction
        task_id=task_id,
        is_cancelled_callback=is_cancelled_callback
    )

    print(f"[INFO] LLM generated {len(llm_results)} annotations")

    # Step 2: Load regex dictionaries for parsing structured annotations
    print("[INFO] Step 2: Loading regex dictionaries for parsing annotations...")
    script_dir = Path(__file__).resolve().parent
    summary_dictionary_path = script_dir / "sarcoma_dictionary.json"
    regexp_dictionary_path = script_dir / "sarcoma_dictionary_regexp.json"

    # Load regex patterns from JSON
    with summary_dictionary_path.open("r", encoding="utf-8") as file:
        summary_dict: dict[str, str] = json.load(file)
    with regexp_dictionary_path.open("r", encoding="utf-8") as file:
        regexp_dict: dict[str, str] = json.load(file)

    # Step 3: Process LLM annotations and extract structured data
    print("[INFO] Step 3: Parsing LLM annotations with regex patterns...")
    new_rows = []

    # Get the biggest record_id from excel_data
    if not excel_data.empty:
        max_record_id = excel_data["record_id"].max()
    else:
        max_record_id = 0

    # Process each LLM annotation
    annotation_count = 0
    for index, llm_row in llm_results.iterrows():
        # Extract data from LLM results
        annotation_text = llm_row["annotation"]
        prompt_type = llm_row["prompt_type"]
        patient_id = llm_row["p_id"]
        note_date = llm_row["date"]  # Original note date
        note_id = llm_row["note_id"]

        # Try to use extracted dates from annotation, fall back to note date
        extracted_dates = llm_row.get("extracted_dates", [])
        if extracted_dates and len(extracted_dates) > 0:
            date = extracted_dates[0]  # Use first extracted date
            print(f"[INFO] Using extracted date: {date} for annotation")
        else:
            date = note_date  # Fall back to original note date
            print(f"[INFO] No date extracted, using note date: {date}")

        # Skip error annotations
        if annotation_text.startswith("ERROR:"):
            print(f"[WARN] Skipping error annotation: {annotation_text}")
            continue

        # Apply regex patterns to the annotation text
        for annotation_id, annotation_data in summary_dict.items():
            # Get the regex pattern for the current annotation
            pattern = regexp_dict.get(annotation_id, "")
            if not pattern:
                continue

            # Find all matches in the annotation text (not the original text!)
            matches: list[str] = re.findall(pattern, annotation_text)
            if matches:
                annotation_count += 1
                for match_index, match in enumerate(matches):
                    # check if there is parameter_data for the current match
                    if len(annotation_data["parameters"]) <= match_index:
                        continue
                    parameter_data = annotation_data["parameters"][match_index]
                    if not parameter_data:
                        continue
                    # check if associated_variable is present in the parameter_data
                    if "associated_variable" not in parameter_data:
                        continue

                    print(
                        f"[INFO] Processing match: '{match}' for annotation_id: {annotation_id}, patient: {patient_id}")

                    # get the entity by the associated variable if associated_variable is present
                    if "associated_variable" in parameter_data:
                        entity = parameter_data["associated_variable"].split(".")[
                            0]
                        # look for other rows in excel_data with the same entity and date
                        existing_rows = excel_data[
                            (excel_data["core_variable"].str.split(".").str[0] == entity) &
                            (excel_data["date_ref"] == date) &
                            (excel_data["patient_id"] == patient_id)
                        ]
                    else:
                        existing_rows = pandas.DataFrame(
                            columns=excel_data.columns)

                    # If there are existing rows, reuse the record_id
                    if not existing_rows.empty:
                        record_id = existing_rows.iloc[0]["record_id"]
                    else:
                        # If no existing rows, use the max_record_id and increment it
                        record_id = max_record_id
                        max_record_id += 1

                    value = match
                    if parameter_data.get("parameter_type") == "ENUM":
                        print(
                            f"[INFO] Processing ENUM type for match: {match}")
                        # find the corresponding value in possible_values array
                        possible_values = parameter_data.get(
                            "possible_values", [])
                        for pos_val in possible_values:
                            if match in pos_val:
                                # If the match is in the possible values, use it
                                value = pos_val[match]
                                print(
                                    f"[INFO] Found value: {value} for match: {match}")
                                break
                    elif parameter_data.get("parameter_type") == "DEFAULT":
                        value = parameter_data.get("value", match)

                    # only append if there is no existing row with the same core_variable, value, date_ref, and patient_id
                    if existing_rows[
                        (existing_rows["core_variable"] == parameter_data["associated_variable"]) &
                        (existing_rows["value"] == value) &
                        (existing_rows["date_ref"] == date) &
                        (existing_rows["patient_id"] == patient_id)
                    ].empty:
                        # Append the new row to the list
                        print(
                            f"[INFO] Appending new row for match: '{match}' with value: '{value}'")

                        new_rows.append({
                            "core_variable": parameter_data["associated_variable"],
                            "value": value,
                            "patient_id": patient_id,
                            "original_source": "NLP_LLM",
                            "date_ref": date,
                            "record_id": record_id,
                            "types": types_map.get(parameter_data.get("parameter_type", "DEFAULT"), "NOT SPECIFIED"),
                            "note_id": note_id,
                            "prompt_type": prompt_type
                        })
                    else:
                        print(
                            f"[INFO] Skipping duplicate: '{match}' already exists")

    # Step 4: Add new rows to excel_data
    print(f"[INFO] Step 4: Adding {len(new_rows)} new rows to excel_data...")
    if new_rows:
        new_rows_df = pandas.DataFrame(new_rows)
        # Ensure the DataFrame has the same columns as excel_data
        for col in excel_data.columns:
            if col not in new_rows_df.columns:
                new_rows_df[col] = None
        # Append the new rows to the existing DataFrame
        excel_data = pandas.concat(
            [excel_data, new_rows_df], ignore_index=True)

    print(
        f"[INFO] Processing complete! Processed {len(llm_results)} LLM annotations, found {annotation_count} regex matches, added {len(new_rows)} new rows")
    return excel_data


def process_texts_old(texts: pandas.DataFrame, excel_data: pandas.DataFrame) -> pandas.DataFrame:
    """
    Function to process texts and extract structured data from them

    Args:
    - texts (pandas.DataFrame): List of texts to process
    - excel_data (pandas.DataFrame): Excel data to integrate structured data into

    return:
    - excel_data (pandas.DataFrame): Excel data with structured data integrated
    """

    # Initialize an NLP model (e.g., Named Entity Recognition)
    # nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    patient_id = 3  # this should come from the texts

    # we load dictionary for LLM output
    # Get the absolute path of the current script
    script_dir = Path(__file__).resolve().parent

    # Construct the path for the JSON file
    file_path = script_dir / "output_regex_dict.json"

    # Load regex patterns from JSON
    with file_path.open("r", encoding="utf-8") as file:
        patterns: dict[str, str] = json.load(file)

    # Define regex patterns for each type of information
    # Extract values using regex from JSON
    identified_values: dict["str", list["str"]] = (
        {}
    )  # we can find a list of values for each variable
    dates = []

    # Example regex patterns
    # loop texts dataframe
    for index, row in texts.iterrows():
        for variable_name, pattern in patterns.items():
            print(pattern)
            matches: list[str] = re.findall(pattern, row["text"])
            if matches:
                identified_values[variable_name] = matches
                # if "date" column is not empty, we can add it to the dates list
                if row["date"] != "":
                    dates.append(row["date"])

    # Ensure DataFrame has the expected structure before appending
    if excel_data.empty:
        excel_data = pd.DataFrame(columns=["core_variable", "value"])

    # Append rows safely
    z = 0
    for variable_name, values in identified_values.items():
        for i, value in enumerate(values):
            row = {
                "core_variable": variable_name,
                "value": value,
                "patient_id": patient_id,
                "original_source": "NLP",
                "date_ref": dates[z],  # Placeholder for date
                # types must be solved.
                # date must be solved
            }
            z += 1
            # Fill in any missing columns explicitly
            for col in excel_data.columns:
                if col not in row:
                    row[col] = None
            excel_data.loc[len(excel_data)] = row

    return excel_data


def llm_annotations_to_excel_format(
    llm_results: pandas.DataFrame,
    excel_data: pandas.DataFrame
) -> pandas.DataFrame:
    """
    Convert LLM annotations to the excel_data format.

    Args:
        llm_results (pandas.DataFrame): Output from process_texts_with_llm
        excel_data (pandas.DataFrame): Existing excel data to append to

    Returns:
        pandas.DataFrame: Updated excel_data with LLM annotations
    """

    # Get the biggest record_id from excel_data
    if not excel_data.empty:
        max_record_id = excel_data["record_id"].max()
    else:
        max_record_id = 0

    new_rows = []

    for index, row in llm_results.iterrows():
        prompt_type = row["prompt_type"]
        annotation = row["annotation"]
        p_id = row["p_id"]
        date = row["date"]

        # Skip error rows
        if annotation.startswith("ERROR:"):
            continue

        # Map prompt_type to core_variable
        # You'll need to define this mapping based on your prompts.json
        core_variable = f"llm.{prompt_type}"

        # Check for existing records
        existing_rows = excel_data[
            (excel_data["core_variable"] == core_variable) &
            (excel_data["date_ref"] == date) &
            (excel_data["patient_id"] == p_id)
        ]

        if existing_rows.empty:
            record_id = max_record_id
            max_record_id += 1
        else:
            record_id = existing_rows.iloc[0]["record_id"]

        # Only append if not duplicate
        if existing_rows[
            (existing_rows["value"] == annotation)
        ].empty:
            new_rows.append({
                "core_variable": core_variable,
                "value": annotation,
                "patient_id": p_id,
                "original_source": "NLP_LLM",
                "date_ref": date,
                "record_id": record_id,
                "types": "string",  # or map based on prompt_type
            })

    # Append new rows
    if new_rows:
        new_rows_df = pandas.DataFrame(new_rows)
        for col in excel_data.columns:
            if col not in new_rows_df.columns:
                new_rows_df[col] = None
        excel_data = pandas.concat(
            [excel_data, new_rows_df], ignore_index=True)

    return excel_data


if __name__ == "__main__":
    # Example usage demonstrating the full pipeline
    print("=" * 80)
    print("EXAMPLE: Full NLP Processing Pipeline")
    print("=" * 80)

    # Sample clinical notes
    sample_texts = pandas.DataFrame([
        {
            "text": "Patient has a history of diabetes mellitus type 2 diagnosed on 2022-05-15. "
                    "Current medications include metformin 500mg twice daily.",
            "date": "2023-10-01",
            "p_id": 1,
            "note_id": 101,
            "report_type": "clinical_note"
        },
        {
            "text": "Patient presents with hypertension. Blood pressure 145/90. "
                    "Started on lisinopril 10mg daily. Follow-up scheduled for 2023-11-15.",
            "date": "2023-10-02",
            "p_id": 2,
            "note_id": 102,
            "report_type": "clinical_note"
        }
    ])

    # Initialize empty excel data structure
    sample_excel_data = pandas.DataFrame(columns=[
        "core_variable", "value", "patient_id",
        "original_source", "date_ref", "record_id", "types"
    ])

    print("\n[STEP 1] Processing texts with full pipeline (LLM + Regex extraction)...")
    print("-" * 80)

    # Main processing function - handles entire workflow:
    # 1. Runs LLM to annotate texts
    # 2. Parses LLM annotations with regex
    # 3. Extracts structured data into excel_data format
    updated_excel_data = process_texts(
        texts=sample_texts,
        excel_data=sample_excel_data
    )

    print("\n[RESULT] Updated Excel Data:")
    print("-" * 80)
    print(updated_excel_data)
    print(f"\nTotal rows in excel_data: {len(updated_excel_data)}")

    # Save results
    updated_excel_data.to_csv("updated_excel_data_demo.csv", index=False)
    print("\n[SAVED] Results saved to 'updated_excel_data_demo.csv'")

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)
