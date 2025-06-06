""" _summary_ """

from transformers import pipeline  # type: ignore  # Example NLP library
import pandas
import re
from pathlib import Path
import json

types_map = {
    "ENUM": "CodeableConcept",
    "NUMBER": "float",
    "DATE": "date in the ISO format ISO8601  https://en.wikipedia.org/wiki/ISO_8601 ",
    "TEXT": "string",
    "DEFAULT": "CodeableConcept",
}


def process_texts(texts: pandas.DataFrame, excel_data: pandas.DataFrame) -> pandas.DataFrame:
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
    patient_id = 3 # this should come from the texts

    # we load dictionary for LLM output
    # Get the absolute path of the current script
    script_dir = Path(__file__).resolve().parent


    summary_dictionary_path = script_dir / "sarcoma_dictionary.json"
    # values_dictionary_path = script_dir / "sarcoma_dictionary_values.json"
    regexp_dictionary_path = script_dir / "sarcoma_dictionary_regexp.json"

    # Load regex patterns from JSON
    with summary_dictionary_path.open("r", encoding="utf-8") as file:
        summary_dict: dict[str, str] = json.load(file)
    # with values_dictionary_path.open("r", encoding="utf-8") as file:
    #     values_dict: dict[str, str] = json.load(file)
    with regexp_dictionary_path.open("r", encoding="utf-8") as file:
        regexp_dict: dict[str, str] = json.load(file)

    new_rows = []
    # get the biggest record_id from excel_data
    if not excel_data.empty:
        max_record_id = excel_data["record_id"].max()
    else:
        max_record_id = 0



    # loop texts dataframe
    for index, row in texts.iterrows():
        # Extract the text and date from the row
        text = row["text"]
        date = row["date"]

        # look for matches in the text using the regex patterns
        identified_values: dict[str, list[str]] = {}
        for annotation_id, annotation_data in summary_dict.items():
            # Get the regex pattern for the current annotation
            pattern = regexp_dict.get(annotation_id, "")
            if not pattern:
                continue
            # Find all matches in the text
            matches: list[str] = re.findall(pattern, text)
            if matches:
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
                    # create an empty dataframe
                    existing_rows = pandas.DataFrame(columns=excel_data.columns)

                    print(f"Processing match: {match} for annotation: {annotation_id}")

                    # get the entity by the associated variable if associated_variable is present
                    if "associated_variable" in parameter_data:
                        entity = parameter_data["associated_variable"].split(".")[0]
                        # look for other rows in excel_data with the same entity exactly until the ""." and date
                        existing_rows = excel_data[
                            # (excel_data["core_variable"].str.startswith(entity)) &
                            (excel_data["core_variable"].str.split(".").str[0] == entity) &
                            (excel_data["date_ref"] == date) &
                            (excel_data["patient_id"] == patient_id)
                        ]
                    # If there are existing rows, we can use the first match
                    if not existing_rows.empty:
                        # get the record id of the first existing row
                        record_id = existing_rows.iloc[0]["record_id"]
                    else:
                        # If no existing rows, use the max_record_id and increment it
                        record_id = max_record_id
                        max_record_id += 1


                    value = match
                    if parameter_data.get("parameter_type") == "ENUM":
                        print(f"Processing ENUM type for match: {match}")
                        # find the corresponding value in possible_values array
                        possible_values = parameter_data.get("possible_values", [])
                        for pos_val in possible_values:
                            if match in pos_val:
                                # If the match is in the possible values, use it
                                value = pos_val[match]
                                print(f"Found value: {value} for match: {match}")
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
                        print(f"Appending new row for match: {match} with value: {value}")

                        new_rows.append({
                            "core_variable": parameter_data["associated_variable"],
                            "value": value,
                            "patient_id": patient_id,
                            "original_source": "NLP",
                            "date_ref": date,  # Use the date from the row
                            "record_id": record_id,  # Placeholder for record_id
                            "types": types_map[parameter_data["parameter_type"]] if "parameter_type" in parameter_data else "NOT SPECIFIED",  # Use the type from the dictionary
                        })
    
    # now add the new rows to the excel_data DataFrame
    if new_rows:
        new_rows_df = pandas.DataFrame(new_rows)
        # Ensure the DataFrame has the same columns as excel_data
        for col in excel_data.columns:
            if col not in new_rows_df.columns:
                new_rows_df[col] = None
        # Append the new rows to the existing DataFrame
        excel_data = pandas.concat([excel_data, new_rows_df], ignore_index=True)
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
    patient_id = 3 # this should come from the texts

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
