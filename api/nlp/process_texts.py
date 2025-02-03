""" _summary_ """

from transformers import pipeline  # type: ignore  # Example NLP library
import pandas
import re
from pathlib import Path
import json


def process_texts(texts: list[str], excel_data: pandas.DataFrame) -> pandas.DataFrame:
    """
    Function to process texts and extract structured data from them

    Args:
    - texts (list): List of texts to process
    - excel_data (pandas.DataFrame): Excel data to integrate structured data into

    return:
    - excel_data (pandas.DataFrame): Excel data with structured data integrated
    """
    # Initialize an NLP model (e.g., Named Entity Recognition)
    # nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    output: str = """
    Patient's BMI is 23.5. The diagnosis was Hypertension. The prescribed medication is Lisinopril.
    """

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
    for variable_name, pattern in patterns.items():
        matches: list[str] = re.findall(pattern, output)
        if matches:
            identified_values[variable_name] = matches

    # now add them to the excel_data
    for variable_name, values in identified_values.items():
        for value in values:
            excel_data.loc[len(excel_data)] = {
                "core_variable": variable_name,
                "value": value,
            }

    return excel_data
