import pandas as pd
import json


def link_rows(data: pd.DataFrame, linking_criteria: dict = None) -> pd.DataFrame:
    """
    Links rows in the DataFrame by populating the 'linked_to' column.

    Args:
        data (pd.DataFrame): Input data containing at least 'record_id' and 'patient_id'.
        linking_criteria (dict): Optional criteria for custom linking rules.
                                 Example: {'by_date': True, 'core_variable': 'some_value'}

    Returns:
        pd.DataFrame: Data with the 'linked_to' column filled.
    """

    entity_mappings = {
    }

    # load from JSON file
    with open("entity_mappings.json", "r") as file:
        entity_mappings = json.load(file)

    # Ensure required columns are present
    if "record_id" not in data.columns or "patient_id" not in data.columns:
        raise ValueError("Data must contain 'record_id' and 'patient_id' columns.")

    # first thing filter empty values for value column
    data = data.dropna(subset=["value"])  # do we want to delete?

    # Create a mapping for records grouped by patient_id
    record_groups = data.groupby("patient_id")["record_id"].apply(list).to_dict()

    # Optional: Incorporate additional linking criteria
    if linking_criteria:
        if linking_criteria.get("by_date"):
            data = data.sort_values(
                "date_ref"
            )  # Ensure data is sorted by date if present

        # Example of filtering by a core variable
        core_variable = linking_criteria.get("core_variable")
        if core_variable:
            data = data[data["core_variable"] == core_variable]

    # Fill the linked_to column
    data["linked_to"] = data.apply(
        lambda row: next(
            (
                rec
                for rec in record_groups.get(row["patient_id"], [])
                if rec != row["record_id"]
            ),
            None,
        ),
        axis=1,
    )

    return data


if __name__ == "__main__":
    # For testing the service
    # Example input DataFrame
    sample_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "patient_id": ["A", "A", "B", "B"],
            "core_variable": ["x", "y", "x", "z"],
            "date_ref": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-03"],
        }
    )

    # Run linking service
    linked_data = link_rows(sample_data, linking_criteria={"by_date": True})
    print(linked_data)
