import pandas as pd
import json
import os
from pathlib import Path

def link_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Links rows in the DataFrame by populating the 'linked_to' column,
    based on entity_mappings.json and date logic.

    Args:
        data (pd.DataFrame): Must contain 'record_id', 'patient_id',
                             'core_variable', 'date_ref' (and optionally 'date_ref.1'), 'value'.

    Returns:
        pd.DataFrame: Copy of data with 'linked_to' filled where possible.
    """
    # 1) Load mappings
    # --- locate the JSON file relative to this script file ---
    script_dir   = Path(__file__).resolve().parent
    mapping_path = script_dir / "entity_mappings.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Could not find mappings at {mapping_path}")
    with open(mapping_path, "r") as f:
        entity_mappings = json.load(f)

    # 2) Check required columns
    for col in ("record_id", "patient_id", "core_variable", "value"):
        if col not in data.columns:
            raise ValueError(f"Data must contain '{col}' column.")

    # 3) Work on a copy; drop rows with no 'value'
    df_orig = data.copy()
    df = df_orig.dropna(subset=["value"]).copy()

    # 4) Extract entity name and parse dates
    df['entity']     = df['core_variable'].str.split('.').str[0]
    df['date_ref']   = pd.to_datetime(df['date_ref'],   errors='coerce')

    result = None

    # loop each row
    for i, row in df.iterrows():
        # 5) Check if the entity is in the mappings
        entity = row['entity']
        if entity not in entity_mappings:
            print(f"Entity '{entity}' not found in mappings. Skipping row {i}.")
            continue
        # 6) Check if the date_ref is valid
        if pd.isna(row['date_ref']):
            print(f"Invalid date_ref for row {i}. Skipping.")
            continue
        # identify target entities
        targets = entity_mappings[entity]
        if not targets:
            print(f"No target entities found for '{entity}'. Skipping row {i}.")
            continue
        # look for the target entities with the latest date_ref and same patient_id
        mask = (
            (df['patient_id'] == row['patient_id']) &
            (df['entity'].isin(targets)) &
            (df['date_ref'] <= row['date_ref'])
        )
        cands = df.loc[mask]
        if cands.empty:
            print(f"No candidates found for row {i}. Skipping.")
            continue
        # pick the record with the max date_ref
        latest_idx = cands['date_ref'].idxmax()
        # assign the linked record_id to the original DataFrame
        df.at[i, 'linked_to'] = df.at[latest_idx, 'record_id']
        # print(f"Row {i} linked to record {df.at[latest_idx, 'record_id']}.")
    
    # 7) Return the original DataFrame with the 'linked_to' column
    return df
    # fallback date column if present
    # df['date_ref.1'] = pd.to_datetime(df.get('date_ref.1', pd.NaT), errors='coerce')
    # # choose whichever exists
    # df['eff_date']   = df['date_ref'].fillna(df['date_ref.1'])

    # # 6) Prepare sorted lookup frame
    # df_lookup = df.sort_values(['patient_id', 'eff_date']).copy()
    # print(df)

    # # 7) Define per-row linker
    # def _find_linked(r):
    #     targets = entity_mappings.get(r['entity'])
    #     if not targets:
    #         return pd.NA
    #     if isinstance(targets, str):
    #         targets = [targets]
    #     mask = (
    #         (df_lookup['patient_id'] == r['patient_id']) &
    #         (df_lookup['entity'].isin(targets)) &
    #         (df_lookup['eff_date'] <= r['eff_date'])
    #     )
    #     cands = df_lookup.loc[mask]
    #     if cands.empty:
    #         return pd.NA
    #     # pick the record with the max eff_date
    #     latest_idx = cands['eff_date'].idxmax()
    #     return df_lookup.at[latest_idx, 'record_id']

    # # 8) Apply and write back into a fresh 'linked_to' column
    # result = df_orig.copy()
    # result['linked_to'] = pd.NA
    # # only assign for rows that survived the dropna
    # linked_vals = df.apply(_find_linked, axis=1)
    # result.loc[df.index, 'linked_to'] = linked_vals.values

    # return result


if __name__ == "__main__":
    # quick smoke‐test
    sample_data = pd.DataFrame({
        "record_id": [1,2,3,4],
        "patient_id": ["A","A","B","B"],
        "core_variable": ["X","Y","X","Z"],
        "date_ref": ["2025-01-01","2025-01-02","2025-01-01","2025-01-03"],
        "value":       [10,      20,      30,       40]
    })
    # linked = link_rows(sample_data, linking_criteria={"by_date": True})

    # load dataframe from file
    data = pd.read_excel(
        "/home/zulaika/IDEA4RC-NLP-Ingestion/test_data/MSCI_DEMO_NT_V2_CLEAN.xlsx",
    )
    # 9) Run the function
    linked = link_rows(data)
    print("###" * 10)
    print(linked)
    # print linked to column
    print(linked["linked_to"])
    # save the file to downloads
    linked.to_excel(
        "/home/zulaika/IDEA4RC-NLP-Ingestion/test_data/MSCI_DEMO_NT_V2_CLEAN_LINKED_TEST.xlsx",
        index=False,
    )
