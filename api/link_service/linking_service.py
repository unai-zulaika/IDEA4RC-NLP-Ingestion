import pandas as pd
import json
import os
from pathlib import Path

def link_rows(data: pd.DataFrame, linking_criteria: dict = None) -> pd.DataFrame:
    """
    Links rows in the DataFrame by populating the 'linked_to' column,
    based on entity_mappings.json and date logic.

    Args:
        data (pd.DataFrame): Must contain 'record_id', 'patient_id',
                             'core_variable', 'date_ref' (and optionally 'date_ref.1'), 'value'.
        linking_criteria (dict): Optional; supports:
            - by_date: if True, will sort by date before linking (our logic always uses dates).
            - core_variable: if provided, filters to only that variable before linking.

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
    # fallback date column if present
    df['date_ref.1'] = pd.to_datetime(df.get('date_ref.1', pd.NaT), errors='coerce')
    # choose whichever exists
    df['eff_date']   = df['date_ref'].fillna(df['date_ref.1'])

    # 5) Optional: filter/sort by linking_criteria
    if linking_criteria:
        if linking_criteria.get("core_variable"):
            df = df[df['core_variable'] == linking_criteria['core_variable']].copy()
        # 'by_date' is implicit in eff_date logic; no extra action needed.

    # 6) Prepare sorted lookup frame
    df_lookup = df.sort_values(['patient_id', 'eff_date']).copy()

    # 7) Define per-row linker
    def _find_linked(r):
        targets = entity_mappings.get(r['entity'])
        if not targets:
            return pd.NA
        if isinstance(targets, str):
            targets = [targets]
        mask = (
            (df_lookup['patient_id'] == r['patient_id']) &
            (df_lookup['entity'].isin(targets)) &
            (df_lookup['eff_date'] <= r['eff_date'])
        )
        cands = df_lookup.loc[mask]
        if cands.empty:
            return pd.NA
        # pick the record with the max eff_date
        latest_idx = cands['eff_date'].idxmax()
        return df_lookup.at[latest_idx, 'record_id']

    # 8) Apply and write back into a fresh 'linked_to' column
    result = df_orig.copy()
    result['linked_to'] = pd.NA
    # only assign for rows that survived the dropna
    linked_vals = df.apply(_find_linked, axis=1)
    result.loc[df.index, 'linked_to'] = linked_vals.values

    return result


if __name__ == "__main__":
    # quick smoke‐test
    sample_data = pd.DataFrame({
        "record_id": [1,2,3,4],
        "patient_id": ["A","A","B","B"],
        "core_variable": ["X","Y","X","Z"],
        "date_ref": ["2025-01-01","2025-01-02","2025-01-01","2025-01-03"],
        "value":       [10,      20,      30,       40]
    })
    linked = link_rows(sample_data, linking_criteria={"by_date": True})
    print(linked)
