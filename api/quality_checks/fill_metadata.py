import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def read_table(path: str, **pd_kwargs) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, **pd_kwargs)
    if ext == ".csv":
        return pd.read_csv(path, **pd_kwargs)
    raise ValueError(f"Unsupported file type: {ext}")


def varid_to_core(var_id: str) -> str:
    # Turn "Entity_variableName" → "Entity.variableName"
    if "_" in var_id:
        ent, rest = var_id.split("_", 1)
        return f"{ent}.{rest}"
    return var_id.replace("_", ".", 1)


def non_empty(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return False
    return True


def compute_center_availability(df_center: pd.DataFrame) -> Dict[str, str]:
    def availability_for(prefix: str) -> str:
        rows = df_center[df_center["core_variable"].str.startswith(
            prefix, na=False)]
        if rows.empty:
            return "Not available"
        has_any_value = rows["value"].apply(non_empty).any()
        return "True" if has_any_value else "False"

    return {
        "availability_d": availability_for("Diagnosis."),
        "availability_p": availability_for("PathologicalStage."),
        "availability_r": availability_for("Radiotherapy."),
    }


def compute_years(df_center: pd.DataFrame) -> str:
    mask = df_center["core_variable"] == "Diagnosis.dateOfDiagnosis"
    if not mask.any():
        return "Unknown"
    yrs = (
        pd.to_datetime(df_center.loc[mask, "value"], errors="coerce")
        .dropna()
        .dt.year
        .astype("Int64")
        .dropna()
    )
    if yrs.empty:
        return "Unknown"
    return f"{int(yrs.min())}-{int(yrs.max())}"


def fill_for_center(
    metadata_items: List[Dict[str, Any]],
    df_long: pd.DataFrame,
    center_code: str,
    datasource_name: str,
    datasource_info: str,
) -> List[Dict[str, Any]]:
    # Filter to center; if original_source missing, assume all rows belong to this center
    if "original_source" in df_long.columns:
        df_center = df_long[df_long["original_source"].astype(
            str) == center_code].copy()
    else:
        df_center = df_long.copy()

    # Ensure required columns exist
    for c in ("patient_id", "core_variable", "value"):
        if c not in df_center.columns:
            df_center[c] = ""

    df_center["patient_id"] = df_center["patient_id"].astype(str)
    df_center["core_variable"] = df_center["core_variable"].astype(str)

    n_cases = int(df_center["patient_id"].nunique()
                  ) if not df_center.empty else 0
    availability = compute_center_availability(df_center)
    years_str = compute_years(df_center)

    def build_center_block(var_core: str) -> Dict[str, Any]:
        # patients with any non-empty value for var_core
        rows = df_center[df_center["core_variable"] == var_core]
        p_with_value = set(
            rows.loc[rows["value"].apply(non_empty), "patient_id"].unique()
        )
        m_cases = max(0, n_cases - len(p_with_value))
        overall_score = int(
            round(((n_cases - m_cases) / n_cases) * 100)) if n_cases > 0 else 0

        # NEW: pick datasource_name as the first non-empty original_source for this variable
        ds_name = "DEFAULT"
        if "original_source" in rows.columns and not rows.empty:
            first_src = (
                rows["original_source"]
                .astype(str)
                .str.strip()
                .replace({"nan": "", "None": "", "none": "", "null": ""})
            )
            first_src = first_src[first_src != ""]
            if not first_src.empty:
                ds_name = first_src.iloc[0]

        return {
            "availability_d": availability["availability_d"],
            "availability_p": availability["availability_p"],
            "availability_r": availability["availability_r"],
            "years": years_str,
            "datasource_name": ds_name,  # was: datasource_name
            "datasource_information": datasource_info,
            "n_cases": n_cases,
            "m_cases": m_cases,
            "plausability_score": "Does not apply",
            "overall_score": overall_score,
        }

    # Update centers array for each variable
    for item in metadata_items:
        var_id = item.get("variable_id", "")
        core_name = varid_to_core(var_id)
        center_block = build_center_block(core_name)

        centers = item.get("centers", [])
        # centers is a list of dicts like [{ "INT": {...} }, { "ISS-FJD": {...} }]
        updated = False
        for idx, c in enumerate(centers):
            if isinstance(c, dict) and center_code in c:
                centers[idx][center_code] = center_block
                updated = True
                break
        if not updated:
            centers.append({center_code: center_block})
        item["centers"] = centers

    return metadata_items


def main():
    # Inputs
    center = os.getenv("CENTER_NAME", "INT")
    datasource_name = os.getenv(
        "METADATA_DATASOURCE_NAME", "National Registry")
    datasource_info = os.getenv(
        "METADATA_DATASOURCE_INFORMATION", "lorem ipsum")

    # Paths
    here = Path(__file__).parent
    metadata_path = os.getenv(
        "EMPTY_METADATA_PATH",
        str(here / "empty_metadata.json"),
    )
    data_file = os.getenv("LONG_DATA_PATH")  # required

    if not data_file or not Path(data_file).exists():
        raise SystemExit(
            "Set LONG_DATA_PATH to the long-format data file (csv/xlsx).")

    # Load data
    df = read_table(data_file, dtype=str)

    # Load metadata skeleton
    with open(metadata_path, encoding="utf-8") as fh:
        metadata = json.load(fh)

    # Compute and fill
    filled = fill_for_center(
        metadata_items=metadata,
        df_long=df,
        center_code=center,
        datasource_name=datasource_name,
        datasource_info=datasource_info,
    )

    # Output
    out_path = os.getenv(
        "FILLED_METADATA_PATH",
        str(Path(metadata_path).with_name(f"filled_metadata_{center}.json")),
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(filled, fh, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
