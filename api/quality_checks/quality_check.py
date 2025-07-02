from collections import Counter
import collections
import os
import sys
import great_expectations as gx
import json
import argparse

import pandas as pd
# from custom_expectations.plugins.expectations.expect_colum_pair_to_be_null_if import ExpectColumnPairToBeNullIf
from quality_checks.custom_expectations.plugins.expectations.expect_colum_pair_to_be_null_if import ExpectColumnPairToBeNullIf
from great_expectations_experimental.expectations.expect_column_values_not_to_be_future_date import ExpectColumnValuesNotToBeFutureDate


import io
from pathlib import Path
import pandas as pd
import csv

# At the top of quality_check.py
BASE_DIR = Path(__file__).parent

if os.name == "nt":                      # only needed on Windows
    def utf8_wrapper(stream): return io.TextIOWrapper(
        stream.buffer, encoding="utf-8", errors="replace"
    )
    sys.stdout = utf8_wrapper(sys.stdout)
    sys.stderr = utf8_wrapper(sys.stderr)
    os.environ["PYTHONUTF8"] = "1"       # downstream libs respect this too
# ────────────────────────────────────────────────────────────────────────────────

def read_table(path: str, **pd_kwargs) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame.
    • .csv  →  try pandas' default reader; on ParserError sniff the delimiter
               (comma / semicolon / tab / pipe) and retry.
    • .xls/.xlsx  →  pd.read_excel
    """
    ext = Path(path).suffix.lower()

    # ---------- CSV -------------------------------------------------------
    if ext == ".csv":
        try:
            return pd.read_csv(path, **pd_kwargs)          # fast path
        except pd.errors.ParserError:
            # ── auto-detect separator and fall back to the python engine ──
            with open(path, "r", newline="", encoding=pd_kwargs.get("encoding", "utf-8")) as fh:
                sample = fh.read(4096)                     # first ~4 KB
                delim = csv.Sniffer().sniff(sample, delimiters=[
                    ",", ";", "\t", "|"]).delimiter
            return pd.read_csv(path,
                               delimiter=delim,
                               engine="python",            # more forgiving
                               **pd_kwargs)

    # ---------- Excel -----------------------------------------------------
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, **pd_kwargs)

    # ---------- unsupported ----------------------------------------------
    raise ValueError(f"Unsupported file type for input data: {ext}")

def prepare_repeatable_entity_data(entity_name, original_data_filepath, repeatable_entities_config, app_path):
    """
    Reads the original long-format data, filters for variables of a specific repeatable entity,
    and pivots it to a wide format where each row is a unique instance of that entity.
    Saves the result to a CSV file.

    Parameters:
        entity_name (str): The name of the repeatable entity (e.g., "PatientFollowUp", "Diagnosis").
        original_data_filepath (str): Path to the original long-format data file (e.g., 'synthetic_idea4rc.xlsx').
        repeatable_entities_config (dict): The dictionary under the "repeteables" key from rep_entities.json.
        app_path (str): Path to the application directory for saving the output CSV.

    Returns:
        pd.DataFrame: A wide-format DataFrame with one row per entity instance, or an empty DataFrame on error.
    """
    try:
        # df_source = pd.read_excel(original_data_filepath, dtype=str)
        df_source = read_table(original_data_filepath, dtype=str)
    except FileNotFoundError:
        print(
            f"Error: Original data file not found at {original_data_filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(
            f"Error reading original data file {original_data_filepath}: {e}")
        return pd.DataFrame()

    if entity_name not in repeatable_entities_config:
        print(
            f"Error: Entity '{entity_name}' not found in the 'repeteables' section of the entities configuration.")
        return pd.DataFrame()

    # Get the list of simple variable names for the entity
    variable_short_names = repeatable_entities_config[entity_name]
    if not variable_short_names:
        print(
            f"Warning: No variables listed for entity '{entity_name}' in the configuration.")
        return pd.DataFrame()

    # Construct the full core_variable names (e.g., "PatientFollowUp.statusOfPatientAtLastFollowUp")
    entity_core_variables = [
        f"{entity_name}.{var_name}" for var_name in variable_short_names]

    # Filter for rows that contain the core variables for the specified entity
    df_entity_long = df_source[df_source['core_variable'].isin(
        entity_core_variables)]

    if df_entity_long.empty:
        print(
            f"No data found for entity '{entity_name}' with core variables: {entity_core_variables}")
        return pd.DataFrame()

    # Pivot the table to have one row per patient_id + record_id,
    # and the entity's variables as columns.
    try:
        df_entity_wide = df_entity_long.pivot_table(
            # Ensures each entity instance is a row
            index=['patient_id', 'record_id'],
            columns='core_variable',          # Turns variables into columns
            values='value',                   # Takes the 'value' for each variable
            aggfunc='first'  # Assumes one value per variable within a patient_id/record_id combo
        ).reset_index()
    except KeyError as e:
        print(
            f"Error pivoting data for entity '{entity_name}'. A key column (patient_id, record_id, core_variable, value) might be missing or misnamed: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error pivoting data for entity '{entity_name}': {e}")
        return pd.DataFrame()

    # Save the prepared DataFrame to a CSV file
    output_filename = f"{entity_name}_prepared_data.csv"
    output_filepath = os.path.join(app_path, output_filename)
    try:
        df_entity_wide.to_csv(output_filepath, index=False)
        print(
            f"Successfully prepared and saved data for entity '{entity_name}' to: {output_filepath}")
    except Exception as e:
        print(
            f"Error saving prepared data for entity '{entity_name}' to CSV: {e}")
        return pd.DataFrame()  # Or return df_entity_wide if saving is optional

    return df_entity_wide


def format_normalized_table(filename, formated_filepath, non_repeatable_vars_set):

    # load filename
    # Force the datetime column to be read as a string to avoid conversion issues with great_expectations. GE needs to have the datetime column in str.
    # input_df = pd.read_excel(filename, dtype=str)
    
    # check if filename is a pandas object already
    if isinstance(filename, pd.DataFrame):
        input_df = filename
    else:
        input_df = read_table(filename, dtype=str)

    output_data = []
    patients_data = {}
    # datatype

    # convert rows into columns by patient
    for _, row in input_df.iterrows():
        p_id = row["patient_id"]
        source = row["original_source"]
        if p_id not in patients_data:
            patients_data[p_id] = {"original_source": source}
        core_variable = row["core_variable"]

        # MODIFICATION: Only process if core_variable is in the non_repeatable_vars_set
        if core_variable in non_repeatable_vars_set:
            patients_data[p_id].update({core_variable: row["value"]})
        # else:
            # Optional: print(f"Skipping repeatable/unknown core_variable: {core_variable} for patient {p_id}")

    # now that we have all data for each patient prepare for doing our dataframe
    for key, value in patients_data.items():
        output_row = {"patient_id": key, **value}
        output_data.append(output_row)

    # save as parquet format for keeping datatypes
    output_df = pd.DataFrame(output_data)
    # output_df.to_csv("idea4rc_data.csv")
    output_df.to_excel(formated_filepath)
    # output_df.to_parquet("data_processor/idea4rc_data.parquet.gzip")


def quality_check(data):

    data_file = data
    repeated_entities_file = BASE_DIR  / "repeteable_entities.json"
    # repeated_entities_file = "./repeteable_entities.json"
    suites_file =  BASE_DIR  / "expectations_data.json"
    app_path =  BASE_DIR  / "app/"
    # environment variable app path
    app_path = os.environ.get("APP_PATH", app_path)
    print(f"App path set to: {app_path}")

    # ─────────────────────────────────────────────────────────────────────────────
    # NEW: allow `data_file` to be a JSON list of many patient files
    #      (["/path/PID01.xlsx", "/path/PID02.xlsx", …]).
    #      If a list is detected we read all the files, concatenate them, and
    #      replace data_file with a single temporary Excel so that everything
    #      below keeps working unchanged.
    # ─────────────────────────────────────────────────────────────────────────────

    try:
        candidate = json.loads(data_file)
        if not isinstance(candidate, list):
            raise ValueError            # it was a plain path string
        data_files = candidate          # ✔ we really have a list
    except (json.JSONDecodeError, ValueError):
        data_files = [data_file]   # fall back → single file
    except Exception as e:
        data_files = [data_file]   # fall back → single file

    if len(data_files) > 1:
        print(f"🛈  Combining {len(data_files)} patient files …")
        dfs = []
        for p in data_files:
            try:
                dfs.append(read_table(p, dtype=str))
            except Exception as e:
                print(f"    ⚠︎  Skipping {p}: {e}")
        if not dfs:
            sys.exit("ERROR: none of the provided data files could be read.")

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_path = os.path.join(app_path, "combined_input.xlsx")
        combined_df.to_excel(combined_path, index=False)
        data_file = combined_path      # 👈 every downstream call sees ONE file
        print(f"    → wrote combined cohort to {combined_path}")
    else:
        # single-file workflow – nothing changes
        data_file = data_files[0]
    

    # Get the absolute path to the current file
    current_file_path = os.path.abspath(__file__)
    plugins_directory = os.path.dirname(
        current_file_path) + "\custom_expectations\plugins"

    data_context_config = gx.data_context.types.base.DataContextConfig(
        store_backend_defaults=gx.data_context.types.base.InMemoryStoreBackendDefaults(),
        plugins_directory=plugins_directory
    )
    context = gx.get_context(project_config=data_context_config)


    # --- Load Entity Definitions (rep_entities.json) ---
    with open(repeated_entities_file, encoding="utf8") as f:
        repeated_entities_data = json.load(f)

    all_non_repeatable_core_variables = set()
    non_repeatable_entity_names = set(
        repeated_entities_data.get("non_repeteables", {}).keys())
    repeatable_entity_names = set(
        repeated_entities_data.get("repeteables", {}).keys())

    if "non_repeteables" in repeated_entities_data:
        for entity_name, variables_list in repeated_entities_data["non_repeteables"].items():
            for var_name in variables_list:
                all_non_repeatable_core_variables.add(f"{entity_name}.{var_name}")

    # --- Load minimum-cardinality per entity -----------------------------------
    card_file = os.path.join(os.path.dirname(repeated_entities_file),
                            "entities_cardinality.json")
    try:
        with open(card_file, encoding="utf-8") as fh:
            # e.g.  {"Patient":1, "Surgery":0, ...}
            entity_min_card = json.load(fh)
    except Exception as e:
        print(f"ERROR loading entities_cardinality.json: {e}")
        entity_min_card = {}          # default → treat as 1 (mandatory)


    # --- Prepare and Validate Non-Repeatable Data ---
    print("\n--- Processing Non-Repeatable Data ---")
    FORMATED_FILEPATH_NON_REP = os.path.join(
        app_path, "idea4rc_data_non_repeatable.xlsx")
    format_normalized_table(
        data_file, FORMATED_FILEPATH_NON_REP, all_non_repeatable_core_variables)
    non_repeatable_validator = context.sources.pandas_default.read_excel(
        FORMATED_FILEPATH_NON_REP, asset_name="non_repeatable_data_asset")
    df_non_repeatable = pd.read_excel(FORMATED_FILEPATH_NON_REP)  # Load the DF
    print(f"Validator 'non_repeatable_validator' created for non-repeatable data.")

    # --- Initialize dictionary to hold validators for repeatable entities ---
    repeatable_entity_validators = {}
    repeatable_entity_checkpoint_results = {}

    # --- Load ALL Expectations (from suites_file, which is expectations_data.json) ---
    print(f"\n--- Loading expectations from {suites_file} ---")
    with open(suites_file, encoding="utf8") as f:
        all_expectations_data = json.load(f)  # This is your expectations_data.json

    # Stores DataFrames for repeatable entities for the Quality Summarization
    repeatable_entity_dataframes = {}

    # --- Process and Route Expectations ---
    for item_definition in all_expectations_data:
        if "expectations" not in item_definition or not item_definition["expectations"]:
            continue

        # Get the primary entity context from the item definition itself (e.g., "Patient_sex" from item_definition["col1"])
        # This will be used as a fallback for table-level expectations.
        context_entity_variable_from_col1 = item_definition.get("col1", "")

        for expectation_details in item_definition["expectations"]:
            ge_name = expectation_details["metadata"]["ge_name"]
            args_list_original = expectation_details["metadata"]["args"]
            dimensions = expectation_details["metadata"]["dimensions"]

            current_validator = non_repeatable_validator  # Default validator
            entity_name_for_routing = None

            # Make a deep copy of args_list to modify it for processing
            # without affecting the original expectations_data if it's used elsewhere.
            args_list = json.loads(json.dumps(args_list_original))

            # Standardize column names in args to Entity.Variable format and attempt type conversion
            # This part is from your existing script, with slight modifications for clarity
            # and to ensure 'column' key is handled if it's Entity_Variable format.
            # Will store the 'Entity.Variable' formatted column name
            target_column_for_expectation = None

            for arg_dict in args_list:
                for key, value in arg_dict.items():
                    new_value = value
                    if isinstance(value, str):
                        # If the key is 'column' and value has '_', convert it to Entity.Variable
                        if key == "column" and '_' in value and '.' not in value:  # e.g. "Patient_sex" -> "Patient.sex"
                            parts = value.split('_', 1)
                            if len(parts) == 2:
                                new_value = f"{parts[0]}.{parts[1]}"
                        # Your original replacement for other string values
                        elif key != "column" and isinstance(value, str):
                            new_value = value.replace('_', '.')

                        # Attempt int conversion for values that are all digits or negative digits
                        # Be cautious with this if some string values are meant to be codes that look like numbers
                        if isinstance(new_value, str) and (new_value.isdigit() or (new_value.startswith('-') and new_value[1:].isdigit())):
                            try:
                                new_value = int(new_value)
                            except ValueError:
                                pass  # Keep as string
                    arg_dict[key] = new_value  # Update the copied arg_dict

                    if key == "column":
                        # Store the (potentially converted) column name
                        target_column_for_expectation = arg_dict[key]

            # Determine entity for routing
            if target_column_for_expectation and '.' in target_column_for_expectation:
                entity_name_for_routing = target_column_for_expectation.split('.')[
                    0]
            elif not target_column_for_expectation and context_entity_variable_from_col1:
                # Fallback for table-level expectations: parse col1
                if '_' in context_entity_variable_from_col1 and '.' not in context_entity_variable_from_col1:
                    entity_name_for_routing = context_entity_variable_from_col1.split('_', 1)[
                        0]
                elif '.' in context_entity_variable_from_col1:  # If col1 was already Entity.Variable
                    entity_name_for_routing = context_entity_variable_from_col1.split('.')[
                        0]
                else:  # col1 does not provide clear entity prefix
                    print(
                        f"  Info: Table-level expectation '{ge_name}' from item '{context_entity_variable_from_col1}' does not have a clear entity prefix in col1. Defaulting to non_repeatable_validator.")
            # else: entity_name_for_routing remains None, will default to non_repeatable_validator

            # Select the validator
            if entity_name_for_routing:
                if entity_name_for_routing in non_repeatable_entity_names:
                    current_validator = non_repeatable_validator
                elif entity_name_for_routing in repeatable_entity_names:
                    if entity_name_for_routing not in repeatable_entity_validators:
                        print(
                            f"  Preparing data and validator for new repeatable entity: {entity_name_for_routing}")
                        df_entity = prepare_repeatable_entity_data(
                            entity_name_for_routing,
                            data_file,
                            repeated_entities_data["repeteables"],
                            app_path
                        )
                        if not df_entity.empty:
                            # Create a unique asset name to avoid conflicts
                            asset_name_for_entity = f"{entity_name_for_routing}_data_asset"
                            # We need to keep this for later use for the quality summarization
                            repeatable_entity_dataframes[entity_name_for_routing] = df_entity
                            validator_instance = context.sources.pandas_default.read_dataframe(
                                df_entity, asset_name=asset_name_for_entity)
                            repeatable_entity_validators[entity_name_for_routing] = validator_instance
                            current_validator = validator_instance
                        else:
                            print(
                                f"  Warning: Could not prepare data for repeatable entity {entity_name_for_routing}. Skipping expectation '{ge_name}'.")
                            continue  # Skip this expectation
                    else:
                        current_validator = repeatable_entity_validators[entity_name_for_routing]
                else:
                    print(f"  Warning: Entity '{entity_name_for_routing}' (derived from '{target_column_for_expectation or context_entity_variable_from_col1}') is not defined in non_repeteables or repeteables. Defaulting to non_repeatable_validator for expectation '{ge_name}'.")
                    # current_validator remains non_repeatable_validator
            # else: current_validator remains non_repeatable_validator

            # Apply the expectation
            method = getattr(current_validator, ge_name, None)
            if method:
                combined_kwargs = {k: v for d in args_list for k,
                                v in d.items()}  # Use the modified args_list
                combined_kwargs["meta"] = {"dimensions": dimensions}

                validator_name_for_log = "non_repeatable_validator"
                if current_validator != non_repeatable_validator:
                    for name, val_instance in repeatable_entity_validators.items():
                        if val_instance == current_validator:
                            validator_name_for_log = f"validator_for_{name}"
                            break

                print(
                    f"  Applying to {validator_name_for_log}: {ge_name} with args: {combined_kwargs}")
                # ------------------------------------------------------------------

                #  ⛔ Skip NOT-NULL checks for optional (min-card 0) non-repeatable
                # ------------------------------------------------------------------
                if (ge_name == "expect_column_values_to_not_be_null"
                        and current_validator is non_repeatable_validator):
                    ent = entity_name_for_routing or "UNKNOWN"
                    if entity_min_card.get(ent, 1) == 0:
                        print(f"    Skipping {ge_name} on optional entity '{ent}'")
                        continue
                # ------------------------------------------------------------------

                try:
                    method(**combined_kwargs)
                except Exception as e:
                    print(
                        f"    ERROR applying expectation {ge_name} to {validator_name_for_log} with args {combined_kwargs}: {e}")

            else:
                print(
                    f"  Warning: Method {ge_name} not found in the selected validator ({validator_name_for_log}).")

    # --- Save Suites and Run Checkpoints ---

    # In main.py

    # ... (after the loop where expectations are added to non_repeatable_validator) ...

    # --- Save Suite and Run Checkpoint for Non-Repeatable Data ---
    non_repeatable_suite_name = "non_repeatable_default_suite"  # Define your suite name

    # Set the name on the ExpectationSuite object held by the validator
    if non_repeatable_validator.expectation_suite:
        non_repeatable_validator.expectation_suite.expectation_suite_name = non_repeatable_suite_name
    else:
        print("Aquie entra?")
        # This should ideally not happen if expectations were added.
        # If it's possible to have no expectations, you might need to create an empty named suite.
        # For now, assuming expectations were added, so validator.expectation_suite exists.
        print(
            f"Warning: non_repeatable_validator does not have an active suite. Creating one named '{non_repeatable_suite_name}'.")
        from great_expectations.core import ExpectationSuite
        new_suite = ExpectationSuite(
            expectation_suite_name=non_repeatable_suite_name)
        # If you created a new suite, you'd typically add it to the context first,
        # then get a validator for it. The current flow adds expectations to the validator's default suite.
        # So, the if block above should normally be sufficient.
        # Assigning a new one if it was None
        non_repeatable_validator.expectation_suite = new_suite

    # Save the suite using the validator. It will use the name set on validator.expectation_suite.
    non_repeatable_validator.save_expectation_suite(
        discard_failed_expectations=False)  # Removed expectation_suite_name kwarg
    print(
        f"Expectation suite '{non_repeatable_suite_name}' saved for non-repeatable data.")


    dataframe_asset = context.sources.add_pandas(
        "non_repeatable_validator_checkpoint"
    ).add_dataframe_asset(
        name="non_repeatable_data_asset",
        dataframe=df_non_repeatable,
    )
    batch_request = dataframe_asset.build_batch_request()


    # Checkpoint configuration still uses expectation_suite_name to load the correct suite
    checkpoint_main = context.add_or_update_checkpoint(
        name="non_repeatable_checkpoint",
        batch_request=batch_request,
        # This tells the checkpoint which suite to use
        expectation_suite_name=non_repeatable_suite_name,
        runtime_configuration={
            "result_format": {"result_format": "COMPLETE", "unexpected_index_column_names": ["patient_id", "original_source"], "return_unexpected_index_query": True}
        }
    )

    print(f"--- Debugging non_repeatable_validator before checkpoint run ---")
    if non_repeatable_validator.active_batch:
        print(
            f"Validator has an active batch with ID: {non_repeatable_validator.active_batch.id}")
        if hasattr(non_repeatable_validator.active_batch.data, "dataframe"):
            print(
                f"Number of rows in validator's active batch: {len(non_repeatable_validator.active_batch.data.dataframe)}")
            print(
                f"Sample of data in validator's active batch:\n{non_repeatable_validator.active_batch.data.dataframe.head()}")
        else:
            print("Validator's active batch data is not a direct Pandas DataFrame.")
    else:
        print("ERROR: non_repeatable_validator has NO active batch before checkpoint run!")

    print(
        f"Expectation suite name in validator: {non_repeatable_validator.expectation_suite.expectation_suite_name}")
    print(
        f"Number of expectations to run: {len(non_repeatable_validator.expectation_suite.expectations)}")

    # Now run the checkpoint
    checkpoint_result_main = checkpoint_main.run()
    results_dict = checkpoint_result_main.to_json_dict()
    with open(os.path.join(app_path, 'results_non_repeatable.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)
    print("\nNon-repeatable data validation complete. Results saved.")


    # For Repeatable Entities
    for entity_name, entity_validator_for_suite_authoring in repeatable_entity_validators.items():  # Renamed for clarity
        print(
            f"\n--- Finalizing validation for repeatable entity: {entity_name} ---")

        if not entity_validator_for_suite_authoring.expectation_suite:
            print(
                f"Warning: Validator for entity {entity_name} does not have an active suite. Skipping checkpoint.")
            continue

        entity_suite_to_run = entity_validator_for_suite_authoring.expectation_suite.expectation_suite_name
        # Ensure the suite name is set on the suite object if not already done during creation
        entity_validator_for_suite_authoring.expectation_suite.expectation_suite_name = entity_suite_to_run
        entity_validator_for_suite_authoring.save_expectation_suite(
            discard_failed_expectations=False)
        print(
            f"Expectation suite '{entity_suite_to_run}' saved for {entity_name}.")

        # === MODIFICATION START: Use the stored DataFrame to create a fresh validator for the checkpoint ===
        df_current_entity = repeatable_entity_dataframes.get(
            entity_name)  # Retrieve the stored DataFrame

        if df_current_entity is None or df_current_entity.empty:
            print(
                f"Warning: DataFrame for entity '{entity_name}' not found or is empty in repeatable_entity_dataframes. Skipping checkpoint.")
            continue

        dataframe_asset = context.sources.add_pandas(
            f"{entity_name}_validator_checkpoint"
        ).add_dataframe_asset(
            name="f{entity_name}_data_asset",
            dataframe=df_current_entity,
        )
        batch_request = dataframe_asset.build_batch_request()

        checkpoint_entity = context.add_or_update_checkpoint(
            name=f"{entity_name}_checkpoint",
            expectation_suite_name=entity_suite_to_run,
            runtime_configuration={
                "result_format": {"result_format": "COMPLETE", "unexpected_index_column_names": ["patient_id", "record_id"], "return_unexpected_index_query": True}
            },
            batch_request=batch_request
        )

        entity_checkpoint_result = checkpoint_entity.run()
        repeatable_entity_checkpoint_results[entity_name] = entity_checkpoint_result.to_json_dict(
        )
        # all_checkpoint_result_dicts.append(entity_checkpoint_result.to_json_dict())

        entity_results_filename = f"results_{entity_name}.json"
        with open(os.path.join(app_path, entity_results_filename), 'w') as f:
            json.dump(entity_checkpoint_result.to_json_dict(), f, indent=4)
        print(f"{entity_name} validation results saved to {entity_results_filename}")


    # In main.py, after all checkpoint.run() calls


    # Add main non-repeatable results
    if 'checkpoint_result_main' in locals() and checkpoint_result_main:
        # Keep this for specific use if needed
        results_dict_main = checkpoint_result_main.to_json_dict()
    else:
        results_dict_main = None  # Ensure it's defined for patient summary fallback

    # Will store tuples: (source_description, result_dict)
    all_checkpoint_results_with_source = []

    # Add main non-repeatable results
    if 'checkpoint_result_main' in locals() and checkpoint_result_main:
        results_dict_main = checkpoint_result_main.to_json_dict()
        all_checkpoint_results_with_source.append(
            ("Non-Repeatable Patient Data", results_dict_main))
    else:
        # Still needed for subsequent summaries if they depend on it by this name
        results_dict_main = None

    # Add results from repeatable entities
    if 'repeatable_entity_checkpoint_results' in locals():
        for entity_name, entity_result_dict in repeatable_entity_checkpoint_results.items():
            if entity_result_dict:  # Ensure it's not None
                all_checkpoint_results_with_source.append(
                    (f"Repeatable Entity: {entity_name}", entity_result_dict))

    if not all_checkpoint_results_with_source:
        print("No checkpoint results found to generate detailed error reports or summaries.")
        # Potentially exit or skip further processing


    # In main.py, after populating all_checkpoint_results_with_source

    print("\n" + "="*30 + " DETAILED EXPECTATION FAILURE REPORT " + "="*30)
    if not all_checkpoint_results_with_source:
        print("No validation results to report on.")
    else:
        any_failure_found_overall = False
        for source_description, results_dict in all_checkpoint_results_with_source:
            if not results_dict:
                print(f"\n--- Validation Results for: {source_description} ---")
                print("  No results dictionary found for this source.")
                continue

            print(f"\n--- Validation Results for: {source_description} ---")
            run_id_info = results_dict.get('run_id', {})
            print(
                f"  Run Name: {run_id_info.get('run_name', 'N/A')}, Run Time: {run_id_info.get('run_time', 'N/A')}")

            # Assume success if key missing (should not happen)
            overall_success_for_this_run = results_dict.get("success", True)

            found_failures_in_this_run = False
            for run_name_key, run_result_data in results_dict.get("run_results", {}).items():
                validation_result = run_result_data.get("validation_result", {})
                if not validation_result:
                    continue

                individual_expectation_results = validation_result.get(
                    "results", [])
                for expectation_outcome in individual_expectation_results:
                    # If this specific expectation failed
                    if not expectation_outcome.get("success", True):
                        found_failures_in_this_run = True
                        # Mark that at least one failure was found globally
                        any_failure_found_overall = True

                        config = expectation_outcome.get("expectation_config", {})
                        ge_name = config.get(
                            "expectation_type", "N/A_EXPECTATION_TYPE")
                        kwargs = config.get("kwargs", {})
                        column_name = kwargs.get(
                            "column", "N/A (Table-Level or Column not in kwargs)")

                        print(f"\n  [FAILED] Expectation: {ge_name}")
                        print(f"    Target Column: {column_name}")
                        if "meta" in config and "dimensions" in config["meta"]:
                            print(
                                f"    Dimensions: {config['meta']['dimensions']}")

                        result_details = expectation_outcome.get("result", {})
                        partial_unexpected_list = result_details.get(
                            "partial_unexpected_list", [])
                        partial_unexpected_index_list = result_details.get(
                            "partial_unexpected_index_list", [])

                        if partial_unexpected_index_list:
                            print(
                                f"    Problematic Entries (up to {len(partial_unexpected_index_list)} shown):")
                            for idx, failing_row_indices_dict in enumerate(partial_unexpected_index_list):
                                # failing_row_indices_dict already contains column names as keys from unexpected_index_column_names
                                # e.g., {"patient_id": "PID1", "original_source": "sourceA", "ActualColumnNameInError": "bad_value"}
                                # Or for repeatable: {"patient_id": "PID1", "record_id": "REC1", "ActualColumnNameInError": "bad_value"}

                                patient_id = failing_row_indices_dict.get(
                                    "patient_id", "PID_Not_Found")
                                record_id = failing_row_indices_dict.get(
                                    "record_id")  # Will be None if not present
                                original_source = failing_row_indices_dict.get(
                                    "original_source")  # Will be None if not present

                                identifier_str = f"Patient ID: {patient_id}"
                                if record_id:
                                    identifier_str += f", Record ID: {record_id}"
                                elif original_source:  # For non-repeatable, you used 'original_source'
                                    identifier_str += f", Source: {original_source}"

                                problem_value_str = "N/A"
                                if idx < len(partial_unexpected_list):
                                    problem_value_str = str(
                                        partial_unexpected_list[idx])

                                # Construct a more detailed reason string
                                reason = ""
                                if ge_name == "expect_column_values_to_not_be_null":
                                    reason = f"Value is NULL in column '{column_name}'"
                                elif ge_name == "expect_column_values_to_be_in_set":
                                    value_set_str = str(
                                        kwargs.get("value_set", "N/A"))
                                    reason = f"Value '{problem_value_str}' from column '{column_name}' not in expected set: {value_set_str}"
                                elif ge_name == "expect_column_values_to_be_between":
                                    min_val, max_val = kwargs.get(
                                        "min_value"), kwargs.get("max_value")
                                    reason = f"Value '{problem_value_str}' from column '{column_name}' not between {min_val} and {max_val}"
                                elif ge_name == "expect_column_values_not_to_be_future_date":
                                    reason = f"Date value '{problem_value_str}' from column '{column_name}' is a future date"
                                # Add more elif for other specific standard or custom expectations
                                elif "expect_radiotherapy_session_completeness" in ge_name or "expect_episode_event_completeness" in ge_name:
                                    # Custom expectations might have their specific structure in partial_unexpected_list
                                    # problem_value_str would be a dict here
                                    reason = f"Completeness check failed. Details: {problem_value_str}"
                                else:  # Generic fallback
                                    reason = f"Unexpected value: '{problem_value_str}' in column '{column_name}'"
                                    if not partial_unexpected_list and "observed_value" in result_details:
                                        reason = f"Observed value '{result_details['observed_value']}' was unexpected for table/column level check on '{column_name}'."

                                print(f"      - {identifier_str} -> {reason}")

                        # No index list, but have values (less common for row-level issues)
                        elif partial_unexpected_list:
                            print(
                                f"    Problematic Values (no specific row indices provided, up to {len(partial_unexpected_list)} shown): {partial_unexpected_list}")

                        elif "observed_value" in result_details:  # For table-level expectations or others without lists
                            print(
                                f"    Observed Value that failed expectation: {result_details['observed_value']}")
                        else:
                            print(
                                f"    No detailed unexpected list or observed value provided for this failure. Check full JSON results for expectation: {ge_name} on column/context: {column_name}.")
                        # Separator for details of next failed expectation
                        print("    " + "-"*20)

            if not found_failures_in_this_run and overall_success_for_this_run:
                print("  All expectations passed for this validation run.")
            elif not found_failures_in_this_run and not overall_success_for_this_run:
                print("  Overall validation for this run is False, but no individual expectation failures were detailed in results (check suite-level issues or evaluation parameters).")

        if not any_failure_found_overall:
            print("\nCongratulations! All expectations passed across all validation runs.")
    print("="*30 + " END OF DETAILED REPORT " + "="*30 + "\n")


    all_checkpoint_result_dicts = [
        rd                                   # the dict itself
        for _src, rd in all_checkpoint_results_with_source
        if rd                                # skip None entries
    ]

    # ------------------------------------------------------------------
    # Figure out which columns actually had a NOT-NULL check
    # ------------------------------------------------------------------
    NULL_EXP = "expect_column_values_to_not_be_null"
    notnull_vars = set()

    for chk in all_checkpoint_result_dicts:
        if not chk:
            continue
        for _, run in chk.get("run_results", {}).items():
            for out in run.get("validation_result", {}).get("results", []):
                cfg = out.get("expectation_config", {})
                if cfg.get("expectation_type") != NULL_EXP:
                    continue
                col = cfg.get("kwargs", {}).get("column")
                if col:
                    notnull_vars.add(col)

    if not notnull_vars:
        print("WARNING: no NOT-NULL expectations found – phase summary will be empty")


    # ---------------------------------------------------------------------------
    # --- Generate Missing-value & Consistency summary per original_source -------
    print("\n--- Generating datasource summary from NOT-NULL QC results ----------")

    try:
        # df_long_all = pd.read_excel(data_file, dtype=str)
        df_long_all = read_table(data_file, dtype=str)

    except Exception as e:
        print(f"ERROR reading long-format file: {e}")
        df_long_all = pd.DataFrame()

    # ── 1.  First-seen rule for canonical source + detect mismatches ────────────
    canonical_src = {}          # core_variable → first source encountered
    var_has_mismatch = set()

    if not df_long_all.empty and {"core_variable", "original_source"}.issubset(df_long_all.columns):
        for _, row in df_long_all.iterrows():
            var = row["core_variable"]
            src = str(row["original_source"]).strip() if pd.notna(
                row["original_source"]) else "NULL"

            # first time we see this variable ⇒ lock the source
            if var not in canonical_src:
                canonical_src[var] = src
            elif src != canonical_src[var]:
                var_has_mismatch.add(var)
    else:
        print("Required columns (core_variable, original_source) missing.")

    # ── 2.  Collect NOT-NULL QC metrics from GE checkpoint results ─────────────
    NULL_EXP = "expect_column_values_to_not_be_null"
    # var → [element_count, unexpected_count]
    var_counts = collections.defaultdict(lambda: [0, 0])

    for chk in all_checkpoint_result_dicts:
        if not chk:
            continue
        for _, run in chk.get("run_results", {}).items():
            for out in run.get("validation_result", {}).get("results", []):
                cfg = out.get("expectation_config", {})
                if cfg.get("expectation_type") != NULL_EXP:
                    continue
                var = cfg.get("kwargs", {}).get("column")
                if not var:
                    continue
                ec = out.get("result", {}).get("element_count", 0)
                uc = out.get("result", {}).get("unexpected_count", 0)
                var_counts[var][0] += ec
                var_counts[var][1] += uc

    # ── 3.  Aggregate counts by datasource (the canonical one) ─────────────────
    ds_stats = {}   # datasource → counters


    def ds_record(src):
        return ds_stats.setdefault(
            src,
            {"Datasource": src, "Total": 0, "Missing": 0, "Mismatch": 0}
        )


    for var, (ec, uc) in var_counts.items():
        src = canonical_src.get(var, "UNKNOWN_SOURCE")
        rec = ds_record(src)
        rec["Total"] += ec          # how many rows GE checked for this variable
        rec["Missing"] += uc          # rows that were NULL
        if var in var_has_mismatch:   # this variable appeared with a second source
            rec["Mismatch"] += 1

    # ── 4.  Finish off derived metrics ─────────────────────────────────────────
    for rec in ds_stats.values():
        rec["Present"] = rec["Total"] - rec["Missing"]
        rec["MissingPercent"] = (
            round(rec["Missing"] / rec["Total"] * 100, 2) if rec["Total"] else 0.0
        )

    # ── 5.  Persist to JSON so the UI can read it ──────────────────────────────
    ds_json = os.path.join(app_path, "datasource_missingness_results.json")
    with open(ds_json, "w", encoding="utf-8") as fh:
        json.dump(list(ds_stats.values()), fh, indent=4)

    print(f"Datasource QC summary saved to: {ds_json}")
    # ---------------------------------------------------------------------------


    # --- Generate Variable‐level QC Summary ---
    print("\n--- Generating Aggregated Variable‐level QC Summary ---")
    variable_qc = {}

    # all_checkpoint_results_with_source is a list of tuples: (source_desc, result_dict)
    for source_desc, results_dict in all_checkpoint_results_with_source:
        for run_name, run_data in results_dict.get("run_results", {}).items():
            validation = run_data.get("validation_result", {})
            for outcome in validation.get("results", []):
                cfg = outcome.get("expectation_config", {})
                col = cfg.get("kwargs", {}).get("column")
                if not col:
                    # skip table‐level or column‐independent checks
                    continue

                res = outcome.get("result", {})
                # element_count may be 0 or missing for table‐level checks; default to 1
                total = res.get("element_count", 1) or 1
                failed = res.get("unexpected_count", 0)

                if col not in variable_qc:
                    variable_qc[col] = {
                        "Variable": col,
                        "Total":    0,
                        "Failed":   0
                    }

                variable_qc[col]["Total"] += total
                variable_qc[col]["Failed"] += failed

    # Calculate Passed and PercentagePass
    for stats in variable_qc.values():
        stats["Passed"] = stats["Total"] - stats["Failed"]
        stats["PercentagePass"] = (
            round((stats["Passed"] / stats["Total"]) * 100, 2)
            if stats["Total"] else 100.0
        )

    # Write out to JSON
    variable_summary_path = os.path.join(
        app_path, "variable_summary_results.json")
    with open(variable_summary_path, "w") as f:
        json.dump(list(variable_qc.values()), f, indent=4)

    print(f"Variable‐level QC summary saved to: {variable_summary_path}")

    # ── Generate Importance-group Summary ( M / O / R ) ────────────────────────
    print("\n--- Generating Importance-group (M/O/R) QC summary ---------------")

    # 0️⃣  Normaliser helper must be in outer scope


    def _norm(txt: str) -> str:
        """lowercase, remove spaces/underscores/-, keep only alphanumerics"""
        return "".join(ch.lower() for ch in txt if ch.isalnum())


    # 1️⃣  Load importance map and normalise keys
    importance_file = os.path.join(os.path.dirname(suites_file),
                                "variable_importance.json")
    importance_map = {}
    try:
        with open(importance_file, encoding="utf-8") as fh:
            raw_map = json.load(fh)

        # normalise keys like  Patient_sex  →  patientsex   (strip dot / underscore)
        importance_map = {_norm(k.replace(".", "_"))                      : v for k, v in raw_map.items()}
    except Exception as e:
        print(f"WARNING: could not load variable_importance.json: {e}")

    # 2️⃣  Helper: map Entity.variable → M/O/R


    def importance_of(col: str) -> str:
        """
        Try full Entity.Variable first; fall back to variable-only.
        Returns 'Unknown' if still not found.
        """
        full_key = _norm(col.replace(".", "_"))      # Patient.sex → patientsex
        if full_key in importance_map:
            return importance_map[full_key]

        short = col.split(".", 1)[1] if "." in col else col   # sex
        return importance_map.get(_norm(short), "Unknown")


    # 3️⃣  Aggregate using variable_qc dict created earlier
    group_stats = {
        g: {"Group": g, "Total": 0, "Failed": 0, "Passed": 0}
        for g in ["M", "O", "R", "Unknown"]
    }

    for var_name, stats in variable_qc.items():

        grp = importance_of(var_name)
        print(f"Variable '{var_name}' classified as group '{grp}'")
        rec = group_stats[grp]
        rec["Total"] += stats["Total"]
        rec["Failed"] += stats["Failed"]
        rec["Passed"] += stats["Passed"]

    for rec in group_stats.values():
        if rec["Total"]:
            rec["PercentagePass"] = round(rec["Passed"] / rec["Total"] * 100, 2)
        else:
            rec["PercentagePass"] = None

    # 4️⃣  Save JSON
    importance_summary_path = os.path.join(
        app_path, "importance_group_summary_results.json"
    )
    with open(importance_summary_path, "w", encoding="utf-8") as fh:
        json.dump(list(group_stats.values()), fh, indent=4)

    print(f"Importance-group QC summary saved to: {importance_summary_path}")
    # ───────────────────────────────────────────────────────────────────────────


    # --- Generate QC (Expectation Name) Summary (Aggregated) ---
    print("\n--- Generating Aggregated QC (Expectation Name) Summary ---")
    qc_summary_data = {}

    for current_results_dict in all_checkpoint_result_dicts:
        if not current_results_dict:
            continue
        for run_name, run_result_data in current_results_dict.get("run_results", {}).items():
            if "validation_result" in run_result_data:
                individual_expectation_results = run_result_data["validation_result"].get(
                    "results", [])
                for expectation_outcome in individual_expectation_results:
                    expectation_config = expectation_outcome.get(
                        'expectation_config', {})
                    ge_name = expectation_config.get('expectation_type')
                    if not ge_name:
                        continue

                    result_details = expectation_outcome.get('result', {})
                    element_count = result_details.get('element_count', 0)
                    unexpected_count = result_details.get('unexpected_count', 0)

                    if ge_name not in qc_summary_data:
                        qc_summary_data[ge_name] = {
                            "ge_name": ge_name, "passed_checks": 0,
                            "failed_checks": 0, "total_checks": 0, "percentage_pass": 0.0
                        }

                    qc_summary_data[ge_name]["total_checks"] += element_count
                    qc_summary_data[ge_name]["failed_checks"] += unexpected_count
                    qc_summary_data[ge_name]["passed_checks"] += (
                        element_count - unexpected_count)

    # Calculate overall percentage for each ge_name
    for ge_name in qc_summary_data:
        summary = qc_summary_data[ge_name]
        if summary["total_checks"] > 0:
            summary["passed_checks"] = summary["total_checks"] - \
                summary["failed_checks"]  # Ensure consistency
            summary["percentage_pass"] = round(
                (summary["passed_checks"] / summary["total_checks"]) * 100, 2)
        elif summary["total_checks"] == 0 and summary["failed_checks"] == 0:
            summary["percentage_pass"] = 100.0
        else:
            summary["percentage_pass"] = 0.0

    final_qc_summary_list = list(qc_summary_data.values())
    qc_summary_output_path = os.path.join(
        app_path, 'qc_summary_results.json')  # New name
    try:
        with open(qc_summary_output_path, 'w') as f:
            json.dump(final_qc_summary_list, f, indent=4)
        print(f"Aggregated QC-specific summary saved to: {qc_summary_output_path}")
    except Exception as e:
        print(f"Error saving aggregated QC-specific summary: {e}")

        # ---------------------------------------------------------------------------


    # --- Phase report: nearest Dx / Progression / Recurrence in time -----------
    print("\n--- Generating phase summary using temporal proximity -------------")

    if df_long_all.empty:
        print("No data, skipping phase report")
    else:
        # 1️⃣  helper to coerce any date‐ish value to a pandas Timestamp
        def to_ts(val):
            try:
                return pd.to_datetime(val, errors="coerce")
            except Exception:
                return pd.NaT

            # 2️⃣  collect phase-defining events for each patient
        #     we now rely on *real* dates:
        #       • Diagnosis.dateOfDiagnosis             → Diagnosis
        #       • EpisodeEvent.diseaseStatus (+ its     → Progression / Recurrence
        #         companion EpisodeEvent.dateOfEpisode)   timestamp)
        # ---------------------------------------------------------------------

        # --- quick look-ups ---------------------------------------------------
        epi_date_lookup = (
            df_long_all[df_long_all.core_variable == "EpisodeEvent.dateOfEpisode"]
            .set_index(["patient_id", "record_id"])["value"]
            .to_dict()
        )

        # Sometimes a disease-status row has no episode date; we fall back to the
        # patient’s diagnosis date (earliest one in the file).
        diag_lookup = (
            df_long_all[df_long_all.core_variable == "Diagnosis.dateOfDiagnosis"]
            .sort_values("value")            # earliest first
            .drop_duplicates("patient_id")   # keep 1 / patient
            .set_index("patient_id")["value"]
            .to_dict()
        )
        # ---------------------------------------------------------------------

        phase_events = []          # rows: patient_id, phase, date

        for _, r in df_long_all.iterrows():

            # ───────────── EpisodeEvent.diseaseStatus  ─────────────
            if r.core_variable == "EpisodeEvent.diseaseStatus":
                code = str(r.value).strip()
                if code == "32949":
                    phase = "Progression"
                elif code == "2000100002":
                    phase = "Recurrence"
                else:
                    phase = "Diagnosis"   # any other value

                # ① try episode-date that belongs to the *same* (pid, rid)
                dt_raw = epi_date_lookup.get((r.patient_id, r.record_id))

                # ② otherwise fall back to patient’s Diagnosis date (if any)
                if dt_raw is None:
                    dt_raw = diag_lookup.get(r.patient_id)

                phase_events.append((r.patient_id, phase, to_ts(dt_raw)))

            # ───────────── Diagnosis.dateOfDiagnosis  ─────────────
            elif r.core_variable == "Diagnosis.dateOfDiagnosis":
                phase_events.append(
                    (r.patient_id, "Diagnosis", to_ts(r.value))
                )

        phase_df = pd.DataFrame(phase_events, columns=[
                                "pid", "phase", "date"]).dropna()

        # keep earliest Diagnosis, keep all Prog/Rec events
        phase_df = (phase_df.sort_values("date")
                            .drop_duplicates(subset=["pid", "phase"], keep="first"))
        # print rows from phase_df where phase is not "Diagnosis"
        test = phase_df[phase_df.phase != "Diagnosis"]

        # 3️⃣  work out a single date for every record
        record_dates = (df_long_all.assign(date=df_long_all.apply(
            lambda r: to_ts(
                r.value) if "date" in r.core_variable.lower() else pd.NaT,
            axis=1))
            .groupby(["patient_id", "record_id"])["date"]
            .min()
            .reset_index())

        # 4️⃣  map each record to the closest *past* status event (same patient)
        def closest_phase(row):
            pid, dt = row.patient_id, row.date
            if pd.isna(dt):                       # record has no usable date
                return "Diagnosis"

            # candidate events for that patient that happened on / before this record
            cand = phase_df[(phase_df.pid == pid) & (phase_df.date <= dt)]

            if cand.empty:                        # no past event ⇒ assume diagnosis
                return "Diagnosis"

            # smallest positive time‐gap (dt – event_date)
            idx = (dt - cand.date).idxmin()
            return cand.loc[idx, "phase"]

        record_dates["phase"] = record_dates.apply(closest_phase, axis=1)

        # 5️⃣  join back to every row
        df_with_phase = df_long_all.merge(
            record_dates[["patient_id", "record_id", "phase"]],
            on=["patient_id", "record_id"],
            how="left"
        )

        # Filter out any rows where phase is NaN
        focus = df_with_phase[
            df_with_phase.core_variable.isin(notnull_vars) &
            df_with_phase.phase.notna()
        ]

        phase_stats = {
            p: {"Phase": p, "Total": 0, "Missing": 0}
            for p in ["Diagnosis", "Progression", "Recurrence"]
        }

        for _, r in focus.iterrows():
            ph = r.phase                           # Diagnosis / Progression / Recurrence
            
            phase_stats[ph]["Total"] += 1
            if pd.isna(r.value) or str(r.value).strip() == "":
                phase_stats[ph]["Missing"] += 1

        for rec in phase_stats.values():
            rec["Present"] = rec["Total"] - rec["Missing"]
            rec["MissingPercent"] = (
                round(rec["Missing"] / rec["Total"] *
                    100, 2) if rec["Total"] else 0.0
            )

        out_path = os.path.join(app_path, "phase_missingness_results.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(list(phase_stats.values()), fh, indent=4)

        print(f"Phase summary saved to: {out_path}")
    # ---------------------------------------------------------------------------


    # --- Phase × Entity presence summary ---------------------------------------
    # Creates phase_entity_results.json for the UI “By phase” tab

    # check if df_with_phase is empty
    if df_long_all.empty:
        print("No data for phase-entity summary, skipping")
    else:
        phase_entity = {}     # (entity_instance, phase) → present-counter

        focus_rows = df_with_phase[df_with_phase.core_variable.isin(notnull_vars)]

        for _, r in focus_rows.iterrows():
            # entity name (e.g. "Surgery") + its record_id to keep instances separate
            # Surgery / RadiotherapySession …
            ent = r.core_variable.split('.')[0]
            ent_inst = f"{ent}{r.record_id or ''}"       # Surgery1, Surgery2 …
            key = (ent_inst, r.phase)

            rec = phase_entity.setdefault(
                key, {"Entity": ent_inst, "Phase": r.phase, "Present": 0}
            )
            # count a variable as “present” if it’s non-null/non-blank
            if pd.notna(r.value) and str(r.value).strip():
                rec["Present"] += 1

        phase_entity_json = os.path.join(app_path, "phase_entity_results.json")
        with open(phase_entity_json, "w", encoding="utf-8") as fh:
            json.dump(list(phase_entity.values()), fh, indent=4)

        print(f"Phase-entity summary saved to: {phase_entity_json}")

        # --- Phase × Entity × Patient summary --------------------------------------
        patient_phase_entity = {}      # (pid, ent_inst, phase) → counters

        for _, r in focus_rows.iterrows():
            ent = r.core_variable.split('.')[0]          # Surgery / Biopsy / …
            ent_inst = f"{ent}{r.record_id or ''}"            # Surgery1, Surgery2 …
            key = (r.patient_id, ent_inst, r.phase)

            rec = patient_phase_entity.setdefault(
                key,
                {
                    "PatientID": str(r.patient_id),
                    "Entity":    ent_inst,
                    "Phase":     r.phase,
                    "Total":     0,
                    "Present":   0,
                    "Missing":   [],  # ⬅︎  new
                }
            )

            rec["Total"] += 1
            if pd.notna(r.value) and str(r.value).strip():
                rec["Present"] += 1
            else:
                rec["Missing"].append(r.core_variable)

        rec["Complete"] = rec["Present"] == rec["Total"]

        # ── decide completeness ────────────────────────────────────────────────────
        for rec in patient_phase_entity.values():
            rec["Complete"] = rec["Present"] == rec["Total"]    # True / False

        out = os.path.join(app_path, "patient_entity_phase_results.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(list(patient_phase_entity.values()), fh, indent=4)
        print(f"Patient-entity-phase summary saved to: {out}")


    # --- Generate Dimension Summary (Aggregated) ---
    print("\n--- Generating Aggregated Dimension Summary ---")
    known_dimensions = ["Plausibility", "Conformance",
                        "Completeness"]  # As defined in your main.py
    dimension_summary_data = {dim: {"Passed": 0, "Failed": 0,
                                    "Total": 0, "PercentagePass": "0.0%"} for dim in known_dimensions}
    dimension_summary_data["Total"] = {
        "Passed": 0, "Failed": 0, "Total": 0, "PercentagePass": "0.0%"}

    for current_results_dict in all_checkpoint_result_dicts:
        if not current_results_dict:
            continue
        for run_name, run_result_data in current_results_dict.get("run_results", {}).items():
            if "validation_result" in run_result_data:
                individual_expectation_results = run_result_data["validation_result"].get(
                    "results", [])
                for expectation_outcome in individual_expectation_results:
                    result_details = expectation_outcome.get('result', {})
                    element_count = result_details.get('element_count', 0)
                    unexpected_count = result_details.get('unexpected_count', 0)
                    current_success = expectation_outcome.get('success', True)

                    if element_count == 0 and not current_success:
                        current_total_for_instance = 1
                        current_failed_for_instance = 1
                    else:
                        current_total_for_instance = element_count
                        current_failed_for_instance = unexpected_count
                    current_passed_for_instance = current_total_for_instance - current_failed_for_instance

                    dimension_summary_data["Total"]["Passed"] += current_passed_for_instance
                    dimension_summary_data["Total"]["Failed"] += current_failed_for_instance
                    dimension_summary_data["Total"]["Total"] += current_total_for_instance

                    expectation_meta = expectation_outcome.get(
                        'expectation_config', {}).get('meta', {})
                    expectation_dimensions = expectation_meta.get('dimensions', [])
                    for dim in expectation_dimensions:
                        if dim in dimension_summary_data:
                            dimension_summary_data[dim]["Passed"] += current_passed_for_instance
                            dimension_summary_data[dim]["Failed"] += current_failed_for_instance
                            dimension_summary_data[dim]["Total"] += current_total_for_instance

    # Calculate percentages
    for category_name, summary in dimension_summary_data.items():
        if summary["Total"] > 0:
            summary["Passed"] = summary["Total"] - \
                summary["Failed"]  # Ensure consistency
            percentage = (summary["Passed"] / summary["Total"]) * 100
            summary["PercentagePass"] = f"{percentage:.1f}%"
        elif summary["Total"] == 0 and summary["Failed"] == 0:
            summary["Passed"] = 0
            summary["PercentagePass"] = "100.0%"
        else:
            summary["Passed"] = 0
            summary["PercentagePass"] = "0.0%"

    dimension_summary_output_path = os.path.join(
        app_path, 'dimension_summary_results.json')  # New name
    try:
        with open(dimension_summary_output_path, 'w') as f:
            json.dump(dimension_summary_data, f, indent=4)
        print(
            f"Aggregated Dimension-specific summary saved to: {dimension_summary_output_path}")
    except Exception as e:
        print(f"Error saving aggregated dimension-specific summary: {e}")


    # --- Generate Patient Summary (Aggregated) ---
    print("\n--- Generating Aggregated Patient Summary ---")
    # PatientID -> {Passed, Failed, Total, ...}
    patient_summary_data_aggregated = {}


    def ensure_patient_in_aggregated_summary(pid, summary_dict):
        pid_str = str(pid)
        if pid_str not in summary_dict:
            summary_dict[pid_str] = {
                "PatientID": pid_str, "Number of Passed Tests": 0, "Failed": 0,
                "Total": 0, "PercentagePass": 0.0
            }


    # Process non-repeatable results
    if results_dict_main:  # This is checkpoint_result_main.to_json_dict()
        try:
            # df_non_rep_patients = pd.read_excel(
            #     FORMATED_FILEPATH_NON_REP, dtype=str)
            df_non_rep_patients = read_table(FORMATED_FILEPATH_NON_REP, dtype=str)

            if 'patient_id' in df_non_rep_patients.columns:
                current_batch_patient_ids = df_non_rep_patients['patient_id'].astype(
                    str).unique()
                for pid in current_batch_patient_ids:
                    ensure_patient_in_aggregated_summary(
                        pid, patient_summary_data_aggregated)

                for run_name, run_result_data in results_dict_main.get("run_results", {}).items():
                    if "validation_result" in run_result_data:
                        individual_expectation_results = run_result_data["validation_result"].get(
                            "results", [])
                        for expectation_outcome in individual_expectation_results:
                            patient_ids_who_failed_this_test = set()
                            unexpected_indices = expectation_outcome.get(
                                "result", {}).get("partial_unexpected_index_list", [])
                            if unexpected_indices:
                                for failed_row_info in unexpected_indices:
                                    if isinstance(failed_row_info, dict):
                                        failed_pid = failed_row_info.get(
                                            "patient_id")
                                        if failed_pid is not None:
                                            patient_ids_who_failed_this_test.add(
                                                str(failed_pid))

                            for pid_str in current_batch_patient_ids:  # This expectation applied to all these patients
                                patient_summary_data_aggregated[pid_str]["Total"] += 1
                                if pid_str in patient_ids_who_failed_this_test:
                                    patient_summary_data_aggregated[pid_str]["Failed"] += 1
                                else:
                                    patient_summary_data_aggregated[pid_str]["Number of Passed Tests"] += 1
            else:
                print(
                    "Warning: 'patient_id' column not found in non-repeatable data for patient summary.")
        except Exception as e:
            print(
                f"Error processing non-repeatable results for aggregated patient summary: {e}")


    # Process repeatable entity results
    if 'repeatable_entity_checkpoint_results' in locals() and 'repeatable_entity_dataframes' in locals():
        for entity_name, entity_results_dict in repeatable_entity_checkpoint_results.items():
            if not entity_results_dict:
                continue

            df_entity = repeatable_entity_dataframes.get(entity_name)
            if df_entity is None or df_entity.empty or 'patient_id' not in df_entity.columns:
                print(
                    f"Warning: DataFrame or patient_id column not found for entity '{entity_name}'. Skipping its contribution to aggregated patient summary.")
                continue

            current_batch_patient_ids_entity = df_entity['patient_id'].astype(
                str).unique()
            for pid in current_batch_patient_ids_entity:
                ensure_patient_in_aggregated_summary(
                    pid, patient_summary_data_aggregated)

            for run_name, run_result_data in entity_results_dict.get("run_results", {}).items():
                if "validation_result" in run_result_data:
                    individual_expectation_results = run_result_data["validation_result"].get(
                        "results", [])
                    for expectation_outcome in individual_expectation_results:
                        patient_ids_who_failed_this_test = set()
                        unexpected_indices = expectation_outcome.get(
                            "result", {}).get("partial_unexpected_index_list", [])
                        if unexpected_indices:
                            for failed_row_info in unexpected_indices:
                                if isinstance(failed_row_info, dict):
                                    # Assumes patient_id is in index for repeatable
                                    failed_pid = failed_row_info.get("patient_id")
                                    if failed_pid is not None:
                                        patient_ids_who_failed_this_test.add(
                                            str(failed_pid))

                        # For repeatable entities, an expectation outcome applies to all patients in that entity's batch
                        for pid_str in current_batch_patient_ids_entity:
                            patient_summary_data_aggregated[pid_str]["Total"] += 1
                            if pid_str in patient_ids_who_failed_this_test:
                                patient_summary_data_aggregated[pid_str]["Failed"] += 1
                            else:
                                patient_summary_data_aggregated[pid_str]["Number of Passed Tests"] += 1

    # Calculate percentages for aggregated patient summary
    for pid_str in patient_summary_data_aggregated:
        summary = patient_summary_data_aggregated[pid_str]
        if summary["Total"] > 0:
            summary["PercentagePass"] = round(
                (summary["Number of Passed Tests"] / summary["Total"]) * 100, 2)
        else:
            summary["PercentagePass"] = 0.0

    final_aggregated_patient_summary_list = list(
        patient_summary_data_aggregated.values())
    agg_patient_summary_output_path = os.path.join(
        app_path, 'patient_summary_results.json')  # New name
    try:
        with open(agg_patient_summary_output_path, 'w') as f:
            json.dump(final_aggregated_patient_summary_list, f, indent=4)
        print(
            f"Aggregated Patient-specific summary saved to: {agg_patient_summary_output_path}")
    except Exception as e:
        print(f"Error saving aggregated patient-specific summary: {e}")


    # ────────────────────────────────────────────────────────────────
    #  Patient  ×  Importance-group summary (High / Medium / Low)
    # ────────────────────────────────────────────────────────────────
    per_patient_imp = {}        # (pid, grp) → counters

    for var, stats in variable_qc.items():
        grp = importance_of(var)       # High / Medium / Low / Unknown
        for pid, pstats in patient_summary_data_aggregated.items():
            # every variable is ‘tested’ for every patient once
            key = (pid, grp)
            rec = per_patient_imp.setdefault(key,
                                            {"PatientID": pid, "Group": grp,
                                            "Passed": 0, "Failed": 0, "Total": 0})
            rec["Total"] += stats["Total"]
            rec["Failed"] += stats["Failed"]

    for rec in per_patient_imp.values():
        rec["Passed"] = rec["Total"] - rec["Failed"]
        rec["PassPercent"] = (
            round(rec["Passed"]/rec["Total"]*100, 2) if rec["Total"] else 100.0)

    out = list(per_patient_imp.values())
    with open(os.path.join(app_path,
            "patient_importance_summary_results.json"), "w") as fh:
        json.dump(out, fh, indent=4)
    print("Patient-importance summary written.")

    # ────────────────────────────────────────────────────────────────
    #  Patient  ×  Phase missingness summary
    # ────────────────────────────────────────────────────────────────
    if df_long_all.empty:
        print("No data for patient-phase summary, skipping")
    else:
        per_patient_phase = {}      # (pid, phase) → counters

        for _, r in df_with_phase.iterrows():
            if r.core_variable not in notnull_vars:
                continue
            key = (r.patient_id, r.phase)
            rec = per_patient_phase.setdefault(key,
                                            {"PatientID": r.patient_id, "Phase": r.phase,
                                                "Present": 0, "Missing": 0, "Total": 0})
            rec["Total"] += 1
            if pd.isna(r.value) or str(r.value).strip() == "":
                rec["Missing"] += 1
            else:
                rec["Present"] += 1

        for rec in per_patient_phase.values():
            rec["MissingPercent"] = (
                round(rec["Missing"]/rec["Total"]*100, 2) if rec["Total"] else 0.0)

        out = list(per_patient_phase.values())
        with open(os.path.join(app_path,
                "patient_phase_summary_results.json"), "w") as fh:
            json.dump(out, fh, indent=4)
        print("Patient-phase summary written.")


    print("Finished generating all reports and summaries.")


    return data, []
