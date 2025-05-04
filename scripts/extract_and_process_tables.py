import os
import json
import pandas as pd
from pathlib import Path
import re
import argparse

# --- Load reference scenes ONCE outside the function ---
try:
    with open("/home/lch/Documents/Matterport3DSimulator/datasets/R2R/annotations/MapGPT_72_scenes_processed.json", "r") as f:
        mapgpt_72_scenes = json.load(f)
except FileNotFoundError:
    print("Error: MapGPT_72_scenes_processed.json not found.")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode MapGPT_72_scenes_processed.json.")
    exit()

# --- Create a lookup dictionary and preserve order ---
scene_lookup = {scene.get("instr_id"): scene for scene in mapgpt_72_scenes if scene.get("instr_id")}
# Preserve the order of instr_id as they appear in the JSON
scene_order = [scene.get("instr_id") for scene in mapgpt_72_scenes if scene.get("instr_id")]
if len(scene_lookup) != len(mapgpt_72_scenes):
    print("Warning: Some scenes in mapgpt_72_scenes might be missing 'instr_id' or have duplicates.")

def process_json_files(input_dir, output_file):
    data = []
    input_path = Path(input_dir)
    processed_instr_ids = set()  # To track which input files were used

    # --- Iterate through JSON files in the directory directly ---
    json_files = list(input_path.glob('*.json'))
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    if not json_files:
        print("No JSON files found to process.")
        return

    for json_file in json_files:
        try:
            # --- Extract instr_id from the filename ---
            match = re.search(r'InstrID_(\d+_\d+)\.json$', json_file.name)
            if not match:
                continue  # Skip if filename doesn't match expected pattern

            file_instr_id = match.group(1)

            # --- Check if this instr_id corresponds to a known scene ---
            if file_instr_id in scene_lookup:
                scene = scene_lookup[file_instr_id]
                scan = scene.get("scan", "UNKNOWN_SCAN")

                # --- Load JSON content ---
                with open(json_file, 'r') as f:
                    content_full = json.load(f)
                    content = content_full.get('evaluation', {})

                # --- Extract data and append ONLY when match is confirmed ---
                traj_steps_list = content.get('trajectory_steps', [])
                traj_lengths_list = content.get('trajectory_lengths', [])
                nav_error_list = content.get('nav_error', [])
                oracle_error_list = content.get('oracle_error', [])
                oracle_success_list = content.get('oracle_success', [])
                success_list = content.get('success', [])
                spl_list = content.get('spl', [])

                entry = {
                    'instr_id': f"{scan}+{file_instr_id}",
                    'trajectory_steps': traj_steps_list[-1] if traj_steps_list else None,
                    'trajectory_lengths': traj_lengths_list[-1] if traj_lengths_list else None,
                    'nav_error': nav_error_list[-1] if nav_error_list else None,
                    'oracle_error': oracle_error_list[-1] if oracle_error_list else None,
                    'oracle_success': oracle_success_list[-1] if oracle_success_list else None,
                    'SR': 1 if spl_list and spl_list[-1] != 0 else 0,  # SR based on spl
                    'spl': spl_list[-1] if spl_list else None
                }
                data.append(entry)
                processed_instr_ids.add(file_instr_id)

        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {json_file.name}. Skipping file.")
        except KeyError as e:
            print(f"Warning: Missing key {str(e)} in {json_file.name}. Skipping file.")
        except IndexError as e:
            print(f"Warning System: : Index error (likely empty list) processing {json_file.name}. Details: {str(e)}")
        except Exception as e:
            print(f"Warning: Unexpected error processing {json_file.name}: {str(e)}. Skipping file.")

    # --- Report Missing Scenes ---
    expected_instr_ids = set(scene_lookup.keys())
    missing_instr_ids = expected_instr_ids - processed_instr_ids
    if missing_instr_ids:
        print(f"\nWarning: Did not find matching JSON files for the following {len(missing_instr_ids)} scene instr_ids:")

    if not data:
        print("No data was successfully processed to save.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Adjust columns
    columns = [
        'instr_id',
        'trajectory_steps',
        'trajectory_lengths',
        'nav_error',
        'oracle_error',
        'SR',
        'oracle_success',
        'spl'
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]

    # --- Sort DataFrame based on the order in MapGPT_72_scenes_processed.json ---
    # Extract the instr_id part after the scan (e.g., '123_456' from 'scan+123_456')
    df['raw_instr_id'] = df['instr_id'].apply(lambda x: x.split('+')[1] if '+' in x else x)
    # Create a categorical type with the order from scene_order
    df['raw_instr_id'] = pd.Categorical(df['raw_instr_id'], categories=scene_order, ordered=True)
    # Sort by raw_instr_id and drop the temporary column
    df = df.sort_values('raw_instr_id').drop('raw_instr_id', axis=1)

    # --- Calculate averages for numeric columns and append as a new row ---
    numeric_columns = [
        'trajectory_steps',
        'trajectory_lengths',
        'nav_error',
        'oracle_error',
        'SR',
        'oracle_success',
        'spl'
    ]
    # Compute mean for numeric columns, ignoring NaN/None values
    averages = df[numeric_columns].mean(numeric_only=True).to_dict()
    # Create a new row with 'instr_id' as 'Average' and averages for numeric columns
    average_row = {'instr_id': 'Average'}
    for col in columns:
        if col in averages:
            average_row[col] = averages[col]
        elif col != 'instr_id':
            average_row[col] = None  # Non-numeric columns get None
    # Append the average row to the DataFrame
    df = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)

    # Save to Excel
    try:
        df.to_excel(output_file, index=False)
        print(f"\nSuccessfully saved {len(df)} records (including average row) to {output_file}")
    except Exception as e:
        print(f"Error saving DataFrame to Excel: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/home/lch/Documents/Matterport3DSimulator/datasets/exprs_map/test/preds",
                        help="Directory containing JSON files to process")
    parser.add_argument('--output_file', type=str, default="test_for_obj_arrow.xlsx",
                        help="Output Excel filename")
    args = parser.parse_args()

    input_directory = args.input_dir
    output_filename = args.output_file
    

    process_json_files(input_directory, output_filename)