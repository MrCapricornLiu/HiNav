import json

# Path to the JSON file
file_path = "/data/lch_zjc/VLN/MapGPT_72_scenes_processed.json"

# Read and process the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract scan and instruction from each sample
extracted_data = [{'scan': sample['scan'], 'instruction': sample['instruction']} for sample in data]

# Optionally, print the extracted data
for item in extracted_data:
    print(f"Scan: {item['scan']}, Instruction: {item['instruction']}")

# Optionally, save the extracted data to a new JSON file
with open('extracted_scan_instruction.json', 'w') as outfile:
    json.dump(extracted_data, outfile, indent=4)