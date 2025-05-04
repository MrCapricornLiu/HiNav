import json

# Path to the JSON file
file_path = "/data/lch_zjc/VLN/MapGPT_72_scenes_processed.json"

# Initialize the list to store instructions
instructions = []

try:
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract instructions into a list of strings
    instructions = [sample['instruction'] for sample in data]
    
    # Save the instructions to a new JSON file
    with open('extracted_instructions.json', 'w') as outfile:
        json.dump(instructions, outfile, indent=4)

except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except json.JSONDecodeError:
    print("Error: The file contains invalid JSON.")
except KeyError:
    print("Error: Missing 'instruction' key in one or more samples.")