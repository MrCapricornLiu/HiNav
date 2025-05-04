import json

# File paths for the input JSON files
mapgpt_file = "../MapGPT_72_scenes_processed_72cases.json"
new_obj_file = "new_obj_list_grok_72cases.json"
output_file = "combined_output.json"

# Read the MapGPT JSON file
with open(mapgpt_file, 'r') as f:
    mapgpt_data = json.load(f)

# Read the new_obj_list JSON file
with open(new_obj_file, 'r') as f:
    new_obj_data = json.load(f)

# Ensure the lengths match
if len(mapgpt_data) != len(new_obj_data):
    raise ValueError(f"Mismatch in number of entries: MapGPT has {len(mapgpt_data)}, new_obj has {len(new_obj_data)}")

# Combine the data
for i in range(len(mapgpt_data)):
    mapgpt_data[i]['direct_objects'] = new_obj_data[i][0]
    mapgpt_data[i]['inferred_objects'] = new_obj_data[i][1]

# Save the combined data to a new JSON file
with open(output_file, 'w') as f:
    json.dump(mapgpt_data, f, indent=4)

print(f"Combined JSON saved to {output_file}")