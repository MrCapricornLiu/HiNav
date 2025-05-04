import json
import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Path to the JSON file
file_path = '/data/lch_zjc/VLN/MapGPT_72_scenes_processed.json'

# Define a set of stop words or non-object terms to filter out
non_object_terms = {'left', 'right', 'forward', 'side', 'step', 'threshold'}

# Function to extract objects from an instruction using NLP
def extract_objects(instruction):
    # Process the instruction with spaCy
    doc = nlp(instruction.lower())
    
    # Initialize list to store found objects
    found_objects = []
    
    # Iterate through tokens
    for token in doc:
        # Check if the token is a noun or proper noun
        if token.pos_ in ('NOUN', 'PROPN'):
            # Get the noun phrase (compound nouns like "glass table")
            phrase = token.text
            # Check if the token is part of a compound noun
            for child in token.children:
                if child.dep_ == 'compound' and child.pos_ in ('NOUN', 'ADJ'):
                    phrase = f"{child.text} {token.text}"
            
            # Filter out non-object terms and duplicates
            if (phrase not in non_object_terms and 
                not any(term in phrase for term in non_object_terms) and 
                phrase not in found_objects):
                found_objects.append(phrase)
    
    return found_objects

# Initialize the list to store extracted data
extracted_data = []

try:
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract scan, instruction, and objects
    for sample in data:
        scan = sample['scan']
        instruction = sample['instruction']
        objects = extract_objects(instruction)
        extracted_data.append({
            'scan': scan,
            'instruction': instruction,
            'objects': objects
        })
    
    # Save the extracted data to a new JSON file
    with open('extracted_scan_instruction_objects_nlp.json', 'w', encoding='utf-8') as outfile:
        json.dump(extracted_data, outfile, indent=4, ensure_ascii=False)

except FileNotFoundError:
    print(f"错误：文件 {file_path} 未找到。")
except json.JSONDecodeError:
    print("错误：文件包含无效的 JSON 格式。")
except KeyError as e:
    print(f"错误：样本中缺少键 {e}。")