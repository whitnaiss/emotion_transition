import json
import csv
import os

def process_json_to_csv(json_file_path, output_csv_path):
    try:
        # Check if the JSON file exists
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"The file {json_file_path} does not exist.")

        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Prepare the data to extract id, text, and assign 'negative' as label
        extracted_data = []
        if isinstance(data, list):
            for item in data:
                if 'id' in item and 'text' in item:
                    extracted_data.append({
                        'id': item['id'],
                        'text': item['text'],
                        'label': 'negative'
                    })
        elif isinstance(data, dict):
            if 'id' in data and 'text' in data:
                extracted_data.append({
                    'id': data['id'],
                    'text': data['text'],
                    'label': 'positive'
                })
        else:
            raise ValueError("Unsupported JSON structure. Expecting a dictionary or a list of dictionaries.")

        # Write the extracted data to CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['id', 'text', 'label'])
            writer.writeheader()
            writer.writerows(extracted_data)

        print(f"CSV file successfully created at: {output_csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_multiple_json_files(input_folder, output_folder):
    try:
        # Check if the input folder exists
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"The folder {input_folder} does not exist.")

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Process each JSON file in the input folder
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.json'):
                json_file_path = os.path.join(input_folder, file_name)
                output_csv_path = os.path.join(output_folder, file_name.replace('.json', '.csv'))
                process_json_to_csv(json_file_path, output_csv_path)

        print(f"All JSON files have been processed and saved to {output_folder}.")
    except Exception as e:
        print(f"An error occurred while processing multiple files: {e}")

# Specify the input folder containing JSON files and the output folder for CSV files
input_folder = 'sampled_positive'  # Replace with the path to your folder containing JSON files
output_folder = 'positives'  # Replace with the desired output folder path

# Call the function to process all JSON files in the folder
process_multiple_json_files(input_folder, output_folder)
