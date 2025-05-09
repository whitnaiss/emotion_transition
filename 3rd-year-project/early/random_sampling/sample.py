import json
import csv
import os
import random
import shutil

def process_json_to_csv(json_file_path, output_csv_path, label):
    try:
        # Check if the JSON file exists
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"The file {json_file_path} does not exist.")

        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Prepare the data to extract id, text, and assign label
        extracted_data = []
        if isinstance(data, list):
            for item in data:
                if 'id' in item and 'text' in item:
                    extracted_data.append({
                        'id': item['id'],
                        'text': item['text'],
                        'label': label
                    })
        elif isinstance(data, dict):
            if 'id' in data and 'text' in data:
                extracted_data.append({
                    'id': data['id'],
                    'text': data['text'],
                    'label': label
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

def process_multiple_json_files(input_folder, output_folder, label):
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
                process_json_to_csv(json_file_path, output_csv_path, label)

        print(f"All JSON files have been processed and saved to {output_folder}.")
    except Exception as e:
        print(f"An error occurred while processing multiple files: {e}")

def sample_json_files(input_folder, output_folder, sample_size=500):
    """
    Randomly sample a specified number of JSON files from a given folder and copy them to a target folder.

    Args:
        input_folder (str): Path to the source folder.
        output_folder (str): Path to the target folder.
        sample_size (int): Number of files to sample.

    Returns:
        list: List of sampled file paths.
    """
    all_json_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.json')]

    if len(all_json_files) < sample_size:
        print(f"Not enough files in {input_folder}. Only {len(all_json_files)} files available.")
        return []
    else:
        sampled_files = random.sample(all_json_files, sample_size)
        os.makedirs(output_folder, exist_ok=True)
        for file_name in sampled_files:
            shutil.copy(file_name, os.path.join(output_folder, os.path.basename(file_name)))

        print(f"Successfully sampled {sample_size} JSON files from {input_folder} to {output_folder}")
        return sampled_files

def split_and_save_files(file_paths, output_folder, mixed_folder, test_ratio=0.2, val_ratio=0.15):
    random.shuffle(file_paths)

    # 确保 shuffled 文件夹存在
    os.makedirs(mixed_folder, exist_ok=True)

    # 复制所有 JSON 文件到 shuffled 文件夹
    for file in file_paths:
        shutil.copy(file, os.path.join(mixed_folder, os.path.basename(file)))

    # 计算拆分比例
    test_size = int(len(file_paths) * test_ratio)
    test_files = file_paths[:test_size]
    remaining_files = file_paths[test_size:]

    val_size = int(len(remaining_files) * val_ratio)
    val_files = remaining_files[:val_size]
    train_files = remaining_files[val_size:]

    # 创建 `train`, `val`, `test` 目录
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")
    test_folder = os.path.join(output_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 处理 JSON 文件并转换成 CSV
    def convert_and_save(json_file, output_dir):
        output_csv = os.path.join(output_dir, os.path.basename(json_file).replace('.json', '.csv'))
        label = "positive" if "positive" in json_file else "negative"
        process_json_to_csv(json_file, output_csv, label)

    for file in train_files:
        convert_and_save(file, train_folder)
    for file in val_files:
        convert_and_save(file, val_folder)
    for file in test_files:
        convert_and_save(file, test_folder)

    print(f"Data split completed with CSV conversion:")
    print(f"Shuffled JSON files: {len(file_paths)} saved in {mixed_folder}")
    print(f"Training set: {len(train_files)} CSV files")
    print(f"Validation set: {len(val_files)} CSV files")
    print(f"Test set: {len(test_files)} CSV files")


# Main script
folder1_path = "5384_negative"
output1_folder = "sampled_negative"

folder2_path = "6493_positive"
output2_folder = "sampled_positive"

output_mixed_folder = "mixed_dataset"
shuffled_folder = "shuffled_files"


# Step 1: Sample 500 files from each folder and save them in separate folders
sampled_files_negative = sample_json_files(folder1_path, output1_folder, sample_size=500)
sampled_files_positive = sample_json_files(folder2_path, output2_folder, sample_size=500)

# Step 2: Combine and shuffle the sampled files
all_sampled_files = sampled_files_negative + sampled_files_positive

# Step 3: Split and save mixed dataset into train, validation, and test sets
split_and_save_files(all_sampled_files, output_mixed_folder, shuffled_folder)

# Step 4: Process sampled files into CSV
process_multiple_json_files(output1_folder, os.path.join(output1_folder, 'csv_files'), label='negative')
process_multiple_json_files(output2_folder, os.path.join(output2_folder, 'csv_files'), label='positive')