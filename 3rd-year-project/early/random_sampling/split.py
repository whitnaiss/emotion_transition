import os
import pandas as pd

def merge_csv_files(input_folder, output_file):
    """
    Merge all CSV files in a given folder into a single CSV file.

    Args:
        input_folder (str): Path to the directory containing CSV files.
        output_file (str): Path to save the merged CSV file.

    Returns:
        None
    """
    try:
        # 获取所有 CSV 文件路径
        csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]

        if not csv_files:
            print(f"No CSV files found in {input_folder}.")
            return

        # 读取所有 CSV 文件并合并
        df_list = [pd.read_csv(file) for file in csv_files]
        merged_df = pd.concat(df_list, ignore_index=True)

        # 保存合并后的 CSV 文件
        merged_df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"Successfully merged {len(csv_files)} CSV files into {output_file}")

        # 可选：删除原来的单个 CSV 文件（如果不想保留）
        for file in csv_files:
            if file != output_file:  # 避免误删自己
                os.remove(file)
        print(f"Original CSV files deleted after merging in {input_folder}.")

    except Exception as e:
        print(f"An error occurred while merging CSV files in {input_folder}: {e}")

# 设定 `mixed_dataset` 目录
mixed_dataset_folder = "mixed_dataset"

# 处理 train, val, test 目录
merge_csv_files(os.path.join(mixed_dataset_folder, "train"), os.path.join(mixed_dataset_folder, "train", "train.csv"))
merge_csv_files(os.path.join(mixed_dataset_folder, "val"), os.path.join(mixed_dataset_folder, "val", "val.csv"))
merge_csv_files(os.path.join(mixed_dataset_folder, "test"), os.path.join(mixed_dataset_folder, "test", "test.csv"))
