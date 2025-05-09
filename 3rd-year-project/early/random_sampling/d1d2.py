import os
import pandas as pd

def split_train_by_label(train_csv, output_folder):
    """
    Split train.csv into two files based on 'label' column.

    Args:
        train_csv (str): Path to train.csv.
        output_folder (str): Folder to save the split files.

    Returns:
        None
    """
    try:
        # 读取 train.csv
        df = pd.read_csv(train_csv)

        # 确保 'label' 列存在
        if 'label' not in df.columns:
            print(f"Error: 'label' column not found in {train_csv}")
            return

        # 拆分数据
        df_positive = df[df['label'] == 'positive']
        df_negative = df[df['label'] == 'negative']

        # 确保输出目录存在
        os.makedirs(output_folder, exist_ok=True)

        # 保存拆分后的文件
        d1_path = os.path.join(output_folder, "D1.csv")
        d2_path = os.path.join(output_folder, "D2.csv")

        df_positive.to_csv(d1_path, index=False, encoding='utf-8')
        df_negative.to_csv(d2_path, index=False, encoding='utf-8')

        print(f"Successfully split train.csv into:\n - {d1_path} (positive)\n - {d2_path} (negative)")

    except Exception as e:
        print(f"An error occurred while splitting train.csv: {e}")

# 设定路径
train_csv_path = "mixed_dataset/train/train.csv"
output_folder = "mixed_dataset/train"

# 执行拆分
split_train_by_label(train_csv_path, output_folder)
