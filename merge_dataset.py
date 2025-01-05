import os
import pandas as pd

# 定义数据文件的根目录
data_dir = 'data'

# 存储所有数据的列表
all_data = []

# 遍历目录下所有的子目录及文件
for root, dirs, files in os.walk(data_dir):
    for file in files:
        # 查找所有的 normalized_dataset.csv 文件
        if file == 'normalized_dataset.csv':
            file_path = os.path.join(root, file)
            # 读取CSV文件
            df = pd.read_csv(file_path)
            # 将文件的数据添加到列表中
            all_data.append(df)

# 将所有读取的数据合并成一个DataFrame
final_data = pd.concat(all_data, ignore_index=True)

# 将合并后的数据保存到新的CSV文件
final_data.to_csv('dataset.csv', index=False)

print("所有数据已成功汇总并保存为 'merged_normalized_dataset.csv'")