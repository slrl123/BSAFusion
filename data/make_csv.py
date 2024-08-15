import os
import csv

def list_files(directory):

    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            files_list.append((file, relative_path))
    return files_list

def write_to_csv(file_list, csv_file):

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'path'])
        for file_info in file_list:
            writer.writerow(file_info)

# 要遍历的文件夹路径
directory_path = '/path/to/your/directory'

# 生成文件列表
files = list_files(directory_path)

# 写入CSV文件
csv_file_path = 'files_list.csv'
write_to_csv(files, csv_file_path)

print(f"CSV file has generated：{csv_file_path}")
