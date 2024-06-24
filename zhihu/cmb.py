import json

# 从三个文件读取数据
data_combined = []
for part_file in [
    "./temp/part1.json",
    "./temp/part2.json",
    "./temp/part3.json",
    "./temp/part4.json",
]:
    with open(part_file, "r", encoding="utf-8") as f:
        data_combined.extend(f.readlines())

# 将数据写入dataset2.json
with open("./temp/dataset2.json", "w", encoding="utf-8") as f:
    f.writelines(data_combined)
