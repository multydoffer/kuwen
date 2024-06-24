import json

# 读取原始 JSON 文件
with open("dataset2.json", "r", encoding="utf-8") as f:
    data = f.readlines()

# 将数据分成四个部分
chunk_size = len(data) // 4
part1 = data[:chunk_size]
part2 = data[chunk_size : 2 * chunk_size]
part3 = data[2 * chunk_size : 3 * chunk_size]
part4 = data[3 * chunk_size :]

# 将数据写入四个新文件
with open("./temp/part1.json", "w", encoding="utf-8") as f:
    f.writelines(part1)

with open("./temp/part2.json", "w", encoding="utf-8") as f:
    f.writelines(part2)

with open("./temp/part3.json", "w", encoding="utf-8") as f:
    f.writelines(part3)

with open("./temp/part4.json", "w", encoding="utf-8") as f:
    f.writelines(part4)
