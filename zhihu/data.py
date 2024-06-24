import json

# 打开JSON文件并加载数据
with open("./dataset2.json", "r", encoding="utf-8") as file:
    json_data = file.readlines()

# 将每四行数据合并为一个JSON对象
novel_data = []
for i in range(0, len(json_data), 4):
    json_str = "".join(json_data[i : i + 4])
    json_obj = json.loads(json_str)
    novel_data.append(json_obj)

for i in novel_data:
    print(f"《{i['name']}》的字数为： {len(i['text'])} 字")

for i in range(len(novel_data)):
    text_data = novel_data[i]["text"]
    start_index = text_data.find("出自专栏")
    if start_index == -1:
        continue
    end_index = text_data.find("\n", start_index)
    if start_index != -1 and end_index != -1:
        novel_data[i]["text"] = text_data[end_index + 1 :]
    else:
        print(f"{novel_data[i]['name']}出自专栏后没有换行符")

for i in range(len(novel_data)):
    text_data = novel_data[i]["text"]
    start_index = text_data.find("备案号")
    if start_index == -1:
        continue
    novel_data[i]["text"] = text_data[:start_index]

# 拆分《呼啸山庄》的文本为 800 字一段的段落
import jieba

paragraphs = []
for i in range(len(novel_data)):
    # 读取数据集中第 4 本小说《呼啸山庄》的文本作为训练集数据来源
    data = novel_data[i]["text"]
    story_name = novel_data[i]["name"]
    # 利用jieba进行句子切分
    sentences = []

    for sentence in data.split("\n"):  # 使用\n作为切分符
        sentences.append(sentence)

    # 将句子合并成800字一段的段落
    current_paragraph = ""
    for sentence in sentences:
        if len(current_paragraph) + len(sentence) <= 800:
            current_paragraph += sentence
        else:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = sentence

    # 将最后一段加入到段落列表中
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    if i % 1000 == 0:
        print(f"已处理{i}本小说")

from loguru import logger
import json
from tqdm import tqdm
import time
import os
from openai import OpenAI

# 配置loguru输出到文件
logger.remove()  # 移除默认的控制台输出
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    level="INFO",
    rotation="00:00",
    retention="10 days",
)


# 使用deepseek-chat api给段落打标签的接口
def get_response(text):
    client = OpenAI(
        api_key="sk-ad4b59759cfd41228839ddc634fa29f6",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://api.deepseek.com",  # 填写DashScope SDK的base_url
    )
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "总结user提交的内容。用一句不超过50字的话总结这段小说的情节。仅回答总结，不需要添加其他内容。",
            },
            {"role": "user", "content": text},
        ],
    )

    return completion.choices[0].message.content


# 设置容错机制，可最多重试 5 次，如果失败记录错误日志
def get_summary_with_retry(text):
    max_retries = 5
    retry_delay = 5  # in seconds
    attempts = 0
    while attempts < max_retries:
        try:
            return get_response(text)
        except Exception as e:
            attempts += 1
            if attempts < max_retries:
                logger.warning(
                    f"Attempt {attempts} failed for text: {text}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"All {max_retries} attempts failed for text: {text}. Error: {e}"
                )
                raise


# 创建文件夹
os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("dataset", exist_ok=True)


# 批量给指定的小说打标签的接口函数
def build_dataset(novel, texts):
    instruction_prompt = (
        "你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。"
    )

    # ... (多进程处理部分)
    num_processes = multiprocessing.cpu_count()
    chunk_size = len(texts) // num_processes

    # 将段落列表分割成多个块
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)

    # 并行处理每个块
    results = pool.starmap(
        process_chunk, [(chunk, novel, instruction_prompt) for chunk in chunks]
    )

    # 合并结果
    dataset = []
    dataset_error = []
    for result in results:
        dataset.extend(result[0])
        dataset_error.extend(result[1])

    with open(f"./data/{novel}.json", "w") as f:
        f.write(json.dumps(dataset, ensure_ascii=False, indent=4))

    with open(f"./data/{novel}_error.txt", "w") as f:
        f.write(json.dumps(dataset_error, ensure_ascii=False, indent=4))
    return dataset


for i in range(1000, len(paragraphs), 1000):
    dataset = build_dataset(i // 1000, paragraphs[i : i + 1000])
