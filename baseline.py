import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# 请到 https://platform.deepseek.com/api_keys 申请api填入下方，新用户赠送10元额度的token完全支持本次baseline速通~
DEEPSEEK_API_KEY = 'sk-8c785b34312444a38f0f62b7aceb5c30'

# 读取参考数据集
import json,time
from functools import reduce
novel_data = []
with open('./参考数据集.json', 'r',encoding='utf-8') as file:
    for line in file:
        novel_data.append(json.loads(line))


# 拆分《呼啸山庄》的文本为 800 字一段的段落
import jieba

paragraphs = []
for i in range(len(novel_data)):
    # 读取数据集中第 4 本小说《呼啸山庄》的文本作为训练集数据来源
    data = novel_data[i]["text"]
    story_name = novel_data[i]["name"]
    # 利用jieba进行句子切分
    sentences = []

    for sentence in data.split('。'):  # 使用句号作为切分符
        sentences.append(sentence)

    # 将句子合并成800字一段的段落
    current_paragraph = ''
    for sentence in sentences:
        if len(current_paragraph) + len(sentence) <= 800:
            current_paragraph += sentence+'。'
        else:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = sentence

    # 将最后一段加入到段落列表中
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())

    # # 打印切分后的段落
    # for idx, paragraph in enumerate(paragraphs):
    #     print(f'段落 {idx + 1}: {paragraph}')

from loguru import logger
import json
from tqdm import tqdm
import time
import os
from openai import OpenAI
# 配置loguru输出到文件
logger.remove()  # 移除默认的控制台输出
logger.add("logs/app_{time:YYYY-MM-DD}.log", level="INFO", rotation="00:00", retention="10 days", compression="zip")

import multiprocessing

# 使用deepseek-chat api给段落打标签的接口
def get_response(text):
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com",  
    )
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                'role': 'system', 
                'content': '总结user提交的内容。用一句不超过50字的话总结这段小说的情节。仅回答总结，不需要添加其他内容。'
            },
            {
                'role': 'user', 
                'content': text
            }
        ])
    
    return completion.choices[0].message.content

# 设置容错机制，可最多重试 5 次，如果失败记录错误日志
def get_summary_with_retry(text):
    max_retries = 5
    retry_delay = 10  # in seconds
    attempts = 0
    while attempts < max_retries:
        try:
            return get_response(text)
        except Exception as e:
            attempts += 1
            if attempts < max_retries:
                logger.warning(f"Attempt {attempts} failed for text: {text}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts failed for text: {text}. Error: {e}")
                raise

# 创建文件夹
os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

def process_chunk(chunk, novel, instruction_prompt):
    """Processes a chunk of texts and returns the processed dataset."""
    dataset = []
    dataset_error = []
    for text in tqdm(chunk, desc=f"Processing {novel}", total=len(chunk)):
        try:
            summary = get_summary_with_retry(text)
            print(summary)
            dataset.append({
                "instruction": instruction_prompt,
                "input": summary,
                "output": text
            })
        except Exception as e:
            dataset_error.append(text)
            logger.error(f"Failed to process text: {text}. Error: {e}")
    return dataset, dataset_error

# 批量给指定的小说打标签的接口函数
def build_dataset(novel,texts):
    instruction_prompt = "你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。"
    
    # ... (多进程处理部分)
    num_processes = multiprocessing.cpu_count() 
    chunk_size = len(texts) // num_processes 

    # 将段落列表分割成多个块
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)

    # 并行处理每个块
    results = pool.starmap(process_chunk, [(chunk, novel, instruction_prompt) for chunk in chunks])

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


dataset = build_dataset(story_name,paragraphs[:len(paragraphs)])

#下载模型
from modelscope import snapshot_download

# 第一次下载时打开
model_dir = snapshot_download('Qwen/Qwen2-1.5B-Instruct',cache_dir='./')

import json
import os

# 设置文件夹路径
directory_path = './data'

# 初始化一个空列表，用于存储合并后的数据
merged_data = []

# 遍历文件夹下的所有文件
for filename in os.listdir(directory_path):
    # 检查文件扩展名是否为.json
    if filename.endswith('.json'):
        # 构建文件的完整路径
        file_path = os.path.join(directory_path, filename)
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            # 加载JSON内容到变量
            data = json.load(file)
            # 将当前文件的数据添加到合并列表中
            merged_data.extend(data)

# 将合并后的数据转换为JSON格式
merged_json = json.dumps(merged_data, ensure_ascii=False, indent=4)

# 可以选择将合并后的数据写入到一个新的JSON文件中
output_file_path = './dataset/merged_story.json'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(merged_json)

# 或者直接输出到控制台
print(merged_json)

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

# 将JSON文件转换为CSV文件
df = pd.read_json('./dataset/merged_story.json')
# df = pd.read_json('./data/story/呼啸山庄.json')
ds = Dataset.from_pandas(df)

model_path = './Qwen/Qwen2-1___5B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

def process_func(example):
    MAX_LENGTH = 2048    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

import torch

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

model = get_peft_model(model, config)

lora_path = "./output/Qwen2-1_5B-Instruct_novel_all"

args = TrainingArguments(
    output_dir=lora_path,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=100,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

lora_path = "./output/Qwen2-1_5B-Instruct_novel_all"
trainer.save_model(lora_path + "/final")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

model_path = './Qwen/Qwen2-1___5B-Instruct'
lora_path = "./output/Qwen2-1_5B-Instruct_novel_all/final"

max_new_tokens = 2048

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

## 傲慢与偏见 data[0]
    # {
    #     "instruction": "你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。",
    #     "input": "一个有钱的单身汉必定想要娶妻，这是社会普遍认同的事实。班纳特太太兴奋地告诉丈夫，尼日斐花园被一位名叫彬格莱的富绅租下，她希望他能成为女儿们的潜在配偶，而班纳特先生则以幽默的方式回应她的期望。",
    #     "output": "凡是有钱的单身汉，总想娶位太太，这已经成了一条举世公认的真理。这样的单身汉，每逢新搬到一个地方，四邻八舍虽然完全不了解他的性情如何，见解如何，可是，既然这样的一条真理早已在人们心目中根深蒂固，因此人们总是把他看作自己某一个女儿理所应得的一笔财产。\n有一天班纳特太太对她的丈夫说：“我的好老爷，尼日斐花园终于租出去了，你听说过没有？”班纳特先生回答道，他没有听说过。\n“的确租出去了，”她说，“朗格太太刚刚上这儿来过，她把这件事的底细，一五一十地告诉了我。”班纳特先生没有理睬她。\n“你难道不想知道是谁租去的吗？”太太不耐烦地嚷起来了。\n“既是你要说给我听，我听听也无妨。”这句话足够鼓励她讲下去了。\n“哦！亲爱的，你得知道，郎格太太说，租尼日斐花园的是个阔少爷，他是英格兰北部的人；听说他星期一那天，乘着一辆驷马大轿车来看房子，看得非常中意，当场就和莫理斯先生谈妥了；他要在‘米迦勒节’以前搬进来，打算下个周未先叫几个佣人来住。”“这个人叫什么名字？”“彬格莱。”“有太太的呢，还是单身汉？”“噢！是个单身汉，亲爱的，确确实实是个单身汉！一个有钱的单身汉；每年有四五千磅的收入。真是女儿们的福气！”“这怎么说？关女儿女儿们什么事？”“我的好老爷，”太太回答道，“你怎么这样叫人讨厌！告诉你吧，我正在盘算，他要是挑中我们一个女儿做老婆，可多好！”“他住到这儿来，就是为了这个打算吗？”“打算！胡扯，这是哪儿的话！不过，他倒作兴看中我们的某一个女儿呢。他一搬来，你就得去拜访拜访他。”“我不用去。你带着女儿们去就得啦，要不你干脆打发她们自己去，那或许倒更好些，因为你跟女儿们比起来，她们哪一个都不能胜过你的美貌，你去了，彬格莱先生倒可能挑中你呢？”“我的好老爷，你太捧我啦。从前也的确有人赞赏过我的美貌，现在我可有敢说有什么出众的地方了。一个女人家有了五个成年的女儿，就不该对自己的美貌再转什么念头。”“这样看来，一个女人家对自己的美貌也转不了多少念头喽。"
    # },


prompt = "一个有钱的单身汉必定想要娶妻，这是社会普遍认同的事实。班纳特太太兴奋地告诉丈夫，尼日斐花园被一位名叫彬格莱的富绅租下，她希望他能成为女儿们的潜在配偶，而班纳特先生则以幽默的方式回应她的期望。"
messages = [
    {"role": "system", "content": "你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=max_new_tokens
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)

stories = [ 
"现代励志故事，一个失业青年如何克服生活困境，终于实现自我突破，成为行业翘楚的心路历程。",
 "一个现代女性穿越到古代某朝代后发生的传奇故事。", 
 "现代背景，一名神探警察遇到了一桩棘手的连环失踪案并将其侦破的故事。", 
 "古代背景，皇家侍卫和公主历经层层考验，突破身份桎梏的爱情故事。", 
 "现代玄幻背景，在一所驯服神兽的魔法学校中，围绕着三个学生小伙伴发生的奇幻冒险故事。", 
 "古代侦探系列，一位才华横溢的年轻学士，在解决一连串神秘案件中揭露皇室阴谋的故事。", 
 "二十一世纪初，一个小镇上发生的一系列神秘事件，让一群青少年开始探索超自然现象，并发现了小镇隐藏的古老秘密的故事。", 
 "现代都市背景，一个名不见经传的漫画家，通过与自己创作的虚拟角色“交流”，解决一系列诡秘案件的故事。", 
 "古代异界背景，一位天赋异禀的少年，在师傅的指导下学习古老的灵术，最终踏上寻找失落的神器，拯救家园的冒险旅程的故事。", 
 "繁华都市背景，一个单亲妈妈如何在抚养孩子和维持生计之间找到平衡，同时保持对自己梦想的追求的故事。", 
 "现代悬疑系列，一位心理学家利用自己的专业知识，帮助警方侦破一系列复杂的心理游戏案件。", 
 "现代心理惊悚背景，一名精神科医生被卷入一连串的脑控实验阴谋，如何在精神与现实的边缘徘徊求生的故事。", 
 "虚构古代背景，一位年轻的书生因缘巧合获得一本神秘典籍，开启了他成为一代宗师的修道之旅。", 
 "古代神话背景，一位勇者如何经过重重试炼，最终获取神器，拯救世界于水深火热之中的传奇故事。", 
 "虚拟现实背景，一群玩家在一款极度真实的VR游戏中探索未知世界并揭露游戏背后隐藏的秘密的故事。", 
 "穿越时空背景，一群来自不同时代的人意外聚集在一个神秘的地方，他们如何互相协作，解开时空之谜的故事。", 
 "科幻背景，一个机器人意识觉醒后，它如何在追求自我身份的同时，挑战人类社会关于存在和自由的根本问题。",
  "20世纪60年代的欧洲，一个侦探在解决一起跨国艺术品盗窃案中，逐渐揭露出一个关于失落宝藏的大阴谋。", 
  "现代都市背景，一位因交通事故失去双腿的舞者，通过先进的义肢技术重新站起来，重新找回舞台与自我的故事。", 
  "古代背景，一个普通医女奋斗成为朝廷高官，最终影响整个王朝政治格局变化的故事。" 
  ]

# 微调模型配置

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = './Qwen/Qwen2-1___5B-Instruct'
lora_path = "./output/Qwen2-1_5B-Instruct_novel_all/final"

max_new_tokens = 2048

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

# 批处理函数
def baseline_model(tasks,model):
    res = []
    for task in tqdm(tasks):
        messages = [
            {"role": "system", "content": "你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。"},
            {"role": "user", "content": task}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        # 为了演示我们只生成三条, 正式提交时,请改为50
        num_gen = 50
        for n in range(num_gen):
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            res.append({
                "instruction":"你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。",
                "input":task,
                "output":response,
            })
    return res

# 启动批处理存为json 这里生成第一个小说为例~
res_novel = baseline_model(stories[:],model)

# 为了保证可以提交我们针对每个小说题目生成了空数据  我们只填入前三条~

import json
with open("submit.json", "w") as file:
    for task_id, task in enumerate(stories):
        print(task)
        for t in range(50):
            response = ''
            data = {
                "instruction":"你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。",
                "input":task,
                "output":response,
            }
            # if(task==stories[0] and t<3):
            #     data =  res_novel[t]
            data = res_novel[task_id*50+t%50]
        # 将每个元素写入文件，并添加换行符
            file.write(json.dumps(data, ensure_ascii=False) + "\n")
