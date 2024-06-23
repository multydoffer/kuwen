from tqdm import tqdm
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

model_path = './Qwen/Qwen2-1___5B-Instruct'
lora_path = "./output/Qwen2-1_5B-Instruct_novel_all/checkpoint-6200"

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

max_new_tokens = 2048

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

# 批处理函数
def baseline_model(tasks, model):
    start_time = time.time()
    res = []
    for task in tqdm(tasks):
        messages = [
            {"role": "system", "content": "你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。"},
            {"role": "user", "content": task}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        # all_gen = 50  # 生成数量
        # num_gen = 50 
        # for i in range(all_gen//num_gen+1):
        #     generated_ids = model.generate(
        #         model_inputs.input_ids,
        #         max_new_tokens=max_new_tokens,
        #         num_return_sequences=num_gen if i!=all_gen//num_gen else all_gen%num_gen # 一次生成多个结果
        #     )
        #     generated_ids = [
        #         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        #     ]
        #     responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        #     for response in responses:
        #         res.append({
        #             "instruction":"你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。",
        #             "input":task,
        #             "output":response,
        #         })
        num_gen = 50  # 生成数量
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_gen  # 一次生成多个结果
        )
        print(generated_ids.shape)
        generated_ids_list = []
        for input_ids in model_inputs.input_ids:
            for i in range(num_gen):
                generated_ids_list.append(generated_ids[i, len(input_ids):])
        generated_ids = generated_ids_list

        # generated_ids = [
        #     output_ids[i, len(input_ids):] for i in range(num_gen) for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for response in responses:
            res.append({
                "instruction":"你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。",
                "input":task,
                "output":response,
            })
        print(len(res))
    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} seconds")
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
