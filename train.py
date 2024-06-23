import os
os.environ["NCCL_P2P_DISABLE"] = "1"
#下载模型
from modelscope import snapshot_download
from tqdm import tqdm

# 第一次下载时打开
model_dir = snapshot_download('Qwen/Qwen2-1.5B-Instruct',cache_dir='./')

import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, GenerationConfig
from transformers import Trainer as tfTrainer

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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningModule, Trainer, loggers, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AdamW

from peft import LoraConfig, TaskType, get_peft_model


lora_path = "./output/Qwen2-1_5B-Instruct_novel_all"

num_gpus = torch.cuda.device_count()

# 定义你的 LightningDataModule
class MyDataModule(LightningDataModule):
    def __init__(self, tokenizer, train_data, batch_size=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.batch_size = batch_size

    def train_dataloader(self):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True)
        train_sampler = DistributedSampler(self.train_data)
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=data_collator,
        )

# 定义你的 LightningModule
class MyLightningModel(LightningModule):
    def __init__(self, tokenizer, lr=1e-4, warmup_steps=1000):
        super().__init__()

        # 创建 device_map
        device_map = {"": f"cuda:{self.local_rank}"} 
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map,torch_dtype=torch.bfloat16)

        model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False, # 训练模式
            r=8, # Lora 秩
            lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1# Dropout 比例
        )

        model = get_peft_model(model, config)
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.save_hyperparameters(ignore=['model'])
        self.num_training_steps = None

    def training_step(self, batch, batch_idx):
        self.move_params_to_device(self.model, self.model.device)
        for k in batch.keys():
            batch[k] = batch[k].to(self.model.device)
        outputs = self.model.forward(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def move_params_to_device(self, module, device):
        """
        递归地将模型参数移动到指定的设备上。

        Args:
            module: 当前模块。
            device: 目标设备。
        """
        for name, param in module.named_parameters():
            if "." in name:
                continue
            param.data = param.data.to(device)
            setattr(module, name, param)  # 更新参数引用

        for _, child in module.named_children():
            self.move_params_to_device(child, device)

    def on_train_start(self):
        self.num_training_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs // self.trainer.accumulate_grad_batches

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# 创建Lightning模型
lightning_model = MyLightningModel( tokenizer)
data_module = MyDataModule(tokenizer, tokenized_id, batch_size=2)

# 设置训练参数
trainer = Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=-1,
    strategy="ddp",
    accumulate_grad_batches=4,
    log_every_n_steps=10,
    callbacks=[
        ModelCheckpoint(dirpath="./output/Qwen2-1_5B-Instruct_novel_all/lightning",  # 保存模型的目录
            filename="{epoch}-{step}-{train_loss:.2f}",  # 文件名格式
            every_n_epochs=5,  # 每隔 1 个 epoch 保存一次模型
            save_weights_only=True,  # 只保存模型权重save_top_k=1, monitor="train_loss"),
        )
    ],
    logger=loggers.TensorBoardLogger("lightning_logs/", name="my_model"),
    precision="bf16"
)

# 开始训练
trainer.fit(
    model=lightning_model,
    datamodule=data_module,
)
trained_model = lightning_model.model 
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

trainer = tfTrainer(
    model=trained_model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

lora_path = "./output/Qwen2-1_5B-Instruct_novel_all"
trainer.save_model(lora_path + "/lightning_final")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

model_path = './Qwen/Qwen2-1___5B-Instruct'
lora_path = "./output/Qwen2-1_5B-Instruct_novel_all/lightning_final"

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
lora_path = "./output/Qwen2-1_5B-Instruct_novel_all/lightning_final"

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
with open("lightning_submit.json", "w") as file:
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
