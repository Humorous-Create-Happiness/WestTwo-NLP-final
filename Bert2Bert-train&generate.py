import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, EncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator
import numpy as np
import torch.nn as nn

# 构建 FictionDataset 类（根据您的数据集）
class FictionDataset(Dataset):
    def __init__(self, txt_file, max_length=128):
        self.data = []  # 存储文本数据的列表
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_length = max_length

        # 从TXT文件中读取数据(并尽可能填充长度至128)
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().replace("\n", "")  # 移除换行符
            start_idx = 0
            while start_idx < len(text):
                chunk = text[start_idx : start_idx + self.max_length]
                self.data.append(chunk)
                start_idx += self.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # 使用BertTokenizer对文本进行分词，并添加特殊标记[CLS]和[SEP]
        # truncation=True 表示截断文本以适配指定的max_length
        # padding='max_length' 表示对文本进行填充以适配指定的max_length
        # return_tensors='pt' 表示返回PyTorch张量
        encoded = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length',
                                             truncation=True, return_tensors='pt')

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        token_type_ids = encoded['token_type_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': input_ids,  # 注意这里将输入的input_ids作为目标序列
            'label_ids': input_ids  # 添加 label_ids，表示要预测的下一个 token
        }


class NextTokenLoss(nn.Module):
    def __init__(self):
        super(NextTokenLoss, self).__init__()

    def forward(self, logits, labels):
        # 根据 labels 的形状获取要预测的下一个 token 位置
        next_token_positions = (labels != -100).nonzero(as_tuple=False)

        # 根据预测的下一个 token 位置，提取相应位置的 logits
        batch_indices = next_token_positions[:, 0]
        token_indices = next_token_positions[:, 1]
        predicted_logits = logits[batch_indices, token_indices]

        # 使用 nn.CrossEntropyLoss 计算损失
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(predicted_logits, labels.view(-1))  # 将 labels 调整为一维向量

        return loss


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = NextTokenLoss()  # 自定义的损失函数
        loss = loss_fn(logits, labels)
        return loss

# 加载 FictionDataset
dataset = FictionDataset(txt_file="综合.txt", max_length=128)


# 初始化 BertTokenizer 和 EncoderDecoderModel
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-chinese", "bert-base-chinese")

# 设置decoder_start_token_id和pad_token_id属性
bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id


# 将模型放置在CPU上
device = torch.device("cpu")
bert2bert.to(device)

# 加载训练好的权重（例如："bert2bert-model.bin"）
model_checkpoint = "bert2bert-model.bin"
bert2bert.load_state_dict(torch.load(model_checkpoint, map_location=device))
bert2bert.eval()

''''''

# 训练参数和配置
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=True  # 启用半精度训练
)

# 构建 Trainer 对象并进行训练
trainer = CustomSeq2SeqTrainer(
    model=bert2bert,
    args=training_args,
    data_collator=default_data_collator,  # 默认的数据整合器用于处理Seq2Seq数据
    train_dataset=dataset,
    compute_metrics=None  # 禁用默认的评估指标计算
)
trainer.train()

save_path = "bert2bert-model2.bin"
torch.save(bert2bert.state_dict(), save_path)
print("Model weights saved to:", save_path)


print('666')

# 生成文本示例
#text = "旧的太黄天世界，早已经毁灭，变成虚无死寂之地，"
#text = "单手持棒抵住进攻，一棒砸过去，另一只手则蓄积法力，随时准备偷袭；六耳猕猴见状，有样学样，也"
#text = "他们此行，武力最强的是伽罗明尊，知识最渊博的是白南轩"
#text = "路瑶珈抹去嘴角的血迹，冷声道：“我能感觉到，那条狗只怕是羿皇"
text = "孔雀心中也震惊万分，当初她带着凤烟柔一起逃婚，那"
def generate_text_with_beam_search(prompt,bad_words, max_length=30, temperature=0.7, top_k=50, top_p=0.95, beam_size=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)

    output = bert2bert.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=beam_size,  # 使用Beam Search
        no_repeat_ngram_size=3,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_beams=beam_size,  # 设置Beam Size
        bad_words_ids = bad_words  # 添加禁止词列表
    )

    generated_texts = []

    for seq in output:
        generated_text = tokenizer.decode(seq, skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts




# 创建禁止词列表
prompt_token_ids = tokenizer.encode(text, add_special_tokens=False)
bad_words = [prompt_token_ids]
badWords = tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
generated_texts = generate_text_with_beam_search(text,bad_words=bad_words)


for i, generated_text in enumerate(generated_texts):
    generated_text = generated_text.replace(' ', '')
    print(f"Generated Text {i+1}:", text + generated_text)
'''
def generate_text_with_context(prefix, max_length=50, temperature=0.8, top_k=25, top_p=0.9):
    generated_text = prefix

    while len(generated_text) < max_length:
        input_ids = tokenizer.encode(generated_text, return_tensors="pt")
        input_ids = input_ids.to(device)

        output = bert2bert.generate(
            input_ids=input_ids,
            max_length=len(generated_text) + 1,  # Generate one token at a time
            num_return_sequences=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size = 2  # Avoid generating repeated n-grams
        )

        next_token_id = output[0, -1]
        next_token = tokenizer.decode(next_token_id, skip_special_tokens=True)

        if next_token == '[SEP]':
            break

        generated_text += next_token
        print("Generated Text:", generated_text)

    return generated_text

# Example usage
prefix = "一到大门口，小短腿突然开始飞奔了起来，嘴里碎碎念"
generated_text = generate_text_with_context(prefix)
#print("Generated Text:", generated_text)'''

