import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,  BertForMaskedLM, AutoTokenizer,AutoModelWithLMHead
import pickle


# 定义 FictionDataset 类
class FictionDataset(Dataset):
    def __init__(self, txt_file, max_length=64, n_gram=8):
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_length = max_length
        self.n_gram = n_gram

        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().replace("\n", "")
            start_idx = 0
            while start_idx < len(text):
                chunk = text[start_idx: start_idx + self.max_length]
                self.data.append(chunk)
                start_idx += self.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        samples = []
        for i in range(len(text) - self.n_gram):
            sample_text = text[i:i + self.n_gram]
            encoded = self.tokenizer.encode_plus(sample_text, max_length=self.max_length, padding='max_length',
                                                 truncation=True, return_tensors='pt')
            input_ids = encoded['input_ids'].squeeze()
            attention_mask = encoded['attention_mask'].squeeze()
            token_type_ids = encoded['token_type_ids'].squeeze()

            samples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            })

        return samples


dataset_save_path = "Fiction-Dataset.pkl"
# 加载 FictionDataset
with open('MyFictionDataset.pkl', 'rb') as f:
    dataset = pickle.load(f)


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

gpt2_tokenizer = AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
gpt2_model = AutoModelWithLMHead.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
gpt2_model.to(device)


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertForMaskedLM.from_pretrained('bert-base-chinese')
bert_model.to(device)




batch_size = 8
learning_rate = 2e-5
epochs = 1

optimizer = torch.optim.Adam(list(bert_model.parameters()) + list(gpt2_model.parameters()), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

with open(dataset_save_path, 'rb') as f:
    dataset = pickle.load(f)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 联合训练
for epoch in range(epochs):
    bert_model.train()
    gpt2_model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        batch_samples = batch

        for sample_idx, sample in enumerate(batch_samples):
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)

            # BERT部分
            bert_labels = input_ids.clone()
            bert_labels[bert_labels == bert_tokenizer.pad_token_id] = -100  # 忽略 padding tokens，以免对结果产生影响

            optimizer.zero_grad()
            bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=bert_labels)
            bert_loss = bert_outputs.loss
            bert_loss.backward()

            # GPT-2部分
            gpt2_input_ids = input_ids[:, :-1].to(device)  # 删掉最后一个 token
            gpt2_labels = input_ids[:, 1:].contiguous().to(device)  # 把 input 变为 labels

            gpt2_outputs = gpt2_model(input_ids=gpt2_input_ids, labels=gpt2_labels)
            gpt2_loss = gpt2_outputs.loss
            gpt2_loss.backward()

            optimizer.step()

            total_loss += (bert_loss.item() + gpt2_loss.item())
            if (sample_idx + 1) % 50==1:
                print(f"Batch [{batch_idx + 1}/{len(dataloader)}], Sample [{sample_idx + 1}/{len(batch_samples)}], Batch Loss: {bert_loss.item() + gpt2_loss.item():.4f}")

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

torch.save({
    'bert_model_state_dict': bert_model.state_dict(),
    'gpt2_model_state_dict': gpt2_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'bert-gpt.pt')

checkpoint = torch.load('bert-gpt.pt')
bert_model.load_state_dict(checkpoint['bert_model_state_dict'])
gpt2_model.load_state_dict(checkpoint['gpt2_model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

bert_model.eval()
gpt2_model.eval()
#text_prompt = "贝冰榆仰头轻笑了一声，嘲讽的看着面前的中年女人，她觉得这个女人脑袋构造完全不能用正常人来判断"
text_prompt = "叶旭等人没有立刻冲出太和大殿，如今他们已经油尽灯枯，别说冲出北海秘境，他们甚至连祭起金箭冲出祭坛的能力"
max_length = 90
temperature = 0.7

input_ids = bert_tokenizer.encode(text_prompt, add_special_tokens=True, return_tensors="pt").to(device)
output = gpt2_model.generate(input_ids, max_length=max_length, num_return_sequences=1)

#generated_text = bert_tokenizer.decode(output[0], skip_special_tokens=True)
#print("Generated Text:", generated_text)


def generate_text(model, tokenizer, input_ids, max_length, temperature, no_repeat_ngram_size):
    generated_ids = input_ids.clone()
    for _ in range(max_length):
        logits = model(input_ids=generated_ids).logits[:, -1, :] / temperature
        next_token_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

        # 检查 no_repeat_ngram 片段
        if no_repeat_ngram_size > 0 and len(generated_ids[0]) >= no_repeat_ngram_size:
            for _ in range(no_repeat_ngram_size):
                if torch.equal(generated_ids[0, -no_repeat_ngram_size:], next_token_id):
                    next_token_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                    break

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

generated_text1 = generate_text(gpt2_model, bert_tokenizer, input_ids, max_length, temperature=0.7, no_repeat_ngram_size=2)
print("Generated Text2:", generated_text1)

print("111")
