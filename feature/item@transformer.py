import pandas as pd
import pickle
train_sessions =  pd.read_csv('./data/sessions_train.csv')
train_pro = pd.read_csv('./data/products_train.csv')
print('read completed')
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle


# 检查 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载 BERT 模型和 Tokenizer，并设置为 float16
bert_model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device).half()  # 使用 float16
bert_model.eval()

# 类别特征词汇表
brand_vocab = {b: idx for idx, b in enumerate(train_pro['brand'].dropna().unique())}
color_vocab = {c: idx for idx, c in enumerate(train_pro['color'].dropna().unique())}
size_vocab = {s: idx for idx, s in enumerate(train_pro['size'].dropna().unique())}
model_vocab = {m: idx for idx, m in enumerate(train_pro['model'].dropna().unique())}
material_vocab = {mat: idx for idx, mat in enumerate(train_pro['material'].dropna().unique())}
author_vocab = {a: idx for idx, a in enumerate(train_pro['author'].dropna().unique())}

# 定义类别特征的 Embedding 层，使用 float16
embedding_dim = 128
brand_embed = nn.Embedding(len(brand_vocab), embedding_dim).to(device).half()
color_embed = nn.Embedding(len(color_vocab), embedding_dim).to(device).half()
size_embed = nn.Embedding(len(size_vocab), embedding_dim).to(device).half()
model_embed = nn.Embedding(len(model_vocab), embedding_dim).to(device).half()
material_embed = nn.Embedding(len(material_vocab), embedding_dim).to(device).half()
author_embed = nn.Embedding(len(author_vocab), embedding_dim).to(device).half()

# 提取 BERT 嵌入
def get_bert_embedding(text):
    """提取文本的 BERT 嵌入表示 (平均池化)，并转换为 float16"""
    if pd.isna(text) or not text.strip():
        return torch.zeros(768, dtype=torch.float16).to(device)  # 返回 float16 的零向量
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=30).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().half()

# 初始化存储 item embeddings 的字典
id_to_embedding = {}

for _, row in tqdm(train_pro.iterrows(), total=len(train_pro)):
    # 类别特征 (GPU 上执行)
    brand_emb = brand_embed(torch.tensor(brand_vocab.get(row['brand'], 0)).to(device))
    color_emb = color_embed(torch.tensor(color_vocab.get(row['color'], 0)).to(device))
    size_emb = size_embed(torch.tensor(size_vocab.get(row['size'], 0)).to(device))
    model_emb = model_embed(torch.tensor(model_vocab.get(row['model'], 0)).to(device))
    material_emb = material_embed(torch.tensor(material_vocab.get(row['material'], 0)).to(device))
    author_emb = author_embed(torch.tensor(author_vocab.get(row['author'], 0)).to(device))

    # 文本特征 (BERT 嵌入)
    title_emb = get_bert_embedding(row['title'])  # 768 维 float16
    desc_emb = get_bert_embedding(row['desc'])   # 768 维 float16
    bert_emb = 0.7 * title_emb + 0.3 * desc_emb  # 加权求和，仍为 float16

    # 拼接所有特征 (128 x 6 类别特征 + 768 文本特征)
    item_emb = torch.cat([brand_emb, color_emb, size_emb, model_emb, material_emb, author_emb, bert_emb], dim=0)
    id_to_embedding[row['id']] = item_emb.cpu()  # 存储为 CPU 数据

# 保存字典，并转换为 float16
with open('id_to_embedding.pkl', 'wb') as f:
    pickle.dump({k: v.half() for k, v in id_to_embedding.items()}, f)  # 转为 float16 存储
print("Item embeddings saved successfully in float16 format.")
