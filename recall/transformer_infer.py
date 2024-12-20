import faiss
import pickle
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformer_train import TransformerXLNet

# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# # 加载已保存的索引
faiss_path = './recall/faiss_index.bin'
item_ids_path = './recall/item_ids.pkl'

# # 加载索引和物品 ID 映射
index = faiss.read_index(faiss_path)
with open(item_ids_path, 'rb') as f:
    item_ids = pickle.load(f)

print(f"Loaded FAISS index with {len(item_ids)} items.")

# 定义召回函数
def search_top_k(index, query_embedding, item_ids, k=100):
    """
    使用 FAISS 索引查找最相似的 Top-K 项目。
    
    Args:
        index: FAISS 索引对象。
        query_embedding: 查询向量，形状为 (1, embedding_dim)。
        item_ids: 物品 ID 列表，与索引顺序一致。
        k: 返回的 Top-K 项目数。
    
    Returns:
        List of Top-K item IDs.
    """
    # 查询索引
    distances, indices = index.search(query_embedding, k)
    # 映射回物品 ID, 这里是批量操作
    results = [[item_ids[i] for i in query] for query in indices]
    return results

# 示例查询
# 假设您有一个查询向量 query_embedding
def generate_query(model, batch, device = 'cpu'):
    model.to(device)
    model.eval()
    # 转换批量会话为嵌入矩阵
    batch_embeddings = []
    for session in batch:
        session_embedding = torch.stack([
            model.item_embeddings[item].to(device) if item in model.item_embeddings else torch.zeros(model.embedding_dim).to(device)
            for item in session
        ])
        batch_embeddings.append(session_embedding)

    # 将批量嵌入堆叠为张量，形状为 [batch_size, seq_len, embedding_dim]
    batch_tensor = torch.stack(batch_embeddings).to(torch.float32).to(device)

    # 使用模型生成查询嵌入
    with torch.no_grad():
        session_output = model.transformer(batch_tensor)  # [batch_size, seq_len, embedding_dim]
        query_embeddings = model.xlnet_layer(session_output[:, -1, :])  # 取最后一个时间步的输出
    return query_embeddings.cpu().numpy()  # 转换为 numpy 数组

class sessionDataset(Dataset):
    def __init__(self, sessions):
        self.sessions = sessions

    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = self.sessions.iloc[idx]
        return session

def session_collate_fn(batch):
    max_len = max(len(session) for session in batch)
    # 对每个会话进行填充
    padded_batch = [session + ["<PAD>"] * (max_len - len(session)) for session in batch]

    return padded_batch


train_sessions = pd.read_csv('./data/sessions_train.csv')
test_sessions = pd.read_csv('./data/sessions_test_task1_phase1.csv')
train_sessions = train_sessions[train_sessions['locale'] == 'DE']
test_sessions = test_sessions[test_sessions['locale'] == 'DE']
df = pd.concat([train_sessions, test_sessions], ignore_index= True)
print(f'Current we have {len(df)} records')

# 处理数据
df['prev_items'] = df['prev_items'].apply(lambda x: re.findall(r'[A-Z0-9]+', x))
# df['prev_items'] = df.apply(lambda row: row['prev_items'] + [row['next_item']], axis=1)
print(df.head())
df = df.iloc[:6]

with open('./feature/id_to_embedding_DE.pkl', 'rb') as f:
    item_embeddings = pickle.load(f)


dataset = sessionDataset(df['prev_items'])
loader = DataLoader(dataset, batch_size= 2, collate_fn= session_collate_fn)
embedding_dim = 1536
model = TransformerXLNet(embedding_dim, item_embeddings)
model.load_state_dict(torch.load('./checkpoint/mlm_checkpoint_1.pt', map_location= torch.device('cpu')))
# device='cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'current running on {device}')

recall_result = []
for batch in loader:
    query_embeddings = generate_query(
        model=model,
        batch=batch,
        device=device
    )
    print(f"Query embeddings shape: {query_embeddings.shape}")  # (batch_size, embedding_dim)
    query_embeddings = query_embeddings.astype(np.float32)
    k = 5
    top_k_items = search_top_k(index, query_embeddings, item_ids, k=k)
    print(f"Top-{k} items: {top_k_items}")
    recall_result.extend(top_k_items)