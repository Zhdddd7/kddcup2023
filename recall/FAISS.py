import faiss
import pickle
import torch
import os
import numpy as np

# 加载物品嵌入
with open('./feature/id_to_embedding_DE.pkl', 'rb') as f:
    item_embeddings = pickle.load(f)

# 准备数据
item_ids = list(item_embeddings.keys())
embedding_matrix = torch.stack([
    item_embeddings[item_id].detach() for item_id in item_ids
]).cpu().numpy()


for item_id, embedding in item_embeddings.items():
    if torch.isnan(embedding).any() or torch.isinf(embedding).any():
        print(f"Invalid embedding found for item ID: {item_id}")

# FAISS 索引路径
faiss_path = './recall/faiss_index.bin'

if not os.path.exists(faiss_path):
    # 构建 FAISS 索引
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # 使用 L2 距离
    index.add(embedding_matrix)  # 将所有物品嵌入添加到索引
    print(f"FAISS index built with {len(item_ids)} items.")

    # 保存索引
    faiss.write_index(index, faiss_path)

    # 保存 item_ids 映射
    with open('./recall/item_ids.pkl', 'wb') as f:
        pickle.dump(item_ids, f)
    print(f"FAISS index and item IDs saved.")
else:
    # 加载已保存的索引
    index = faiss.read_index(faiss_path)
    with open('./recall/item_ids.pkl', 'rb') as f:
        item_ids = pickle.load(f)
    print(f"Loaded FAISS index from {faiss_path}.")

# 验证索引和映射的一一对应性
test_embedding = embedding_matrix[0].reshape(1, -1)
distances, indices = index.search(test_embedding.astype(np.float32), 10)
print("Test query results:", [item_ids[i] for i in indices[0]])
