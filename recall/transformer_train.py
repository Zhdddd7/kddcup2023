import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import pickle
import re
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
    
def custom_collate_fn(batch):
    # 解构批量数据
    masked_sessions = [item['masked_session'] for item in batch]
    labels = [item['labels'] for item in batch]
    neg_samples = [item['neg_samples'] for item in batch]

    # 返回合并后的批量数据
    return {
        'masked_session': masked_sessions,
        'labels': labels,
        'neg_samples': neg_samples,
    }

def sampled_softmax_loss(logits, labels):
    return F.cross_entropy(logits, labels)

class MaskedSessionDataset(Dataset):
    def __init__(self, sessions: pd.Series, item_embeddings: dict, mask_prob=0.35, batch_size=1024):
        self.item_to_int = {item: idx for idx, item in enumerate(item_embeddings.keys())}
        self.int_to_item = {idx: item for item, idx in self.item_to_int.items()}
        self.sessions = sessions
        self.item_embeddings = item_embeddings
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.item_ids = list(item_embeddings.keys())
        self.max_session_length = max(self.sessions.apply(len))  # 获取最长 session 的长度

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions.iloc[idx]
        masked_session, labels = self._mask_session(session)

        # 如果 `labels` 全为 `<PAD>`，重新生成
        while all(label == "<PAD>" for label in labels):
            masked_session, labels = self._mask_session(session)

        # Padding masked_session 和 labels
        masked_session = self._pad_sequence(masked_session, self.max_session_length, pad_value="<PAD>")
        labels = self._pad_sequence(labels, self.max_session_length, pad_value="<PAD>")

        return {
            'masked_session': masked_session,
            'labels': labels,
            'neg_samples': self._negative_sampling(labels)
        }

    def _mask_session(self, session):
        masked_session = []
        labels = []
        num_to_mask = max(1, int(len(session) * self.mask_prob))  # 至少遮蔽一个物品
        mask_indices = random.sample(range(len(session)), num_to_mask)

        for i, item in enumerate(session):
            if i in mask_indices:
                masked_session.append("<MASK>")
                labels.append(item)
            else:
                masked_session.append(item)
                labels.append("<PAD>")
        return masked_session, labels


    def _negative_sampling(self, labels):
        # 过滤掉无效的标签
        valid_labels = [label for label in labels if label != "<PAD>"]

        # 确保生成的负样本数量一致
        in_batch_negatives = random.sample(valid_labels, min(len(valid_labels), self.batch_size))
        uniform_negatives = random.choices(self.item_ids, k=self.batch_size * 4)

        # 合并负样本，并去除正样本
        negatives = list(set(in_batch_negatives + uniform_negatives) - set(valid_labels))

        # 固定负采样数量（不足时填充）
        while len(negatives) < self.batch_size:
            negatives.append(random.choice(self.item_ids))  # 随机填充
        return negatives[:self.batch_size]  # 返回固定数量的负样本

    def _pad_sequence(self, sequence, max_length, pad_value):
        return sequence + [pad_value] * (max_length - len(sequence))

class TransformerXLNet(nn.Module):
    def __init__(self, embedding_dim, item_embeddings, num_heads=4, num_layers=2):
        super(TransformerXLNet, self).__init__()
        self.item_embeddings = item_embeddings
         
        self.embedding_dim = embedding_dim

        # 无激活 MLP
        self.mlp_no_activation = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Transformer + XLNet
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.xlnet_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Candidate Tower：有激活函数的 MLP
        self.candidate_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, masked_session, labels, negative_samples, sampled_softmax=True):
        """
        Forward pass with sampled softmax.
        :param masked_session: Tensor - Masked session sequences [batch_size, seq_len, embedding_dim]
        :param labels: Tensor - Positive labels embeddings [batch_size, seq_len, embedding_dim]
        :param negative_samples: Tensor - Negative samples embeddings [batch_size, neg_samples_per_batch, embedding_dim]
        :param sampled_softmax: bool - Whether to apply sampled softmax
        :return: logits and labels
        """
        # Step 1: Transformer 提取会话表示
        session_output = self.transformer(masked_session)[:, -1, :]  # [batch_size, embedding_dim]
        session_representation = self.xlnet_layer(session_output)  # [batch_size, embedding_dim]

        # Step 2: 拼接正样本和负样本
        all_candidates = torch.cat((labels[:, :1, :], negative_samples), dim=1)  # [batch_size, 1 + neg_samples_per_batch, embedding_dim]

        # Step 3: 计算 logits
        logits = torch.bmm(
            session_representation.unsqueeze(1), all_candidates.transpose(1, 2)
        ).squeeze(1)  # [batch_size, 1 + neg_samples_per_batch]

        if sampled_softmax:
            # Logits for sampled softmax: 正样本在第 0 位，其他为负样本
            return logits, torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        else:
            return logits

def train(model, loader, epochs=30, neg_samples_per_batch=3, saved = True):
    model.to(device)  # 将模型转移到 GPU
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        print(f'begin epoch {epoch}')
        total_loss = 0
        for batch in loader:
            # 转移输入到 GPU
            
            sessions = [
                torch.stack([model.item_embeddings[item].to(device) if item in model.item_embeddings else torch.zeros(model.embedding_dim).to(device)
                             for item in session])
                for session in batch['masked_session']
            ]
            sessions = torch.stack(sessions)  # [batch_size, seq_len, embedding_dim]

            labels = [
                torch.stack([model.item_embeddings[item].to(device) if item in model.item_embeddings else torch.zeros(model.embedding_dim).to(device)
                             for item in label])
                for label in batch['labels']
            ]
            labels = torch.stack(labels)  # [batch_size, seq_len, embedding_dim]

            neg_samples = [
                torch.stack([model.item_embeddings[item].to(device) if item in model.item_embeddings else torch.zeros(model.embedding_dim).to(device)
                             for item in neg_sample[:neg_samples_per_batch]])
                for neg_sample in batch['neg_samples']
            ]
            neg_samples = torch.stack(neg_samples)  # [batch_size, neg_samples_per_batch, embedding_dim]

            logits, targets = model(sessions, labels, neg_samples)

            # 转移目标张量到 GPU
            targets = targets.to(device)
            # 计算损失并优化
            loss = sampled_softmax_loss(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
    if saved:      
        torch.save(model.state_dict(), './checkpoint/mlm_checkpoint_1.pt')

# 训练模型
if __name__ ==  '__main__':

    # 检查 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载数据
    train_sessions = pd.read_csv('./data/sessions_train.csv')
    test_sessions = pd.read_csv('./data/sessions_test_task1_phase1.csv')
    train_sessions = train_sessions[train_sessions['locale'] == 'DE']
    test_sessions = test_sessions[test_sessions['locale'] == 'DE']
    df = pd.concat([train_sessions, test_sessions], ignore_index= True)
    print(f'Current we have {len(df)} records')

    # 处理数据
    df['prev_items'] = df['prev_items'].apply(lambda x: re.findall(r'[A-Z0-9]+', x))
    df['prev_items'] = df.apply(lambda row: row['prev_items'] + [row['next_item']], axis=1)
    print(df.head())
    # 加载物品嵌入
    with open('./feature/id_to_embedding_DE.pkl', 'rb') as f:
        item_embeddings = pickle.load(f)

    embedding_dim = 1536
    model = TransformerXLNet(embedding_dim, item_embeddings)

    dataset = MaskedSessionDataset(df['prev_items'], item_embeddings)
    loader = DataLoader(dataset, batch_size=128, collate_fn=custom_collate_fn)

    train(model, loader, epochs=30, neg_samples_per_batch=10)