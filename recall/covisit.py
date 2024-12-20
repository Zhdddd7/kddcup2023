
import heapq
from collections import defaultdict
import pandas as pd
import re
import pickle

train_sessions = pd.read_csv('./data/sessions_train.csv')
test_df = train_sessions.iloc[0]
print(test_df)
breakpoint()
test_df['prev_items'] = re.findall(r'[A-Z0-9]+', test_df['prev_items'])

def topk_recall(test_session, covisit_matrix, top_k=10):
    """
    基于共现矩阵进行 Top-K 召回
    :param test_session: 包含 prev_items 列表的测试会话
    :param covisit_matrix: 共现矩阵 (字典类型)
    :param top_k: 返回的候选物品数量
    :return: 推荐物品列表
    """
    prev_items = test_session['prev_items']
    print(prev_items)
    print(covisit_matrix[('B09W9FND7K', 'B09JSPLN1M')])
    candidate_scores = defaultdict(float)

    # 遍历用户历史交互物品
    for item in prev_items:
        # 获取与当前物品共现的物品
        for other_item, score in covisit_matrix.get(item, {}).items():
            candidate_scores[other_item] += score

    # 排除用户已经交互过的物品
    # for item in prev_items:
    #     if item in candidate_scores:
    #         del candidate_scores[item]

    # 按分数排序并取 Top-K
    top_k_items = heapq.nlargest(top_k, candidate_scores.items(), key=lambda x: x[1])
    return [item for item, score in top_k_items]

# 示例测试数据


# 示例共现矩阵 (可以根据实际情况调整为更高效的数据结构)
with open('./feature/covisit_matrix.pkl', 'rb') as f:
    covisit_matrix = pickle.load(f)

print(len(covisit_matrix))


# 召回结果
top_k_result = topk_recall(test_df, covisit_matrix, top_k=3)
print("Top-K 召回结果:", top_k_result)
