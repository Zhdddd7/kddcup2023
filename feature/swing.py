import heapq
import pickle
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import re
import csv

def recommend_topk_from_swing(prev_items, swing_matrix, top_k=5):
    """
    基于 Swing 矩阵推荐 Top-K 物品
    :param prev_items: 用户历史交互物品列表
    :param swing_matrix: Swing 矩阵，格式为 { (item_i, item_j): score }
    :param top_k: 推荐物品的数量
    :return: 推荐物品列表
    """
    candidate_scores = defaultdict(float)

    # 遍历用户历史物品，查找 Swing 矩阵
    for item in prev_items:
        for (item_i, item_j), score in swing_matrix.items():
            if item == item_i and item_j not in prev_items:
                candidate_scores[item_j] += score
            elif item == item_j and item_i not in prev_items:
                candidate_scores[item_i] += score

    # 获取 Top-K 候选物品
    top_k_items = heapq.nlargest(top_k, candidate_scores.items(), key=lambda x: x[1])

    return [item for item, score in top_k_items]


with open('./feature/swing_matrix.pkl', 'rb') as f:
    swing_matrix = pickle.load(f)

test_df = pd.read_csv('./data/sessions_test_task1_phase1.csv')
gt_df = pd.read_csv('./data/gt_task1_phase1.csv')
rec_list = []
gt = gt_df['next_item'].to_list()
for _, row in tqdm(test_df.iterrows()):
    prev_items = row['prev_items']
    prev_items = re.findall(r'[A-Z0-9]+', prev_items)
    top_k_recommendations = recommend_topk_from_swing(prev_items, swing_matrix, top_k=100)
    rec_list.append(top_k_recommendations)

from utils import compute_mrr
mrr = compute_mrr(rec_list, gt)
print(f"the current rec list has a mrr of {mrr}")
rec_list_df = pd.DataFrame(rec_list)
rec_list_df.to_csv('./recall/rec_by_swing.csv')

