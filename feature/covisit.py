import pandas as pd
from collections import defaultdict
import ast
import pickle
from tqdm import tqdm
from itertools import combinations
import re
# 读取训练数据
train_sessions = pd.read_csv('./data/sessions_train.csv')

dic_list = ['locale', 'brand', 'color', 'size', 'model', 'material']

with open('./preprocessing/id2locale.pkl', 'rb') as f:
    id2locale = pickle.load(f)

with open('./preprocessing/id2brand.pkl', 'rb') as f:
    id2brand = pickle.load(f)

with open('./preprocessing/id2color.pkl', 'rb') as f:
    id2color = pickle.load(f)

with open('./preprocessing/id2size.pkl', 'rb') as f:
    id2size = pickle.load(f)

with open('./preprocessing/id2model.pkl', 'rb') as f:
    id2model = pickle.load(f)

with open('./preprocessing/id2material.pkl', 'rb') as f:
    id2material = pickle.load(f)


# 构建 id Co-Visit 矩阵
def build_id_covisit_matrix(train_sessions):
    covisit_matrix = defaultdict(int)
    locale_covisit_matrix = defaultdict(int)
    brand_covisit_matrix = defaultdict(int)
    color_covisit_matrix = defaultdict(int)
    size_covisit_matrix = defaultdict(int)
    model_covisit_matrix = defaultdict(int)
    material_covisit_matrix = defaultdict(int)
    swing_matrix = defaultdict(float)
    for _, row in tqdm(train_sessions.iterrows()):
        prev_item = row['prev_items']
        # print(prev_item)
        prev_item = re.findall(r'[A-Z0-9]+', prev_item)
        prev_item.append(row['next_item'])
        for item_i, item_j in combinations(prev_item, 2):
            if item_i != item_j:
                pair = tuple(sorted((item_i, item_j)))
                swing_matrix[pair] += 1/(len(prev_item) - 1)

                locale_covisit_matrix[(id2locale[pair[0]], id2locale[pair[1]])] += 1
                brand_covisit_matrix[(id2brand[pair[0]], id2brand[pair[1]])] += 1
                color_covisit_matrix[(id2color[pair[0]], id2color[pair[1]])] += 1
                size_covisit_matrix[(id2size[pair[0]], id2size[pair[1]])] += 1
                model_covisit_matrix[(id2model[pair[0]], id2model[pair[1]])] += 1
                material_covisit_matrix[(id2material[pair[0]], id2material[pair[1]])] += 1
                

    with open("./feature/swing_matrix.pkl", 'wb') as f:
        pickle.dump(swing_matrix, f)

    with open("./feature/covisit_matrix.pkl", 'wb') as f:
        pickle.dump(covisit_matrix, f)

    with open("./feature/locale_covisit_matrix.pkl", 'wb') as f:
        pickle.dump(locale_covisit_matrix, f)

    with open("./feature/brand_covisit_matrix.pkl", 'wb') as f:
        pickle.dump(brand_covisit_matrix, f)

    with open("./feature/color_covisit_matrix.pkl", 'wb') as f:
        pickle.dump(color_covisit_matrix, f)

    with open("./feature/size_covisit_matrix", 'wb') as f:
        pickle.dump(size_covisit_matrix, f)
    
    with open("./feature/model_covisit_matrix.pkl", 'wb') as f:
        pickle.dump(model_covisit_matrix, f)

    with open("./feature/material_covisit_matrix.pkl", 'wb') as f:
        pickle.dump(material_covisit_matrix, f)
 
    
build_id_covisit_matrix(train_sessions)
# 调用函数，构建矩阵
# print("开始构建 id Co-Visit 矩阵...")
# id_covisit_matrix = build_id_covisit_matrix(train_df['prev_items'])

# # 保存 Co-Visit 矩阵
# with open('id_covisit_matrix.pkl', 'wb') as f:
#     pickle.dump(id_covisit_matrix, f)

# print(f"id Co-Visit 矩阵构建完成，共包含 {len(id_covisit_matrix)} 对商品")
