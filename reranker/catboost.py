import pandas as pd
import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# 1. 数据加载
data = pd.read_csv('candidate_data.csv')  # 包含用户-物品对和相关特征
features = [col for col in data.columns if col not in ['user_id', 'item_id', 'label']]

# 2. 数据划分
X_train, X_val, y_train, y_val = train_test_split(data[features], data['label'], test_size=0.2, random_state=42)

# 3. 模型配置
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',  # 或 'PairLogit' 用于排序
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

# 4. 模型训练
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=[],  # 如果有分类特征，填入其列索引
    use_best_model=True
)

# 5. 模型评估
y_pred = model.predict_proba(X_val)[:, 1]
print(f"AUC: {roc_auc_score(y_val, y_pred)}")

# 6. 模型保存
model.save_model('catboost_rerank_model.cbm')

# 7. 在线推理
test_data = pd.read_csv('test_candidate_data.csv')  # 新的候选集
test_features = test_data[features]
test_scores = model.predict_proba(test_features)[:, 1]
test_data['score'] = test_scores
final_ranking = test_data.sort_values(by='score', ascending=False)
