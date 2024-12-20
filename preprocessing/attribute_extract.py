import pandas as pd
from tqdm import tqdm
import pickle

pro_df = pd.read_csv('./data/products_train.csv')
print(pro_df.head()['size'])
id2locale = {}
id2brand = {}
id2color = {}
id2size = {}
id2model = {}
id2material = {}

for _, item in tqdm(pro_df.iterrows()):
    id2locale[item['id']] = item['locale']
    id2brand[item['id']] = item['brand']
    id2color[item['id']] = item['color']
    id2size[item['id']] = item['size']
    id2model[item['id']] = item['model']
    id2material[item['id']] = item['material']

with open("./preprocessing/id2locale.pkl", 'wb') as f:
    pickle.dump(id2locale, f)

with open("./preprocessing/id2brand.pkl", 'wb') as f:
    pickle.dump(id2brand, f)

with open("./preprocessing/id2color.pkl", 'wb') as f:
    pickle.dump(id2color, f)

with open("./preprocessing/id2size.pkl", 'wb') as f:
    pickle.dump(id2size, f)

with open("./preprocessing/id2model.pkl", 'wb') as f:
    pickle.dump(id2model, f)

with open("./preprocessing/id2material.pkl", 'wb') as f:
    pickle.dump(id2material, f)