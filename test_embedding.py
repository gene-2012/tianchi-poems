from utils import client, get_embedding
import json
import os
import tqdm
import dotenv
import numpy as np

dotenv.load_dotenv()
model = 'netease-youdao/bce-embedding-base_v1'
print(model)

def handle_data(data):
    for item in data:
        embedding = get_embedding(item['title'], model)
        item['embedding'] = embedding.tolist()

# 遍历 train-data 下所有 json 文件，包括子文件夹

result = []

for root, dirs, files in os.walk('train-data'):
    for file in files:
        if file.endswith('.json') and file != 'result.json':
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                handle_data(data)
                result.extend(data)
            print(f"处理完成 {file_path}")

with open('train-data/result.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)