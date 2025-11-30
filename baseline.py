import json
from tqdm import tqdm
import re
import dotenv
import numpy as np
import logging
import os
from utils import (
    client, 
    get_prompt, 
    k_nearest_neighbors, 
    get_embedding, 
    filter_content,
    create_model_client
)

# 获取当前模块的logger
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# 通过环境变量确定使用的模型客户端类型，默认为 siliconflow
MODEL_CLIENT_TYPE = os.getenv("MODEL_CLIENT_TYPE", "siliconflow")  # "siliconflow" 或 "ollama"
model_client = create_model_client(MODEL_CLIENT_TYPE)

# 模型名称可以通过环境变量配置
model = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
print(f"Using model: {model} with client: {MODEL_CLIENT_TYPE}")

prompt = get_prompt("version2.txt")
with open("train-data/result.json", "r", encoding="utf-8") as f:
    train_set = json.load(f)
    train_embeddings = np.array([item['embedding'] for item in train_set])

def get_response(data):
    global prompt, model_client
    try:
        embedding = get_embedding(data['content'])
        if embedding.size == 0:
            logger.warning(f"Empty embedding for data index: {data.get('index', 'unknown')}")
            return None
            
        nearest_neighbors = k_nearest_neighbors(embedding, train_embeddings)
        # 构造邻居提示内容，排除embedding字段
        neighbor_prompts = '\n'.join(
            str(filter_content(train_set[i], exclude_keys=['embedding'])) 
            for i in nearest_neighbors
        )
        cur_prompt = prompt % (data, neighbor_prompts)
        logger.debug(f"Generated prompt for index {data.get('index', 'unknown')}: {cur_prompt}")
        
        for attempt in range(3):  # 三次重传机制
            try:
                # 使用统一的模型客户端接口
                response = model_client.chat_completion(
                    messages=[{"role": "user", "content": cur_prompt}],
                    model=model
                )
                content = response.choices[0].message.content.strip()
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    answer = match.group(0)
                    answer = answer.strip()
                    answer = json.loads(answer)
                    logger.info(f"Successfully generated response for index: {data.get('index', 'unknown')}")
                    return answer
                else:
                    logger.warning(f"No JSON found in response for index: {data.get('index', 'unknown')}, attempt: {attempt+1}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error for index: {data.get('index', 'unknown')}, attempt: {attempt+1}, error: {e}")
                continue
            except Exception as e:
                logger.error(f"API call failed for index: {data.get('index', 'unknown')}, attempt: {attempt+1}, error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing data index {data.get('index', 'unknown')}: {e}")
        
    return {
        "idx": data.get("index", ""),
        "ans_qa_words": {},
        "ans_qa_sents": {},
        "choose_id": -1
    }

def main():
    # 读取输入数据
    output_path = 'output.json'
    input_path = 'eval-data/eval_data.json'
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        logger.info(f"Loaded {len(input_data)} items from input data")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return

    output_data = []
    try:
        for data in tqdm(input_data, desc="Processing"):
            answer = get_response(data)
            if answer:
                output_data.append(answer)
            else:
                logger.warning(f"Failed to generate answer for index: {data.get('index', '')}")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("已停止处理。")
    except Exception as e:
        logger.error(f"Error during processing: {e}")

    # 写入输出文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Processing completed. Output saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
    
    print(f"处理完成。共处理 {len(output_data)} 条数据，输出已保存到 {output_path}。")

if __name__ == "__main__":
    main()