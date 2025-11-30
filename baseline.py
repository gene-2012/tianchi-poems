# 使用 vllm,使用方式见GitHub vllm
#这是我的服务器启动
# vllm serve /mnt/home/user04/CCL/model/Qwen2.5-7B-Instruct  --served-model-name qwen2.5-7b   --max_model_len 20000 
import os
import json
from openai import OpenAI
from tqdm import tqdm
import re
import dotenv
dotenv.load_dotenv()
# 设置 HTTP 代理
openai_api_key = os.getenv("API_KEY")
openai_api_base = "https://api.siliconflow.cn/v1"
client = OpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key
)
model = 'Qwen/Qwen3-8B'
print(model)
with open("prompts/version1.txt", "r", encoding="utf-8") as f:
    prompt = f.read()
def get_response(data):
    global prompt
    for _ in range(3):  # 三次重传机制
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt % data}],
                stream=False
            )
            # print(response)
            content = response.choices[0].message.content.strip()
            # 使用正则提取 JSON 格式的内容（匹配 { } 之间的内容）
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                answer = match.group(0)  # 获取匹配的 JSON 字符串
                answer = answer.strip()  # 去除前后空格
                answer = json.loads(answer)
                return answer
        except json.JSONDecodeError:
            continue
    return {
        "idx": data.get("index", ""),
        "ans_qa_words": {},
        "ans_qa_sents": {},
        "choose_id": -1
    }

def main():
    # 读取输入数据
    output_path = 'output.json'
    input_path =  'eval-data/eval_data.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)[:10] #  [:10] 可以打印几条数据测试
    
    output_data = []
    for data in tqdm(input_data, desc="Processing"):
        answer = get_response(data)
        print(answer)
        if answer:
            output_data.append(answer)
        else:
            print(f"未能生成答案的数据")

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成。共处理 {len(output_data)} 条数据，输出已保存到 {output_path}。")

if __name__ == "__main__":
    main()