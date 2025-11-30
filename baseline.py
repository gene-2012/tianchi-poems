import json
from tqdm import tqdm
import re
import dotenv
from utils import client, get_prompt

dotenv.load_dotenv()
model = 'Qwen/Qwen3-8B'
print(model)

prompt = get_prompt("version2.txt")

def get_response(data):
    global prompt
    for _ in range(3):  # 三次重传机制
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt % data}],
                stream=False
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                answer = match.group(0)
                answer = answer.strip()
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