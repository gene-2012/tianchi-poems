import dotenv
import os
from openai import OpenAI
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables")

base_url = os.getenv("API_BASE_URL", 'https://api.siliconflow.cn/v1')
try:
    client = OpenAI(base_url=base_url, api_key=API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

def get_embedding(text, model='netease-youdao/bce-embedding-base_v1'):
    """
    获取文本的embedding向量
    
    Args:
        text (str): 输入文本
        model (str): embedding模型名称
        
    Returns:
        np.ndarray: 文本的embedding向量
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid input text for embedding")
        return np.array([])
    
    try:
        response = client.embeddings.create(input=text, model=model)
        embedding = np.array(response.data[0].embedding)
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return np.array([])

def get_prompt(filename):
    """
    从prompts目录读取提示词文件
    
    Args:
        filename (str): 提示词文件名
        
    Returns:
        str: 提示词内容
        
    Raises:
        FileNotFoundError: 文件未找到
        IOError: 文件读取错误
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
        
    file_path = os.path.join('prompts', filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
        logger.info(f"Successfully loaded prompt from {filename}")
        return prompt
    except FileNotFoundError:
        error_msg = f"Prompt file {filename} not found in prompts directory"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except IOError as e:
        error_msg = f"Failed to read prompt file {filename}: {e}"
        logger.error(error_msg)
        raise IOError(error_msg)
    
def k_nearest_neighbors(embedding, embeddings, k=5):
    return np.argsort(np.linalg.norm(embedding - embeddings, axis=1))[:k]

def filter_content(item, exclude_keys):
    return {k: v for k, v in item.items() if k not in exclude_keys}
