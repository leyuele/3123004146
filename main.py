import sys
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_file(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if not os.path.isfile(file_path):
            raise IsADirectoryError(f"路径不是文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # 尝试其他编码（如GBK）
        with open(file_path, 'r', encoding='gbk', errors='ignore') as file:
            return file.read()
    except Exception as e:
        print(f"读取文件错误: {e}", file=sys.stderr)
        sys.exit(1)

"""
def preprocess_text(text):
    #文本预处理：分词
    # 使用jieba进行中文分词
    words = jieba.cut(text)
    # 过滤空字符串并将分词结果用空格连接
    return ' '.join([word for word in words if word.strip()])
"""
def preprocess_text(text):
    with open("stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())
    words = [word for word in jieba.cut(text) if word.strip() and word not in stopwords]
    return ' '.join(words)

""""
def calculate_similarity(original_text, copied_text):
    #计算两篇文本的相似度
    # 预处理文本
    original_processed = preprocess_text(original_text)
    copied_processed = preprocess_text(copied_text)

    # 创建TF-IDF向量器
    vectorizer = TfidfVectorizer()

    # 拟合并转换文本为TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform([original_processed, copied_processed])

    # 计算余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return similarity
"""
def calculate_similarity(original_text, copied_text):
    original_processed = preprocess_text(original_text)
    copied_processed = preprocess_text(copied_text)
    vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_matrix = vectorizer.fit_transform([original_processed, copied_processed])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

def write_result(result_path, similarity):
    """将结果写入文件"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        with open(result_path, 'w', encoding='utf-8') as file:
            # 保留两位小数
            file.write(f"{similarity:.2f}")
    except Exception as e:
        print(f"写入结果错误: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) != 4:
        print("用法: python main.py [原文文件路径] [抄袭版论文文件路径] [答案文件路径]", file=sys.stderr)
        sys.exit(1)

    # 获取文件路径
    original_path = sys.argv[1]
    copied_path = sys.argv[2]
    result_path = sys.argv[3]

    # 读取文件内容
    original_text = read_file(original_path)
    copied_text = read_file(copied_path)

    # 计算相似度
    similarity = calculate_similarity(original_text, copied_text)

    print(f"重复率：{similarity:.2f}")
    # 写入结果
    write_result(result_path, similarity)


if __name__ == "__main__":
    main()
