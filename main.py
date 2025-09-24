import sys
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_file(file_path):
    """
    读取文件内容，处理编码和路径异常
    file_path: 要读取的文件路径
    return: 文件内容字符串
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        # 检查路径是否为文件
        if not os.path.isfile(file_path):
            raise IsADirectoryError(f"路径不是文件: {file_path}")
        # 尝试以UTF-8编码读取
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # UTF-8解码失败时，尝试GBK编码
        with open(file_path, 'r', encoding='gbk', errors='ignore') as file:
            return file.read()
    except Exception as e:
        # 捕获其他未知异常并提示退出
        print(f"读取文件错误: {e}", file=sys.stderr)
        sys.exit(1)


def preprocess_text(text):
    """
    文本预处理：分词、停用词过滤
    text: 原始文本字符串
    return: 预处理后的分词字符串（空格分隔）
    """
    try:
        # 加载停用词表
        with open("stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = set(f.read().splitlines())
        print(f"成功加载{len(stopwords)}个停用词")  # 验证是否加载成功
    except FileNotFoundError:
        # 停用词文件不存在时，降级为无停用词过滤
        print("Error: stopwords.txt not found!")
        stopwords = set()  # 停用词文件不存在时，设为空集合
    # 使用jieba分词并过滤停用词、空字符串
    words = [word for word in jieba.cut(text) if word.strip() and word not in stopwords]
    return ' '.join(words)


def calculate_similarity(original_text, copied_text):
    """
    计算两篇文本的相似度（基于TF-IDF和余弦相似度）
    original_text: 原文文本字符串
    copied_text: 抄袭文本字符串
    return: 相似度值（0~1之间）
    """
    # 对原文和抄袭文本分别进行预处理
    original_processed = preprocess_text(original_text)
    copied_processed = preprocess_text(copied_text)

    # 初始化TF-IDF向量化器，限制最大特征数为3000
    vectorizer = TfidfVectorizer(max_features=3000)

    # 拟合并转换文本为TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform([original_processed, copied_processed])

    # 计算余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity


def write_result(result_path, similarity):
    """
    将相似度结果写入指定文件
    result_path: 结果文件路径
    similarity: 相似度值
    """
    try:
        # 仅当目录非空时创建
        dir_name = os.path.dirname(result_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        # 将相似度结果保留两位小数并写入文件
        with open(result_path, 'w', encoding='utf-8') as file:
            # 保留两位小数
            file.write(f"{similarity:.2f}")
    except Exception as e:
        # 捕获写入异常并提示退出
        print(f"写入结果错误：{e}", file=sys.stderr)
        sys.exit(1)



def main():
    """主函数：处理命令行参数，执行全流程"""
    # 检查命令行参数数量（需传入3个文件路径）
    if len(sys.argv) != 4:
        print("用法: python main.py [原文文件路径] [抄袭版论文文件路径] [答案文件路径]", file=sys.stderr)
        sys.exit(1)

    # 解析命令行参数为文件路径
    original_path = sys.argv[1]
    copied_path = sys.argv[2]
    result_path = sys.argv[3]

    # 打印文件出处
    print(f"原文文件路径：{os.path.abspath(original_path)}")
    print(f"抄袭版论文文件路径：{os.path.abspath(copied_path)}")
    print(f"结果文件路径：{os.path.abspath(result_path)}")

    # 读取文件内容
    original_text = read_file(original_path)
    copied_text = read_file(copied_path)

    # 计算相似度
    similarity = calculate_similarity(original_text, copied_text)

    # 打印并写入相似度结果
    print(f"重复率：{similarity:.2f}")
    write_result(result_path, similarity)


if __name__ == "__main__":
    main()
