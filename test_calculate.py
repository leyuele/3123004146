import pytest
import os
import tempfile
from main import read_file, preprocess_text, calculate_similarity, write_result, main

@pytest.fixture
def test_files():
    """
    测试数据夹具：提供原文、抄袭文本和结果文件的默认路径
    """
    orig_path = "orig.txt"
    copied_path = "orig_0.8_dis_15.txt"
    result_path = "test_result.txt"
    yield orig_path, copied_path, result_path

def test_read_file_all_branches():
    """
    测试read_file函数的所有分支：正常读取、异常场景、编码兼容
    """
    # 1. 正常读取（UTF-8编码）
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        f.write("测试UTF-8编码的文本内容")
        utf8_path = f.name
    content = read_file(utf8_path)
    assert "测试UTF-8编码的文本内容" in content # 验证内容读取正确
    os.remove(utf8_path) # 清理临时文件

    # 2. 异常：文件不存在（FileNotFoundError）
    non_existent = "non_existent_file_1234.txt"
    with pytest.raises(SystemExit): # 验证文件不存在时触发程序退出
        read_file(non_existent)

    # 3. 异常：路径是目录（IsADirectoryError）
    with tempfile.TemporaryDirectory() as dir_path: # 创建临时目录
        with pytest.raises(SystemExit): # 验证路径为目录时触发程序退出
            read_file(dir_path)

    # 4. 异常：UTF-8解码失败，尝试GBK编码
    gbk_content = "测试GBK编码的文本内容：中文测试".encode('gbk')
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(gbk_content)
        gbk_path = f.name
    content = read_file(gbk_path)
    assert "中文测试" in content # 验证GBK编码兼容读取
    os.remove(gbk_path) # 清理临时文件

    # 5. 异常：其他未知异常（触发通用Exception）
    try:
        read_file(123)  # 传入非字符串路径 # 传入整数（非字符串）路径，触发通用异常
    except SystemExit:
        assert True, "未知异常触发了SystemExit"

def test_preprocess_text_all_branches():
    """
    测试preprocess_text函数的所有分支：分词、停用词过滤、边缘场景
    """
    # 1. 正常分词（含停用词过滤）
    raw_text = "这是一篇关于人工智能的测试文本，用于验证预处理功能！"
    processed = preprocess_text(raw_text)
    assert "人工智能" in processed, "核心术语未保留"
    assert "测试" in processed, "有效词汇‘测试’未保留"
    assert "文本" in processed, "有效词汇‘文本’未保留"
    assert "的" not in processed, "停用词‘的’未过滤"

    # 2. 包含英文的文本分词
    eng_text = "Artificial Intelligence是计算机科学的分支，AI技术发展迅速。"
    eng_processed = preprocess_text(eng_text)
    assert "Artificial" in eng_processed, "英文词汇未保留"
    assert "Intelligence" in eng_processed, "英文词汇未保留"
    assert "计算机科学" in eng_processed, "中文术语未保留"

    # 3. 异常：stopwords.txt不存在时的降级逻辑
    if os.path.exists("stopwords.txt"):
        os.rename("stopwords.txt", "stopwords_temp.txt")
    try:
        empty_text = "的 地 得 了 在"  # 纯停用词文本
        empty_processed = preprocess_text(empty_text)
        # 当停用词文件不存在时，stopwords为空集合，因此这些词不会被过滤
        assert empty_processed == "的 地 得 了 在", "停用词文件不存在时过滤逻辑异常"
    finally:
        if os.path.exists("stopwords_temp.txt"):
            os.rename("stopwords_temp.txt", "stopwords.txt") # 恢复停用词文件

    # 4.  异常：空文本处理
    empty_text = ""
    empty_processed = preprocess_text(empty_text)
    assert empty_processed == "", "空文本处理异常"

def test_calculate_similarity(test_files):
    """
    测试calculate_similarity函数：验证相似度计算的合理性
    """
    orig_path, copied_path, _ = test_files
    orig_text = read_file(orig_path)
    copied_text = read_file(copied_path)
    similarity = calculate_similarity(orig_text, copied_text)
    # 预设抄袭文本相似度在0.5~0.9之间，验证计算结果合理性
    assert 0.5 <= similarity <= 0.9, "高相似度计算异常"

def test_write_result_all_branches():
    """
    测试write_result函数的所有分支：正常写入、目录创建、异常场景
    """
    # 1. 正常写入（当前目录文件）
    result_path = "test_normal_result.txt"
    write_result(result_path, 0.85)
    assert os.path.exists(result_path), "正常写入失败"
    with open(result_path, "r") as f:
        assert f.read() == "0.85", "内容写入错误"
    os.remove(result_path) # 清理临时文件

    # 2. 带子目录的路径（验证目录创建）
    subdir_path = "test_subdir/result.txt"
    write_result(subdir_path, 0.7)
    assert os.path.exists(subdir_path), "子目录写入失败"
    os.remove(subdir_path)
    os.rmdir("test_subdir") # 清理临时目录

    # 3. 异常：文件写入失败（只读文件）
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            read_only_path = f.name
        os.chmod(read_only_path, 0o444)  # 设置文件为只读
        with pytest.raises(SystemExit):  # 验证写入只读文件时触发程序退出
            write_result(read_only_path, 0.5)
    finally:
        os.chmod(read_only_path, 0o644) # 恢复文件权限
        os.remove(read_only_path)       # 清理临时文件

def test_main(monkeypatch, test_files):
    """
    测试main函数：验证全流程（读取→预处理→计算→写入）的完整性
    """
    orig_path, copied_path, result_path = test_files
    # 模拟命令行参数
    monkeypatch.setattr("sys.argv", ["main.py", orig_path, copied_path, result_path])
    main()
    assert os.path.exists(result_path), "主函数未生成结果文件"
    os.remove(result_path) # 清理结果文件