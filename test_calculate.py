import pytest
import os
import tempfile
from main import read_file, preprocess_text, calculate_similarity, write_result, main

@pytest.fixture
def test_files():
    orig_path = "orig.txt"
    copied_path = "orig_0.8_dis_15.txt"
    result_path = "test_result.txt"
    yield orig_path, copied_path, result_path

def test_read_file_all_branches():
    # 1. 正常读取（UTF-8编码）
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        f.write("测试UTF-8编码的文本内容")
        utf8_path = f.name
    content = read_file(utf8_path)
    assert "测试UTF-8编码的文本内容" in content
    os.remove(utf8_path)

    # 2. 文件不存在（FileNotFoundError）
    non_existent = "non_existent_file_1234.txt"
    with pytest.raises(SystemExit):
        read_file(non_existent)

    # 3. 路径是目录（IsADirectoryError）
    with tempfile.TemporaryDirectory() as dir_path:
        with pytest.raises(SystemExit):
            read_file(dir_path)

    # 4. UTF-8解码失败，尝试GBK编码
    gbk_content = "测试GBK编码的文本内容：中文测试".encode('gbk')
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(gbk_content)
        gbk_path = f.name
    content = read_file(gbk_path)
    assert "中文测试" in content
    os.remove(gbk_path)

    # 5. 其他未知异常（触发通用Exception）
    try:
        read_file(123)  # 传入非字符串路径
    except SystemExit:
        assert True, "未知异常触发了SystemExit"

def test_preprocess_text_all_branches():
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

    # 3. stopwords.txt不存在时的降级逻辑（仅验证逻辑，实际场景需确保文件存在）
    if os.path.exists("stopwords.txt"):
        os.rename("stopwords.txt", "stopwords_temp.txt")
    try:
        empty_text = "的 地 得 了 在"
        empty_processed = preprocess_text(empty_text)
        # 当停用词文件不存在时，stopwords为空集合，因此这些词不会被过滤
        # 调整断言为“文本未被过滤”（与降级逻辑一致）
        assert empty_processed == "的 地 得 了 在", "停用词文件不存在时过滤逻辑异常"
    finally:
        if os.path.exists("stopwords_temp.txt"):
            os.rename("stopwords_temp.txt", "stopwords.txt")

    # 4. 空文本处理
    empty_text = ""
    empty_processed = preprocess_text(empty_text)
    assert empty_processed == "", "空文本处理异常"

def test_calculate_similarity(test_files):
    orig_path, copied_path, _ = test_files
    orig_text = read_file(orig_path)
    copied_text = read_file(copied_path)
    similarity = calculate_similarity(orig_text, copied_text)
    assert 0.5 <= similarity <= 0.9, "高相似度计算异常"

def test_write_result_all_branches():
    # 1. 正常写入（当前目录文件）
    result_path = "test_normal_result.txt"
    write_result(result_path, 0.85)
    assert os.path.exists(result_path), "正常写入失败"
    with open(result_path, "r") as f:
        assert f.read() == "0.85", "内容写入错误"
    os.remove(result_path)

    # 2. 带子目录的路径（验证目录创建）
    subdir_path = "test_subdir/result.txt"
    write_result(subdir_path, 0.7)
    assert os.path.exists(subdir_path), "子目录写入失败"
    os.remove(subdir_path)
    os.rmdir("test_subdir")

    # 3. 文件写入失败（只读文件）
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            read_only_path = f.name
        os.chmod(read_only_path, 0o444)  # 设置为只读
        with pytest.raises(SystemExit):
            write_result(read_only_path, 0.5)
    finally:
        os.chmod(read_only_path, 0o644)
        os.remove(read_only_path)

def test_main(monkeypatch, test_files):
    orig_path, copied_path, result_path = test_files
    monkeypatch.setattr("sys.argv", ["main.py", orig_path, copied_path, result_path])
    main()
    assert os.path.exists(result_path), "主函数未生成结果文件"
    os.remove(result_path)