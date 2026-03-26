# 词类比任务评估模块（top-5 版本）
import numpy as np
from typing import Dict, List, Tuple

def load_embeddings(file_path: str) -> Tuple[Dict[str, np.ndarray], int, int]:
    """加载训练好的词向量

    Args:
        file_path: embeddings文件路径

    Returns:
        word2vec: 词到向量的映射字典
        vocab_size: 词表大小
        emb_size: 向量维度
    """
    word2vec = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split()
        vocab_size = int(header[0])
        emb_size = int(header[1])

        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            word2vec[word] = vector

    return word2vec, vocab_size, emb_size


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def find_topn_similar(word2vec: Dict[str, np.ndarray],
                       target_vec: np.ndarray,
                       exclude_words: List[str],
                       topn: int = 5) -> List[Tuple[str, float]]:
    """找到与目标向量最相似的 top-n 个词（排除指定词）

    Args:
        word2vec: 词向量字典
        target_vec: 目标向量
        exclude_words: 需要排除的词列表
        topn: 返回前 n 个最相似的词

    Returns:
        List of (word, similarity_score) tuples, sorted by similarity descending
    """
    candidates = []
    for word, vec in word2vec.items():
        if word in exclude_words:
            continue
        sim = cosine_similarity(target_vec, vec)
        candidates.append((word, sim))

    # Sort by similarity descending and take top-n
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:topn]


def evaluate_word_analogy(word2vec: Dict[str, np.ndarray],
                          test_file: str,
                          topn: int = 5) -> Tuple[float, List, List]:
    """评估词类比任务（top-n 版本）

    任务形式: Word1 is to Word2 as Word3 is to ?
    公式: vec(Word4) ≈ vec(Word2) - vec(Word1) + vec(Word3)

    Args:
        word2vec: 词向量字典
        test_file: 测试文件路径（CSV格式）
        topn: 将正确答案是否出现在前 topn 个候选中视为正确（默认5）

    Returns:
        accuracy: 准确率
        correct_cases: 正确预测的样例列表
        wrong_cases: 错误预测的样例列表
    """
    correct = 0
    total = 0
    correct_cases = []
    wrong_cases = []

    with open(test_file, 'r', encoding='utf-8') as f:
        # 跳过标题行
        header = f.readline()

        for line in f:
            parts = line.strip().split(',')
            # parts[4] 是 Subject 列（如 capital-common-countries / family 等），不影响评估
            if len(parts) < 4:
                continue

            word1, word2, word3, word4 = parts[0], parts[1], parts[2], parts[3]

            # 检查所有词是否在词表中
            if word1 not in word2vec or word2 not in word2vec or \
               word3 not in word2vec or word4 not in word2vec:
                continue

            # 计算目标向量: vec(word2) - vec(word1) + vec(word3)
            target_vec = word2vec[word2] - word2vec[word1] + word2vec[word3]

            # 找到前 topn 个最相似的词（排除输入的三个词）
            top_candidates = find_topn_similar(
                word2vec,
                target_vec,
                exclude_words=[word1, word2, word3],
                topn=topn
            )

            total += 1

            # topn 列表中的词（用于显示）
            predicted_words = [w for w, _ in top_candidates]
            top1_word, top1_sim = top_candidates[0] if top_candidates else (None, 0.0)

            if word4 in predicted_words:
                correct += 1
                correct_cases.append({
                    'word1': word1, 'word2': word2,
                    'word3': word3, 'word4': word4,
                    'predicted': top1_word,
                    'similarity': top1_sim,
                    'top5': predicted_words,
                })
            else:
                wrong_cases.append({
                    'word1': word1, 'word2': word2,
                    'word3': word3, 'word4': word4,
                    'predicted': top1_word,
                    'similarity': top1_sim,
                    'top5': predicted_words,
                })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct_cases, wrong_cases


def print_analogy_results(accuracy: float, correct_cases: List, wrong_cases: List, set_name: str):
    """打印词类比评估结果"""
    print(f"\n{'='*60}")
    print(f"{set_name} 词类比任务评估结果 (top-5)")
    print(f"{'='*60}")
    print(f"总样本数: {len(correct_cases) + len(wrong_cases)}")
    print(f"正确预测: {len(correct_cases)}")
    print(f"错误预测: {len(wrong_cases)}")
    print(f"准确率: {accuracy*100:.2f}%")

    print(f"\n部分正确样例:")
    for i, case in enumerate(correct_cases[:5]):
        top5_str = ', '.join(case['top5'][:5])
        print(f"  {i+1}. {case['word1']} : {case['word2']} = {case['word3']} : {case['word4']} "
              f"(预测: {case['predicted']}, top-5: [{top5_str}], 相似度: {case['similarity']:.4f})")

    print(f"\n部分错误样例:")
    for i, case in enumerate(wrong_cases[:5]):
        top5_str = ', '.join(case['top5'][:5])
        print(f"  {i+1}. {case['word1']} : {case['word2']} = {case['word3']} : {case['word4']} "
              f"(预测: {case['predicted']}, 正确答案: {case['word4']}, top-5: [{top5_str}], 相似度: {case['similarity']:.4f})")
