from typing import Dict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

from .logger import get_logger

logger = get_logger(__name__)


def compute_pairwise_f1(
    author_embeddings: Dict[str, np.ndarray],
    author_labels: Dict[str, np.ndarray],
    db_eps: float = 1e-6,
    db_min: int = 5,
):
    predict_results = {}
    for author_name in author_embeddings:
        embeddings = author_embeddings[author_name]

        # 计算相似度矩阵
        similarity_matrix = pairwise_distances(embeddings, metric="cosine")

        # DBSCAN聚类
        local_labels = DBSCAN(
            eps=db_eps, min_samples=db_min, metric="precomputed"
        ).fit_predict(similarity_matrix)

        pred = local_labels.tolist()
        predict_results[author_name] = pred

    # 计算F1
    avg_f1 = evaluate(predict_results, author_labels)
    return avg_f1, predict_results


def evaluate(predict_dict, true_dict):
    """
    新版评估函数（带数据校验）

    参数格式示例:
    predict_dict = {
        "author1": [0, 0, 1, 1],  # 预测标签列表
        "author2": [1, 1, 0, 0]
    }

    true_dict = {
        "author1": [0, 0, 0, 1],  # 真实标签列表
        "author2": [0, 0, 1, 1]
    }
    """
    total_f1 = 0
    valid_authors = 0

    for author in true_dict:
        # 数据校验
        if author not in predict_dict:
            logger.warning(f"Warning: Author {author} missing in predictions")
            continue

        true_labels = np.array(true_dict[author])
        pred_labels = np.array(predict_dict[author])

        if len(true_labels) != len(pred_labels):
            logger.error(f"Error: Label length mismatch for {author}")
            continue

        if len(np.unique(true_labels)) == 1:
            logger.warning(f"Warning: All samples same class for {author}")
            continue

        # 计算指标
        precision, recall, f1 = pairwise_evaluate(true_labels, pred_labels)
        total_f1 += f1
        valid_authors += 1

        # 打印详细结果
        logger.info(
            f"[{author}] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

    if valid_authors == 0:
        logger.error("No valid authors for evaluation")
        return 0

    avg_f1 = total_f1 / valid_authors
    logger.info(
        f"\nAverage Pairwise F1: {avg_f1:.4f} ({valid_authors} authors evaluated)"
    )
    return avg_f1


def pairwise_evaluate(correct_labels, pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    # iterate over all pairs of papers (i, j)
    # and compute metrics based on the pairs
    # so the metrics are not affected by the numerics of clusters
    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (
                pred_labels[i] == pred_labels[j]
            ):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (
            pairwise_precision + pairwise_recall
        )

    return pairwise_precision, pairwise_recall, pairwise_f1


if __name__ == "__main__":
    predict_dict = {
        "author1": [0, 0, 1, 1],  # 预测标签列表
        "author2": [1, 1, 0, 0],
    }

    true_dict = {
        "author1": [0, 0, 0, 1],  # 真实标签列表
        "author2": [0, 0, 1, 1],
    }

    evaluate(predict_dict, true_dict)
