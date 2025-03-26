from typing import Dict

import numpy as np
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

from .logger import get_logger

logger = get_logger(__name__)


from tqdm import tqdm


def hybrid_evaluate(
    author_embeddings: Dict[str, np.ndarray],
    author_labels: Dict[str, np.ndarray],
):
    best_results = {
        "db_eps": None,
        "db_min": None,
        "avg_f1": 0,
        "predict_results": None,
    }

    # db_eps_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    # db_eps_list.extend(list(np.linspace(0.02, 0.5, 20)))

    # db_min_list = list(range(2, 11))
    
    db_eps_list = [0.1]
    db_min_list = [1]


    # 外层循环：db_eps
    for db_eps in tqdm(db_eps_list, desc="DBSCAN eps"):
        # 内层循环：db_min
        for db_min in tqdm(db_min_list, desc="DBSCAN min_samples", leave=False):
            avg_f1, predict_results, detailed_info = compute_pairwise_f1(
                author_embeddings, author_labels, db_eps, db_min
            )
            if avg_f1 > best_results["avg_f1"]:
                best_results["db_eps"] = db_eps
                best_results["db_min"] = db_min
                best_results["avg_f1"] = avg_f1
                best_results["predict_results"] = predict_results

                logger.info("New best results:")
                logger.info(
                    f"New best DBSCAN parameters: eps={db_eps}, min_samples={db_min}"
                )
                logger.info(f"New best Pairwise F1: {avg_f1:.4f}")
                logger.info(f"Detailed Evaluation Info: {detailed_info}")

    logger.info(
        f"Best DBSCAN parameters: eps={best_results['db_eps']}, min_samples={best_results['db_min']}"
    )
    logger.info(f"Best Pairwise F1: {best_results['avg_f1']:.4f}")

    return best_results


def compute_pairwise_f1(
    author_embeddings: Dict[str, np.ndarray],
    author_labels: Dict[str, np.ndarray],
    db_eps: float = 1e-6,
    db_min: int = 5,
):
    predict_results = {}
    distance_detailed_info = ""

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

        distance_detailed_info += (
            f"{author_name}: pred labels{pred}, true labels {author_labels}\n"
        )
        distance_detailed_info += f"Similarity Matrix:\n{similarity_matrix}\n"

    # 计算F1
    avg_f1, detailed_info = evaluate(predict_results, author_labels)
    # detailed_info += distance_detailed_info
    return avg_f1, predict_results, detailed_info


def predict(
    author_embeddings: Dict[str, np.ndarray],
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

    return predict_results


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
    detailed_info = "\n"

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
            # logger.warning(f"Warning: All samples same class for {author}")
            # continue
            pass

        # 计算指标
        precision, recall, f1 = pairwise_evaluate(true_labels, pred_labels)
        total_f1 += f1
        valid_authors += 1

        detailed_info += (
            f"{author}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n"
        )

    if valid_authors == 0:
        logger.error("No valid authors for evaluation")
        return 0

    avg_f1 = total_f1 / valid_authors
    # logger.info(
    #     f"\nAverage Pairwise F1: {avg_f1:.4f} ({valid_authors} authors evaluated)"
    # )
    return avg_f1, detailed_info


def pairwise_evaluate(correct_labels, pred_labels):
    """Macro Pairwise F1

    Pairwise precision: \frac{#Pairs Correctly Predicted To SameAuthor}{Total Pairs Predicted To SameAuthor}
    Pairwise recall: \frac{#Pairs Correctly Predicted To SameAuthor}{Total Pairs To SameAuthor}
    Pairwise F1: \frac{2 \times Pairwise Precision \times Pairwise Recall}{Pairwise Precision + Pairwise Recall}

    为了避免受聚类数目的影响，此处采用pairwise的方式计算指标，即将聚类问题转化为关系问题，只考虑每对样本之间的关系。
    同时因为负样本数目远远大于正样本数目，所以此处采用属于同一作者的样本作为正样本，只进行正样本的计算。

    Args:
        correct_labels (_type_): _description_
        pred_labels (_type_): _description_

    Returns:
        _type_: _description_
    """
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    # iterate over all pairs of papers (i, j)
    # and compute metrics based on the pairs
    # so the metrics are not affected by the numbers of clusters
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
    sample_num = 100
    author_embeddings = {
        "author1": np.random.rand(sample_num, 768),
        "author2": np.random.rand(sample_num, 768),
    }

    author_labels = {
        "author1": np.random.randint(0, 10, sample_num),
        "author2": np.random.randint(0, 10, sample_num),
    }

    best_results = hybrid_evaluate(author_embeddings, author_labels)
    logger.info(best_results)
