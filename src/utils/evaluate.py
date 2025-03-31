import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .logger import get_logger

logger = get_logger(__name__)


# 线程安全的进度条包装器
class TqdmMultiThreadWrapper:
    def __init__(self, total):
        self.pbar = tqdm(total=total)
        self._lock = threading.Lock()

    def update(self, n=1):
        with self._lock:
            self.pbar.update(n)


# 定义聚类算法配置基类
@dataclass
class ClusteringConfig:
    name: str
    estimator: Callable
    param_grid: dict
    precomputed: bool = True


# 预定义算法配置
CLUSTERING_METHODS = {
    "dbscan": ClusteringConfig(
        name="DBSCAN",
        estimator=DBSCAN,
        param_grid={
            "eps": [0.05] + list(np.linspace(0.02, 0.5, 20)),
            "min_samples": list(range(1, 6)),
        },
    ),
    "hdbscan": ClusteringConfig(
        name="HDBSCAN",
        estimator=HDBSCAN,
        param_grid={
            "min_cluster_size": [2, 3, 4, 5, 6],
        },
    ),
}


def evaluate_single_params(
    params: dict,
    config: ClusteringConfig,
    author_embeddings: Dict[str, np.ndarray],
    author_labels: Dict[str, np.ndarray],
    pbar_wrapper: TqdmMultiThreadWrapper = None,
) -> Tuple[dict, float, Dict[str, np.ndarray]]:
    """单组参数的评估函数（线程安全）"""
    try:
        # 创建聚类器
        metric = "precomputed" if config.precomputed else "cosine"
        clusterer = config.estimator(**params, metric=metric)
        # 执行评估
        avg_f1, predict_results, detailed_info = compute_clustering_f1(
            author_embeddings,
            author_labels,
            clusterer=clusterer,
            precomputed=config.precomputed,
        )
        if pbar_wrapper:
            pbar_wrapper.update(1)
        return params, avg_f1, predict_results
    except Exception as e:
        logger.error(f"Error evaluating params {params}: {str(e)}")
        return params, 0.0, {}


def hybrid_evaluate(
    author_embeddings: Dict[str, np.ndarray],
    author_labels: Dict[str, np.ndarray],
    method: str = "dbscan",
    auto_eps: bool = False,
    max_workers: int = 4,
) -> Dict:
    """支持多线程的聚类评估框架"""

    # 获取算法配置
    config = CLUSTERING_METHODS.get(method.lower())
    if not config:
        raise ValueError(
            f"Unsupported method: {method}. Available: {list(CLUSTERING_METHODS.keys())}"
        )
    # 自动参数估计
    if auto_eps and method == "dbscan":
        config.param_grid["eps"] = [estimate_auto_eps(author_embeddings)]
    # 生成参数搜索空间
    param_grid = generate_param_grid(config)
    total_params = len(param_grid)

    # 初始化结果存储
    best_results = {
        "params": None,
        "avg_f1": 0,
        "predict_results": None,
        "method": config.name,
    }

    # 线程安全的进度条
    pbar = TqdmMultiThreadWrapper(total=total_params)

    # 多线程评估
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for params in param_grid:
            futures.append(
                executor.submit(
                    evaluate_single_params,
                    params=params,
                    config=config,
                    author_embeddings=author_embeddings,
                    author_labels=author_labels,
                    pbar_wrapper=pbar,
                )
            )
        # 收集结果
        for future in as_completed(futures):
            params, avg_f1, predict_results = future.result()
            if avg_f1 > best_results["avg_f1"]:
                best_results.update(
                    {
                        "params": params,
                        "avg_f1": avg_f1,
                        "predict_results": predict_results,
                    }
                )
                logger.info(f"New best params: {params}, F1: {avg_f1:.4f}")
    logger.info(f"Best {config.name} params: {best_results['params']}")
    logger.info(f"Best Pairwise F1: {best_results['avg_f1']:.4f}")
    return best_results


def compute_clustering_f1(
    author_embeddings: Dict[str, np.ndarray],
    author_labels: Dict[str, np.ndarray],
    clusterer: object,
    precomputed: bool = True,
):
    """通用聚类评估核心函数

    Args:
        author_embeddings (Dict[str, np.ndarray]): _description_
        author_labels (Dict[str, np.ndarray]): _description_
        clusterer (object): _description_
        precomputed (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    predict_results = {}
    detailed_info = ""

    for author_name, embeddings in author_embeddings.items():
        # 数据预处理
        processed_embeddings = preprocess_embeddings(embeddings)

        # 距离矩阵计算
        if precomputed:
            similarity_matrix = pairwise_distances(
                processed_embeddings, metric="cosine"
            )
            labels = clusterer.fit_predict(similarity_matrix)
        else:
            labels = clusterer.fit_predict(processed_embeddings)

        # 后处理: outlier处理
        processed_labels = postprocess_labels(labels)
        predict_results[author_name] = processed_labels

        # 记录详细信息
        detailed_info += f"{author_name}:\n"
        detailed_info += (
            f"True: {author_labels[author_name]}\nPred: {processed_labels}\n"
        )

    # 评估指标计算
    avg_f1, eval_details = evaluate(predict_results, author_labels)
    return avg_f1, predict_results, detailed_info + eval_details


def generate_param_grid(config: ClusteringConfig) -> list:
    """生成参数组合网格"""
    from itertools import product

    keys = config.param_grid.keys()
    values = product(*config.param_grid.values())
    return [dict(zip(keys, v)) for v in values]


def estimate_auto_eps(embeddings: Dict[str, np.ndarray], k: int = 5) -> float:
    """自动估计DBSCAN的eps参数"""
    all_distances = []
    for emb in embeddings.values():
        nn = NearestNeighbors(n_neighbors=k).fit(emb)
        distances, _ = nn.kneighbors()
        all_distances.extend(distances[:, -1].tolist())
    return np.percentile(all_distances, 95)


def preprocess_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """标准化预处理"""
    return embeddings


def postprocess_labels(labels: np.ndarray) -> list:
    """标签后处理（处理噪声点）"""
    labels = labels.tolist()
    max_label = max(labels) + 1
    return [x if x != -1 else max_label + i for i, x in enumerate(labels)]


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
