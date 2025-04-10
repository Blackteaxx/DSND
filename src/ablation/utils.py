import os
from typing import Dict

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger(__name__)


# 计算同一个author下embedding之间的余弦相似度
def cosine_similarity(embeddings):
    embeddings = torch.tensor(embeddings)
    normd_embeddings = F.normalize(embeddings, p=2, dim=-1)
    return torch.mm(normd_embeddings, normd_embeddings.t())


def get_checkpoint_sim_stats(
    author_embeddings: Dict[str, torch.Tensor], author_labels: Dict[str, torch.Tensor]
):
    """


    Args:
        author_embeddings (Dict[str, torch.Tensor]): A dictionary containing author embeddings.
        author_labels (Dict[str, torch.Tensor]): A dictionary containing author labels.

    """
    author_similarities_stats = {
        "avg_pos": [],
        "avg_neg": [],
        "std_pos": [],
        "std_neg": [],
        "min_pos": [],
        "max_neg": [],
    }

    for author_name, author_embedding in author_embeddings.items():
        n_samples = len(author_embedding)
        author_label = author_labels[author_name]
        label_mask = (author_label == author_label.unsqueeze(1)).float()
        label_similarity = cosine_similarity(author_embedding)

        # 计算统计量（处理可能的空张量）
        pos_sim = label_similarity[label_mask == 1]
        neg_sim = (
            label_similarity[label_mask == 0] if n_samples > 1 else torch.tensor([])
        )

        avg_pos = pos_sim.mean().item() if len(pos_sim) > 0 else 1.0
        avg_neg = neg_sim.mean().item() if len(neg_sim) > 0 else 0.0
        std_pos = pos_sim.std().item() if len(pos_sim) > 0 else 0.0
        std_neg = neg_sim.std().item() if len(neg_sim) > 0 else 0.0
        min_pos = pos_sim.min().item() if len(pos_sim) > 0 else 1.0
        max_neg = neg_sim.max().item() if len(neg_sim) > 0 else 0.0

        author_similarities_stats["avg_pos"].append(avg_pos)
        author_similarities_stats["avg_neg"].append(avg_neg)
        author_similarities_stats["std_pos"].append(std_pos)
        author_similarities_stats["std_neg"].append(std_neg)
        author_similarities_stats["min_pos"].append(min_pos)
        author_similarities_stats["max_neg"].append(max_neg)

    results = {
        "avg_pos": torch.tensor(author_similarities_stats["avg_pos"]).mean().item(),
        "avg_neg": torch.tensor(author_similarities_stats["avg_neg"]).mean().item(),
        "std_pos": torch.tensor(author_similarities_stats["std_pos"]).mean().item(),
        "std_neg": torch.tensor(author_similarities_stats["std_neg"]).mean().item(),
        "min_pos": torch.tensor(author_similarities_stats["min_pos"]).min().item(),
        "max_neg": torch.tensor(author_similarities_stats["max_neg"]).max().item(),
    }
    return results


def get_all_checkpoints_sim_stats(
    checkpoint_dir: str,
):
    """
    Args:
        checkpoint_path (str): The path to the checkpoint directory.

    """
    # 列出所有检查点，即checkpoint_dir 下所有包含有 checkpoint名称的文件夹
    dev_embeddings_dir = os.path.join(checkpoint_dir, "dev_embeddings")
    dev_labels_dir = os.path.join(checkpoint_dir, "dev_labels")
    checkpoint_names = [f for f in os.listdir(dev_embeddings_dir)]

    all_sim_stats = {}
    for checkpoint_name in tqdm(checkpoint_names):
        step_num = int(checkpoint_name.split("-")[1].split(".")[0])

        author_embeddings_path = os.path.join(dev_embeddings_dir, checkpoint_name)
        author_labels_path = os.path.join(dev_labels_dir, checkpoint_name)

        author_embeddings = load_file(author_embeddings_path)
        author_labels = load_file(author_labels_path)
        sim_stats = get_checkpoint_sim_stats(
            author_embeddings=author_embeddings,
            author_labels=author_labels,
        )
        all_sim_stats[step_num] = sim_stats

    # 根据step_num排序
    all_sim_stats = dict(sorted(all_sim_stats.items(), key=lambda item: item[0]))

    return all_sim_stats
