import math
import os
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.manifold import TSNE
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


def plot_sim_progress(data):
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.sort_index()
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Step"}, inplace=True)

    # 设置学术风格
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6), dpi=100)

    # 创建画布
    # ax = plt.gca()

    # 绘制正样本曲线
    sns.lineplot(
        x="Step",
        y="avg_pos",
        data=df,
        color="#1f77b4",
        linewidth=2.5,
        label="Positive Pairs",
    )
    # 绘制负样本曲线
    sns.lineplot(
        x="Step",
        y="avg_neg",
        data=df,
        color="#ff7f0e",
        linewidth=2.5,
        label="Negative Pairs",
    )

    # 添加标准差区域
    plt.fill_between(
        df["Step"],
        df["avg_pos"] - df["std_pos"],
        df["avg_pos"] + df["std_pos"],
        color="#1f77b4",
        alpha=0.2,
    )
    plt.fill_between(
        df["Step"],
        df["avg_neg"] - df["std_neg"],
        df["avg_neg"] + df["std_neg"],
        color="#ff7f0e",
        alpha=0.2,
    )

    # 坐标轴优化
    plt.xticks(np.arange(0, 11000, 1000))  # 设置x轴刻度
    plt.xlabel("Training Steps", labelpad=10)
    plt.ylabel("Similarity Score", labelpad=10)
    plt.ylim(0.3, 1.0)  # 根据数据范围调整

    # 添加图例
    plt.legend(loc="upper left", frameon=True)

    # 设置标题
    plt.title("Similarity Score Progression with Standard Deviation", pad=20)

    # 保存高清图片
    plt.show()


def visualize_author_embeddings_plotly(
    author_embeddings,
    author_labels,
    title="3D Embeddings",
    point_size=2,  # 新增点大小参数
    opacity=0.8,  # 新增透明度参数
):
    # 合并数据（保持原有处理逻辑不变）
    data = []
    for auth, emb in author_embeddings.items():
        X = emb.cpu().numpy() if hasattr(emb, "cpu") else emb
        labels = (
            author_labels[auth].cpu().numpy()
            if hasattr(author_labels[auth], "cpu")
            else author_labels[auth]
        )
        data.extend(
            [
                {
                    "author": auth,
                    "label": str(lbl),
                    "x": vec[0],
                    "y": vec[1],
                    "z": vec[2],
                }
                for vec, lbl in zip(X, labels)
            ]
        )

    df = pd.DataFrame(data)

    # 创建3D散点图
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="label",
        symbol="author",
        title=title,
        labels={"x": "t-SNE 1", "y": "t-SNE 2", "z": "t-SNE 3"},
        size_max=point_size,  # 控制最大点尺寸
        opacity=opacity,  # 设置透明度
        hover_data={"author": True, "label": True},  # 优化悬停信息
    )

    # 微调标记样式
    fig.update_traces(
        marker=dict(
            line=dict(width=0.3, color="Gray"),  # 添加细边框
            sizemode="diameter",  # 尺寸模式
        ),
        selector=dict(mode="markers"),
    )

    # 调整布局
    fig.update_layout(
        scene=dict(
            xaxis=dict(title_font=dict(size=14)),
            yaxis=dict(title_font=dict(size=14)),
            zaxis=dict(title_font=dict(size=14)),
        ),
        legend=dict(
            title_font=dict(size=12),
            font=dict(size=10),
            itemsizing="constant",  # 统一图例标记大小
        ),
    )

    fig.show()


def visualize_author_embeddings_3d(
    author_embeddings: Dict[str, Union[torch.Tensor, np.ndarray]],
    author_labels: Dict[str, Union[torch.Tensor, np.ndarray]],
    perplexity: int = 30,
    figsize_scaling: float = 1.2,
    title: str = "3D Embedding Visualization by Author",
    elev: float = 100,
    azim: float = -35,
):
    """
    可视化不同作者的嵌入向量（使用3D t-SNE降维）

    参数:
        author_embeddings: 字典 {作者名: 嵌入矩阵}, 矩阵形状为 [n_samples, embedding_dim]
        author_labels: 字典 {作者名: 标签向量}, 形状为 [n_samples]
        perplexity: t-SNE的困惑度参数 (默认: 30)
        figsize_scaling: 图形大小的缩放系数 (默认: 1.2)
        title: 整个图形的标题 (默认: "3D Embedding Visualization by Author")
        elev: 3D视图仰角 (默认: 20)
        azim: 3D视图方位角 (默认: -35)
    """

    # 参数检查
    if not isinstance(author_embeddings, dict) or not isinstance(author_labels, dict):
        raise TypeError("author_embeddings和author_labels必须是字典")

    if set(author_embeddings.keys()) != set(author_labels.keys()):
        raise ValueError("author_embeddings和author_labels的键(作者名)必须一致")

    # 设置3D可视化风格
    sns.set_theme(style="whitegrid")
    plt.rcParams["axes.grid"] = False

    # 计算子图行列数
    n_authors = len(author_embeddings)
    n_cols = min(3, n_authors)
    n_rows = math.ceil(n_authors / n_cols)

    # 创建画布并预生成所有子图
    fig = plt.figure(
        figsize=(7 * n_cols * figsize_scaling, 6 * n_rows * figsize_scaling)
    )
    axes = []
    total_plots = n_rows * n_cols  # 总子图数量（包含空位）

    # 生成所有子图（包括空位）
    for i in range(total_plots):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        axes.append(ax)

    # 转换为二维数组以便行列访问
    axes = np.array(axes).reshape(n_rows, n_cols)

    # 对每个作者处理
    for idx, (author_name, embeddings) in enumerate(author_embeddings.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # 数据转换
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        current_labels = author_labels[author_name]
        if isinstance(current_labels, torch.Tensor):
            current_labels = current_labels.cpu().numpy()

        # 调整perplexity
        actual_perplexity = min(perplexity, len(embeddings) - 1)

        # 3D t-SNE降维
        tsne = TSNE(
            n_components=3,
            perplexity=actual_perplexity,
            metric="cosine",
            random_state=42,
            init="pca",
            max_iter=1000,
        )
        embeddings_3d = tsne.fit_transform(embeddings)

        # 颜色映射
        unique_labels = np.unique(current_labels).tolist()
        color_palette = sns.color_palette("hsv", len(unique_labels))
        label_to_color = {
            label: color_palette[i] for i, label in enumerate(unique_labels)
        }

        # 绘制3D散点图
        for label in np.unique(current_labels):
            mask = current_labels == label
            ax.scatter3D(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                color=label_to_color[label],
                label=f"Label {label}",
                alpha=0.8,
                s=40,
                depthshade=True,
                edgecolor="w",
                linewidth=0.3,
            )

        # 设置视角和标签
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("t-SNE 1", labelpad=10)
        ax.set_ylabel("t-SNE 2", labelpad=10)
        ax.set_zlabel("t-SNE 3", labelpad=10)
        ax.set_title(
            f"{author_name}\n(n={len(embeddings)}, perplexity={actual_perplexity})",
            fontsize=12,
            pad=20,
        )

        # 调整刻度
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
        ax.zaxis.set_tick_params(labelsize=8)

    # 隐藏空余子图
    for idx in range(n_authors, total_plots):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    # # 统一图例
    # if len(unique_labels) > 0:
    #     handles, labels = ax.get_legend_handles_labels()
    #     fig.legend(
    #         handles,
    #         labels,
    #         loc="upper right",
    #         bbox_to_anchor=(0.98, 0.92),
    #         fontsize=10,
    #         title="Class Labels",
    #     )

    # 调整布局
    plt.subplots_adjust(
        wspace=0.15, hspace=0.2, left=0.05, right=0.95, bottom=0.05, top=0.9
    )

    # # 添加总标题
    # fig.suptitle(title, fontsize=16, y=0.97)

    plt.show()


def visualize_author_embeddings(
    author_embeddings: Dict[str, Union[torch.Tensor, np.ndarray]],
    author_labels: Dict[str, Union[torch.Tensor, np.ndarray]],
    perplexity: int = 30,
    figsize_scaling: float = 1.0,
    title: str = "Embedding Visualization by Author",
) -> None:
    """
    可视化不同作者的嵌入向量（使用t-SNE降维）

    参数:
        author_embeddings: 字典 {作者名: 嵌入矩阵}, 矩阵形状为 [n_samples, embedding_dim]
        author_labels: 字典 {作者名: 标签向量}, 形状为 [n_samples]
        perplexity: t-SNE的困惑度参数 (默认: 30)
        figsize_scaling: 图形大小的缩放系数 (默认: 1.0)
        title: 整个图形的标题 (默认: "Embedding Visualization by Author")
    """
    # 参数检查
    if not isinstance(author_embeddings, dict) or not isinstance(author_labels, dict):
        raise TypeError("author_embeddings和author_labels必须是字典")

    if set(author_embeddings.keys()) != set(author_labels.keys()):
        raise ValueError("author_embeddings和author_labels的键(作者名)必须一致")

    # 设置seaborn风格
    sns.set_theme(style="whitegrid")

    # 计算合理的行列数（每行最多3个子图）
    n_authors = len(author_embeddings)
    n_cols = min(3, n_authors)
    n_rows = math.ceil(n_authors / n_cols)

    # 创建画布 - 动态调整大小
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols * figsize_scaling, 5.5 * n_rows * figsize_scaling),
    )

    # 处理单子图的情况
    if n_authors == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # 对每个作者单独处理
    for idx, (author_name, embeddings) in enumerate(author_embeddings.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # 确保数据是numpy数组
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        current_labels = author_labels[author_name]
        if isinstance(current_labels, torch.Tensor):
            current_labels = current_labels.cpu().numpy()

        # 自动调整perplexity
        actual_perplexity = min(perplexity, len(embeddings) - 1)

        # t-SNE降维
        tsne = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            metric="cosine",
            random_state=42,
            init="pca",
            max_iter=500,  # 增加迭代次数保证收敛
        )
        embeddings_2d = tsne.fit_transform(embeddings)

        # 获取唯一标签并创建颜色映射
        unique_labels = np.unique(current_labels).tolist()
        color_palette = sns.color_palette("hsv", len(unique_labels))
        label_to_color = {
            label: color_palette[i] for i, label in enumerate(unique_labels)
        }

        # 为每个label单独绘制
        for label in np.unique(current_labels):
            mask = current_labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                color=label_to_color[label],
                label=f"Label {label}",
                alpha=0.8,
                s=50,
                edgecolor="white",
                linewidth=0.7,
            )

        # 添加子图标题和信息
        ax.set_title(f"Author: {author_name}", fontsize=14, fontweight="bold")
        # ax.text(
        #     0.5,
        #     0.97,
        #     f"Samples: {len(embeddings)}\nPerplexity: {actual_perplexity}",
        #     horizontalalignment="center",
        #     transform=ax.transAxes,
        #     fontsize=10,
        # )
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)

        # 添加图例
        # ax.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=8)

    # 隐藏多余的子图
    for idx in range(n_authors, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    # 调整布局
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # 添加大标题
    fig.suptitle(title, fontsize=16, y=0.98 if n_rows > 1 else 1.05)

    plt.show()


def visualize_all_author_embeddings(
    checkpoint_dir: str,
):
    """
    可视化所有作者的嵌入向量（使用t-SNE降维）

    参数:
        checkpoint_dir: 检查点目录
    """
    # 列出所有检查点，即checkpoint_dir 下所有包含有 checkpoint名称的文件夹
    dev_embeddings_dir = os.path.join(checkpoint_dir, "dev_embeddings")
    dev_labels_dir = os.path.join(checkpoint_dir, "dev_labels")
    checkpoint_names = [f for f in os.listdir(dev_embeddings_dir)]
    # 根据step_num排序
    checkpoint_names = sorted(
        checkpoint_names, key=lambda x: int(x.split("-")[1].split(".")[0])
    )

    for checkpoint_name in tqdm(checkpoint_names):
        step_num = int(checkpoint_name.split("-")[1].split(".")[0])

        author_embeddings_path = os.path.join(dev_embeddings_dir, checkpoint_name)
        author_labels_path = os.path.join(dev_labels_dir, checkpoint_name)

        author_embeddings = load_file(author_embeddings_path)
        author_labels = load_file(author_labels_path)
        visualize_author_embeddings(
            author_embeddings=author_embeddings,
            author_labels=author_labels,
            perplexity=30,
            figsize_scaling=1.0,
            title=f"Embedding Visualization by Author - Step {step_num}",
        )

    return True


def visualize_single_author_embeddings(
    checkpoint_dir: str,
    author_name: str = None,
    author_idx: int = None,
    step: int = None,
):
    """
    可视化单个作者的嵌入向量（使用t-SNE降维）

    参数:
        checkpoint_dir: 检查点目录
        author_name: 作者名称
        author_idx: 作者索引
    """
    # 列出所有检查点，即checkpoint_dir 下所有包含有 checkpoint名称的文件夹
    dev_embeddings_dir = os.path.join(checkpoint_dir, "dev_embeddings")
    dev_labels_dir = os.path.join(checkpoint_dir, "dev_labels")
    checkpoint_names = [f for f in os.listdir(dev_embeddings_dir)]
    # 根据step_num排序
    checkpoint_names = sorted(
        checkpoint_names, key=lambda x: int(x.split("-")[1].split(".")[0])
    )

    for checkpoint_name in tqdm(checkpoint_names):
        # 如果指定了step，则只可视化该step的嵌入向量

        step_num = int(checkpoint_name.split("-")[1].split(".")[0])

        if step is None:
            logger.info(
                "You have not specified a step, so no steps will be visualized."
            )
            return False
        if step != step_num:
            continue

        author_embeddings_path = os.path.join(dev_embeddings_dir, checkpoint_name)
        author_labels_path = os.path.join(dev_labels_dir, checkpoint_name)

        author_embeddings = load_file(author_embeddings_path)
        author_labels = load_file(author_labels_path)

        author_name = (
            author_name if author_name else list(author_embeddings.keys())[author_idx]
        )
        logger.info(
            f"Visualizing author {author_name} embeddings at step {step_num}..."
        )

        # 只可视化指定作者的嵌入向量
        visualize_author_embeddings_plotly(
            author_embeddings={author_name: author_embeddings[author_name]},
            author_labels={author_name: author_labels[author_name]},
            title=f"Embedding Visualization by Author - Step {step_num}",
        )

    return True
