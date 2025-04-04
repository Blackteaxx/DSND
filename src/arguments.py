import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


def generate_snd_run_name(
    model_name_or_path: str,
    positive_num: Optional[int] = None,
    packing_size: int = 4,
    temperature: float = 0.05,
    temperature_decay: float = 1.0,
    temperature_decay_step: int = 1000,
    learning_rate: float = 1e-5,
    shuffle: bool = False,
    use_graph: bool = False,
    dynamic_weight: bool = False,
    use_cluster_loss: bool = True,
    *args,
    **kwargs,
) -> str:
    """
    根据SND任务的参数配置自动生成唯一的run_name

    参数:
        model_name_or_path: 模型路径，从中提取模型大小
        positive_num: 正样本数量
        packing_size: packing大小
        temperature: 对比学习温度
        temperature_decay: 温度衰减系数
        temperature_decay_step: 温度衰减步长
        learning_rate: 学习率
        shuffle: 是否打乱数据
        use_graph: 是否使用图数据
        dynamic_weight: 是否使用动态权重
        use_cluster_loss: 是否使用聚类损失

    返回:
        生成的run_name字符串
    """
    # 提取模型大小信息
    expected_model_sizes = [
        "1.5B",
        "3B",
        "7B",
        "14B",
    ]
    model_size = "custom"
    for size in expected_model_sizes:
        if size in model_name_or_path:
            model_size = size
            break

    # 提取模型名称
    expected_model_names = [
        "LLaMA",
        "Llama",
        "Mistral",
        "Mistral-7B",
        "Llama-2",
        "Qwen",
        "bge",
        "BGE",
    ]
    model_name = "custom"
    for name in expected_model_names:
        if name in model_name_or_path:
            model_name = name
            break

    # 构造名称组件
    components = [
        model_size,
        model_name,
        f"{positive_num if positive_num is not None else 'auto'}Pos",
        f"{packing_size}Pack",
        f"{temperature:.2f}Temp",
    ]

    # 添加温度衰减信息（如果不是默认值）
    if temperature_decay != 1.0:
        components.extend(
            [f"{temperature_decay:.2f}Decay", f"{temperature_decay_step}DecayStep"]
        )

    # 添加学习率信息
    components.append(f"{learning_rate}LR")

    # 添加数据处理和模型配置信息
    components.append("Sequential" if not shuffle else "Shuffled")
    if use_graph:
        components.append("Graph")
    if dynamic_weight:
        components.append("DynamicW")
    if use_cluster_loss:
        components.append("ClusterLoss")

    # 生成最终名称
    return "-".join(components)


def get_snd_output_dir(
    model_name_or_path: str,
    base_output_dir: Optional[str] = None,
    run_name: str = None,
    **kwargs,
) -> str:
    """
    根据SND任务参数生成输出目录路径

    参数:
        model_name_or_path: 模型路径
        base_output_dir: 基础输出目录（可选）
        **kwargs: 传递给generate_snd_run_name的参数

    返回:
        完整的输出目录路径
    """

    if base_output_dir is None:
        # 默认放在模型同级目录的SND文件夹下
        base_output_dir = os.path.join(
            os.path.dirname(os.path.dirname(model_name_or_path)), "SND"
        )

    return os.path.join(base_output_dir, run_name)


@dataclass
class SpecialToken:
    GRAPH_TOKEN: str = "<graph token>"


@dataclass
class DataArguments:
    data_dir: str = field(
        default="data/SND",
        metadata={
            "help": "The input data dir. Should contain the .csv files (or other data files) for the task."
        },
    )

    names_pub_dir: str = field(
        default="data/SND/names-pub",
        metadata={
            "help": "The input names_pub dir. Should contain the .csv files (or other data files) for the task."
        },
    )

    feature_dir: str = field(
        default="data/SND/features",
        metadata={
            "help": "The input feature dir. Should contain the .csv files (or other data files) for the task."
        },
    )

    graph_feature_path: str = field(
        default="data/SND/pub_graph_embs.safetensors",
        metadata={
            "help": "The input graph feature path. Should be .safetensors file (or other data files) for the task."
        },
    )

    packing_size: int = field(
        default=4,
        metadata={"help": "The packing size of the dataset."},
    )

    positive_ratio: float = field(
        default=None,
        metadata={"help": "The positive ratio of the dataset."},
    )

    positive_num: int = field(
        default=None,
        metadata={"help": "The positive number of the dataset."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "The model checkpoint for weights initialization."},
    )

    padding_side: str = field(
        default="left",
        metadata={"help": "The padding side of the tokenizer."},
    )

    use_graph: bool = field(
        default=False,
        metadata={"help": "Whether to use graph data."},
    )

    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use lora."},
    )

    enable_quantization: bool = field(
        default=False,
        metadata={"help": "Whether to enable quantization."},
    )

    lora_module_path: str = field(
        default=None,
        metadata={"help": "The path to the pretrained lora module."},
    )

    graph_hidden_size: int = field(
        default=768,
        metadata={"help": "The hidden size of the graph projection module."},
    )

    graph_proj_module_path: str = field(
        default=None,
        metadata={"help": "The path to the pretrained graph projection module."},
    )


@dataclass
class SNDTrainingArguments(TrainingArguments):
    db_eps: float = field(
        default=0.1,
        metadata={"help": "The eps of the DBSCAN."},
    )

    db_min: int = field(
        default=5,
        metadata={"help": "The min of the DBSCAN."},
    )

    temperature: float = field(
        default=0.05,
        metadata={"help": "The temperature of the contrastive loss."},
    )

    temperature_decay: float = field(
        default=1,
        metadata={"help": "The temperature decay of the contrastive loss."},
    )

    temperature_decay_step: int = field(
        default=1000,
        metadata={"help": "The temperature decay step of the contrastive loss."},
    )

    shuffle: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the dataset."},
    )

    dynamic_weight: bool = field(
        default=False,
        metadata={"help": "Whether to use dynamic weight."},
    )

    use_cluster_loss: bool = field(
        default=True,
        metadata={"help": "Whether to use cluster loss."},
    )

@dataclass
class InferenceArguments:
    mode: str = field(
        default="train",
        metadata={"help": "The mode of the inference."},
    )