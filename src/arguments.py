from dataclasses import dataclass, field

from transformers import TrainingArguments


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
