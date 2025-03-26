import json
import os
from typing import Literal

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import PreTrainedTokenizer

from .arguments import DataArguments
from .utils.hierarchical_sampler import HierarchicalSampler
from .utils.logger import get_logger
from .utils.prompt_template import format_paper_for_llm

VALID_MODES = ["train", "dev", "valid", "test"]

logger = get_logger(__name__)


class SNDPackingCollator:
    def __call__(self, batch):
        return batch[0]


class SNDPackingDataset(Dataset):
    """The Dataset can be used for supervised name disambiguation and unsupervised name disambiguation
    with contrastive learning.

    Supervised name disambiguation dataset is designed as packing dataset as features and labels

    Unsupervised name disambiguation dataset is designed as packing dataset as features.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        data_args: DataArguments = None,
        mode: Literal["train", "dev", "valid", "test"] = "train",
    ):
        self.data_args: DataArguments
        self.names_pub: dict
        self.feature_dir: str
        self.labels: dict
        self.pub_to_author: dict
        self.tokenizer: PreTrainedTokenizer

        self.data_args = data_args
        self.tokenizer = tokenizer
        self.mode = mode

        assert self.mode in VALID_MODES, f"Invalid mode: {self.mode}"

        # Load the data
        names_pub_path = os.path.join(data_args.names_pub_dir, f"{self.mode}.json")
        self.feature_dir = os.path.join(data_args.feature_dir, self.mode)

        # Load the names_pub data
        with open(names_pub_path, "r") as f:
            self.names_pub = json.load(f)

        label_path = None

        author_pub_labels = {}
        if self.mode in ["train", "dev"]:  # 统一处理监督模式
            label_filename = "train_labels.json"
            label_path = os.path.join(data_args.data_dir, label_filename)

            if not os.path.exists(label_path):
                raise FileNotFoundError(
                    f"Label file required for {self.mode} mode: {label_path}"
                )
            with open(label_path, "r") as f:
                all_author_pub_labels = json.load(f)

            for author_name in all_author_pub_labels:
                if author_name in self.names_pub:
                    author_pub_labels[author_name] = all_author_pub_labels[author_name]

        self.has_labels = label_path is not None
        self.pub_to_author = {}

        if self.has_labels:
            self.labels = {}
            for author, pubs in author_pub_labels.items():
                for pub, label in pubs.items():
                    self.pub_to_author[pub] = author
                    self.labels[pub] = label

        self.data = []
        if self.mode == "train":
            # packing: "text_feature", "label"
            self.sampler = HierarchicalSampler(
                names_pub=self.names_pub,
                author_pub_labels=author_pub_labels,
                packing_size=data_args.packing_size,
                positive_ratio=data_args.positive_ratio,
                positive_num=data_args.positive_num,
            )
            pubs_packing_data = self.sampler.sampling()
            for packing_data in tqdm(
                pubs_packing_data, desc="Constructing the training features"
            ):
                data = []
                for pub in packing_data:
                    features = self._get_features(pub)
                    data.append(features)
                self.data.append(data)
        else:
            for author_name, pubs in tqdm(
                self.names_pub.items(), desc=f"Constructing the {self.mode} features"
            ):
                data = []
                for pub in pubs:
                    features = self._get_features(pub)
                    features["author_name"] = author_name
                    if self.has_labels:
                        features["label"] = self.labels[pub]
                    data.append(features)
                for i in range(0, len(data), data_args.packing_size):
                    self.data.append(data[i : i + data_args.packing_size])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        text_features = [d["text_feature"] for d in data]

        # Tokenize the text features
        tokenized_text_features = self.tokenizer(
            text_features,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        input_ids, attention_mask = (
            tokenized_text_features["input_ids"],
            tokenized_text_features["attention_mask"],
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.has_labels:
            labels = [d["label"] for d in data]
            labels = torch.LongTensor(labels)
            batch["labels"] = labels

        if self.mode != "train":
            batch["author_names"] = [d["author_name"] for d in data]

        return batch

    def _get_features(self, pub: str):
        author_name = self.pub_to_author[pub]

        papers_dict = self.names_pub[author_name]
        paper_dict = papers_dict[pub]
        text_feature = format_paper_for_llm(paper_dict, author_name)

        features = {"text_feature": text_feature}

        if self.has_labels:
            features["label"] = self.labels[pub]
        return features


class SNDInferenceDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments):
        self.tokenizer = tokenizer
        self.data_args = data_args

        with open(os.path.join(data_args.names_pub_dir, "valid.json"), "r") as f:
            self.valid_names_pub = json.load(f)

        unpacked_data = []
        for author_name, pubs in tqdm(
            self.valid_names_pub.items(), desc="Constructing the inference features"
        ):
            for pub in pubs:
                papar_dict = pubs[pub]
                features = self._get_features(papar_dict, author_name)
                unpacked_data.append(features)

        self.data = []
        for i in range(0, len(unpacked_data), data_args.packing_size):
            self.data.append(unpacked_data[i : i + data_args.packing_size])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        text_features = [d["text_feature"] for d in data]

        # Tokenize the text features
        tokenized_text_features = self.tokenizer(
            text_features,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        input_ids, attention_mask = (
            tokenized_text_features["input_ids"],
            tokenized_text_features["attention_mask"],
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        batch["author_names"] = [d["author_name"] for d in data]
        batch["pub_ids"] = [d["pub_id"] for d in data]
        return batch

    def _get_features(self, paper_dict, author_name):
        text_feature = format_paper_for_llm(paper_dict, author_name)
        features = {"text_feature": text_feature}
        features["author_name"] = author_name
        features["pub_id"] = paper_dict["id"]
        return features
