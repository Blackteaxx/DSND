import json
import os
from typing import Literal

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import PreTrainedTokenizer

from .arguments import DataArguments
from .utils.logger import get_logger
from .utils.prompt_template import format_paper_for_bert, format_paper_for_llm
from .utils.triplet_collector import TripletCollector

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
        shuffle: bool = True,
        seed: int = 42,
        use_graph: bool = False,
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
        self.shuffle = shuffle
        self.seed = seed
        self.use_graph = use_graph

        assert self.mode in VALID_MODES, f"Invalid mode: {self.mode}"

        # Load the data
        names_pub_path = os.path.join(data_args.names_pub_dir, f"{self.mode}.json")
        self.feature_dir = os.path.join(data_args.feature_dir, self.mode)
        graph_embedding_path = data_args.graph_feature_path

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

        # 读取graph embedding
        if self.use_graph:
            if not os.path.exists(graph_embedding_path):
                raise FileNotFoundError(
                    f"Graph embedding file required for {self.mode} mode: {graph_embedding_path}"
                )
            self.pub_graph_embeddings = load_file(graph_embedding_path)

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
            self.sampler = TripletCollector(
                names_pub=self.names_pub,
                author_pub_labels=author_pub_labels,
                packing_size=data_args.packing_size,
                positive_ratio=data_args.positive_ratio,
                positive_num=data_args.positive_num,
                shuffle=shuffle,
                reuse_pos_samples=False,
                random_seed=seed,
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
        batch = {}

        text_features = [d["text_feature"] for d in data]

        # Tokenize the text features
        tokenized_text_features = self.tokenizer(
            text_features,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        for k, v in tokenized_text_features.items():
            batch[k] = v

        if self.has_labels:
            labels = [d["label"] for d in data]
            labels = torch.LongTensor(labels)
            batch["labels"] = labels

        if self.use_graph:
            graph_embeddings = [d["graph_embedding"] for d in data]
            graph_embeddings = torch.stack(graph_embeddings)
            batch["graph_embeddings"] = graph_embeddings

        if self.mode != "train":
            batch["author_names"] = [d["author_name"] for d in data]

        return batch

    def _get_features(self, pub: str):
        """

        Args:
            pub (str): _description_

        Returns:
            _type_: _description_
        """
        author_name = self.pub_to_author[pub]

        papers_dict = self.names_pub[author_name]
        paper_dict = papers_dict[pub]
        text_feature = format_paper_for_llm(
            paper_dict, author_name, use_graph=self.use_graph
        )

        features = {"text_feature": text_feature}

        if self.has_labels:
            features["label"] = self.labels[pub]

        if self.use_graph:
            graph_embedding = self.pub_graph_embeddings[pub]
            features["graph_embedding"] = graph_embedding

        return features


class SNDInferenceDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        mode: Literal["train", "dev", "valid", "test", "private"] = "train",
        use_graph: bool = False,
    ):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.use_graph = use_graph

        graph_embedding_path = data_args.graph_feature_path

        with open(os.path.join(data_args.names_pub_dir, f"{mode}.json"), "r") as f:
            self.valid_names_pub = json.load(fp=f)

        if self.use_graph:
            if not os.path.exists(graph_embedding_path):
                raise FileNotFoundError(
                    f"Graph embedding file required for {self.mode} mode: {graph_embedding_path}"
                )
            self.pub_graph_embeddings = load_file(graph_embedding_path)

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
        tokenized_text_features = (
            self.tokenizer(
                text_features,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            if self.tokenizer
            else {"text_features": text_features}
        )
        batch = {k: v for k, v in tokenized_text_features.items()}

        batch["author_names"] = [d["author_name"] for d in data]
        batch["pub_ids"] = [d["pub_id"] for d in data]

        if self.use_graph:
            graph_embeddings = [d["graph_embedding"] for d in data]
            graph_embeddings = torch.stack(graph_embeddings)
            batch["graph_embeddings"] = graph_embeddings

        return batch

    def _get_features(self, paper_dict, author_name):
        text_feature = format_paper_for_llm(
            paper_dict, author_name, use_graph=self.use_graph
        )
        features = {"text_feature": text_feature}
        features["author_name"] = author_name
        features["pub_id"] = paper_dict["id"]

        if self.use_graph:
            graph_embedding = self.pub_graph_embeddings[paper_dict["id"]]
            features["graph_embedding"] = graph_embedding

        return features


class SNDAuthorPackingDataset(Dataset):
    """The Dataset can be used for supervised name disambiguation and unsupervised name disambiguation, with author name packing

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        data_args: DataArguments = None,
        mode: Literal["train", "dev", "valid", "test"] = "train",
        seed: int = 42,
        use_graph: bool = False,
        *args,
        **kwargs,
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
        self.seed = seed
        self.use_graph = use_graph

        assert self.mode in VALID_MODES, f"Invalid mode: {self.mode}"

        # Load the data
        names_pub_path = os.path.join(data_args.names_pub_dir, f"{self.mode}.json")
        self.feature_dir = os.path.join(data_args.feature_dir, self.mode)
        graph_embedding_path = data_args.graph_feature_path

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

        # 读取graph embedding
        if self.use_graph:
            if not os.path.exists(graph_embedding_path):
                raise FileNotFoundError(
                    f"Graph embedding file required for {self.mode} mode: {graph_embedding_path}"
                )
            self.pub_graph_embeddings = load_file(graph_embedding_path)

        self.has_labels = label_path is not None
        self.pub_to_author = {}

        if self.has_labels:
            self.labels = {}
            for author, pubs in author_pub_labels.items():
                for pub, label in pubs.items():
                    self.pub_to_author[pub] = author
                    self.labels[pub] = label

        self.data = []
        # 依据author_name进行packing
        for author_name, pubs in tqdm(
            author_pub_labels.items(), desc=f"Constructing the {self.mode} features"
        ):
            data = []
            for pub in pubs:
                features = self._get_features(pub)
                data.append(features)
            packing_size = len(data) // 6

            # logger.info(
            #     f"Packing {len(data)} papers for author {author_name} into {packing_size} batches"
            # )
            if packing_size > 150:
                logger.warning(
                    f"Packing size for author {author_name} is too large: {packing_size}"
                )
                continue
            for i in range(0, packing_size * 6, packing_size):
                self.data.append(data[i : i + packing_size])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        batch = {}

        text_features = [d["text_feature"] for d in data]

        # Tokenize the text features
        tokenized_text_features = self.tokenizer(
            text_features,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        for k, v in tokenized_text_features.items():
            batch[k] = v

        batch["author_names"] = [d["author_name"] for d in data]

        if self.has_labels:
            labels = [d["label"] for d in data]
            labels = torch.LongTensor(labels)
            batch["labels"] = labels

        if self.use_graph:
            graph_embeddings = [d["graph_embedding"] for d in data]
            graph_embeddings = torch.stack(graph_embeddings)
            batch["graph_embeddings"] = graph_embeddings

        return batch

    def _get_features(self, pub: str):
        """

        Args:
            pub (str): _description_

        Returns:
            _type_: _description_
        """
        author_name = self.pub_to_author[pub]

        papers_dict = self.names_pub[author_name]
        paper_dict = papers_dict[pub]
        text_feature = format_paper_for_bert(
            paper_dict, author_name, use_graph=self.use_graph
        )

        features = {"text_feature": text_feature}

        if self.has_labels:
            features["label"] = self.labels[pub]

        if self.use_graph:
            graph_embedding = self.pub_graph_embeddings[pub]
            features["graph_embedding"] = graph_embedding

        features["author_name"] = author_name

        return features
