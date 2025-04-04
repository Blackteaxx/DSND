import os

import numpy as np
import torch
from gensim.models import Word2Vec
from safetensors.torch import save_file

from transformers import HfArgumentParser

from ..arguments import DataArguments
from ..dataset import SNDInferenceDataset
from ..utils.logger import get_logger

logger = get_logger(__name__)


def sentence_embedding_inference(
    sentence: str,
    model: Word2Vec,
):
    """
    使用Word2Vec模型生成句子嵌入
    Args:
        sentence: 输入句子
        model: 训练好的Word2Vec模型
    Returns:
        嵌入向量
    """
    # 分词
    tokens = sentence.split(" ")
    # 获取嵌入向量
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    # 计算句子向量
    sentence_vector = np.mean(vectors, axis=0)
    return torch.tensor(sentence_vector)


data_args: DataArguments
data_args = HfArgumentParser(DataArguments).parse_args_into_dataclasses()[0]

logger.info(f"data_args: {data_args}, model_args: {None}, training_args: {None}")

train_inference_dataset = SNDInferenceDataset(
    tokenizer=None,
    data_args=data_args,
    mode="train",
    use_graph=False,
)
dev_inference_dataset = SNDInferenceDataset(
    tokenizer=None,
    data_args=data_args,
    mode="dev",
    use_graph=False,
)

# collect corpus
corpus = []

for idx in range(len(train_inference_dataset)):
    data = train_inference_dataset[idx]
    text_features = data["text_features"]
    pub_ids = data["pub_ids"]

    corpus.extend(text_features)

for idx in range(len(dev_inference_dataset)):
    data = dev_inference_dataset[idx]
    text_features = data["text_features"]
    pub_ids = data["pub_ids"]

    corpus.extend(text_features)

# tokenize corpus
tokenized_corpus = []
for text in corpus:
    tokenized_text = text.split(" ")
    tokenized_corpus.append(tokenized_text)

# train word2vec model
word2vec_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,  # Skip-gram
)


pub_embeddings = {}
# get sentence vectors
for idx in range(len(train_inference_dataset)):
    data = train_inference_dataset[idx]
    text_features = data["text_features"]
    pub_ids = data["pub_ids"]

    for i in range(len(text_features)):
        sentence_vector = sentence_embedding_inference(text_features[i], word2vec_model)
        pub_embeddings[pub_ids[i]] = sentence_vector
        print(f"pub_id: {pub_ids[i]}, vector: {sentence_vector}")

for idx in range(len(dev_inference_dataset)):
    data = dev_inference_dataset[idx]
    text_features = data["text_features"]
    pub_ids = data["pub_ids"]

    for i in range(len(text_features)):
        sentence_vector = sentence_embedding_inference(text_features[i], word2vec_model)
        pub_embeddings[pub_ids[i]] = sentence_vector
        print(f"pub_id: {pub_ids[i]}, vector: {sentence_vector}")

# save sentence vectors
save_file(
    pub_embeddings,
    os.path.join(data_args.data_dir, "word2vec_pub_embs.safetensors"),
)
