# %%
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

from src.modeling import Qwen2ModelForSNDPubEmbedding

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = Qwen2ModelForSNDPubEmbedding.from_pretrained(
    "/data/name_disambiguation/model/Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb_config,
)

model


# %%
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model, model.print_trainable_parameters()

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/data/name_disambiguation/model/Qwen/Qwen2.5-3B-Instruct"
)

inputs = tokenizer(
    ["Hello, my dog is cute", "Hello, my", "You are cute", "Hello, you"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# %%
from torch.nn import functional as F

labels = torch.tensor([1, 0, 1, 0]).to(model.device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# get the embeddings
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
last_hidden_states = outputs.embeddings

# 标准化特征向量
norm_hidden = F.normalize(last_hidden_states, p=2, dim=1)

# 计算相似度矩阵
logits = torch.mm(norm_hidden, norm_hidden.T) / 0.07

# 构建标签掩码
label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
filtered_label_mask = label_mask.fill_diagonal_(0)
num_pos_samples = filtered_label_mask.sum(dim=1, keepdim=True)
valid_samples = num_pos_samples.squeeze() > 0

# 计算InfoNCE损失
# 排除自身样本，使用负无穷大填充对角线
true_logits = logits.clone()
true_logits[torch.eye(logits.size(0), dtype=torch.bool)] = float("-inf")
log_prob = F.log_softmax(true_logits, dim=1)
filtered_log_prob = log_prob.clone().fill_diagonal_(0)

per_sample_loss = (
    -(filtered_label_mask * filtered_log_prob) / (num_pos_samples + 1e-8)
).sum(dim=1)
contrastive_loss = per_sample_loss[valid_samples].mean()

print("contrastive_loss", contrastive_loss)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.grad)

# %%
from transformers import AutoTokenizer

from src.arguments import DataArguments
from src.dataset import SNDPackingCollator, SNDPackingDataset

dataset = SNDPackingDataset(data_args=DataArguments(), tokenizer=tokenizer)


# %%
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    collate_fn=SNDPackingCollator(),
    shuffle=True,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for batch in dataloader:
    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"].to(model.device)

    # get the embeddings
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    last_hidden_states = outputs.embeddings

    # 标准化特征向量
    norm_hidden = F.normalize(last_hidden_states, p=2, dim=1)

    # 计算相似度矩阵
    logits = torch.mm(norm_hidden, norm_hidden.T) / 0.07

    # 构建标签掩码
    label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    filtered_label_mask = label_mask.fill_diagonal_(0)
    num_pos_samples = filtered_label_mask.sum(dim=1, keepdim=True)
    valid_samples = num_pos_samples.squeeze() > 0

    # 计算InfoNCE损失
    # 排除自身样本，使用负无穷大填充对角线
    true_logits = logits.clone()
    true_logits[torch.eye(logits.size(0), dtype=torch.bool)] = float("-inf")
    log_prob = F.log_softmax(true_logits, dim=1)
    filtered_log_prob = log_prob.clone().fill_diagonal_(0)

    per_sample_loss = (
        -(filtered_label_mask * filtered_log_prob) / (num_pos_samples + 1e-8)
    ).sum(dim=1)
    contrastive_loss = per_sample_loss[valid_samples].mean()

    optimizer.zero_grad()
    contrastive_loss.backward()
    optimizer.step()

    print(contrastive_loss)
    i = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            i += 1
            print(name, param.grad)
        if i > 10:
            break
