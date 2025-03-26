from argparse import ArgumentParser

import torch
from peft import LoraConfig, get_peft_model
from src.arguments import DataArguments, ModelArguments, SNDTrainingArguments
from src.dataset import SNDInferenceDataset, SNDPackingCollator, SNDPackingDataset
from src.modeling import Qwen2ModelForSNDPubEmbedding
from src.trainer import SNDTrainer
from src.utils.logger import distributed_logging, get_logger

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)

logger = get_logger(__name__)
parser = ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="The path to the config file.",
)

args = parser.parse_args()

config_path = args.config

data_args, model_args, training_args = HfArgumentParser(
    (DataArguments, ModelArguments, SNDTrainingArguments)
).parse_json_file(config_path)


distributed_logging(
    logger,
    f"Data Arguments: {data_args}",
    f"Model Arguments: {model_args}",
    f"Training Arguments: {training_args}",
)


tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
)
tokenizer.padding_side = model_args.padding_side
if "Llama" in model_args.model_name_or_path:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = SNDPackingDataset(
    data_args=data_args,
    tokenizer=tokenizer,
    mode="train",
)

eval_dataset = SNDPackingDataset(
    data_args=data_args,
    tokenizer=tokenizer,
    mode="dev",
)

inference_dataset = SNDInferenceDataset(
    data_args=data_args,
    tokenizer=tokenizer,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = Qwen2ModelForSNDPubEmbedding.from_pretrained(
    model_args.model_name_or_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

modules_to_save = []
modules_to_save += "lora"


target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CASUAL_LM",
)

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

if torch.distributed.is_initialized():
    if torch.distributed.get_rank() == 0:
        model.print_trainable_parameters()
else:
    model.print_trainable_parameters()


def verify_lora_weights(model):
    """检查LoRA权重是否正确初始化"""
    empty_weights = []
    valid_weights = []

    for name, param in model.named_parameters():
        if "lora_" in name:
            if param.numel() == 0 or any(s == 0 for s in param.shape):
                empty_weights.append(name)
            else:
                valid_weights.append(name)

    logger.info(
        f"找到 {len(valid_weights)} 个有效LoRA权重，{len(empty_weights)} 个空LoRA权重"
    )
    if empty_weights:
        logger.info(f"空LoRA权重示例:, {empty_weights[:3]}")
    if valid_weights:
        logger.info(f"有效LoRA权重示例: {valid_weights[:3]}")

    return len(valid_weights) > 0


if torch.distributed.is_initialized():
    if torch.distributed.get_rank() == 0:
        is_valid = verify_lora_weights(model)
        if not is_valid:
            raise ValueError("LoRA权重未正确初始化！请检查配置")
else:
    is_valid = verify_lora_weights(model)
    if not is_valid:
        raise ValueError("LoRA权重未正确初始化！请检查配置")


model.gradient_checkpointing_enable()
model.enable_input_require_grads()


# 在开始训练前添加这段代码
def log_grad_hook(module, grad_input, grad_output):
    if hasattr(module, "_name") and "lora" in module._name:
        distributed_logging(
            logger,
            f"Module: {module._name}",
            f"Grad Input: {grad_input}",
            f"Grad Output: {grad_output}",
        )


trainer = SNDTrainer(
    model=model,
    args=training_args,
    data_collator=SNDPackingCollator(),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    eval_dataset={"eval": eval_dataset, "inference": inference_dataset},
)

def train():
    # trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.train()
    
if __name__ == "__main__":
    train()
