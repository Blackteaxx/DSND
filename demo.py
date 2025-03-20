import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)

from src.arguments import DataArguments, SNDTrainingArguments
from src.dataset import SNDPackingCollator, SNDPackingDataset
from src.modeling import Qwen2ModelForSNDPubEmbedding
from src.trainer import SNDTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

data_args, training_args = HfArgumentParser(
    (DataArguments, SNDTrainingArguments)
).parse_json_file("/data/name_disambiguation/configs/snd_packing.json")

tokenizer = AutoTokenizer.from_pretrained(
    "/data/name_disambiguation/model/Qwen/Qwen2.5-7B"
)
tokenizer.padding_side = "left"

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

logger.info(eval_dataset[0])


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = Qwen2ModelForSNDPubEmbedding.from_pretrained(
    "/data/name_disambiguation/model/Qwen/Qwen2.5-7B",
    # quantization_config=bnb_config,
    trust_remote_code=True,
    # attn_implementation="flash_attention_2",
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

model.print_trainable_parameters()

# for name, param in model.named_parameters():
#     if "lora" in name:
#         logger.info(name)
#         logger.info(param.data.shape)


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


# 使用方法
is_valid = verify_lora_weights(model)
if not is_valid:
    raise ValueError("LoRA权重未正确初始化！请检查配置")


model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# model.train()


# 在开始训练前添加这段代码
def log_grad_hook(module, grad_input, grad_output):
    if hasattr(module, "_name") and "lora" in module._name:
        logger.info(f"模块 {module._name} 梯度范数: {grad_output[0].norm().item():.4e}")


# 为模型的关键部分注册钩子
# for name, module in model.named_modules():
#     if any(target in name for target in target_modules) and "lora" in name:
#         module._name = name  # 存储名称以便在钩子中使用
#         module.register_full_backward_hook(log_grad_hook)


# logger.info(model.state_dict().keys())

trainer = SNDTrainer(
    model=model,
    args=training_args,
    data_collator=SNDPackingCollator(),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    eval_dataset=eval_dataset,
)

trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
# trainer.train()
