import os
from argparse import ArgumentParser

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from src.arguments import (
    DataArguments,
    ModelArguments,
    SNDTrainingArguments,
    SpecialToken,
    generate_snd_run_name,
    get_snd_output_dir,
)
from src.dataset import SNDInferenceDataset, SNDPackingCollator, SNDPackingDataset
from src.modeling import Qwen2ModelForSNDPubEmbedding
from src.trainer import SNDTrainer
from src.utils.add_special_token import smart_tokenizer_and_embedding_resize
from src.utils.logger import distributed_logging, get_logger

from transformers import (
    AutoConfig,
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
).parse_yaml_file(config_path)

# 设置run_name
training_args.run_name = generate_snd_run_name(
    model_name_or_path=model_args.model_name_or_path,
    positive_num=data_args.positive_num,
    packing_size=data_args.packing_size,
    temperature=training_args.temperature,
    temperature_decay=training_args.temperature_decay,
    temperature_decay_step=training_args.temperature_decay_step,
    learning_rate=training_args.learning_rate,
    shuffle=training_args.shuffle,
    use_graph=model_args.use_graph,
    dynamic_weight=training_args.dynamic_weight,
    use_contrastive_loss=training_args.use_contrastive_loss,
    use_cluster_loss=training_args.use_cluster_loss,
    loss_weight=training_args.loss_weight,
    sentence_pooling_method=model_args.sentence_pooling_method,
)
training_args.output_dir = get_snd_output_dir(
    model_name_or_path=model_args.model_name_or_path,
    base_output_dir=training_args.output_dir,
    run_name=training_args.run_name,
)
training_args.logging_dir = os.path.join(
    training_args.output_dir,
    "logs",
)

distributed_logging(
    logger,
    f"Data Arguments: {data_args}",
    f"Model Arguments: {model_args}",
    f"Training Arguments: {training_args}",
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
)
tokenizer.padding_side = model_args.padding_side

if "Llama" in model_args.model_name_or_path:
    tokenizer.pad_token = tokenizer.eos_token

# load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
)

model_config.use_cache = False
model_config.model_args = model_args
dtype = torch.bfloat16

model = Qwen2ModelForSNDPubEmbedding.from_pretrained(
    model_args.model_name_or_path,
    config=model_config,
    quantization_config=bnb_config if model_args.enable_quantization else None,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=dtype,
)

# add special tokens
if model_args.use_graph:
    special_tokens_dict = {"additional_special_tokens": [SpecialToken.GRAPH_TOKEN]}
    # add speical token to embedding
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    model.add_special_tokens(tokenizer)


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

# load dataset
train_dataset = SNDPackingDataset(
    data_args=data_args,
    tokenizer=tokenizer,
    mode="train",
    shuffle=training_args.shuffle,
    use_graph=model_args.use_graph,
)

eval_dataset = SNDPackingDataset(
    data_args=data_args,
    tokenizer=tokenizer,
    mode="dev",
    shuffle=training_args.shuffle,
    use_graph=model_args.use_graph,
)

inference_dataset = SNDInferenceDataset(
    data_args=data_args,
    tokenizer=tokenizer,
    use_graph=model_args.use_graph,
)


# set the module to be trained
modules_to_save = []
if model_args.use_lora:
    modules_to_save.append("lora")
if model_args.use_graph:
    modules_to_save.append("graph_proj")

training_args.modules_to_save = modules_to_save

logger.info(f"模型 {model_args.model_name_or_path} 的训练模块为: {modules_to_save}")
model.requires_grad = False


# 精确匹配模块层级结构
def set_trainable_params(model, patterns):
    for name, param in model.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad = True
            logger.info(f"✅ 可训练参数: {name}")
        else:
            param.requires_grad = False


set_trainable_params(model, modules_to_save)


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


# load LoRA weights
if model_args.lora_module_path:
    module_state_dict = load_file(
        filename=model_args.lora_module_path,
    )

    # 使用peft加载LoRA权重
    set_peft_model_state_dict(model, module_state_dict)
    logger.info(f"LoRA weights loaded with {len(module_state_dict)} keys.")

if model_args.graph_proj_module_path:
    graph_proj_state_dict = load_file(
        filename=model_args.graph_proj_module_path,
    )

    # 使用peft加载图投影权重
    missing_keys, unexpected_keys = model.load_state_dict(
        graph_proj_state_dict,
        strict=False,
    )

    logger.info(
        f"Graph projection weights loaded with {len(graph_proj_state_dict)} keys, with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys."
    )

# enable gradient checkpointing
model.gradient_checkpointing_enable()
model.enable_input_require_grads()


# create hook for gradient checkpointing
def print_grad_norms(model):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
            logger.info(f"Gradient norm for {name}: {param_norm:.4f}")

    total_norm = total_norm ** (1.0 / 2)
    logger.info(f"Total gradient norm: {total_norm:.4f}")


# register the hook
def register_hook(model):
    def hook(module, grad_input, grad_output):
        print_grad_norms(model)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_full_backward_hook(hook)


# register_hook(model)


trainer = SNDTrainer(
    model=model,
    args=training_args,
    data_collator=SNDPackingCollator(tokenizer=tokenizer),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    eval_dataset={"eval": eval_dataset, "inference": inference_dataset},
)


def train():
    # trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.train()


if __name__ == "__main__":
    train()
