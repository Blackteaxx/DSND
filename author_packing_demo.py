import os
from argparse import ArgumentParser

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from src.arguments import (
    DataArguments,
    ModelArguments,
    SNDTrainingArguments,
    SpecialToken,
    generate_snd_run_name,
    get_snd_output_dir,
)
from src.dataset import SNDAuthorPackingDataset, SNDInferenceDataset, SNDPackingCollator
from src.modeling import XLMRobertaModelForSNDPubEmbedding
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
    use_cluster_loss=training_args.use_cluster_loss,
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

model_config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
)

model_config.use_cache = False
model_config.model_args = model_args
dtype = torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = XLMRobertaModelForSNDPubEmbedding.from_pretrained(
    model_args.model_name_or_path,
    config=model_config,
    trust_remote_code=True,
    torch_dtype=dtype,
    # quantization_config=bnb_config,
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


# load dataset
train_dataset = SNDAuthorPackingDataset(
    data_args=data_args,
    tokenizer=tokenizer,
    mode="train",
    shuffle=training_args.shuffle,
    use_graph=model_args.use_graph,
)

eval_dataset = SNDAuthorPackingDataset(
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
modules_to_save = ["full"]

training_args.modules_to_save = modules_to_save

logger.info(f"模型 {model_args.model_name_or_path} 的训练模块为: {modules_to_save}")
model.requires_grad = True

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


trainer = SNDTrainer(
    model=model,
    args=training_args,
    data_collator=SNDPackingCollator(),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    eval_dataset={"eval": eval_dataset, "inference": inference_dataset},
)


def train():
    trainer.train()


if __name__ == "__main__":
    train()
