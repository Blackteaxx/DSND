import time
from argparse import ArgumentParser

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from src.arguments import (
    DataArguments,
    InferenceArguments,
    ModelArguments,
    SNDTrainingArguments,
    SpecialToken,
)
from src.dataset import SNDInferenceDataset
from src.inference_agent import InferenceAgent
from src.modeling import Qwen2ModelForSNDPubEmbedding
from src.utils.add_special_token import smart_tokenizer_and_embedding_resize
from src.utils.logger import distributed_logging, get_logger
from tqdm import tqdm

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

data_args, model_args, training_args, infer_args = HfArgumentParser(
    (DataArguments, ModelArguments, SNDTrainingArguments, InferenceArguments)
).parse_yaml_file(config_path)

model_path = model_args.model_name_or_path
inference_embedding_name = model_path.split("/")[-1] + ".safetensors"


distributed_logging(
    logger,
    f"Data Arguments: {data_args}",
    f"Model Arguments: {model_args}",
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
model.cpu()
model.eval()

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


# load LoRA weights
if model_args.lora_module_path:
    module_state_dict = load_file(
        filename=model_args.lora_module_path,
    )

    # 使用peft加载LoRA权重
    set_peft_model_state_dict(model, module_state_dict)
    logger.info(f"LoRA weights loaded with {len(module_state_dict)} keys.")
    inference_embedding_name = "lora_" + inference_embedding_name

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
    inference_embedding_name = "graph_proj_" + inference_embedding_name


def main():
    inference_agent = InferenceAgent(
        model=model,
    )

    # load dataset
    if infer_args.mode == "train":
        train_inference_dataset = SNDInferenceDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            mode="train",
            use_graph=model_args.use_graph,
        )
        dev_inference_dataset = SNDInferenceDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            mode="dev",
            use_graph=model_args.use_graph,
        )

        packing_sentences = []
        for i in tqdm(range(len(train_inference_dataset)), desc="Packing sentences."):
            packing_sentences.append(train_inference_dataset[i])
        for i in tqdm(range(len(dev_inference_dataset)), desc="Packing sentences."):
            packing_sentences.append(dev_inference_dataset[i])

        start_time = time.time()

        logger.info("Starting inferencing with train and dev dataset.")
        results = inference_agent.encode(
            sentences=packing_sentences,
        )
        end_time = time.time()
        logger.info(f"Time taken for inferencing: {end_time - start_time:.2f} seconds.")
        pt_embeddings = results["embeddings"]
        pub_ids = results["pub_ids"]
        embeddings = {}

        for i in range(len(pub_ids)):
            pub_id = pub_ids[i]
            embedding = pt_embeddings[i]
            embeddings[pub_id] = embedding.cpu()

        # save embeddings
        save_file(embeddings, f"data/SND/infer-features/{inference_embedding_name}")

    elif infer_args.mode == "private":
        private_inference_dataset = SNDInferenceDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            mode="private",
            use_graph=model_args.use_graph,
        )

        embeddings = {}
        for i in tqdm(
            range(len(private_inference_dataset)), desc="Private Inferencing."
        ):
            data = private_inference_dataset[i]
            pub_ids = data["pub_ids"]
            print(f"pub_ids: {pub_ids}")
            print(f"data: {data}")
            print("data")

            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda()

            with torch.no_grad():
                outputs = model(
                    **data,
                )

            pub_embeddings = outputs.embeddings

            for j in range(len(pub_ids)):
                pub_id = pub_ids[j]
                embeddings[pub_id] = pub_embeddings[j].cpu()

        # save embeddings
        save_file(embeddings, f"data/private/{inference_embedding_name}")


if __name__ == "__main__":
    main()
