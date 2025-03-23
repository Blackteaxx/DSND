import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Trainer
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import has_length

from .utils.evaluate import compute_pairwise_f1, hybrid_evaluate
from .utils.logger import get_logger
from .criterion import ContrastiveLossCalculator


logger = get_logger(__name__)


class SNDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss = ContrastiveLossCalculator(self.args.temperature)

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # 获取嵌入
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        last_hidden_states = outputs.embeddings

        # 计算对比损失 - 所有设备都计算相同的损失
        contrastive_loss = self.contrastive_loss(last_hidden_states, labels)

        # 不需要额外的梯度同步了，torch.distributed.nn.all_gather会处理
        return (contrastive_loss, outputs) if return_outputs else contrastive_loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()
        args = self.args
        dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        batch_size = self.args.eval_batch_size

        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        # Start evaluation
        model.eval()
        eval_result = []
        observed_num_examples = 0
        with torch.no_grad():
            for inputs in tqdm(
                dataloader, desc=f"Evaluating, step: {self.state.global_step}"
            ):
                # Update the observed num examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                        batch_size = observed_batch_size

                # compute the embeddings
                inputs_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                author_names = inputs["author_names"]
                labels = inputs["labels"]

                outputs = model(input_ids=inputs_ids, attention_mask=attention_mask)
                embeddings = outputs.embeddings
                res = [
                    {
                        "author_name": author_names[i],
                        "embedding": embeddings[i].detach().cpu().float().numpy(),
                        "label": labels[i].item(),
                    }
                    for i in range(len(author_names))
                ]

                # gather the results
                raw = self.accelerator.gather_for_metrics(res)
                if self.accelerator.is_main_process:
                    eval_result.extend(raw)

        # logger.info("Observing samples = %d", len(eval_result))

        # compute the metrics
        if self.accelerator.is_main_process:
            author_embeddings = {}
            author_labels = {}
            for pub in eval_result:
                author_name = pub["author_name"]
                embedding = pub["embedding"]
                label = pub["label"]
                if author_name not in author_embeddings:
                    author_embeddings[author_name] = []
                if author_name not in author_labels:
                    author_labels[author_name] = []
                author_embeddings[author_name].append(embedding)
                author_labels[author_name].append(label)

            # with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
            #     f.write(f"step:{self.state.global_step}, epoch:{self.state.epoch}\n")
            #     for author_name in author_embeddings:
            #         f.write(f"{author_name}: {len(author_embeddings[author_name])}\n")
            #         f.write(f"{author_name}: {author_embeddings[author_name]}\n")
            #         f.write(f"{author_name}: {len(author_labels[author_name])}\n")

            for author_name in author_embeddings:
                author_embeddings[author_name] = np.stack(
                    author_embeddings[author_name]
                )

            best_results = hybrid_evaluate(
                author_embeddings, author_labels, self.args.db_eps, self.args.db_min
            )
            
            avg_f1 = best_results["avg_f1"]
            predict_results = best_results["predict_results"]
            db_eps = best_results["db_eps"]
            db_min = best_results["db_min"]

            os.makedirs(os.path.join(args.output_dir, "result"), exist_ok=True)
            with open(
                os.path.join(
                    args.output_dir, f"result/step-{self.state.global_step}.json"
                ),
                "w",
            ) as f:
                for author_name in predict_results:
                    f.write(
                        json.dumps(
                            {
                                "author_name": author_name,
                                "pred": predict_results[author_name],
                                "labels": author_labels[author_name],
                            }
                        )
                        + "\n"
                    )

            # update best metric
            if self.state.best_metric is None:
                self.state.best_metric = avg_f1
                self.state.best_model_checkpoint = os.path.join(
                    self.args.output_dir,
                    f"checkpoint-{self.state.global_step}",
                )
            elif avg_f1 > self.state.best_metric:
                self.state.best_metric = avg_f1
                self.state.best_model_checkpoint = os.path.join(
                    self.args.output_dir,
                    f"checkpoint-{self.state.global_step}",
                )

            output = {
                "avg_f1": avg_f1,
                "step": self.state.global_step,
                "epoch": self.state.epoch,
                "db_eps": db_eps,
                "db_min": db_min,
            }
            self.log(output)
            with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
                f.write(
                    f"step:{self.state.global_step}, epoch:{self.state.epoch}, AVG-F1:{avg_f1}\n"
                )

        # return output
