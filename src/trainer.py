import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from peft import get_peft_model_state_dict
from safetensors.torch import save_file
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from torch.utils.data import Dataset, RandomSampler, Sampler, SequentialSampler
from tqdm import tqdm

from transformers import Trainer
from transformers.integrations.deepspeed import deepspeed_init
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import has_length

from .criterion import ClusterLossCalculator, ContrastiveLossCalculator
from .utils.evaluate import hybrid_evaluate, predict
from .utils.logger import get_logger

logger = get_logger(__name__)


class SNDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss = ContrastiveLossCalculator()
        self.cluster_loss = ClusterLossCalculator()

        self.temperature = self.args.temperature
        self.temperature_decay = self.args.temperature_decay
        self.temperature_decay_step = self.args.temperature_decay_step

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.last_contrastive_loss = 0.0
        self.last_cluster_loss = 0.0

        self.loss_weight = torch.tensor(0.5, device=self.args.device)
        self.loss_weight.requires_grad = True if self.args.dynamic_weight else False

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        graph_embeddings = inputs.get("graph_embeddings", None)
        labels = inputs["labels"]

        # 计算温度衰减：逐步增加训练难度
        if (
            self.state.global_step != 0
            and self.state.global_step % self.temperature_decay_step == 0
        ):
            self.temperature *= self.temperature_decay
            logger.info(
                f"Step: {self.state.global_step}, Temperature decay to {self.temperature}"
            )

        # 获取嵌入
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, graph_embeddings=graph_embeddings
        )
        last_hidden_states = outputs.embeddings

        all_hidden = self._dist_gather_tensor(last_hidden_states)
        all_labels = self._dist_gather_tensor(labels)

        # 计算对比损失 - 所有设备都计算相同的损失
        contrastive_loss = self.contrastive_loss(
            all_hidden, all_labels, self.temperature
        )

        # 聚类损失: 提纯聚类结果
        if self.args.use_cluster_loss:
            distances = pairwise_distances(
                all_hidden.detach().cpu().float().numpy(),
                metric="cosine",
            )
            pseudo_labels = DBSCAN(
                eps=self.args.db_eps, min_samples=self.args.db_min, metric="precomputed"
            ).fit_predict(distances)
            # 将-1的结果作为outlier处理
            unique_label = max(pseudo_labels) + 1
            for i in range(len(pseudo_labels)):
                if pseudo_labels[i] == -1:
                    pseudo_labels[i] = unique_label
                    unique_label += 1
            pseudo_labels = torch.tensor(pseudo_labels).to(all_hidden.device)
            # 计算聚类损失
            cluster_loss = self.cluster_loss(
                all_hidden,
                pseudo_labels,
                1,
            )
        else:
            cluster_loss = torch.tensor(0.0).to(all_hidden.device)

        self.last_contrastive_loss = contrastive_loss.item()
        self.last_cluster_loss = cluster_loss.item()

        loss_weight = self.loss_weight.to(
            all_hidden.device
        )  # 确保loss_weight在正确的设备上

        loss = loss_weight * contrastive_loss + (1 - loss_weight) * cluster_loss

        # 不需要额外的梯度同步了，torch.distributed.nn.all_gather会处理
        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()
        args = self.args
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        inference_dataset = eval_dataset["inference"]
        eval_dataset = eval_dataset["eval"]
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

        # ======================== Evaluate the results of dev set ========================
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

        logger.info(f"Eval results len: {len(eval_result)}")

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

            for author_name in author_embeddings:
                author_embeddings[author_name] = np.stack(
                    author_embeddings[author_name]
                )

            # save the embeddings and labels for further evaluation
            author_embeddings_torch = {
                author_name: torch.tensor(author_embeddings[author_name])
                for author_name in author_embeddings
            }
            author_labels_torch = {
                author_name: torch.tensor(author_labels[author_name])
                for author_name in author_labels
            }
            os.makedirs(os.path.join(args.output_dir, "dev_embeddings"), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, "dev_labels"), exist_ok=True)
            save_file(
                author_embeddings_torch,
                os.path.join(
                    args.output_dir,
                    "dev_embeddings",
                    f"step-{self.state.global_step}.safetensors",
                ),
            )
            save_file(
                author_labels_torch,
                os.path.join(
                    args.output_dir,
                    "dev_labels",
                    f"step-{self.state.global_step}.safetensors",
                ),
            )

            # evaluate the results
            best_results = hybrid_evaluate(
                author_embeddings=author_embeddings,
                author_labels=author_labels,
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
                    f"step:{self.state.global_step}, epoch:{self.state.epoch}, AVG-F1:{avg_f1}, DB-EPS:{db_eps}, DB-MIN:{db_min}\n"
                )

        # ======================== Predict the results of validation set ========================
        if self.args.do_predict:
            db_eps = self.args.db_eps
            db_min = self.args.db_min
            inference_dataloader = self.get_eval_dataloader(inference_dataset)
            inference_result = []
            with torch.no_grad():
                for inputs in tqdm(
                    inference_dataloader,
                    desc=f"Predicting, step: {self.state.global_step}",
                ):
                    # compute the embeddings
                    inputs_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    author_names = inputs["author_names"]
                    pub_ids = inputs["pub_ids"]

                    outputs = model(input_ids=inputs_ids, attention_mask=attention_mask)
                    embeddings = outputs.embeddings
                    res = [
                        {
                            "author_name": author_names[i],
                            "embedding": embeddings[i].detach().cpu().float().numpy(),
                            "pub_id": pub_ids[i],
                        }
                        for i in range(len(author_names))
                    ]

                    # gather the results
                    raw = self.accelerator.gather_for_metrics(res)
                    if self.accelerator.is_main_process:
                        inference_result.extend(raw)

            logger.info(f"Inference results len: {len(inference_result)}")

            if self.accelerator.is_main_process:
                author_embeddings = {}
                author_pub_ids = {}
                for pub in inference_result:
                    author_name = pub["author_name"]
                    embedding = pub["embedding"]
                    if author_name not in author_embeddings:
                        author_embeddings[author_name] = []
                    if author_name not in author_pub_ids:
                        author_pub_ids[author_name] = []
                    author_embeddings[author_name].append(embedding)
                    author_pub_ids[author_name].append(pub["pub_id"])

                logger.info(f"author_pub_ids: {author_pub_ids['haifeng_qian']}")

                for author_name in author_embeddings:
                    author_embeddings[author_name] = np.stack(
                        author_embeddings[author_name]
                    )

                logger.info("db_eps: %f, db_min: %d", db_eps, db_min)
                inference_predict_results = predict(author_embeddings, db_eps, db_min)
                logger.info(
                    f"haifeng_qian: {len(inference_predict_results['haifeng_qian'])}"
                )
                logger.info(
                    f"haifeng_qian: {inference_predict_results['haifeng_qian']}"
                )

                # save the valid results for uploading
                valid_results = {}
                for author_name, labels in inference_predict_results.items():
                    pub_ids = author_pub_ids[author_name]

                    cluster_pub_ids = {}

                    for pub_id, label in zip(pub_ids, labels):
                        if label not in cluster_pub_ids:
                            cluster_pub_ids[label] = []
                        cluster_pub_ids[label].append(pub_id)

                    # 将-1的结果作为outlier处理
                    non_outlier_results = {}
                    for label in cluster_pub_ids:
                        if label == -1:
                            continue
                        non_outlier_results[label] = cluster_pub_ids[label]

                    sorted_cluster_pub_ids = [
                        non_outlier_results[label]
                        for label in sorted(non_outlier_results.keys())
                    ]
                    if -1 in cluster_pub_ids.keys():
                        for pub_id in cluster_pub_ids[-1]:
                            sorted_cluster_pub_ids.append([pub_id])

                    valid_results[author_name] = sorted_cluster_pub_ids

                os.makedirs(
                    os.path.join(args.output_dir, "valid_result"), exist_ok=True
                )
                with open(
                    os.path.join(
                        args.output_dir,
                        f"valid_result/step-{self.state.global_step}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(valid_results, f, indent=4)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = unwrap_model(self.model)

        model_to_save_state_dict = model_to_save.state_dict()

        modules_to_save = self.args.modules_to_save
        if modules_to_save is not None:
            if "lora" in modules_to_save:
                lora_state_dict = get_peft_model_state_dict(
                    model_to_save, state_dict=model_to_save_state_dict
                )

                save_file(
                    lora_state_dict,
                    os.path.join(output_dir, "lora.safetensors"),
                )
            if "graph_proj" in modules_to_save:
                graph_proj_state_dict = {}
                for name, param in model_to_save.named_parameters():
                    if "graph_proj" in name:
                        graph_proj_state_dict[name] = param.data
                save_file(
                    graph_proj_state_dict,
                    os.path.join(output_dir, "graph_proj.safetensors"),
                )

    # copied from https://github.com/shyoulala/Kaggle_Eedi_2024_sayoulala/blob/main/recall_code/qwen2_qlora_v1.py#L762
    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    # copied from https://blog.csdn.net/qq_42729417/article/details/142309057
    def _get_train_sampler(self) -> Optional[Sampler]:
        """Return SequentialSampler if distributed training, else RandomSampler.

        Returns:
            Optional[Sampler]: _description_
        """

        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.shuffle:
            # If shuffle is True, we use RandomSampler
            return RandomSampler(self.train_dataset)

        if self.args.local_rank != -1 or self.args.world_size > 1:
            # When using distributed training, we need to use a DistributedSampler
            # to ensure that each process gets a different subset of the data.
            return SequentialSampler(self.train_dataset)
        else:
            # If not using distributed training, we can use a RandomSampler
            return RandomSampler(self.train_dataset)

    # copied from transformers/trainer.py
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            # if is_torch_xla_available():
            #     xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["contrastive_loss"] = round(
                self.last_contrastive_loss,
                4,
            )
            logs["cluster_loss"] = round(
                self.last_cluster_loss,
                4,
            )

            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.detach().item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            logs["learning_rate"] = self._get_learning_rate()

            loss_weight = self.loss_weight
            logs["loss_weight"] = loss_weight.item()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )
