from typing import Optional
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from .utils.logger import get_logger

logger = get_logger(__name__)

class ContrastiveLossCalculator(nn.Module):
    def __init__(self, temperature: float=0.05):
        super().__init__()
        self.temperature = temperature
        self.rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self.world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 获取所有隐藏状态和标签
        if torch.distributed.is_initialized():
            all_hidden = self._dist_gather_tensor(hidden_states)
            all_labels = self._dist_gather_tensor(labels)
        else:
            all_hidden = hidden_states.clone()
            all_labels = labels.clone()

        # 计算相似度矩阵
        logits = torch.mm(all_hidden, all_hidden.T) / self.temperature

        # 构建标签掩码
        label_mask = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()
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

        return contrastive_loss
    
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
