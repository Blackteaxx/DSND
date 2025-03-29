import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.logger import get_logger

logger = get_logger(__name__)


class ContrastiveLossCalculator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        all_hidden = hidden_states
        all_labels = labels

        # 计算相似度矩阵
        logits = torch.mm(all_hidden, all_hidden.T) / temperature

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
        
        if torch.isnan(contrastive_loss):
            logger.warning("contrastive_loss is nan")
            logger.warning(f"Possibly caused by num_pos_samples: {num_pos_samples}")
            logger.warning(f"Labels: {all_labels}")
            contrastive_loss = torch.tensor(0.0).to(per_sample_loss.device)

        return contrastive_loss
