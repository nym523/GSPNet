from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class AuxiliaryBalancedLoss(nn.Module):
    """Bernoulli-sampled auxiliary NLL with equal expected class counts."""

    def __init__(
        self,
        class_counts: Sequence[float],
        ignore_label: Optional[int] = -1,
        eps: float = 1e-8,
    ):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32).reshape(-1)
        if counts.numel() == 0:
            raise ValueError("ABSH class_counts must not be empty")
        if torch.any(counts < 0):
            raise ValueError("ABSH class_counts must be non-negative")

        positive = counts > 0
        if positive.any():
            min_count = counts[positive].min()
            retention = torch.zeros_like(counts)
            retention[positive] = min_count / counts[positive]
        else:
            retention = torch.ones_like(counts)

        self.ignore_label = ignore_label
        self.eps = float(eps)
        self.register_buffer("class_counts", counts)
        self.register_buffer("retention", retention.clamp(0.0, 1.0))
        self.last_masks: List[torch.Tensor] = []

    def forward(self, outputs: Sequence[torch.Tensor], labels: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(outputs) != len(labels):
            raise ValueError("ABSH needs one label tensor for every auxiliary output")
        if len(outputs) == 0:
            raise ValueError("ABSH needs at least one auxiliary output")

        total = outputs[0].sum() * 0.0
        masks = []
        for output, target in zip(outputs, labels):
            target = target.reshape(-1).long()
            if output.shape[0] != target.shape[0]:
                raise ValueError("ABSH output and label point counts must match")
            valid = (target >= 0) & (target < self.retention.numel())
            if self.ignore_label is not None:
                valid &= target != int(self.ignore_label)

            probability = output.new_zeros(target.shape[0])
            probability[valid] = self.retention[target[valid]].to(output.dtype)
            mask = valid & (torch.rand_like(probability) < probability)
            masks.append(mask)
            if mask.any():
                total = total + F.nll_loss(output[mask], target[mask], reduction="mean")

        self.last_masks = masks
        return total
