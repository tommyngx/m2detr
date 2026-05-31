# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Losses for M2DETR multitask training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.utils.loss import RTDETRDetectionLoss


class M2DETRLoss(nn.Module):
    """RT-DETR detection loss with an image-level classification term."""

    def __init__(
        self,
        nc: int = 80,
        cls_nc: int = 2,
        cls_loss_gain: float = 1.0,
        cls_label_smoothing: float = 0.0,
        use_vfl: bool = True,
    ):
        """Initialize the multitask loss."""
        super().__init__()
        self.det_loss = RTDETRDetectionLoss(nc=nc, use_vfl=use_vfl)
        self.cls_nc = cls_nc
        self.cls_loss_gain = cls_loss_gain
        self.cls_label_smoothing = cls_label_smoothing

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        cls_logits: torch.Tensor | None = None,
        image_cls: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute detection and optional image classification losses."""
        losses = self.det_loss(preds, batch, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta)
        device = preds[0].device

        if cls_logits is None or image_cls is None:
            losses["loss_image_class"] = torch.tensor(0.0, device=device)
            return losses

        image_cls = image_cls.to(device=device, dtype=torch.long).view(-1)
        valid = (image_cls >= 0) & (image_cls < self.cls_nc)
        if valid.any():
            loss_cls = F.cross_entropy(
                cls_logits[valid],
                image_cls[valid],
                label_smoothing=self.cls_label_smoothing,
            )
            losses["loss_image_class"] = loss_cls * self.cls_loss_gain
        else:
            losses["loss_image_class"] = cls_logits.sum() * 0.0
        return losses
