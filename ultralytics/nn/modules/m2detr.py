# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""M2DETR modules: timm backbone wrapper and RT-DETR multitask decoder."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import RTDETRDecoder

__all__ = ("M2RTDETRDecoder", "M2TimmBackbone")


class M2TimmBackbone(nn.Module):
    """Feature-pyramid wrapper for timm backbones.

    The module returns a list of BCHW feature maps, so YAML configs should select
    each level with `Index` before passing them into FPN/PAN/RT-DETR heads.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = False,
        out_indices: Iterable[int] = (2, 3, 4),
        in_chans: int = 3,
        freeze: bool = False,
        model_kwargs: dict | None = None,
        pad_divisor: int = 1,
    ):
        """Create a timm features-only backbone.

        Args:
            model_name (str): timm model name, e.g. ``resnet18`` or ``convnextv2_tiny.fcmae_ft_in22k_in1k``.
            pretrained (bool): Load pretrained weights through timm.
            out_indices (Iterable[int]): Feature levels returned by timm.
            in_chans (int): Number of input image channels.
            freeze (bool): Freeze the backbone parameters.
            model_kwargs (dict, optional): Extra keyword arguments passed to ``timm.create_model``.
            pad_divisor (int): Pad input height/width to this divisor before the backbone.
        """
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
            raise ModuleNotFoundError(
                "M2TimmBackbone requires the 'timm' package. Install it with `pip install timm`."
            ) from exc

        self.model_name = model_name
        self.out_indices = tuple(out_indices)
        self.pad_divisor = int(pad_divisor or 1)
        model_kwargs = model_kwargs or {}
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
            in_chans=in_chans,
            **model_kwargs,
        )
        self.channels = tuple(self.model.feature_info.channels())
        self.reductions = tuple(self.model.feature_info.reduction())

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return selected feature maps in BCHW format."""
        if self.pad_divisor > 1:
            h, w = x.shape[-2:]
            pad_h = (self.pad_divisor - h % self.pad_divisor) % self.pad_divisor
            pad_w = (self.pad_divisor - w % self.pad_divisor) % self.pad_divisor
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
        feats = self.model(x)
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        elif isinstance(feats, dict):
            feats = list(feats.values())

        out = []
        for feat, channels in zip(feats, self.channels):
            if feat.ndim != 4:
                raise ValueError(f"M2TimmBackbone expected 4D feature maps, got shape {tuple(feat.shape)}.")
            if feat.shape[1] != channels and feat.shape[-1] == channels:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out.append(feat)
        return out


class M2RTDETRDecoder(RTDETRDecoder):
    """RT-DETR decoder with an additional image-level classification head."""

    def __init__(
        self,
        nc: int = 80,
        ch: tuple[int, ...] = (512, 1024, 2048),
        cls_nc: int = 2,
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        cls_dropout: float = 0.0,
        cls_loss_gain: float = 1.0,
    ):
        """Initialize multitask RT-DETR.

        Args mirror :class:`RTDETRDecoder`; ``cls_nc`` controls image-level classification classes.
        """
        super().__init__(
            nc=nc,
            ch=ch,
            hd=hd,
            nq=nq,
            ndp=ndp,
            nh=nh,
            ndl=ndl,
            d_ffn=d_ffn,
            dropout=dropout,
            act=act,
            eval_idx=eval_idx,
            nd=nd,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
        )
        self.cls_nc = cls_nc
        self.cls_loss_gain = cls_loss_gain
        cls_in = sum(ch)
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(cls_in),
            nn.Linear(cls_in, hd),
            nn.SiLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(hd, cls_nc),
        )

    def forward_cls(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Compute image-level logits from pooled multiscale features."""
        pooled = [self.cls_pool(feat).flatten(1) for feat in x]
        return self.cls_head(torch.cat(pooled, dim=1))

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Return RT-DETR detection predictions plus image classification logits."""
        cls_logits = self.forward_cls(x)
        det = super().forward(x, batch=batch)

        if self.training:
            return (*det, cls_logits)
        if self.export:
            return det
        y, raw = det
        return y, {"det": raw, "cls_logits": cls_logits}
