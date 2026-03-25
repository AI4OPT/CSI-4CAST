"""Ablation A4b: replace ShuffleNet embedding with MobileNet-style blocks (TDD)."""

from einops import rearrange
import torch.nn as nn

from src.cp.models.ablation.base import AblationLightningModel, AblationTDDModel
from src.cp.models.common.dataembedding import DataEmbedding


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, squeeze_ratio: int = 4):
        super().__init__()
        squeeze_channels = max(1, in_channels // max(1, squeeze_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)
        self.act2 = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        scale = self.act2(scale)
        return x * scale


class MobileNetV3InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 2.0,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = True,
        use_hs: bool = True,
        se_ratio: int = 4,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        expanded_channels = max(1, round(in_channels * max(1.0, float(expand_ratio))))
        activation = nn.Hardswish if use_hs else nn.ReLU
        padding = (kernel_size - 1) // 2

        layers: list[nn.Module] = []
        if expanded_channels != in_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(expanded_channels),
                    activation(inplace=True),
                ]
            )

        layers.extend(
            [
                nn.Conv2d(
                    expanded_channels,
                    expanded_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=expanded_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded_channels),
                activation(inplace=True),
            ]
        )

        if use_se:
            layers.append(SqueezeExcitation(expanded_channels, squeeze_ratio=se_ratio))

        layers.extend(
            [
                nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.block = nn.Sequential(*layers)
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            out = out + x
        return out


class MobileNetStem(nn.Module):
    def __init__(self, out_channels: int, use_hs: bool = True):
        super().__init__()
        activation = nn.Hardswish if use_hs else nn.ReLU
        self.layers = nn.Sequential(
            nn.Conv2d(2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class MobileNetEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        hist_len: int,
        dim_data: int,
        embed: str,
        freq: str,
        dropout: float,
        num_blocks: int = 4,
        base_channels: int = 32,
        expand_ratio: int = 2,
        kernel_size: int = 3,
        use_se: bool = True,
        use_hs: bool = True,
        se_ratio: int = 4,
    ):
        super().__init__()

        self.delay_branch = self._build_branch(
            num_blocks=num_blocks,
            base_channels=base_channels,
            expand_ratio=expand_ratio,
            kernel_size=kernel_size,
            use_se=use_se,
            use_hs=use_hs,
            se_ratio=se_ratio,
        )
        self.freq_branch = self._build_branch(
            num_blocks=num_blocks,
            base_channels=base_channels,
            expand_ratio=expand_ratio,
            kernel_size=kernel_size,
            use_se=use_se,
            use_hs=use_hs,
            se_ratio=se_ratio,
        )

        self.embedding = DataEmbedding(dim_data, dim_model, embed, freq, dropout)
        self.predict_linear_pre = nn.Linear(hist_len, hist_len)

    def _build_branch(
        self,
        num_blocks: int,
        base_channels: int,
        expand_ratio: int,
        kernel_size: int,
        use_se: bool,
        use_hs: bool,
        se_ratio: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [MobileNetStem(out_channels=base_channels, use_hs=use_hs)]
        for _ in range(num_blocks):
            layers.append(
                MobileNetV3InvertedResidual(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    expand_ratio=expand_ratio,
                    kernel_size=kernel_size,
                    stride=1,
                    use_se=use_se,
                    use_hs=use_hs,
                    se_ratio=se_ratio,
                )
            )

        layers.append(nn.Conv2d(base_channels, 2, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x_delay, x_freq):
        x_delay = rearrange(x_delay, "b l (k o) -> b o l k", o=2)
        x_delay = self.delay_branch(x_delay)

        x_freq = rearrange(x_freq, "b l (k o) -> b o l k", o=2)
        x_freq = self.freq_branch(x_freq)

        x = x_delay + x_freq
        x = rearrange(x, "b o l k -> b l (k o)", o=2)
        x = self.embedding(x)
        x = self.predict_linear_pre(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class Model(AblationTDDModel):
    def _build_embedding(self) -> nn.Module:
        p = self._p
        return MobileNetEmbedding(
            dim_model=self.dim_model,
            hist_len=self.hist_len,
            dim_data=self.dim_data,
            embed=p.get("embedding_embed", "timeF"),
            freq=p.get("embedding_freq", "h"),
            dropout=p.get("embedding_dropout", 0.1),
            num_blocks=p.get("embedding_mobilenet_num_blocks", 4),
            base_channels=p.get("embedding_mobilenet_base_channels", 32),
            expand_ratio=p.get("embedding_mobilenet_expand_ratio", 2),
            kernel_size=p.get("embedding_mobilenet_kernel_size", 3),
            use_se=p.get("embedding_mobilenet_use_se", True),
            use_hs=p.get("embedding_mobilenet_use_hs", True),
            se_ratio=p.get("embedding_mobilenet_se_ratio", 4),
        )


class MOBILENET_REPLACE_EMBED_TDD(AblationLightningModel):
    model_class = Model
    model_display_name = "MOBILENET_REPLACE_EMBED"
