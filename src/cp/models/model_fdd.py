import torch
import torch.nn as nn
from einops import rearrange

from src.cp.config.config import ExperimentConfig
from src.cp.models.common.activation import get_activation
from src.cp.models.common.base import BaseCSIModel
from src.cp.models.common.dataembedding import DataEmbedding
from src.cp.models.common.mlp import MLP
from src.cp.models.common.normalizer import batch_denormalize, batch_normalizer
from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, PRED_LEN
from src.utils.real_n_complex import complex_to_real_flat, real_flat_to_complex


"""
▗▄▄▄ ▗▄▄▄▖▗▖  ▗▖ ▗▄▖ ▗▄▄▄▖ ▗▄▄▖▗▄▄▄▖▗▄▄▖ 
▐▌  █▐▌   ▐▛▚▖▐▌▐▌ ▐▌  █  ▐▌   ▐▌   ▐▌ ▐▌
▐▌  █▐▛▀▀▘▐▌ ▝▜▌▐▌ ▐▌  █   ▝▀▚▖▐▛▀▀▘▐▛▀▚▖
▐▙▄▄▀▐▙▄▄▖▐▌  ▐▌▝▚▄▞▘▗▄█▄▖▗▄▄▞▘▐▙▄▄▖▐▌ ▐▌
"""


class Denoiser(nn.Module):
    def __init__(
        self,
        num_filters_2d: int = 3,
        filter_size_2d: int = 3,
        filter_size_1d: int = 3,
        activation: str = "tanh",  # relu, tanh, sigmoid, gelu
        is_post_processor: bool = True,
        is_residual: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.num_filters_2d = num_filters_2d
        self.filter_size_2d = filter_size_2d
        self.filter_size_1d = filter_size_1d
        self.activation = activation
        self.is_post_processor = is_post_processor
        self.is_residual = is_residual

        list_n_filters_2d = [2 ** (i + 1) for i in range(num_filters_2d)]
        list_filter_sizes_2d = [filter_size_2d for _ in range(num_filters_2d)]

        self.encoder = nn.ModuleList()
        for i in range(len(list_n_filters_2d) - 1):
            conv = nn.Conv2d(
                in_channels=list_n_filters_2d[i],
                out_channels=list_n_filters_2d[i + 1],
                kernel_size=list_filter_sizes_2d[i],
                stride=1,
                padding=(list_filter_sizes_2d[i] - 1) // 2,
            )
            nn.init.uniform_(conv.weight, -1.0 / (list_n_filters_2d[i] ** 0.5), 1.0 / (list_n_filters_2d[i] ** 0.5))
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

            bn = nn.BatchNorm2d(num_features=list_n_filters_2d[i + 1])
            self.encoder.append(nn.Sequential(conv, bn))

        self.decoder = nn.ModuleList()
        for i in range(len(list_n_filters_2d) - 1, 0, -1):
            conv = nn.Conv2d(
                in_channels=list_n_filters_2d[i],
                out_channels=list_n_filters_2d[i - 1],
                kernel_size=list_filter_sizes_2d[i - 1],
                stride=1,
                padding=(list_filter_sizes_2d[i - 1] - 1) // 2,
            )
            nn.init.uniform_(conv.weight, -1.0 / (list_n_filters_2d[i] ** 0.5), 1.0 / (list_n_filters_2d[i] ** 0.5))
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
            bn = nn.BatchNorm2d(num_features=list_n_filters_2d[i - 1])
            self.decoder.append(nn.Sequential(conv, bn))

        if is_post_processor:
            self.post_processor = nn.Conv1d(
                in_channels=HIST_LEN,
                out_channels=HIST_LEN,
                kernel_size=filter_size_1d,
                stride=1,
                padding=(filter_size_1d - 1) // 2,
            )
            self.post_bn = nn.BatchNorm1d(num_features=HIST_LEN)

        self.activation_fn = get_activation(activation)

    def forward(self, x):
        x_noisy = x
        x = rearrange(x, "b l (s i) -> b i l s", i=2)  # [512, 2, 16, 48]
        # Encoder
        for layer in self.encoder:
            x = self.activation_fn(layer(x))
        # Decoder
        for layer in self.decoder:
            x = self.activation_fn(layer(x))
        # Postprocessor
        x = rearrange(x, "b i l s -> b l (s i)", i=2)  # [512, 16, 96]

        if self.is_post_processor:
            x = self.post_processor(x)  # [512, 16, 96]
            x = self.post_bn(x)  # [512, 16, 96]
        else:
            x = x

        if self.is_residual:
            x_clean = x_noisy - x
        else:
            x_clean = x

        return x_clean


"""
 ▗▄▖ ▗▄▄▖ ▗▖   ▗▄▄▖ ▗▄▄▖  ▗▄▖  ▗▄▄▖▗▄▄▄▖ ▗▄▄▖ ▗▄▄▖ ▗▄▖ ▗▄▄▖ 
▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌   ▐▌   ▐▌   ▐▌ ▐▌▐▌ ▐▌
▐▛▀▜▌▐▛▀▚▖▐▌   ▐▛▀▘ ▐▛▀▚▖▐▌ ▐▌▐▌   ▐▛▀▀▘ ▝▀▚▖ ▝▀▚▖▐▌ ▐▌▐▛▀▚▖
▐▌ ▐▌▐▌ ▐▌▐▙▄▄▖▐▌   ▐▌ ▐▌▝▚▄▞▘▝▚▄▄▖▐▙▄▄▖▗▄▄▞▘▗▄▄▞▘▝▚▄▞▘▐▌ ▐▌                                                    
"""


class AdaptiveReweightingLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
        is_arl: bool = False,
        output_activation_name: str = "relu",
        arl_operation: str = "multiply",  # Options: "multiply" or "add"
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_arl = is_arl
        self.output_activation_name = output_activation_name
        self.output_activation = get_activation(output_activation_name)

        if is_arl and arl_operation not in ["multiply", "add"]:
            raise ValueError(f"Unsupported ARL operation: {arl_operation}. Must be 'multiply' or 'add'")
        self.arl_operation = arl_operation

        # MLP that either:
        # - generates weights for ARL mode (when is_arl=True)
        # - transforms the input directly (when is_arl=False)
        self.mlp = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            output_activation=self.output_activation,
        )

        self.layernorm = nn.LayerNorm(out_dim, eps=1e-5)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.mlp(x)
        if self.is_arl:
            if self.arl_operation == "multiply":
                # ARL mode: multiply input by learned weights
                # This allows the network to learn importance weights for different features
                out = out * x
            else:  # self.arl_operation == "add"
                # ARL mode: add learned correction to input
                # This allows the network to learn residual corrections
                out = out + x
        else:
            # Normal mode: just use the MLP output
            pass

        out = self.layernorm(out)
        out = self.dropout(out)
        return out


class AdaptiveReweightingLayerProcessor(nn.Module):
    def __init__(
        self,
        # data
        hist_len: int = HIST_LEN,
        dim_data: int = NUM_SUBCARRIERS * 2,  # real and imag
        # is_U2D
        is_U2D: bool = False,  # TDD do not need arl for subcarrier
        # temporal projection
        temporal_proj_num_layers: int = 2,
        temporal_proj_hidden_dim: int = 256,
        temporal_proj_is_arl: bool = False,
        temporal_proj_output_activation_name: str = "none",
        temporal_proj_arl_operation: str = "add",
        # subcarrier projection
        subcarrier_proj_num_layers: int = 2,
        subcarrier_proj_hidden_dim: int = 256,
        subcarrier_proj_is_arl: bool = True,
        subcarrier_proj_output_activation_name: str = "sigmoid",
        subcarrier_proj_arl_operation: str = "add",
    ):
        super().__init__()

        # data
        self.hist_len = hist_len
        self.dim_data = dim_data
        # is_U2D
        self.is_U2D = is_U2D
        # temporal projection
        self.temporal_proj_num_layers = temporal_proj_num_layers
        self.temporal_proj_hidden_dim = temporal_proj_hidden_dim
        self.temporal_proj_is_arl = temporal_proj_is_arl
        self.temporal_proj_output_activation_name = temporal_proj_output_activation_name
        self.temporal_proj_arl_operation = temporal_proj_arl_operation
        # subcarrier projection
        self.subcarrier_proj_num_layers = subcarrier_proj_num_layers
        self.subcarrier_proj_hidden_dim = subcarrier_proj_hidden_dim
        self.subcarrier_proj_is_arl = subcarrier_proj_is_arl
        self.subcarrier_proj_output_activation_name = subcarrier_proj_output_activation_name
        self.subcarrier_proj_arl_operation = subcarrier_proj_arl_operation

        # temporal projection - freq
        self.temporal_proj_freq = AdaptiveReweightingLayer(
            in_dim=self.hist_len,
            out_dim=self.hist_len,
            num_layers=self.temporal_proj_num_layers,
            hidden_dim=self.temporal_proj_hidden_dim,
            is_arl=self.temporal_proj_is_arl,
            output_activation_name=self.temporal_proj_output_activation_name,
            arl_operation=self.temporal_proj_arl_operation,
        )

        # temporal projection - delay
        self.temporal_proj_delay = AdaptiveReweightingLayer(
            in_dim=self.hist_len,
            out_dim=self.hist_len,
            num_layers=self.temporal_proj_num_layers,
            hidden_dim=self.temporal_proj_hidden_dim,
            is_arl=self.temporal_proj_is_arl,
            output_activation_name=self.temporal_proj_output_activation_name,
            arl_operation=self.temporal_proj_arl_operation,
        )

        # subcarrier projection - freq
        if self.is_U2D:
            self.subcarrier_proj_freq = AdaptiveReweightingLayer(
                in_dim=self.dim_data,
                out_dim=self.dim_data,
                num_layers=self.subcarrier_proj_num_layers,
                hidden_dim=self.subcarrier_proj_hidden_dim,
                is_arl=self.subcarrier_proj_is_arl,
                output_activation_name=self.subcarrier_proj_output_activation_name,
                arl_operation=self.subcarrier_proj_arl_operation,
            )

        # subcarrier projection - delay
        if self.is_U2D:
            self.subcarrier_proj_delay = AdaptiveReweightingLayer(
                in_dim=self.dim_data,
                out_dim=self.dim_data,
                num_layers=self.subcarrier_proj_num_layers,
                hidden_dim=self.subcarrier_proj_hidden_dim,
                is_arl=self.subcarrier_proj_is_arl,
                output_activation_name=self.subcarrier_proj_output_activation_name,
            )

    def forward(self, x):
        # x: [B, L, D]
        # D = 2 * K = 2 * NUM_SUBCARRIERS

        x_complex = real_flat_to_complex(x)  # [B, L, K]
        x_delay = torch.fft.ifft(x_complex, dim=2)  # [B, L, K]
        x_delay = complex_to_real_flat(x_delay)  # [B, L, 2 * K]

        x_freq = x.permute(0, 2, 1)  # [B, D, L]
        x_freq = self.temporal_proj_freq(x_freq)
        x_freq = x_freq.permute(0, 2, 1)  # [B, L, D]
        if self.is_U2D:
            x_freq = self.subcarrier_proj_freq(x_freq)  # [B, L, D]

        x_delay = x_delay.permute(0, 2, 1)  # [B, D, L]
        x_delay = self.temporal_proj_delay(x_delay)  # [B, D, L]
        x_delay = x_delay.permute(0, 2, 1)  # [B, L, D]
        if self.is_U2D:
            x_delay = self.subcarrier_proj_delay(x_delay)  # [B, L, D]

        return x_delay, x_freq


"""
 ▗▄▄▖▗▖ ▗▖▗▖ ▗▖▗▄▄▄▖▗▄▄▄▖▗▖   ▗▄▄▄▖▗▄▄▖ ▗▖    ▗▄▖  ▗▄▄▖▗▖ ▗▖ ▗▄▄▖
▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌   ▐▌   ▐▌   ▐▌ ▐▌▐▌   ▐▌ ▐▌▐▌   ▐▌▗▞▘▐▌   
 ▝▀▚▖▐▛▀▜▌▐▌ ▐▌▐▛▀▀▘▐▛▀▀▘▐▌   ▐▛▀▀▘▐▛▀▚▖▐▌   ▐▌ ▐▌▐▌   ▐▛▚▖  ▝▀▚▖
▗▄▄▞▘▐▌ ▐▌▝▚▄▞▘▐▌   ▐▌   ▐▙▄▄▖▐▙▄▄▖▐▙▄▞▘▐▙▄▄▖▝▚▄▞▘▝▚▄▄▖▐▌ ▐▌▗▄▄▞▘
"""


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x has shape B, C, H, W
        # self.avg_pool(x) has shape B, C, 1, 1
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ShuffleBlockCA(nn.Module):
    def __init__(
        self,
        in_channels=64,
        groups=4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.groups = groups
        assert in_channels % groups == 0, "in_channels must be divisible by groups"

        # grouped pointwise conv for input
        self.g_pw_conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            groups=groups,
            bias=False,
        )
        self.g_pw_conv_in_bn = nn.BatchNorm2d(num_features=in_channels)
        self.g_pw_conv_in_relu = nn.ReLU(inplace=True)

        # depthwise conv
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.dw_conv_bn = nn.BatchNorm2d(num_features=in_channels)

        # grouped pointwise conv for output
        self.g_pw_conv_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            groups=groups,
            bias=False,
        )
        self.g_pw_conv_out_bn = nn.BatchNorm2d(num_features=in_channels)

        # channel attention
        self.ca = ChannelAttention(in_planes=in_channels, ratio=1)
        self.ca_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs = self.g_pw_conv_in(x)
        rs = self.g_pw_conv_in_bn(rs)
        rs = self.g_pw_conv_in_relu(rs)

        rs = self.channel_shuffle(rs)

        rs = self.dw_conv(rs)
        rs = self.dw_conv_bn(rs)

        rs = self.g_pw_conv_out(rs)
        rs = self.g_pw_conv_out_bn(rs)

        channel_attn = self.ca(rs)
        rs = channel_attn * rs
        out = torch.add(x, rs)

        out = self.ca_relu(out)
        return out

    def channel_shuffle(self, x):
        B, C, H, W = x.size()
        assert C % self.groups == 0, "C must be divisible by groups"
        group_channels = C // self.groups
        x = x.view(B, self.groups, group_channels, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, H, W)
        return x


class CSIEmbeddingShuffleNet(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_res_layers: int = 4,
        res_dim: int = 64,
        res_groups: int = 4,
        hist_len: int = HIST_LEN,
        dim_data: int = NUM_SUBCARRIERS * 2,
        embed: str = "timeF",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_res_layers = num_res_layers
        self.res_dim = res_dim
        self.res_groups = res_groups
        self.hist_len = hist_len
        self.dim_data = dim_data
        self.dim_model = dim_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout

        list_RB_delay = []
        list_RB_freq = []
        list_RB_delay.append(nn.Conv2d(2, res_dim, 3, 1, 1))
        list_RB_freq.append(nn.Conv2d(2, res_dim, 3, 1, 1))
        for _ in range(num_res_layers):
            list_RB_delay.append(ShuffleBlockCA(in_channels=res_dim, groups=res_groups))
            list_RB_freq.append(ShuffleBlockCA(in_channels=res_dim, groups=res_groups))
        list_RB_delay.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        list_RB_freq.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        self.RB_delay = nn.Sequential(*list_RB_delay)
        self.RB_freq = nn.Sequential(*list_RB_freq)

        self.embedding = DataEmbedding(dim_data, dim_model, embed, freq, dropout)

        self.predict_linear_pre = nn.Linear(hist_len, hist_len)

    def forward(self, x_delay, x_freq):
        # process in delay domain
        x_delay = rearrange(x_delay, "b l (k o) -> b o l k", o=2)  # [B, 2, L, D/2] (512, 2, 16, 48)
        x_delay = self.RB_delay(x_delay)  # [B, 2, L, D/2] (512, 2, 16, 48)

        # process in frequency domain
        x_freq = rearrange(x_freq, "b l (k o) -> b o l k", o=2)  # [B, 2, L, D/2] (512, 2, 16, 48)
        x_freq = self.RB_freq(x_freq)  # [B, 2, L, D/2] (512, 2, 16, 48)

        x = x_freq + x_delay  # [B, 2, L, D/2] (512, 2, 16, 48)
        x = rearrange(x, "b o l k -> b l (k o)", o=2)  # [B, L, D] (512, 16, 96)
        x = self.embedding(x)  # [B, L, 768]
        x = self.predict_linear_pre(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, 768]   (512, 16, 768)

        return x


"""
▗▄▄▄▖▗▄▄▖  ▗▄▖ ▗▖  ▗▖ ▗▄▄▖▗▄▄▄▖ ▗▄▖ ▗▄▄▖ ▗▖  ▗▖▗▄▄▄▖▗▄▄▖ 
  █  ▐▌ ▐▌▐▌ ▐▌▐▛▚▖▐▌▐▌   ▐▌   ▐▌ ▐▌▐▌ ▐▌▐▛▚▞▜▌▐▌   ▐▌ ▐▌
  █  ▐▛▀▚▖▐▛▀▜▌▐▌ ▝▜▌ ▝▀▚▖▐▛▀▀▘▐▌ ▐▌▐▛▀▚▖▐▌  ▐▌▐▛▀▀▘▐▛▀▚▖
  █  ▐▌ ▐▌▐▌ ▐▌▐▌  ▐▌▗▄▄▞▘▐▌   ▝▚▄▞▘▐▌ ▐▌▐▌  ▐▌▐▙▄▄▖▐▌ ▐▌
"""


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        hidden_dim: int = 1024,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.dim_model = dim_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_prob,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x):
        x = self.transformer(x)
        return x


"""
▗▖  ▗▖▗▖   ▗▄▄▖ ▗▄▄▖ ▗▄▄▖  ▗▄▖  ▗▄▄▖▗▄▄▄▖ ▗▄▄▖ ▗▄▄▖ ▗▄▖ ▗▄▄▖ 
▐▛▚▞▜▌▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌   ▐▌   ▐▌   ▐▌ ▐▌▐▌ ▐▌
▐▌  ▐▌▐▌   ▐▛▀▘ ▐▛▀▘ ▐▛▀▚▖▐▌ ▐▌▐▌   ▐▛▀▀▘ ▝▀▚▖ ▝▀▚▖▐▌ ▐▌▐▛▀▚▖
▐▌  ▐▌▐▙▄▄▖▐▌   ▐▌   ▐▌ ▐▌▝▚▄▞▘▝▚▄▄▖▐▙▄▄▖▗▄▄▞▘▗▄▄▞▘▝▚▄▞▘▐▌ ▐▌
"""


class MLPProcessor(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_data: int,
        hist_len: int,
        pred_len: int,
    ):
        super().__init__()

        self.dim_model = dim_model
        self.dim_data = dim_data

        self.pred_to_data = nn.Linear(dim_model, dim_data)
        self.hist_to_pred = nn.Linear(hist_len, pred_len)

    def forward(self, x):
        x = self.pred_to_data(x)  # [batch_size, hist_len, dim_model] -> [batch_size, hist_len, dim_data]
        x = self.hist_to_pred(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # [batch_size, hist_len, dim_data] -> [batch_size, pred_len, dim_data]
        return x


"""
▗▖  ▗▖ ▗▄▖ ▗▄▄▄ ▗▄▄▄▖▗▖   
▐▛▚▞▜▌▐▌ ▐▌▐▌  █▐▌   ▐▌   
▐▌  ▐▌▐▌ ▐▌▐▌  █▐▛▀▀▘▐▌   
▐▌  ▐▌▝▚▄▞▘▐▙▄▄▀▐▙▄▄▖▐▙▄▄▖
"""


class Model(nn.Module):
    def __init__(
        self,
        # data
        hist_len: int = HIST_LEN,
        pred_len: int = PRED_LEN,
        dim_data: int = NUM_SUBCARRIERS * 2,
        dim_model: int = 768,
        # Denoiser
        denoiser_num_filters_2d: int = 3,
        denoiser_filter_size_2d: int = 3,
        denoiser_filter_size_1d: int = 3,
        denoiser_activation: str = "tanh",
        denoiser_is_post_processor: bool = True,
        denoiser_is_residual: bool = True,
        # ARL
        arl_is_U2D: bool = False,  # TDD as default
        arl_temporal_proj_num_layers: int = 2,
        arl_temporal_proj_hidden_dim: int = 256,
        arl_temporal_proj_is_arl: bool = False,
        arl_temporal_proj_output_activation_name: str = "none",
        arl_temporal_proj_arl_operation: str = "add",
        arl_subcarrier_proj_num_layers: int = 2,
        arl_subcarrier_proj_hidden_dim: int = 256,
        arl_subcarrier_proj_is_arl: bool = True,
        arl_subcarrier_proj_output_activation_name: str = "sigmoid",
        arl_subcarrier_proj_arl_operation: str = "add",
        # Shuffle Embedding
        embedding_num_res_layers: int = 4,
        embedding_res_dim: int = 64,
        embedding_res_groups: int = 4,
        embedding_embed: str = "timeF",
        embedding_freq: str = "h",
        embedding_dropout: float = 0.1,
        # TransformerPredictor
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 4,
        transformer_hidden_dim: int = 1024,
        transformer_dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.hist_len = hist_len
        self.pred_len = pred_len
        self.dim_data = dim_data
        self.dim_model = dim_model

        if denoiser_num_filters_2d > 1:
            self.denoiser = Denoiser(
                num_filters_2d=denoiser_num_filters_2d,
                filter_size_2d=denoiser_filter_size_2d,
                filter_size_1d=denoiser_filter_size_1d,
                activation=denoiser_activation,
                is_post_processor=denoiser_is_post_processor,
                is_residual=denoiser_is_residual,
            )
        else:
            self.denoiser = nn.Identity()

        self.arl = AdaptiveReweightingLayerProcessor(
            hist_len=hist_len,
            dim_data=dim_data,
            is_U2D=arl_is_U2D,
            # temporal projection
            temporal_proj_num_layers=arl_temporal_proj_num_layers,
            temporal_proj_hidden_dim=arl_temporal_proj_hidden_dim,
            temporal_proj_is_arl=arl_temporal_proj_is_arl,
            temporal_proj_output_activation_name=arl_temporal_proj_output_activation_name,
            temporal_proj_arl_operation=arl_temporal_proj_arl_operation,
            # subcarrier projection
            subcarrier_proj_num_layers=arl_subcarrier_proj_num_layers,
            subcarrier_proj_hidden_dim=arl_subcarrier_proj_hidden_dim,
            subcarrier_proj_is_arl=arl_subcarrier_proj_is_arl,
            subcarrier_proj_output_activation_name=arl_subcarrier_proj_output_activation_name,
            subcarrier_proj_arl_operation=arl_subcarrier_proj_arl_operation,
        )

        self.embedding = CSIEmbeddingShuffleNet(
            dim_model=dim_model,
            num_res_layers=embedding_num_res_layers,
            res_dim=embedding_res_dim,
            res_groups=embedding_res_groups,
            hist_len=hist_len,
            dim_data=dim_data,
            embed=embedding_embed,
            freq=embedding_freq,
            dropout=embedding_dropout,
        )

        self.transformer = TransformerPredictor(
            dim_model=dim_model,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            hidden_dim=transformer_hidden_dim,
            dropout_prob=transformer_dropout_prob,
        )

        self.mlp = MLPProcessor(
            dim_model=dim_model,
            dim_data=dim_data,
            hist_len=hist_len,
            pred_len=pred_len,
        )

    def forward(self, x):
        x = self.denoiser(x)
        x, mean, std = batch_normalizer(x)
        x_delay, x_freq = self.arl(x)
        x = self.embedding(x_delay, x_freq)
        x = self.transformer(x)
        x = self.mlp(x)
        x = batch_denormalize(x, mean, std)
        return x


class MODEL_fdd_pl(BaseCSIModel):
    """Complete model for CSI prediction."""

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )

        self.name = "MODEL"
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})

        self.model = Model(**config.model.params)

    def __str__(self):
        return self.name

    def forward(self, x):
        return self.model(x)
