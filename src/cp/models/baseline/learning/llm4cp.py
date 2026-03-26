"""LLM4CP baseline built on top of GPT-2 backbones.

This module adapts pretrained GPT-2 models to CSI forecasting. It
contains the core predictor, which combines frequency- and delay-domain
processing with transformer layers, and the Lightning wrapper used by
the rest of the codebase.
"""

import logging

from einops import rearrange
import torch
import torch.nn as nn

# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import GPT2Model

from src.cp.config.config import ExperimentConfig
from src.cp.models.common.base import BaseCSIModel
from src.cp.models.common.dataembedding import DataEmbedding
from src.cp.models.common.normalizer import batch_denormalize, batch_normalizer
from src.cp.models.common.resblocks import ResBlock
from src.utils.dirs import DIR_HUGGINGFACE_TRANSFORMERS


logger = logging.getLogger(__name__)


class LLM4CP(nn.Module):
    """Predict CSI sequences with a GPT-2 based backbone.

    The model preprocesses CSI inputs in both frequency and delay
    domains, embeds them into the GPT-2 hidden space, and projects the
    transformer output to the target prediction horizon.
    """

    def __init__(
        self,
        gpt_type="gpt2",
        d_ff=768,
        d_model=768,
        gpt_layers=6,
        pred_len=4,
        prev_len=16,
        mlp=0,
        res_layers=4,
        K=300,
        UQh=4,
        UQv=1,
        BQh=2,
        BQv=1,
        patch_size=4,
        stride=1,
        res_dim=64,
        embed="timeF",
        freq="h",
        dropout=0.1,
        **kwargs,
    ):
        """Initialize the LLM4CP predictor.

        Args:
            gpt_type: GPT-2 checkpoint variant to load.
            d_ff: Feed-forward dimension used by the output projection.
            d_model: Embedding dimension used before the transformer.
            gpt_layers: Number of GPT-2 blocks to retain.
            pred_len: Number of future steps to predict.
            prev_len: Number of historical steps provided as input.
            mlp: Whether to unfreeze GPT-2 MLP layers.
            res_layers: Number of residual blocks for CSI preprocessing.
            K: Number of subcarriers.
            UQh: User antenna horizontal dimension.
            UQv: User antenna vertical dimension.
            BQh: Base-station antenna horizontal dimension.
            BQv: Base-station antenna vertical dimension.
            patch_size: Patch size used in the preprocessing layers.
            stride: Reserved stride parameter.
            res_dim: Channel width for residual preprocessing blocks.
            embed: Embedding mode for temporal features.
            freq: Time-feature frequency code.
            dropout: Dropout probability for embeddings.
            **kwargs: Additional keyword arguments.

        """
        super().__init__()
        # self.device = torch.device('cuda:{}'.format(gpu_id))

        self.is_separate_antennas = True
        self.name = "LLM4CP"

        self.mlp = mlp
        self.res_layers = res_layers
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.patch_size = patch_size
        self.stride = stride

        self.K = K
        self.UQh = UQh
        self.UQv = UQv
        self.BQh = BQh
        self.BQv = BQv
        self.Nt = UQh * UQv
        self.Nr = BQh * BQv
        self.mul = prev_len * K * UQh * UQv * BQh * BQv
        self.enc_in = K * UQh * UQv * BQh * BQv
        self.c_out = K * UQh * UQv * BQh * BQv

        if gpt_type == "gpt2-medium":
            self.gpt2 = GPT2Model.from_pretrained(
                "gpt2-medium", output_attentions=True, output_hidden_states=True, cache_dir=DIR_HUGGINGFACE_TRANSFORMERS
            )
            self.gpt2.h = self.gpt2.h[:gpt_layers]  # type: ignore
            self.gpt_dim = 1024
        elif gpt_type == "gpt2-large":
            self.gpt2 = GPT2Model.from_pretrained(
                "gpt2-large", output_attentions=True, output_hidden_states=True, cache_dir=DIR_HUGGINGFACE_TRANSFORMERS
            )
            self.gpt2.h = self.gpt2.h[:gpt_layers]  # type: ignore
            self.gpt_dim = 1280
        elif gpt_type == "gpt2-xl":
            self.gpt2 = GPT2Model.from_pretrained(
                "gpt2-xl", output_attentions=True, output_hidden_states=True, cache_dir=DIR_HUGGINGFACE_TRANSFORMERS
            )
            self.gpt2.h = self.gpt2.h[:gpt_layers]  # type: ignore
            self.gpt_dim = 1600
        else:
            self.gpt2 = GPT2Model.from_pretrained(
                "gpt2", output_attentions=True, output_hidden_states=True, cache_dir=DIR_HUGGINGFACE_TRANSFORMERS
            )
            self.gpt2.h = self.gpt2.h[:gpt_layers]  # type: ignore
            self.gpt_dim = 768

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):  # type: ignore
            if "ln" in name or "wpe" in name or ("mlp" in name and mlp == 1):  # or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        d_ff = self.gpt_dim
        d_model = self.gpt_dim
        self.d_ff = d_ff
        self.d_model = d_model

        self.enc_embedding1 = DataEmbedding(2 * self.enc_in, self.d_model, embed, freq, dropout)

        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        self.patch_layer_fre = nn.Linear(self.patch_size, self.patch_size)
        self.predict_linear_pre = nn.Linear(self.prev_len, self.prev_len)
        self.out_layer_dim = nn.Linear(d_ff, self.c_out * 2)
        self.output_layer_time = nn.Sequential(nn.Linear(self.prev_len, self.pred_len))

        self.RB_e = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        self.RB_f = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        for i in range(self.res_layers):
            self.RB_e.append(ResBlock(res_dim))
            self.RB_f.append(ResBlock(res_dim))
        self.RB_e.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        self.RB_f.append(nn.Conv2d(res_dim, 2, 3, 1, 1))

    def __str__(self):
        """Return the model name used in logs and registries."""
        return self.name

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """Run the LLM4CP forward pass.

        Args:
            x_enc: Input CSI history tensor.
            x_mark_enc: Optional time markers aligned with ``x_enc``.
            x_dec: Unused decoder input placeholder for compatibility.
            x_mark_dec: Unused decoder marker placeholder.
            mask: Optional attention mask placeholder.

        Returns:
            Predicted CSI sequence for the configured forecast horizon.

        """
        x_enc, mean, std = batch_normalizer(x_enc)  # [B, L, D] (512, 16, 96)

        B, L, enc_in = x_enc.shape  # [B, L, D] (512, 16, 96)

        # process in delay domain
        x_enc_r = rearrange(x_enc, "b l (k o) -> b l k o", o=2)  # [B, L, D/2, 2] (512, 16, 48, 2)
        x_enc_complex = torch.complex(x_enc_r[:, :, :, 0], x_enc_r[:, :, :, 1])  # [B, L, D/2] (512, 16, 48)
        x_enc_delay = torch.fft.ifft(x_enc_complex, dim=2)  # [B, L, D/2] (512, 16, 48)
        x_enc_delay = torch.cat([torch.real(x_enc_delay), torch.imag(x_enc_delay)], dim=2)  # [B, L, D] (512, 16, 96)
        x_enc_delay = x_enc_delay.reshape(
            B, L // self.patch_size, self.patch_size, enc_in
        )  # [B, L/4, 4, D] (512, 4, 4, 96)
        x_enc_delay = self.patch_layer(x_enc_delay.permute(0, 1, 3, 2)).permute(
            0, 1, 3, 2
        )  # [B, L/4, 4, D] (512, 4, 4, 96)
        x_enc_delay = x_enc_delay.reshape(B, L, enc_in)  # [B, L, D] (512, 16, 96)
        x_enc_delay = rearrange(x_enc_delay, "b l (k o) -> b o l k", o=2)  # [B, 2, L, D/2] (512, 2, 16, 48)
        x_enc_delay = self.RB_f(x_enc_delay)  # [B, 2, L, D/2] (512, 2, 16, 48)

        # process in frequency domain
        x_enc_fre = x_enc.reshape(B, L // self.patch_size, self.patch_size, enc_in)  # [B, L/4, 4, D] (512, 4, 4, 96)
        x_enc_fre = self.patch_layer(x_enc_fre.permute(0, 1, 3, 2)).permute(
            0, 1, 3, 2
        )  # [B, L/4, 4, D] (512, 4, 4, 96)
        x_enc_fre = x_enc_fre.reshape(B, L, enc_in)  # [B, L, D] (512, 16, 96)
        x_enc_fre = rearrange(x_enc_fre, "b l (k o) -> b o l k", o=2)  # [B, 2, L, D/2] (512, 2, 16, 48)
        x_enc_fre = self.RB_e(x_enc_fre)  # [B, 2, L, D/2] (512, 2, 16, 48)

        x_enc = x_enc_fre + x_enc_delay  # [B, 2, L, D/2] (512, 2, 16, 48)
        x_enc = rearrange(x_enc, "b o l k -> b l (k o)", o=2)  # [B, L, D] (512, 16, 96)

        enc_out = self.enc_embedding1(x_enc, x_mark_enc)  # [B, L, 768]
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, 768]   (512, 16, 768)
        enc_out = torch.nn.functional.pad(
            enc_out, (0, self.gpt_dim - enc_out.shape[-1])
        )  # [B, L, 768]   (512, 16, 768)

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state  # [B, L, 768] (512, 16, 768)  # type: ignore
        # logger.info("dec_out.shape: {}".format(dec_out.shape))
        # import pdb; pdb.set_trace()

        dec_out = self.out_layer_dim(dec_out)  # [B, L, D] (512, 16, 96)
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L', D] (512, 4, 96)

        dec_out = batch_denormalize(dec_out, mean, std)

        return dec_out[:, -self.pred_len :, :]  # [B, L', D] (512, 4, 96)


class LLM4CP_pl(BaseCSIModel):
    """Lightning wrapper for the GPT-based CSI predictor.

    This class attaches the core LLM4CP model to the shared optimizer,
    scheduler, and loss setup used throughout the repository.
    """

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        """Build the Lightning wrapper from an experiment config.

        Args:
            config: Experiment configuration containing model and
                optimization settings.
            *args: Passed to the parent class.
            **kwargs: Passed to the parent class.

        """
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )

        self.name = "LLM4CP"
        self.is_separate_antennas = True
        self.save_hyperparameters({"model": config.model})

        self.model = LLM4CP(**config.model.params)

    def __str__(self):
        """Return the model name used in logs and registries."""
        return self.name

    def forward(self, x):
        """Delegate the forward pass to the wrapped predictor.

        Args:
            x: Input CSI tensor.

        Returns:
            Model prediction for the next CSI steps.

        """
        x = self.model(x)
        return x


if __name__ == "__main__":
    model = LLM4CP(
        gpt_type="gpt2-large",
        gpt_layers=2,
        res_layers=2,
        res_dim=128,
        UQh=1,
        UQv=1,
        BQh=1,
        BQv=1,
    )

    fake_input = torch.randn(4, 16, 600)
    out = model(fake_input)
    print(out.shape)  # Expected output shape: [4, 4, 600]
