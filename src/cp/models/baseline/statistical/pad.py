"""PAD (Prony Analysis Decomposition) baseline for CSI prediction.

Pure-PyTorch translation of the NumPy implementation (pad_old.py).
All nested Python loops for matrix building are replaced with
vectorised einops rearrangements; the algorithm is otherwise identical.
"""

import math

from einops import rearrange
import torch
import torch.nn as nn


_CPU = torch.device("cpu")


def _np_fallback(sample: torch.Tensor, pred_len: int) -> torch.Tensor:
    """Repeat the last observed CSI slot, matching the NP baseline behavior."""
    return sample[:, -1:, :].repeat(1, pred_len, 1)


def _solve_prony_coefficients(calH: torch.Tensor, pL: torch.Tensor, ridge_rel: float) -> torch.Tensor:
    """Solve the Prony least-squares system with adaptive ridge damping.

    ``calH`` is often close to rank deficient under robustness noise. A raw
    pseudo-inverse can amplify small perturbations in ``calH`` / ``pL`` into
    huge coefficients, which then explode during recursive rollout.
    """
    p = calH.shape[1]
    gram = calH.conj().T @ calH
    rhs = calH.conj().T @ pL
    eye = torch.eye(p, dtype=gram.dtype, device=gram.device)

    # Scale the ridge term by the average Gram diagonal so the damping tracks
    # the energy of the current sample instead of relying on a fixed absolute
    # threshold.
    trace_scale = torch.trace(gram).real / max(p, 1)
    ridge = max(float(ridge_rel), torch.finfo(calH.real.dtype).eps) * max(float(trace_scale), 1.0)

    phat = -torch.linalg.solve(gram + ridge * eye, rhs)
    if torch.isfinite(phat).all():
        return phat

    # Fall back to progressively stronger damping if the first solve still
    # produces non-finite values.
    for multiplier in (1e2, 1e4, 1e6):
        phat = -torch.linalg.solve(gram + (ridge * multiplier) * eye, rhs)
        if torch.isfinite(phat).all():
            return phat

    return torch.zeros((p, 1), dtype=calH.dtype, device=calH.device)


def _dft_matrix(
    n: int,
    *,
    dtype: torch.dtype = torch.complex128,
    device: torch.device = _CPU,
) -> torch.Tensor:
    """Build the NxN unitary DFT matrix W where W[i,j] = exp(-2πij/N)/√N."""
    idx = torch.arange(n, device=device, dtype=torch.float64)
    W = torch.exp(-2j * math.pi / n * idx.unsqueeze(1) * idx.unsqueeze(0)) / math.sqrt(n)
    return W.to(dtype)


def _pronyvec(
    y: torch.Tensor,
    p: int,
    pred_len: int,
    startidx: int,
    subcarriernum: int,
    Nt: int,
    Nr: int,
    ridge_rel: float,
) -> torch.Tensor | None:
    """Prony vector prediction (vectorised).

    Equivalent to the original ``pronyvec()`` but replaces the three nested
    Python loops that build ``calH`` / ``pL`` and unpack ``hpredict`` with
    single ``einops.rearrange`` calls.

    Args:
        y: ``[subcarriernum, hist_len, Nr, Nt]`` complex tensor.

    Returns:
        ``[subcarriernum, pred_len, Nt*Nr]`` complex tensor.
    """
    y = y.reshape(subcarriernum, -1, Nr, Nt)
    T = y.shape[1]

    # -- Build calH [D, p] ------------------------------------------------
    # Original indexes y[k, startidx-1-p+lag, nr, nt] for lag in 0..p-1.
    # When startidx-1-p < 0 (e.g. 16-1-16 = -1), NumPy element access wraps
    # via negative indexing but slicing doesn't.  Use explicit modular indices.
    time_idx = torch.arange(startidx - 1 - p, startidx - 1, device=y.device) % T
    y_slice = y[:, time_idx, :, :]  # [K, p, Nr, Nt]
    calH = rearrange(y_slice, "k p nr nt -> (nt k nr) p")  # [D, p]

    # -- Build pL [D, 1] --------------------------------------------------
    y_last = y[:, (startidx - 1) % T, :, :]  # [K, Nr, Nt]
    pL = rearrange(y_last, "k nr nt -> (nt k nr) 1")  # [D, 1]

    # -- Prony coefficients ------------------------------------------------
    phat = _solve_prony_coefficients(calH, pL, ridge_rel)  # [p, 1]
    if not torch.isfinite(phat).all():
        return None

    calH = torch.cat([calH[:, 1:p], pL], dim=1)  # [D, p]
    hpredict = -calH @ phat  # [D, 1]
    if not torch.isfinite(hpredict).all():
        return None

    # -- Unpack predictions ------------------------------------------------
    hp2 = torch.zeros(subcarriernum, pred_len, Nr, Nt, dtype=y.dtype, device=y.device)
    hp2[:, 0, :, :] = rearrange(
        hpredict.squeeze(-1),
        "(nt k nr) -> k nr nt",
        nt=Nt,
        k=subcarriernum,
        nr=Nr,
    )

    for t in range(pred_len - 1):
        calH = torch.cat([calH[:, 1:p], hpredict], dim=1)
        hpredict = -calH @ phat
        if not torch.isfinite(hpredict).all():
            return None
        hp2[:, t + 1, :, :] = rearrange(
            hpredict.squeeze(-1),
            "(nt k nr) -> k nr nt",
            nt=Nt,
            k=subcarriernum,
            nr=Nr,
        )

    return hp2.reshape(subcarriernum, pred_len, Nt * Nr)


def _pad3(
    y: torch.Tensor,
    p: int,
    pred_len: int,
    startidx: int,
    subcarriernum: int,
    Nt: int,
    Nr: int,
    S: torch.Tensor,
    ridge_rel: float,
    fallback_pred_ratio_threshold: float,
) -> torch.Tensor | None:
    """PAD3 prediction for a single sample (vectorised).

    Args:
        y: ``[subcarriernum, hist_len, Nr, Nt]`` complex tensor (single sample).
        S: Pre-computed Kronecker DFT matrix ``[K*Nt, K*Nt]``.

    Returns:
        ``[subcarriernum, pred_len, Nt*Nr]`` complex tensor.
    """
    y = y.reshape(subcarriernum, -1, Nr, Nt)

    hp2 = torch.zeros(subcarriernum, pred_len, Nr, Nt, dtype=y.dtype, device=y.device)

    for nridx in range(Nr):
        history_abs_max = max(float(y[:, :, nridx, :].abs().amax()), 1.0)

        # Build ypad — replaces inner loop over subcarriers
        ypad = rearrange(y[:, :, nridx, :], "k t nt -> (k nt) t")  # [K*Nt, T]

        gu = S.conj().T @ ypad  # [K*Nt, T]
        gu = rearrange(gu, "(k nt) t -> k t 1 nt", k=subcarriernum, nt=Nt)  # [K, T, 1, Nt]

        ghat = _pronyvec(
            gu,
            p,
            pred_len,
            startidx,
            subcarriernum,
            Nt,
            1,
            ridge_rel,
        )  # [K, pred_len, Nt]
        if ghat is None:
            return None
        ghat = rearrange(ghat, "k pl nt -> (k nt) pl")  # [K*Nt, pred_len]

        hhat = S @ ghat  # [K*Nt, pred_len]
        if (not torch.isfinite(hhat).all()) or (
            float(hhat.abs().amax()) > fallback_pred_ratio_threshold * history_abs_max
        ):
            return None
        hp2[:, :, nridx, :] = rearrange(hhat, "(k nt) pl -> k pl nt", k=subcarriernum, nt=Nt)

    return hp2.reshape(subcarriernum, pred_len, Nt * Nr)


class PADMODEL(nn.Module):
    def __init__(
        self,
        p=8,
        pred_len=4,
        startidx=16,
        subcarriernum=300,
        Nt=32,
        Nr=1,
        ridge_rel: float = 1e-4,
        fallback_pred_ratio_threshold: float = 5.0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.is_separate_antennas = False
        self.name = "PAD"

        self.p = p
        self.pred_len = pred_len
        self.startidx = startidx
        self.subcarriernum = subcarriernum
        self.Nt = Nt
        self.Nr = Nr
        self.ridge_rel = float(ridge_rel)
        self.fallback_pred_ratio_threshold = float(fallback_pred_ratio_threshold)

        # Pre-compute the Kronecker DFT matrix (constant for given K, Nt)
        S = torch.kron(
            _dft_matrix(subcarriernum),
            _dft_matrix(Nt),
        )  # [K*Nt, K*Nt]
        self.register_buffer("S", S, persistent=False)

    def __str__(self) -> str:
        return self.name

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            y: ``[batch_size, num_antennas, hist_len, num_subcarriers]`` complex.

        Returns:
            ``[batch_size, num_antennas, pred_len, num_subcarriers]`` complex.
        """
        y_device = y.device
        # Work in complex128 for numerical stability during the damped solve.
        y_work = rearrange(y, "b n l k -> b k l n").to(dtype=torch.complex128)
        S = self.S.to(dtype=torch.complex128, device=y_work.device)

        results = []
        for i in range(y_work.shape[0]):
            pad_out = _pad3(
                y_work[i],
                self.p,
                self.pred_len,
                self.startidx,
                self.subcarriernum,
                self.Nt,
                self.Nr,
                S,
                self.ridge_rel,
                self.fallback_pred_ratio_threshold,
            )
            if pad_out is None:
                results.append(rearrange(_np_fallback(y[i], self.pred_len), "n l k -> k l n").to(torch.complex128))
            else:
                results.append(pad_out)
        out = torch.stack(results, dim=0)  # [B, K, pred_len, Nt*Nr]
        return rearrange(out.to(dtype=torch.complex64, device=y_device), "b k l n -> b n l k")


if __name__ == "__main__":
    input_shape = (1, 32, 16, 300)
    x = torch.randn(input_shape) + 1j * torch.randn(input_shape)

    model = PADMODEL()
    output = model(x)
    print(f"output shape: {output.shape}")
    print(f"output dtype: {output.dtype}")
