import json
from pathlib import Path
from typing import ClassVar

import torch

from src.noise.noise import (
    gen_burst_noise_nd,
    gen_packagedrop_noise_nd,
    gen_phase_noise_nd,
    gen_vanilla_noise_snr,
)


class Noise:
    list_vanilla_snr: ClassVar[list[int]] = [0, 5, 10, 15, 20, 25]
    list_phase_snr: ClassVar[list[int]] = [10, 15, 20, 25]
    list_burst_snr: ClassVar[list[int]] = [10, 15, 20, 25]
    list_packagedrop_nd: ClassVar[list[float]] = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

    path_decide_nd_json: ClassVar[Path] = Path("src/noise/results/decide_nd.json")
    with open(path_decide_nd_json) as f:
        decide_nd: ClassVar[dict[str, dict[str, dict[str, float]]]] = json.load(f)

    def gen_phase_noise_snr(self, data: torch.Tensor, snr: int) -> torch.Tensor:
        assert snr in self.list_phase_snr, f"snr must be in {self.list_phase_snr}"
        nd = self.decide_nd["phase"][str(snr)]["noise_degree"]
        return gen_phase_noise_nd(data, nd)

    def gen_burst_noise_snr(self, data: torch.Tensor, snr: int) -> torch.Tensor:
        assert snr in self.list_burst_snr, f"snr must be in {self.list_burst_snr}"
        nd = self.decide_nd["burst"][str(snr)]["noise_degree"]
        return gen_burst_noise_nd(data, nd)

    def gen_packagedrop_noise_nd(self, data: torch.Tensor, nd: float) -> torch.Tensor:
        return gen_packagedrop_noise_nd(data, nd)

    def gen_vanilla_noise_snr(self, data: torch.Tensor, snr: int) -> torch.Tensor:
        return gen_vanilla_noise_snr(data, snr)

    vanilla = gen_vanilla_noise_snr
    phase = gen_phase_noise_snr
    burst = gen_burst_noise_snr
    packagedrop = gen_packagedrop_noise_nd


if __name__ == "__main__":
    noise = Noise()

    print("list_phase_snr: ", noise.list_phase_snr)
    print("list_burst_snr: ", noise.list_burst_snr)
    print("list_packagedrop_nd: ", noise.list_packagedrop_nd)
    print("list_vanilla_snr: ", noise.list_vanilla_snr)

    print("decide_nd: ", noise.decide_nd)

    data = torch.complex(torch.randn(4, 32, 16, 300), torch.randn(4, 32, 16, 300))
    print("data: ", data.shape)

    for snr in noise.list_phase_snr:
        noise_data = noise.phase(data, snr)
        print("phase_noise_data: ", noise_data.shape)
    print("--------------------------------")

    for snr in noise.list_burst_snr:
        noise_data = noise.burst(data, snr)
        print("burst_noise_data: ", noise_data.shape)
    print("--------------------------------")

    for nd in noise.list_packagedrop_nd:
        noise_data = noise.packagedrop(data, nd)
        print("packagedrop_noise_data: ", noise_data.shape)
    print("--------------------------------")

    for snr in noise.list_vanilla_snr:
        noise_data = noise.vanilla(data, snr)
        print("vanilla_noise_data: ", noise_data.shape)
