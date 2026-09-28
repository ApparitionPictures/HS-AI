"""Actor-critic policy: Nature-CNN on stacked frames + MLP on the state
vector, fused into a GRU (hand-written cell so ONNX/TensorRT export keeps the
parameter names for weight refit), with a hybrid action head:

* mouse: 2-d Gaussian, mean = tanh(linear), state-independent log-std
* keys : one categorical head per discrete action group
* value: scalar

``act_step`` is the single-step inference path (batch 1, used by the actor and
exported to ONNX).  ``forward_seq`` runs truncated-BPTT sequences for PPO.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicySpec:
    img_channels: int
    img_size: int
    state_dim: int
    discrete_sizes: List[int]
    n_continuous: int = 2
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    conv_out: int = 256
    state_hidden: int = 128
    gru_hidden: int = 512
    logstd_init: float = -0.5
    logstd_min: float = -3.0
    logstd_max: float = 0.5

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]

    @staticmethod
    def from_dict(d: Dict) -> "PolicySpec":
        return PolicySpec(**d)


def _init_linear(m: nn.Linear, gain: float = math.sqrt(2)) -> None:
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, size: int, chans: List[int], out: int):
        super().__init__()
        c1, c2, c3 = chans
        self.conv1 = nn.Conv2d(in_ch, c1, 8, stride=4)
        self.conv2 = nn.Conv2d(c1, c2, 4, stride=2)
        self.conv3 = nn.Conv2d(c2, c3, 3, stride=1)
        with torch.no_grad():
            n = self._conv(torch.zeros(1, in_ch, size, size)).numel()
        self.fc = nn.Linear(n, out)
        for m in (self.conv1, self.conv2, self.conv3):
            nn.init.orthogonal_(m.weight, math.sqrt(2))
            nn.init.zeros_(m.bias)
        _init_linear(self.fc)

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(self._conv(x)))


class GRUCellManual(nn.Module):
    """Same parameterisation as nn.GRUCell, written with plain ops."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.ih = nn.Linear(in_dim, 3 * hidden)
        self.hh = nn.Linear(hidden, 3 * hidden)
        for lin in (self.ih, self.hh):
            nn.init.orthogonal_(lin.weight, 1.0)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gi = self.ih(x)
        gh = self.hh(h)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h


class Policy(nn.Module):
    def __init__(self, spec: PolicySpec):
        super().__init__()
        self.spec = spec
        self.encoder = ConvEncoder(spec.img_channels, spec.img_size, spec.conv_channels, spec.conv_out)
        self.state_fc1 = nn.Linear(spec.state_dim, spec.state_hidden)
        self.state_fc2 = nn.Linear(spec.state_hidden, spec.state_hidden)
        self.fuse = nn.Linear(spec.conv_out + spec.state_hidden, spec.gru_hidden)
        self.gru = GRUCellManual(spec.gru_hidden, spec.gru_hidden)
        self.mouse_mean = nn.Linear(spec.gru_hidden, spec.n_continuous)
        self.logstd = nn.Parameter(torch.full((spec.n_continuous,), float(spec.logstd_init)))
        self.n_disc = int(sum(spec.discrete_sizes))
        self.disc = nn.Linear(spec.gru_hidden, max(1, self.n_disc))
        self.value = nn.Linear(spec.gru_hidden, 1)
        for m in (self.state_fc1, self.state_fc2, self.fuse):
            _init_linear(m)
        _init_linear(self.mouse_mean, 0.01)
        _init_linear(self.disc, 0.01)
        _init_linear(self.value, 1.0)

    # ---- pieces -----------------------------------------------------------
    def features(self, img: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        z_img = self.encoder(img)
        z_s = F.relu(self.state_fc2(F.relu(self.state_fc1(state))))
        return F.relu(self.fuse(torch.cat([z_img, z_s], dim=-1)))

    def heads(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.mouse_mean(h))
        logstd = self.logstd.clamp(self.spec.logstd_min, self.spec.logstd_max).expand_as(mean)
        logits = self.disc(h)
        v = self.value(h)
        return mean, logstd, logits, v

    def initial_state(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch, self.spec.gru_hidden, device=device, dtype=dtype)

    # ---- single step (inference / export) ---------------------------------
    def act_step(self, img: torch.Tensor, state: torch.Tensor, h: torch.Tensor):
        """img: [B,C,H,W] float in [0,1]; state: [B,D]; h: [B,G] -> (mean, logstd, logits, value, h_new)"""
        x = self.features(img, state)
        h_new = self.gru(x, h)
        mean, logstd, logits, v = self.heads(h_new)
        return mean, logstd, logits, v, h_new

    # ---- sequence (training) ----------------------------------------------
    def forward_seq(self, img: torch.Tensor, state: torch.Tensor, h0: torch.Tensor, first: torch.Tensor):
        """img: [T,B,C,H,W]; state: [T,B,D]; h0: [B,G]; first: [T,B] (1 where a new episode starts at t).
        Returns mean [T,B,2], logstd [T,B,2], logits [T,B,ND], value [T,B]."""
        T, B = img.shape[:2]
        x = self.features(img.flatten(0, 1), state.flatten(0, 1)).view(T, B, -1)
        h = h0
        hs = []
        for t in range(T):
            h = h * (1.0 - first[t]).unsqueeze(-1).to(h.dtype)
            h = self.gru(x[t], h)
            hs.append(h)
        H = torch.stack(hs, 0)
        mean, logstd, logits, v = self.heads(H)
        return mean, logstd, logits, v.squeeze(-1)

    def split_logits(self, logits: torch.Tensor) -> List[torch.Tensor]:
        return list(torch.split(logits, self.spec.discrete_sizes, dim=-1))

    # ---- distribution math ------------------------------------------------
    def log_prob(self, mean, logstd, logits, mouse_a, disc_a) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (total log-prob, entropy) with shapes [...] matching the batch dims."""
        var = torch.exp(2.0 * logstd)
        lp_c = (-0.5 * ((mouse_a - mean) ** 2) / var - logstd - 0.5 * math.log(2 * math.pi)).sum(-1)
        ent_c = (0.5 + 0.5 * math.log(2 * math.pi) + logstd).sum(-1)
        lp_d = torch.zeros_like(lp_c)
        ent_d = torch.zeros_like(lp_c)
        for i, lg in enumerate(self.split_logits(logits)):
            logp = F.log_softmax(lg.float(), dim=-1)
            lp_d = lp_d + logp.gather(-1, disc_a[..., i:i + 1].long()).squeeze(-1)
            ent_d = ent_d - (logp.exp() * logp).sum(-1)
        return lp_c + lp_d, ent_c, ent_d

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_policy(img_channels: int, img_size: int, state_dim: int, discrete_sizes: List[int], model_cfg) -> Policy:
    spec = PolicySpec(img_channels=img_channels, img_size=img_size, state_dim=state_dim, discrete_sizes=list(discrete_sizes),
                      conv_channels=list(model_cfg.conv_channels), conv_out=model_cfg.conv_out,
                      state_hidden=model_cfg.state_hidden, gru_hidden=model_cfg.gru_hidden,
                      logstd_init=model_cfg.logstd_init, logstd_min=model_cfg.logstd_min, logstd_max=model_cfg.logstd_max)
    return Policy(spec)
