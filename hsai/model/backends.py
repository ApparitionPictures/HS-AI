"""Inference backends for the single-step policy (batch 1).

* EagerBackend      : plain PyTorch (any device), fp16/bf16/fp32
* CudaGraphBackend  : the whole act_step captured in a CUDA graph -> ~0.3-0.6 ms
                      on an RTX 4090, no Python/launch overhead per kernel
* TensorRTBackend   : see trt.py (optional; falls back to CudaGraphBackend)

All backends expose:
    step(img_u8[C,H,W] tensor on device, state np.float32[D]) -> (mean, logstd, logits, value) as numpy
    reset_state() / load_weights(state_dict) / name
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .policy import Policy


_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class EagerBackend:
    name = "eager"

    def __init__(self, policy: Policy, device: str = "cuda", precision: str = "fp16"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.dtype = _DTYPES.get(precision, torch.float16) if self.device.type == "cuda" else torch.float32
        self.model = Policy(policy.spec).to(self.device).to(self.dtype).eval()
        self.load_weights(policy.state_dict())
        self.h = self.model.initial_state(1, self.device, self.dtype)
        self._state_buf = torch.zeros(1, policy.spec.state_dim, device=self.device, dtype=self.dtype)
        self.last_ms = 0.0

    def reset_state(self) -> None:
        self.h.zero_()

    def hidden(self) -> np.ndarray:
        """Current GRU state (before the next step) as float16 numpy."""
        return self.h[0].detach().float().cpu().numpy().astype(np.float16)

    @torch.no_grad()
    def load_weights(self, sd: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def step(self, img_u8: torch.Tensor, state: np.ndarray):
        t0 = time.perf_counter()
        img = img_u8.unsqueeze(0).to(self.dtype).div_(255.0)
        self._state_buf.copy_(torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0))
        mean, logstd, logits, v, h = self.model.act_step(img, self._state_buf, self.h)
        self.h = h
        out = torch.cat([mean.float(), logstd.float(), logits.float(), v.float()], dim=-1)[0].cpu().numpy()
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return self._split(out)

    def _split(self, out: np.ndarray):
        n = self.model.spec.n_continuous
        nd = self.model.n_disc
        return out[:n], out[n:2 * n], out[2 * n:2 * n + nd], float(out[2 * n + nd])


class CudaGraphBackend(EagerBackend):
    name = "cuda_graph"

    def __init__(self, policy: Policy, device: str = "cuda", precision: str = "fp16"):
        super().__init__(policy, device, precision)
        if self.device.type != "cuda":
            raise RuntimeError("CUDA graphs need a CUDA device")
        spec = policy.spec
        self.s_img = torch.zeros(1, spec.img_channels, spec.img_size, spec.img_size, device=self.device, dtype=self.dtype)
        self.s_state = torch.zeros(1, spec.state_dim, device=self.device, dtype=self.dtype)
        self.s_h = torch.zeros(1, spec.gru_hidden, device=self.device, dtype=self.dtype)
        self.stream = torch.cuda.Stream(priority=-1)  # high priority stream for the actor
        self._capture()

    @torch.no_grad()
    def _capture(self) -> None:
        s = self.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # warm-up (cudnn autotune, allocator)
                self.model.act_step(self.s_img, self.s_state, self.s_h)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            mean, logstd, logits, v, h_new = self.model.act_step(self.s_img, self.s_state, self.s_h)
            self.o_out = torch.cat([mean.float(), logstd.float(), logits.float(), v.float()], dim=-1)
            self.o_h = h_new
        self._out_pinned = torch.zeros(self.o_out.shape, dtype=torch.float32).pin_memory()

    def reset_state(self) -> None:
        self.s_h.zero_()

    def hidden(self) -> np.ndarray:
        return self.s_h[0].detach().float().cpu().numpy().astype(np.float16)

    @torch.no_grad()
    def step(self, img_u8: torch.Tensor, state: np.ndarray):
        t0 = time.perf_counter()
        with torch.cuda.stream(self.stream):
            self.s_img.copy_(img_u8.unsqueeze(0).to(self.dtype).div_(255.0))
            self.s_state.copy_(torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0))
            self.graph.replay()
            self.s_h.copy_(self.o_h)
            self._out_pinned.copy_(self.o_out, non_blocking=True)
        self.stream.synchronize()
        out = self._out_pinned[0].numpy().copy()
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return self._split(out)


def make_backend(policy: Policy, backend: str = "auto", device: str = "cuda", precision: str = "fp16",
                 trt_workspace_mb: int = 1024, cache_dir: Optional[str] = None, log=print):
    """Pick the fastest backend that works on this machine."""
    dev = device if torch.cuda.is_available() else "cpu"
    order = {"auto": ["tensorrt", "cuda_graph", "eager"], "tensorrt": ["tensorrt", "cuda_graph", "eager"],
             "cuda_graph": ["cuda_graph", "eager"], "eager": ["eager"]}[backend]
    last: Optional[Exception] = None
    for name in order:
        try:
            if name == "tensorrt":
                if dev != "cuda":
                    continue
                from .trt import TensorRTBackend
                b = TensorRTBackend(policy, device=dev, precision=precision, workspace_mb=trt_workspace_mb,
                                    cache_dir=cache_dir, log=log)
            elif name == "cuda_graph":
                if dev != "cuda":
                    continue
                b = CudaGraphBackend(policy, device=dev, precision=precision)
            else:
                b = EagerBackend(policy, device=dev, precision=precision)
            log(f"[backend] using {b.name} on {dev} ({precision})")
            return b
        except Exception as e:  # pragma: no cover - depends on machine
            last = e
            if backend != "auto" and name == backend:
                log(f"[backend] {name} failed: {e!r}; falling back")
            elif backend == "auto":
                log(f"[backend] {name} unavailable ({type(e).__name__}: {e}); trying next")
    raise RuntimeError(f"no inference backend could be created: {last!r}")
