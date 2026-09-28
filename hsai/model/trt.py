"""TensorRT backend with in-place weight refit.

Build once (ONNX export -> FP16 engine with the REFIT flag), then every time
the learner publishes new weights we call ``Refitter.set_named_weights`` for
each parameter and ``refit_cuda_engine`` (tens of milliseconds) instead of
rebuilding the engine (tens of seconds).  Parameter names survive the ONNX
export because the network only uses Conv/Gemm/elementwise ops with the
weights used directly (see GRUCellManual).

Requires: pip install tensorrt onnx   (TensorRT >= 10, Windows or Linux)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .backends import EagerBackend
from .policy import Policy


class _ActStepWrapper(torch.nn.Module):
    def __init__(self, p: Policy):
        super().__init__()
        self.p = p

    def forward(self, img, state, h):
        mean, logstd, logits, v, h_new = self.p.act_step(img, state, h)
        return mean, logstd, logits, v, h_new


def export_onnx(policy: Policy, path: str) -> None:
    spec = policy.spec
    m = _ActStepWrapper(policy).cpu().float().eval()
    img = torch.zeros(1, spec.img_channels, spec.img_size, spec.img_size)
    st = torch.zeros(1, spec.state_dim)
    h = torch.zeros(1, spec.gru_hidden)
    torch.onnx.export(m, (img, st, h), path, input_names=["img", "state", "h"],
                      output_names=["mean", "logstd", "logits", "value", "h_new"], opset_version=17,
                      do_constant_folding=False, dynamo=False)


class TensorRTBackend(EagerBackend):
    name = "tensorrt"

    def __init__(self, policy: Policy, device: str = "cuda", precision: str = "fp16", workspace_mb: int = 1024,
                 cache_dir: Optional[str] = None, log=print):
        import tensorrt as trt  # noqa: F401  (import error -> caller falls back)
        self.trt = trt
        self.log = log
        # keep an eager fp32 twin for validation / fallback of hidden state math
        super().__init__(policy, device, "fp32")
        self.precision = precision
        self.workspace_mb = workspace_mb
        self.cache_dir = Path(cache_dir or ".")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.spec = policy.spec
        self.stream = torch.cuda.Stream(priority=-1)
        self._build_or_load(policy)
        self._bind()
        self.refit_ms = 0.0
        self.load_weights(policy.state_dict())
        self._validate(policy)

    # ------------------------------------------------------------------
    def _arch_key(self) -> str:
        d = json.dumps(self.spec.to_dict(), sort_keys=True) + self.precision + str(self.trt.__version__)
        return hashlib.sha1(d.encode()).hexdigest()[:12]

    def _build_or_load(self, policy: Policy) -> None:
        trt = self.trt
        key = self._arch_key()
        engine_path = self.cache_dir / f"policy_{key}.engine"
        onnx_path = self.cache_dir / f"policy_{key}.onnx"
        runtime = trt.Runtime(self.logger)
        if engine_path.exists():
            with open(engine_path, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is not None:
                self.log(f"[trt] loaded cached engine {engine_path.name}")
                return
        self.log("[trt] building engine (one-time, ~30-90 s)...")
        export_onnx(policy, str(onnx_path))
        builder = trt.Builder(self.logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, self.logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("ONNX parse failed: " + "; ".join(errs))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_mb << 20)
        if self.precision in ("fp16", "bf16") and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if hasattr(trt.BuilderFlag, "REFIT_INDIVIDUAL"):
            config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
        else:
            config.set_flag(trt.BuilderFlag.REFIT)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build failed")
        with open(engine_path, "wb") as f:
            f.write(serialized)
        self.engine = runtime.deserialize_cuda_engine(serialized)
        self.log(f"[trt] engine built and cached at {engine_path}")

    def _bind(self) -> None:
        trt = self.trt
        self.context = self.engine.create_execution_context()
        spec = self.spec
        dev = self.device
        self.t_img = torch.zeros(1, spec.img_channels, spec.img_size, spec.img_size, device=dev, dtype=torch.float32)
        self.t_state = torch.zeros(1, spec.state_dim, device=dev, dtype=torch.float32)
        self.t_h = torch.zeros(1, spec.gru_hidden, device=dev, dtype=torch.float32)
        self.t_mean = torch.zeros(1, spec.n_continuous, device=dev, dtype=torch.float32)
        self.t_logstd = torch.zeros(1, spec.n_continuous, device=dev, dtype=torch.float32)
        self.t_logits = torch.zeros(1, max(1, sum(spec.discrete_sizes)), device=dev, dtype=torch.float32)
        self.t_value = torch.zeros(1, 1, device=dev, dtype=torch.float32)
        self.t_hnew = torch.zeros(1, spec.gru_hidden, device=dev, dtype=torch.float32)
        binds = {"img": self.t_img, "state": self.t_state, "h": self.t_h, "mean": self.t_mean, "logstd": self.t_logstd,
                 "logits": self.t_logits, "value": self.t_value, "h_new": self.t_hnew}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name not in binds:
                raise RuntimeError(f"unexpected tensor {name}")
            dt = self.engine.get_tensor_dtype(name)
            if dt != trt.float32:
                # keep IO in fp32 for simplicity; engines built from fp32 ONNX keep fp32 IO
                raise RuntimeError(f"tensor {name} has dtype {dt}, expected float32")
            self.context.set_tensor_address(name, binds[name].data_ptr())
        self._out_pinned = torch.zeros(1, 2 * spec.n_continuous + max(1, sum(spec.discrete_sizes)) + 1,
                                       dtype=torch.float32).pin_memory()
        self._out_gpu = torch.zeros_like(self._out_pinned, device=dev)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def load_weights(self, sd: Dict[str, torch.Tensor]) -> None:
        super().load_weights(sd)
        if not hasattr(self, "engine"):
            return
        trt = self.trt
        t0 = time.perf_counter()
        refitter = trt.Refitter(self.engine, self.logger)
        names: List[str] = list(refitter.get_all_weights())
        keep = []
        missing = []
        cpu_sd = {k: v.detach().float().cpu().contiguous().numpy() for k, v in sd.items()}
        for name in names:
            arr = cpu_sd.get(name)
            if arr is None:
                # torch.onnx may prefix names (e.g. 'p.encoder.conv1.weight') -> strip module prefixes
                alt = name.split(".", 1)[1] if name.startswith("p.") else None
                arr = cpu_sd.get(alt) if alt else None
            if arr is None:
                missing.append(name)
                continue
            w = trt.Weights(arr)
            keep.append(arr)
            if not refitter.set_named_weights(name, w):
                missing.append(name)
        if missing:
            self.log(f"[trt] {len(missing)} weights not refit: {missing[:5]}...")
        if not refitter.refit_cuda_engine():
            raise RuntimeError("TensorRT refit failed")
        self._keep = keep
        self.refit_ms = (time.perf_counter() - t0) * 1000.0

    def reset_state(self) -> None:
        self.t_h.zero_()

    def hidden(self) -> np.ndarray:
        return self.t_h[0].detach().float().cpu().numpy().astype(np.float16)

    @torch.no_grad()
    def step(self, img_u8: torch.Tensor, state: np.ndarray):
        t0 = time.perf_counter()
        with torch.cuda.stream(self.stream):
            self.t_img.copy_(img_u8.unsqueeze(0).float().div_(255.0))
            self.t_state.copy_(torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0))
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute failed")
            self.t_h.copy_(self.t_hnew)
            torch.cat([self.t_mean, self.t_logstd, self.t_logits, self.t_value], dim=-1, out=self._out_gpu)
            self._out_pinned.copy_(self._out_gpu, non_blocking=True)
        self.stream.synchronize()
        out = self._out_pinned[0].numpy().copy()
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return self._split(out)

    def _validate(self, policy: Policy) -> None:
        """Compare one step against eager fp32; large mismatch -> refuse the backend."""
        spec = self.spec
        img = (torch.rand(spec.img_channels, spec.img_size, spec.img_size, device=self.device) * 255).to(torch.uint8)
        st = np.random.randn(spec.state_dim).astype(np.float32) * 0.5
        self.reset_state()
        m1, ls1, lg1, v1 = self.step(img, st)
        self.h.zero_()
        m2, ls2, lg2, v2 = EagerBackend.step(self, img, st)
        err = max(np.abs(m1 - m2).max(), np.abs(lg1 - lg2).max(), abs(v1 - v2))
        self.log(f"[trt] validation max abs diff vs fp32 eager: {err:.4f}")
        if not np.isfinite(err) or err > 0.15:
            raise RuntimeError(f"TensorRT output mismatch ({err:.3f})")
        self.reset_state()
