"""Strict backward adapters for the unchanged Fit40 V3/V4 objective.

The native CUDA forward kernels remain the source of every score. PyTorch
2.5.1 CUDA backward operators that strict deterministic mode rejects are
routed through their deterministic ATen CPU implementations. The objective
implementation is delegated byte-for-byte to adapter attempt03.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from scripts.audio_enhancement import differentiable_native_crop as _native_crop
from scripts.audio_enhancement import v3_v4_native_runtime_adapter_attempt03 as _objective


SCHEMA = "cross_tfg_v3_v4_native_runtime_adapter_v7_attempt01"
PREPARATION_ONLY = True
STRICT_DETERMINISM_REQUIRED = True
EXPECTED_FROZEN_OBJECTIVE_SHA256 = "4f4f6f496bd15e073268694673b8df5678404e672aafa625ff9a6abc135248aa"
EXPECTED_NATIVE_CROP_SHA256 = "1e89da5bdf2bee83938b717a084b55a699f1902925be3290d23317bfa562fa27"
EXPECTED_V3_HEAD_SHA256 = "77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87"
_SUPPORTED_POOL_NAMES = {"max_pool3d_with_indices_backward"}
_SUPPORTED_BILINEAR_NAMES = {"upsample_bilinear2d_backward"}


def score_with_v4_native_path(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _objective.score_with_v4_native_path(*args, **kwargs)


def v4_objective_for_v3_candidate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Call the frozen attempt03 objective without changing its arithmetic."""

    return _objective.v4_objective_for_v3_candidate(*args, **kwargs)


def _triple(value: Any) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value),) * 3
    values = tuple(int(item) for item in value)
    if len(values) != 3:
        raise RuntimeError(f"MaxPool3d parameter is not a length-three tuple: {value!r}")
    return values


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class _QualificationLedger:
    def __init__(self, path: str | os.PathLike[str], *, torch: Any) -> None:
        self.path = Path(path)
        if self.path.exists() or self.path.is_symlink():
            raise RuntimeError(f"refusing to reuse backward qualification path: {self.path}")
        self.torch = torch
        self.value: dict[str, Any] = {
            "schema": "cross_tfg_fit40_strict_backward_operator_qualification_v1",
            "status": "RUNNING_NATIVE_FORWARD_QUALIFICATION",
            "adapter_schema": SCHEMA,
            "torch_version": str(torch.__version__),
            "strict_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "strict_warn_only": False,
            "autocast_or_dtype_conversion": False,
            "native_cuda_forward_checks": [],
            "cpu_backward_checks": [],
            "full_objective_backward": None,
        }
        self.capture_backward_diagnostics = True
        self._flush()

    def _flush(self) -> None:
        raw = (json.dumps(self.value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        temporary = self.path.with_name(self.path.name + f".tmp.{os.getpid()}")
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def add_native_check(self, value: dict[str, Any]) -> None:
        self.value["native_cuda_forward_checks"].append(value)
        self._flush()

    def add_backward_check(self, value: dict[str, Any]) -> None:
        self.value["cpu_backward_checks"].append(value)
        self._flush()

    def finish_graph(self, value: dict[str, Any]) -> None:
        self.value["full_objective_backward"] = value
        self.value["status"] = "PASS_FULL_OBJECTIVE_BACKWARD_BEFORE_OPTIMIZER"
        self._flush()


class _StrictMaxPool3d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: Any,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        dilation: tuple[int, int, int],
        ceil_mode: bool,
        recorder: _QualificationLedger,
        module_name: str,
        torch: Any,
    ) -> Any:
        if input_tensor.device.type != "cuda" or input_tensor.dtype != torch.float32:
            raise RuntimeError("strict MaxPool3d adapter only accepts CUDA float32 activations")
        output, indices = torch.nn.functional.max_pool3d(
            input_tensor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )
        ctx.save_for_backward(indices)
        ctx.input_shape = tuple(int(value) for value in input_tensor.shape)
        ctx.input_dtype = input_tensor.dtype
        ctx.input_device = input_tensor.device
        ctx.output_shape = tuple(int(value) for value in output.shape)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = bool(ceil_mode)
        ctx.recorder = recorder
        ctx.module_name = module_name
        ctx.torch = torch
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
        torch = ctx.torch
        (indices,) = ctx.saved_tensors
        if grad_output.device != ctx.input_device or grad_output.dtype != ctx.input_dtype:
            raise RuntimeError("MaxPool3d gradient changed device or dtype before deterministic backward")
        grad_cpu = grad_output.detach().contiguous().to(device="cpu")
        indices_cpu = indices.detach().contiguous().to(device="cpu")
        input_shape_cpu = torch.empty(ctx.input_shape, dtype=ctx.input_dtype, device="cpu")
        op = torch.ops.aten.max_pool3d_with_indices_backward.default
        grad_input_cpu = op(
            grad_cpu,
            input_shape_cpu,
            list(ctx.kernel_size),
            list(ctx.stride),
            list(ctx.padding),
            list(ctx.dilation),
            ctx.ceil_mode,
            indices_cpu,
        )
        if not torch.isfinite(grad_input_cpu).all().item():
            raise FloatingPointError("ATen CPU MaxPool3d backward returned non-finite gradients")
        if ctx.recorder.capture_backward_diagnostics:
            replay_cpu = op(
                grad_cpu,
                input_shape_cpu,
                list(ctx.kernel_size),
                list(ctx.stride),
                list(ctx.padding),
                list(ctx.dilation),
                ctx.ceil_mode,
                indices_cpu,
            )
            if not torch.equal(grad_input_cpu, replay_cpu):
                raise RuntimeError("ATen CPU MaxPool3d backward replay was not bit-identical")

            # Independently verify the saved CUDA winners and overlap
            # accumulation against serial output-index-order scatter during the
            # pre-optimizer qualification only. Training uses the same ATen CPU
            # backward without repeating the diagnostic scatter and full-ledger
            # writes at every optimizer step.
            import numpy as np

            batch, channels, depth, height, width = ctx.input_shape
            plane_size = depth * height * width
            indices_np = indices_cpu.numpy().reshape(batch * channels, -1)
            grad_np = grad_cpu.numpy().reshape(batch * channels, -1)
            if indices_np.size and (int(indices_np.min()) < 0 or int(indices_np.max()) >= plane_size):
                raise RuntimeError("MaxPool3d saved CUDA indices escaped the input spatial plane")
            serial = np.zeros((batch * channels, plane_size), dtype=np.float32)
            for plane in range(batch * channels):
                np.add.at(serial[plane], indices_np[plane], grad_np[plane])
            serial_tensor = torch.from_numpy(serial.reshape(ctx.input_shape))
            if not torch.equal(grad_input_cpu, serial_tensor):
                raise RuntimeError("ATen CPU MaxPool3d backward differs from the independent serial scatter")

            duplicate_winner_count = int(indices_np.size - sum(np.unique(row).size for row in indices_np))
            ctx.recorder.add_backward_check(
                {
                    "operator": "aten::max_pool3d_with_indices_backward",
                    "module_name": ctx.module_name,
                    "route": "official ATen CPU backward with exact saved native CUDA indices",
                    "input_shape": list(ctx.input_shape),
                    "output_shape": list(ctx.output_shape),
                    "grad_output_shape": list(grad_cpu.shape),
                    "kernel_size": list(ctx.kernel_size),
                    "stride": list(ctx.stride),
                    "padding": list(ctx.padding),
                    "dilation": list(ctx.dilation),
                    "ceil_mode": ctx.ceil_mode,
                    "dtype": "float32; no cast",
                    "device_returned": str(ctx.input_device),
                    "saved_indices_sha256": _sha(indices_np.astype("<i8", copy=False).tobytes()),
                    "duplicate_winner_count": duplicate_winner_count,
                    "repeat_bit_identical": True,
                    "independent_serial_scatter_bit_identical": True,
                    "grad_input_sha256": _sha(grad_input_cpu.numpy().astype("<f4", copy=False).tobytes()),
                }
            )
        return (
            grad_input_cpu.to(device=ctx.input_device),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _StrictBilinear2d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: Any,
        output_height: int,
        output_width: int,
        align_corners: bool,
        native_functional: Any,
        recorder: _QualificationLedger,
        torch: Any,
    ) -> Any:
        if input_tensor.device.type != "cuda" or input_tensor.dtype != torch.float32:
            raise RuntimeError("strict bilinear adapter only accepts CUDA float32 patches")
        if input_tensor.ndim != 4 or align_corners:
            raise RuntimeError("strict bilinear adapter received an unqualified shape/layout/coordinate rule")
        output = native_functional.interpolate(
            input_tensor,
            size=(int(output_height), int(output_width)),
            mode="bilinear",
            align_corners=False,
        )
        ctx.input_shape = tuple(int(value) for value in input_tensor.shape)
        ctx.output_size = (int(output_height), int(output_width))
        ctx.align_corners = bool(align_corners)
        ctx.input_device = input_tensor.device
        ctx.input_dtype = input_tensor.dtype
        ctx.recorder = recorder
        ctx.torch = torch
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
        torch = ctx.torch
        if grad_output.device != ctx.input_device or grad_output.dtype != ctx.input_dtype:
            raise RuntimeError("bilinear gradient changed device or dtype before deterministic backward")
        grad_cpu = grad_output.detach().contiguous().to(device="cpu")
        input_size = list(ctx.input_shape)
        output_size = list(ctx.output_size)
        op = torch.ops.aten.upsample_bilinear2d_backward.default
        grad_input_cpu = op(grad_cpu, output_size, input_size, False, None, None)
        if ctx.recorder.capture_backward_diagnostics:
            replay_cpu = op(grad_cpu, output_size, input_size, False, None, None)
            if not torch.equal(grad_input_cpu, replay_cpu):
                raise RuntimeError("ATen CPU bilinear backward replay was not bit-identical")
        if not torch.isfinite(grad_input_cpu).all().item():
            raise FloatingPointError("ATen CPU bilinear backward returned non-finite gradients")
        if ctx.recorder.capture_backward_diagnostics:
            ctx.recorder.add_backward_check(
                {
                    "operator": "aten::upsample_bilinear2d_backward",
                    "route": "official ATen CPU backward after exact native CUDA bilinear forward",
                    "input_shape": list(ctx.input_shape),
                    "output_size": list(ctx.output_size),
                    "grad_output_shape": list(grad_cpu.shape),
                    "align_corners": False,
                    "dtype": "float32; no cast",
                    "device_returned": str(ctx.input_device),
                    "repeat_bit_identical": True,
                    "grad_input_sha256": _sha(grad_input_cpu.numpy().astype("<f4", copy=False).tobytes()),
                }
            )
        return grad_input_cpu.to(device=ctx.input_device), None, None, None, None, None, None


class _StrictReflectionPad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: Any,
        padding: tuple[int, ...],
        native_functional: Any,
        recorder: _QualificationLedger,
        torch: Any,
    ) -> Any:
        if input_tensor.device.type != "cuda" or input_tensor.dtype != torch.float32:
            raise RuntimeError("strict reflection adapter only accepts CUDA float32 values")
        if input_tensor.ndim not in (3, 4):
            raise RuntimeError("strict reflection adapter supports rank-3 or rank-4 input only")
        expected_pad_items = 2 * (input_tensor.ndim - 2)
        if len(padding) != expected_pad_items or any(value < 0 for value in padding):
            raise RuntimeError("strict reflection adapter received an unsupported padding tuple")
        output = native_functional.pad(input_tensor, padding, mode="reflect")
        ctx.input_shape = tuple(int(value) for value in input_tensor.shape)
        ctx.input_stride = tuple(int(value) for value in input_tensor.stride())
        ctx.padding = tuple(int(value) for value in padding)
        ctx.input_device = input_tensor.device
        ctx.input_dtype = input_tensor.dtype
        ctx.input_ndim = int(input_tensor.ndim)
        ctx.recorder = recorder
        ctx.torch = torch
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
        torch = ctx.torch
        if grad_output.device != ctx.input_device or grad_output.dtype != ctx.input_dtype:
            raise RuntimeError("reflection gradient changed device or dtype before deterministic backward")
        grad_cpu = grad_output.detach().contiguous().to(device="cpu")
        input_cpu = torch.empty_strided(
            ctx.input_shape,
            ctx.input_stride,
            dtype=ctx.input_dtype,
            device="cpu",
        )
        operator_dim = ctx.input_ndim - 2
        op = getattr(torch.ops.aten, f"reflection_pad{operator_dim}d_backward").default
        grad_input_cpu = op(grad_cpu, input_cpu, list(ctx.padding))
        if not torch.isfinite(grad_input_cpu).all().item():
            raise FloatingPointError("ATen CPU reflection-pad backward returned non-finite gradients")
        if ctx.recorder.capture_backward_diagnostics:
            replay_cpu = op(grad_cpu, input_cpu, list(ctx.padding))
            if not torch.equal(grad_input_cpu, replay_cpu):
                raise RuntimeError("ATen CPU reflection-pad backward replay was not bit-identical")
            ctx.recorder.add_backward_check(
                {
                    "operator": f"aten::reflection_pad{operator_dim}d_backward",
                    "route": "official ATen CPU backward after exact native CUDA reflection-pad forward",
                    "input_shape": list(ctx.input_shape),
                    "input_stride": list(ctx.input_stride),
                    "output_shape": list(grad_cpu.shape),
                    "padding": list(ctx.padding),
                    "mode": "reflect",
                    "dtype": "float32; no cast",
                    "device_returned": str(ctx.input_device),
                    "repeat_bit_identical": True,
                    "grad_input_sha256": _sha(grad_input_cpu.contiguous().numpy().astype("<f4", copy=False).tobytes()),
                }
            )
        return grad_input_cpu.to(device=ctx.input_device), None, None, None, None


class _FunctionalProxy:
    def __init__(self, native: Any, *, torch: Any, recorder: _QualificationLedger) -> None:
        self._native = native
        self._torch = torch
        self._recorder = recorder
        self._seen_bilinear_shapes: set[tuple[tuple[int, ...], tuple[int, int]]] = set()
        self._seen_reflection_shapes: set[tuple[tuple[int, ...], tuple[int, ...], str]] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)

    def pad(self, input_tensor: Any, pad: Any, mode: str = "constant", value: Any = None) -> Any:
        if not (mode == "reflect" and input_tensor.device.type == "cuda" and input_tensor.dtype == self._torch.float32):
            return self._native.pad(input_tensor, pad, mode=mode, value=value)
        if input_tensor.ndim not in (3, 4):
            raise RuntimeError("strict reflect padding saw an unsupported CUDA tensor rank")
        padding = tuple(int(item) for item in pad)
        expected_pad_items = 2 * (input_tensor.ndim - 2)
        if len(padding) != expected_pad_items or value is not None:
            raise RuntimeError("strict reflect padding saw an unsupported pad/value argument")
        output = _StrictReflectionPad.apply(
            input_tensor,
            padding,
            self._native,
            self._recorder,
            self._torch,
        )
        key = (tuple(int(item) for item in input_tensor.shape), padding, mode)
        if key not in self._seen_reflection_shapes:
            with self._torch.no_grad():
                reference = self._native.pad(input_tensor, padding, mode=mode, value=value)
            if not self._torch.equal(output, reference):
                raise RuntimeError("strict reflection adapter changed native CUDA forward values")
            self._seen_reflection_shapes.add(key)
            self._recorder.add_native_check(
                {
                    "operator": f"aten::reflection_pad{input_tensor.ndim - 2}d",
                    "route": "native CUDA reflect forward retained",
                    "input_shape": list(key[0]),
                    "output_shape": list(output.shape),
                    "padding": list(padding),
                    "mode": mode,
                    "dtype": "float32; no cast",
                    "bit_identical_to_direct_native_functional_pad": True,
                }
            )
        return output

    def interpolate(
        self,
        input_tensor: Any,
        size: Any = None,
        scale_factor: Any = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
        antialias: bool = False,
    ) -> Any:
        if not (
            mode == "bilinear"
            and input_tensor.ndim == 4
            and input_tensor.device.type == "cuda"
            and input_tensor.dtype == self._torch.float32
            and size is not None
            and scale_factor is None
            and align_corners is False
            and recompute_scale_factor is None
            and not antialias
        ):
            return self._native.interpolate(
                input_tensor,
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor,
                antialias=antialias,
            )
        if isinstance(size, int):
            output_size = (int(size), int(size))
        else:
            output_size = tuple(int(value) for value in size)
        if len(output_size) != 2 or min(output_size) <= 0:
            raise RuntimeError(f"bilinear output size is invalid: {output_size!r}")
        key = (tuple(int(value) for value in input_tensor.shape), output_size)
        output = _StrictBilinear2d.apply(
            input_tensor,
            output_size[0],
            output_size[1],
            False,
            self._native,
            self._recorder,
            self._torch,
        )
        if key not in self._seen_bilinear_shapes:
            reference = self._native.interpolate(
                input_tensor,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            if not self._torch.equal(output, reference):
                raise RuntimeError("strict bilinear adapter changed the native CUDA forward values")
            self._seen_bilinear_shapes.add(key)
            self._recorder.add_native_check(
                {
                    "operator": "aten::upsample_bilinear2d",
                    "route": "native CUDA forward retained",
                    "input_shape": list(key[0]),
                    "output_size": list(output_size),
                    "dtype": "float32; no cast",
                    "align_corners": False,
                    "bit_identical_to_direct_native_interpolate": True,
                }
            )
        return output


def install_strict_backward_compatibility(
    adapter: Any,
    *,
    head_module: Any,
    torch: Any,
    qualification_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Patch only the qualified SyncNet, crop, and V3-head backward routes."""

    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("strict deterministic algorithms must be enabled before installing the adapter")
    if bool(torch.is_deterministic_algorithms_warn_only_enabled()):
        raise RuntimeError("warn_only deterministic mode is forbidden")
    recorder = _QualificationLedger(qualification_path, torch=torch)
    recorder.value["strict_deterministic_algorithms"] = True
    recorder.value["strict_warn_only"] = False
    objective_path = Path(_objective.__file__).resolve(strict=True)
    objective_raw = objective_path.read_bytes()
    objective_sha = _sha(objective_raw)
    if objective_sha != EXPECTED_FROZEN_OBJECTIVE_SHA256:
        raise RuntimeError("frozen V3/V4 objective source differs from the byte-pinned attempt03 implementation")
    recorder.value["frozen_objective_source"] = {
        "module": _objective.__name__,
        "path": str(objective_path),
        "bytes": len(objective_raw),
        "sha256": objective_sha,
    }

    if not hasattr(adapter, "syncnet_model") or not hasattr(adapter, "F") or not hasattr(head_module, "F"):
        raise RuntimeError("runtime adapter lacks the qualified SyncNet or functional namespace")
    head_module_path = Path(head_module.__file__).resolve(strict=True)
    head_module_raw = head_module_path.read_bytes()
    head_module_sha256 = _sha(head_module_raw)
    if head_module_sha256 != EXPECTED_V3_HEAD_SHA256:
        raise RuntimeError("shared-recipient V3 head source differs from its whole-file frozen pin")
    crop_module_path = Path(_native_crop.__file__).resolve(strict=True)
    crop_module_raw = crop_module_path.read_bytes()
    crop_module_sha256 = _sha(crop_module_raw)
    if crop_module_sha256 != EXPECTED_NATIVE_CROP_SHA256:
        raise RuntimeError("differentiable native-crop source differs from its whole-file frozen pin")
    if getattr(_native_crop.render_native_syncnet_crops, "__globals__", None) is not _native_crop.__dict__:
        raise RuntimeError("native crop renderer does not resolve F from the pinned module namespace")

    syncnet = adapter.syncnet_model
    pools: list[tuple[str, Any]] = []
    for name, module in syncnet.named_modules():
        if isinstance(module, torch.nn.MaxPool3d):
            pools.append((name, module))
    if len(pools) != 3:
        raise RuntimeError(f"expected exactly three official SyncNet MaxPool3d layers, observed {len(pools)}")

    pool_manifest = []
    for name, module in pools:
        kernel = _triple(module.kernel_size)
        stride = _triple(module.stride if module.stride is not None else module.kernel_size)
        padding = _triple(module.padding)
        dilation = _triple(module.dilation)
        if (
            kernel != (1, 3, 3)
            or stride != (1, 2, 2)
            or dilation != (1, 1, 1)
            or padding not in {(0, 0, 0), (0, 1, 1)}
            or bool(module.ceil_mode)
            or bool(module.return_indices)
        ):
            raise RuntimeError(f"unqualified official SyncNet MaxPool3d module: {name}")
        native_forward = module.forward

        def _patched_forward(
            this: Any,
            input_tensor: Any,
            *,
            _native_forward: Any = native_forward,
            _name: str = name,
            _kernel: tuple[int, int, int] = kernel,
            _stride: tuple[int, int, int] = stride,
            _padding: tuple[int, int, int] = padding,
            _dilation: tuple[int, int, int] = dilation,
            _ceil_mode: bool = bool(module.ceil_mode),
        ) -> Any:
            output = _StrictMaxPool3d.apply(
                input_tensor,
                _kernel,
                _stride,
                _padding,
                _dilation,
                _ceil_mode,
                recorder,
                _name,
                torch,
            )
            shape_key = (tuple(int(value) for value in input_tensor.shape), tuple(int(value) for value in output.shape))
            if not any(
                row.get("operator") == "aten::max_pool3d_with_indices"
                and row.get("module_name") == _name
                and row.get("input_shape") == list(shape_key[0])
                and row.get("output_shape") == list(shape_key[1])
                for row in recorder.value["native_cuda_forward_checks"]
            ):
                with torch.no_grad():
                    reference = _native_forward(input_tensor)
                if not torch.equal(output, reference):
                    raise RuntimeError(f"strict MaxPool3d adapter changed native CUDA values: {_name}")
                recorder.add_native_check(
                    {
                        "operator": "aten::max_pool3d_with_indices",
                        "module_name": _name,
                        "route": "native CUDA forward retained; exact winners saved for deterministic backward",
                        "input_shape": list(shape_key[0]),
                        "output_shape": list(shape_key[1]),
                        "kernel_size": list(_kernel),
                        "stride": list(_stride),
                        "padding": list(_padding),
                        "dilation": list(_dilation),
                        "ceil_mode": _ceil_mode,
                        "dtype": "float32; no cast",
                        "bit_identical_to_original_module_forward": True,
                    }
                )
            return output

        module.forward = _patched_forward.__get__(module, type(module))
        pool_manifest.append(
            {
                "module_name": name,
                "kernel_size": list(kernel),
                "stride": list(stride),
                "padding": list(padding),
                "dilation": list(dilation),
                "ceil_mode": bool(module.ceil_mode),
                "return_indices": bool(module.return_indices),
            }
        )

    native_functional = adapter.F
    crop_native_functional = _native_crop.F
    head_native_functional = head_module.F
    if crop_native_functional is not native_functional or head_native_functional is not native_functional:
        raise RuntimeError("adapter, V3 head, and native-crop modules do not share the pinned native torch.nn.functional object")
    functional_proxy = _FunctionalProxy(native_functional, torch=torch, recorder=recorder)
    adapter.F = functional_proxy
    head_module.F = functional_proxy
    # render_native_syncnet_crops was imported by the frozen V4 runtime, but
    # Python function globals remain this module's dictionary. Patch that
    # isolated module attribute so the real crop path is intercepted too.
    _native_crop.F = functional_proxy
    result = {
        "schema": SCHEMA,
        "status": "INSTALLED_STRICT_BACKWARD_COMPATIBILITY",
        "native_crop_module": {
            "module": _native_crop.__name__,
            "path": str(crop_module_path),
            "bytes": len(crop_module_raw),
            "sha256": crop_module_sha256,
            "renderer_globals_are_module_namespace": True,
            "functional_attribute_patched_in_isolated_training_process": True,
            "direct_native_functional_object_preserved": True,
        },
        "v3_head_module": {
            "module": head_module.__name__,
            "path": str(head_module_path),
            "bytes": len(head_module_raw),
            "sha256": head_module_sha256,
            "functional_attribute_patched_in_isolated_training_process": True,
            "direct_native_functional_object_preserved": True,
        },
        "syncnet_pool_layers": pool_manifest,
        "maxpool_backward_operator": "aten::max_pool3d_with_indices_backward on CPU using exact saved CUDA indices",
        "bilinear_backward_operator": "aten::upsample_bilinear2d_backward on CPU after direct CUDA interpolate forward",
        "reflection_pad_backward_operator": "aten::reflection_pad1d_backward and aten::reflection_pad2d_backward on CPU after direct CUDA reflect forward",
        "strict_deterministic_algorithms": True,
        "warn_only": False,
        "dtype_conversion": False,
        "qualification_path": str(Path(qualification_path).resolve(strict=False)),
    }
    recorder.value["installation"] = result
    recorder._flush()
    adapter._strict_backward_qualification_recorder = recorder
    return result


def complete_full_objective_backward_qualification(
    qualification_path: str | os.PathLike[str],
    *,
    loss: Any,
    gradient: Any,
    theta_before: Any,
    theta_after: Any,
    recorder: Any,
    torch: Any,
) -> dict[str, Any]:
    """Seal the actual pre-optimizer whole-objective backward qualification."""

    path = Path(qualification_path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("strict backward qualification ledger is missing or is a symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    if Path(recorder.path).resolve(strict=True) != path.resolve(strict=True):
        raise RuntimeError("in-memory qualification recorder does not match the expected artifact path")
    if value.get("schema") != "cross_tfg_fit40_strict_backward_operator_qualification_v1":
        raise RuntimeError("strict backward qualification ledger schema changed")
    if value.get("status") != "RUNNING_NATIVE_FORWARD_QUALIFICATION":
        raise RuntimeError("strict backward qualification ledger is not in its expected running state")
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError("strict deterministic mode changed during whole-objective backward qualification")
    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise FloatingPointError("whole-objective backward qualification loss is non-finite or non-scalar")
    if gradient is None or tuple(gradient.shape) != (3, 3) or not torch.isfinite(gradient).all().item():
        raise FloatingPointError("whole-objective backward qualification gradient is absent or invalid")
    theta_unchanged = torch.equal(theta_before, theta_after)
    if not theta_unchanged:
        raise RuntimeError("whole-objective qualification mutated the frozen zero-initialized transition head")

    checks = value.get("cpu_backward_checks")
    if not isinstance(checks, list):
        raise RuntimeError("strict backward qualification ledger lost its operator records")
    observed_pools = {
        str(row.get("module_name"))
        for row in checks
        if row.get("operator") == "aten::max_pool3d_with_indices_backward"
    }
    installed_pools = {
        str(row.get("module_name"))
        for row in value.get("installation", {}).get("syncnet_pool_layers", [])
    }
    if len(installed_pools) != 3 or observed_pools != installed_pools:
        raise RuntimeError("full-objective qualification did not exercise all three official SyncNet pool backward layers")
    bilinear_checks = [
        row for row in checks if row.get("operator") == "aten::upsample_bilinear2d_backward"
    ]
    if not bilinear_checks:
        raise RuntimeError("full-objective qualification did not exercise native crop bilinear backward")
    reflection2d_checks = [
        row for row in checks if row.get("operator") == "aten::reflection_pad2d_backward"
    ]
    if not reflection2d_checks:
        raise RuntimeError("full-objective qualification did not exercise the V3 reflect-mask backward")
    reflection_forward_checks = value.get("native_cuda_forward_checks")
    if not isinstance(reflection_forward_checks, list):
        raise RuntimeError("strict backward qualification ledger lost its native forward records")
    for operator in ("aten::reflection_pad1d", "aten::reflection_pad2d"):
        if not any(
            row.get("operator") == operator
            and row.get("bit_identical_to_direct_native_functional_pad") is True
            for row in reflection_forward_checks
        ):
            raise RuntimeError(f"whole-objective qualification did not verify exact native CUDA forward for {operator}")
    if any(not row.get("repeat_bit_identical") for row in checks):
        raise RuntimeError("a deterministic CPU backward route failed exact replay")

    gradient_bytes = gradient.detach().contiguous().to(device="cpu").numpy().astype("<f4", copy=False).tobytes()
    complete = {
        "loss_float32": float(loss.detach().item()),
        "gradient_shape": [3, 3],
        "gradient_dtype": "float32; no cast",
        "gradient_nonzero_scalar_count": int(torch.count_nonzero(gradient).item()),
        "gradient_sha256": _sha(gradient_bytes),
        "zero_head_unchanged": theta_unchanged,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "strict_deterministic_algorithms": True,
        "warn_only": False,
        "all_three_syncnet_pool_backward_layers_exercised": True,
        "native_crop_bilinear_backward_exercised": True,
        "v3_reflection_pad2d_backward_exercised": True,
        "v3_reflection_pad1d_forward_exercised": True,
        "v3_reflection_pad2d_forward_bit_identical": True,
        "all_backward_replays_bit_identical": True,
    }
    recorder.value["full_objective_backward"] = complete
    recorder.value["status"] = "PASS_FULL_OBJECTIVE_BACKWARD_BEFORE_OPTIMIZER"
    recorder.capture_backward_diagnostics = False
    value = recorder.value
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return complete


__all__ = [
    "SCHEMA",
    "PREPARATION_ONLY",
    "STRICT_DETERMINISM_REQUIRED",
    "EXPECTED_FROZEN_OBJECTIVE_SHA256",
    "score_with_v4_native_path",
    "v4_objective_for_v3_candidate",
    "install_strict_backward_compatibility",
    "complete_full_objective_backward_qualification",
]
