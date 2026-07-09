"""MlxLoraTrainer — LoRA training/merge adapter over mlx_lm (theory doc
§11-12). mlx_lm has no in-process Python training API this repo can rely on
across versions - its real entry points are `mlx_lm.lora` (training) and
`mlx_lm.fuse` (adapter merging), both invoked as subprocess CLI commands.
`train_fn`/`merge_fn` are injected via constructor - the same
dependency-injection principle as Module 6's `MLXRuntime` - so tests
substitute fakes without needing mlx-lm installed or Apple Silicon (this
repo's machine constraint: no model runtime installed here at all).

Enabling this for real (on Apple Silicon):
    1. In pyproject.toml, uncomment `"mlx-lm>=0.19"` (already the entry from
       Module 6 - it covers both text generation and LoRA fine-tuning), then
       run `uv sync`.
    2. Pick a base model, e.g. `mlx-community/Qwen2.5-1.5B-Instruct-4bit`,
       and a dataset directory formatted per `mlx_lm.lora`'s expectations
       (train.jsonl / valid.jsonl / test.jsonl).
    3. Construct with no overrides: `MlxLoraTrainer()` - `train_fn` and
       `merge_fn` already default to the real subprocess-based
       `_real_train`/`_real_merge` below; only tests inject fakes.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

TrainFn = Callable[..., int]
MergeFn = Callable[..., int]


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str
    dataset_dir: str
    adapter_output_dir: str
    rank: int = 8
    alpha: int = 16
    iterations: int = 1000


@dataclass(frozen=True)
class TrainingResult:
    adapter_output_dir: str
    exit_code: int

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class MergeResult:
    merged_model_dir: str
    exit_code: int

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


def _real_train(config: TrainingConfig) -> int:
    result = subprocess.run(
        [
            "python",
            "-m",
            "mlx_lm.lora",
            "--model",
            config.base_model,
            "--train",
            "--data",
            config.dataset_dir,
            "--adapter-path",
            config.adapter_output_dir,
            "--num-layers",
            str(config.rank),
            "--iters",
            str(config.iterations),
        ],
        check=False,
    )
    return result.returncode


def _real_merge(base_model: str, adapter_dir: str, merged_output_dir: str) -> int:
    result = subprocess.run(
        [
            "python",
            "-m",
            "mlx_lm.fuse",
            "--model",
            base_model,
            "--adapter-path",
            adapter_dir,
            "--save-path",
            merged_output_dir,
        ],
        check=False,
    )
    return result.returncode


class MlxLoraTrainer:
    def __init__(
        self,
        *,
        train_fn: TrainFn = _real_train,
        merge_fn: MergeFn = _real_merge,
    ) -> None:
        self._train_fn = train_fn
        self._merge_fn = merge_fn

    def train_lora_adapter(self, config: TrainingConfig) -> TrainingResult:
        exit_code = self._train_fn(config)
        return TrainingResult(adapter_output_dir=config.adapter_output_dir, exit_code=exit_code)

    def merge_adapter(self, *, base_model: str, adapter_dir: str, merged_output_dir: str) -> MergeResult:
        exit_code = self._merge_fn(base_model, adapter_dir, merged_output_dir)
        return MergeResult(merged_model_dir=merged_output_dir, exit_code=exit_code)


def is_mlx_lm_available() -> bool:
    """Honest-skip check for real execution - True only if mlx-lm is
    actually importable (theory doc's Machine note: not installed on this
    Mac by design).
    """
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True
