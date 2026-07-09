"""Real LoRA parameter-count and memory-savings math (theory doc §3). A
genuine, computable reduction (`rank * (d_in + d_out)` per adapted layer vs.
`d_in * d_out` for full fine-tuning of that layer) - not an assertion that
"LoRA is more efficient."
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerShape:
    name: str
    d_in: int
    d_out: int


@dataclass(frozen=True)
class LoraParamReport:
    full_finetune_params: int
    lora_trainable_params: int

    @property
    def reduction_ratio(self) -> float:
        """How many times smaller the LoRA trainable-parameter count is
        than full fine-tuning's - undefined (returns 0.0) when full
        fine-tuning itself has zero parameters to compare against.
        """
        if self.full_finetune_params == 0:
            return 0.0
        return self.lora_trainable_params / self.full_finetune_params


def full_finetune_params(layers: list[LayerShape]) -> int:
    return sum(layer.d_in * layer.d_out for layer in layers)


def lora_trainable_params(layers: list[LayerShape], *, rank: int) -> int:
    """Each adapted layer contributes two low-rank matrices, A (d_in x rank)
    and B (rank x d_out), for rank * (d_in + d_out) trainable parameters -
    the base weight matrix itself stays frozen and is not counted here.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    return sum(rank * (layer.d_in + layer.d_out) for layer in layers)


def compare_full_finetune_and_lora(layers: list[LayerShape], *, rank: int) -> LoraParamReport:
    return LoraParamReport(
        full_finetune_params=full_finetune_params(layers),
        lora_trainable_params=lora_trainable_params(layers, rank=rank),
    )
