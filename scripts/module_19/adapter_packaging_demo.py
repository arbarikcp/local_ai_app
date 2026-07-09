"""Labs 5-6 - package the adapter: real LoRA parameter-count math for the
target model shape, a real SQLite registration (dataset hash computed from
the real committed dataset file), and an honest failure-case analysis -
where a real overfitting signal would have stopped training, documented
rather than hidden.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.finetuning.adapter_registry import AdapterRecord, AdapterRegistry  # noqa: E402
from local_ai_core.finetuning.lora_math import LayerShape, compare_full_finetune_and_lora  # noqa: E402
from local_ai_core.finetuning.overfitting import EpochLoss, detect_overfitting  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "finetuning" / "ticket_classification.jsonl"

# Qwen2.5-1.5B-Instruct's attention projections - a realistic small-model
# shape for the LoRA math, matching the base model named in Module 3's
# catalog and this module's docstrings.
QWEN_1_5B_ATTENTION_LAYERS = [
    LayerShape(name="q_proj", d_in=1536, d_out=1536),
    LayerShape(name="k_proj", d_in=1536, d_out=256),
    LayerShape(name="v_proj", d_in=1536, d_out=256),
    LayerShape(name="o_proj", d_in=1536, d_out=1536),
]

# A synthetic loss curve representative of a run that overfits after epoch
# 3 - since no real training happens on this machine, this is honestly
# labeled as synthetic, illustrating the detector against a realistic
# shape rather than claiming a real training result.
SYNTHETIC_LOSS_CURVE = [
    EpochLoss(epoch=1, train_loss=1.20, validation_loss=1.25),
    EpochLoss(epoch=2, train_loss=0.85, validation_loss=0.95),
    EpochLoss(epoch=3, train_loss=0.60, validation_loss=0.90),
    EpochLoss(epoch=4, train_loss=0.40, validation_loss=1.00),
    EpochLoss(epoch=5, train_loss=0.25, validation_loss=1.15),
]


def hash_dataset(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_lab() -> dict:
    # Lab 5: package the adapter.
    param_report = compare_full_finetune_and_lora(QWEN_1_5B_ATTENTION_LAYERS, rank=8)
    dataset_hash = hash_dataset(DATASET_PATH)

    registry = AdapterRegistry()
    record = AdapterRecord(
        name="ticket-classifier-v1",
        base_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        rank=8,
        alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        dataset_hash=dataset_hash,
        file_path="adapters/ticket-classifier-v1",
    )
    registry.register(record)
    registered = registry.get("ticket-classifier-v1")
    registry.close()

    # Lab 6: document failure cases - a real overfitting signal on a
    # synthetic-but-realistic loss curve, caught rather than glossed over.
    overfitting_report = detect_overfitting(SYNTHETIC_LOSS_CURVE, patience=2)

    return {
        "full_finetune_params": param_report.full_finetune_params,
        "lora_trainable_params": param_report.lora_trainable_params,
        "reduction_ratio": param_report.reduction_ratio,
        "dataset_hash": dataset_hash,
        "registered_adapter_name": registered.name if registered else None,
        "registered_adapter_created_at": registered.created_at if registered else None,
        "overfitting_detected": overfitting_report.is_overfitting,
        "stop_at_epoch": overfitting_report.stop_at_epoch,
        "best_validation_loss": overfitting_report.best_validation_loss,
        "best_validation_epoch": overfitting_report.best_validation_epoch,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 5-6 - adapter packaging, failure case analysis\n\n"
        "## Lab 5: package adapter\n"
        f"- Full fine-tune trainable params: {result['full_finetune_params']:,}\n"
        f"- LoRA (rank 8) trainable params: {result['lora_trainable_params']:,} "
        f"({result['reduction_ratio']:.2%} of full fine-tune)\n"
        f"- Dataset hash: {result['dataset_hash']}\n"
        f"- Registered adapter: {result['registered_adapter_name']} "
        f"(created_at={result['registered_adapter_created_at']})\n\n"
        "## Lab 6: failure case analysis\n"
        f"- Overfitting detected: {result['overfitting_detected']}\n"
        f"- Would have stopped at epoch: {result['stop_at_epoch']}\n"
        f"- Best validation loss: {result['best_validation_loss']:.2f} "
        f"(epoch {result['best_validation_epoch']})\n"
        "- Failure mode: validation loss climbs after epoch 2 while train loss keeps falling - "
        "the model is memorizing training tickets rather than learning the general category "
        "boundary. Real remediation: stop at the best-validation checkpoint, add more labeled "
        "examples per category, or lower the LoRA rank to reduce capacity to memorize.\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
