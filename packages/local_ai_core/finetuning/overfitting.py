"""Real overfitting detection over a loss curve (theory doc §8) - the
standard early-stopping signal, implemented as a real, testable function
rather than "watch the loss curves."
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochLoss:
    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class OverfittingReport:
    is_overfitting: bool
    best_validation_loss: float
    best_validation_epoch: int
    worsening_streak: int
    stop_at_epoch: int | None


def detect_overfitting(history: list[EpochLoss], *, patience: int = 2) -> OverfittingReport:
    """Tracks the best validation loss seen so far and flags overfitting
    once validation loss has increased for `patience` consecutive epochs
    while training loss kept decreasing - the training-loss condition rules
    out noisy validation loss on a model that is still genuinely improving.
    """
    if not history:
        raise ValueError("history must not be empty")
    if patience <= 0:
        raise ValueError("patience must be positive")

    best_validation_loss = history[0].validation_loss
    best_validation_epoch = history[0].epoch
    worsening_streak = 0
    stop_at_epoch: int | None = None

    for previous, current in zip(history, history[1:]):
        validation_worsened = current.validation_loss > best_validation_loss
        train_still_improving = current.train_loss < previous.train_loss

        if validation_worsened and train_still_improving:
            worsening_streak += 1
        else:
            worsening_streak = 0

        if current.validation_loss < best_validation_loss:
            best_validation_loss = current.validation_loss
            best_validation_epoch = current.epoch

        if worsening_streak >= patience and stop_at_epoch is None:
            stop_at_epoch = current.epoch

    return OverfittingReport(
        is_overfitting=stop_at_epoch is not None,
        best_validation_loss=best_validation_loss,
        best_validation_epoch=best_validation_epoch,
        worsening_streak=worsening_streak,
        stop_at_epoch=stop_at_epoch,
    )
