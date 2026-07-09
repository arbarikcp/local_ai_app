from local_ai_core.finetuning.overfitting import EpochLoss, detect_overfitting


class TestDetectsGenuineOverfitting:
    def test_flags_overfitting_when_validation_worsens_while_train_keeps_improving(self):
        history = [
            EpochLoss(epoch=1, train_loss=1.0, validation_loss=1.0),
            EpochLoss(epoch=2, train_loss=0.8, validation_loss=0.9),
            EpochLoss(epoch=3, train_loss=0.6, validation_loss=0.95),
            EpochLoss(epoch=4, train_loss=0.4, validation_loss=1.0),
        ]
        report = detect_overfitting(history, patience=2)
        assert report.is_overfitting is True
        assert report.stop_at_epoch == 4
        assert report.best_validation_loss == 0.9
        assert report.best_validation_epoch == 2

    def test_patience_of_one_stops_after_a_single_worsening_epoch(self):
        history = [
            EpochLoss(epoch=1, train_loss=1.0, validation_loss=1.0),
            EpochLoss(epoch=2, train_loss=0.9, validation_loss=1.1),
        ]
        report = detect_overfitting(history, patience=1)
        assert report.is_overfitting is True
        assert report.stop_at_epoch == 2


class TestDoesNotFlagHealthyTraining:
    def test_no_overfitting_when_validation_keeps_improving(self):
        history = [
            EpochLoss(epoch=1, train_loss=1.0, validation_loss=1.0),
            EpochLoss(epoch=2, train_loss=0.8, validation_loss=0.9),
            EpochLoss(epoch=3, train_loss=0.6, validation_loss=0.7),
        ]
        report = detect_overfitting(history, patience=2)
        assert report.is_overfitting is False
        assert report.stop_at_epoch is None
        assert report.best_validation_loss == 0.7

    def test_noisy_validation_does_not_count_when_train_loss_has_plateaued(self):
        # Worsening validation while training loss also stalls should not
        # trigger the overfitting signal - only a widening train/val gap
        # from a model still improving on train counts.
        history = [
            EpochLoss(epoch=1, train_loss=1.0, validation_loss=1.0),
            EpochLoss(epoch=2, train_loss=1.0, validation_loss=1.1),
            EpochLoss(epoch=3, train_loss=1.0, validation_loss=1.2),
        ]
        report = detect_overfitting(history, patience=2)
        assert report.is_overfitting is False


class TestInputValidation:
    def test_empty_history_raises(self):
        try:
            detect_overfitting([], patience=2)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_nonpositive_patience_raises(self):
        history = [EpochLoss(epoch=1, train_loss=1.0, validation_loss=1.0)]
        try:
            detect_overfitting(history, patience=0)
            assert False, "expected ValueError"
        except ValueError:
            pass
