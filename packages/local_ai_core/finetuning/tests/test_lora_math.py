from local_ai_core.finetuning.lora_math import (
    LayerShape,
    compare_full_finetune_and_lora,
    full_finetune_params,
    lora_trainable_params,
)


class TestFullFinetuneParams:
    def test_sums_d_in_times_d_out_across_layers(self):
        layers = [LayerShape(name="q_proj", d_in=4096, d_out=4096), LayerShape(name="v_proj", d_in=4096, d_out=4096)]
        assert full_finetune_params(layers) == 4096 * 4096 * 2

    def test_empty_layer_list_is_zero(self):
        assert full_finetune_params([]) == 0


class TestLoraTrainableParams:
    def test_matches_rank_times_d_in_plus_d_out(self):
        layers = [LayerShape(name="q_proj", d_in=4096, d_out=4096)]
        assert lora_trainable_params(layers, rank=8) == 8 * (4096 + 4096)

    def test_sums_across_multiple_layers(self):
        layers = [
            LayerShape(name="q_proj", d_in=4096, d_out=4096),
            LayerShape(name="v_proj", d_in=4096, d_out=1024),
        ]
        expected = 8 * (4096 + 4096) + 8 * (4096 + 1024)
        assert lora_trainable_params(layers, rank=8) == expected

    def test_nonpositive_rank_raises(self):
        layers = [LayerShape(name="q_proj", d_in=4096, d_out=4096)]
        try:
            lora_trainable_params(layers, rank=0)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestCompareFullFinetuneAndLora:
    def test_lora_is_a_genuine_reduction_for_realistic_shapes(self):
        layers = [LayerShape(name="q_proj", d_in=4096, d_out=4096), LayerShape(name="v_proj", d_in=4096, d_out=4096)]
        report = compare_full_finetune_and_lora(layers, rank=8)
        assert report.lora_trainable_params < report.full_finetune_params
        assert 0 < report.reduction_ratio < 1

    def test_reduction_ratio_is_zero_when_full_finetune_has_no_params(self):
        report = compare_full_finetune_and_lora([], rank=8)
        assert report.reduction_ratio == 0.0
