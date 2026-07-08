import pytest

from model_shapes import KNOWN_SHAPES, get_shape


def test_known_shapes_registry_is_nonempty():
    assert len(KNOWN_SHAPES) >= 3


@pytest.mark.parametrize("model_id", list(KNOWN_SHAPES))
def test_every_known_shape_has_positive_architecture_fields(model_id):
    shape = KNOWN_SHAPES[model_id]
    assert shape.n_params > 0
    assert shape.n_layers > 0
    assert shape.n_kv_heads > 0
    assert shape.head_dim > 0
    assert shape.source_note  # every entry must document where the number came from


def test_get_shape_returns_the_registered_shape():
    shape = get_shape("llama3.1-8b")
    assert shape.model_id == "llama3.1-8b"
    assert shape.n_layers == 32
    assert shape.n_kv_heads == 8
    assert shape.head_dim == 128


def test_get_shape_raises_key_error_with_helpful_message_for_unknown_model():
    with pytest.raises(KeyError) as exc_info:
        get_shape("not-a-real-model")
    assert "not-a-real-model" in str(exc_info.value)
