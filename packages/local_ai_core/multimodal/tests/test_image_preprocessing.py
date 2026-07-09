from PIL import Image

from local_ai_core.multimodal.image_preprocessing import (
    enhance_contrast,
    resize_max_dimension,
    rotate,
    to_grayscale,
)


def make_image(width: int = 200, height: int = 100, color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


class TestToGrayscale:
    def test_converts_to_a_single_channel_image(self):
        image = make_image()
        result = to_grayscale(image)
        assert result.mode == "L"

    def test_preserves_dimensions(self):
        image = make_image(200, 100)
        result = to_grayscale(image)
        assert result.size == (200, 100)


class TestEnhanceContrast:
    def test_factor_one_returns_the_same_pixel_values(self):
        image = make_image(color=(128, 64, 32))
        result = enhance_contrast(image, factor=1.0)
        assert result.getpixel((0, 0)) == image.getpixel((0, 0))

    def test_a_different_factor_changes_pixel_values(self):
        image = make_image(color=(128, 64, 32))
        result = enhance_contrast(image, factor=2.0)
        assert result.getpixel((0, 0)) != image.getpixel((0, 0))


class TestResizeMaxDimension:
    def test_scales_down_a_larger_image(self):
        image = make_image(400, 200)
        result = resize_max_dimension(image, max_dimension=200)
        assert max(result.size) == 200

    def test_preserves_aspect_ratio(self):
        image = make_image(400, 200)  # 2:1
        result = resize_max_dimension(image, max_dimension=100)
        assert result.size == (100, 50)

    def test_never_scales_up_a_smaller_image(self):
        image = make_image(100, 50)
        result = resize_max_dimension(image, max_dimension=500)
        assert result.size == (100, 50)


class TestRotate:
    def test_a_90_degree_rotation_swaps_dimensions(self):
        image = make_image(200, 100)
        result = rotate(image, 90)
        assert result.size == (100, 200)

    def test_a_zero_degree_rotation_preserves_dimensions(self):
        image = make_image(200, 100)
        result = rotate(image, 0)
        assert result.size == (200, 100)
