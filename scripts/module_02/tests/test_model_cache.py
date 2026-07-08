from pathlib import Path

from model_cache import (
    CacheLocation,
    caches_to_markdown_table,
    default_cache_paths,
    directory_size_bytes,
    format_bytes,
    scan_caches,
)


def test_default_cache_paths_uses_injected_home():
    home = Path("/tmp/fake-home")
    paths = default_cache_paths(home)
    assert paths["ollama"] == home / ".ollama" / "models"
    assert paths["huggingface (used by mlx-lm)"] == home / ".cache" / "huggingface" / "hub"


def test_directory_size_bytes_sums_files_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 250)
    assert directory_size_bytes(tmp_path) == 350


def test_directory_size_bytes_ignores_symlinks(tmp_path):
    real = tmp_path / "real.bin"
    real.write_bytes(b"z" * 1000)
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    # Only the real file's size should count once, not doubled via the symlink.
    assert directory_size_bytes(tmp_path) == 1000


def test_format_bytes_scales_units():
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.0 MB"
    assert format_bytes(3 * 1024 * 1024 * 1024) == "3.0 GB"


def test_scan_caches_marks_nonexistent_paths(tmp_path):
    locations = scan_caches(home=tmp_path)
    assert all(isinstance(loc, CacheLocation) for loc in locations)
    assert all(loc.exists is False for loc in locations)
    assert all(loc.size_bytes is None for loc in locations)


def test_scan_caches_sizes_existing_path(tmp_path):
    ollama_dir = tmp_path / ".ollama" / "models"
    ollama_dir.mkdir(parents=True)
    (ollama_dir / "weights.bin").write_bytes(b"0" * 4096)

    locations = scan_caches(home=tmp_path)
    ollama_loc = next(loc for loc in locations if loc.runtime == "ollama")
    assert ollama_loc.exists is True
    assert ollama_loc.size_bytes == 4096


def test_caches_to_markdown_table_renders_all_locations():
    locations = [
        CacheLocation(runtime="ollama", path=Path("/x"), exists=True, size_bytes=1024),
        CacheLocation(runtime="huggingface", path=Path("/y"), exists=False, size_bytes=None),
    ]
    table = caches_to_markdown_table(locations)
    assert "ollama" in table
    assert "huggingface" in table
    assert "n/a" in table
