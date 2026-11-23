from __future__ import annotations

import hashlib
from pathlib import Path
import warnings

import numpy as np
import pytest

import vedo
from vedo.file_io.numpy_io import NumpyIntegrityError, UnsafePickleError, _warn_unsafe_pickle
from vedo.shapes import text_utils
from vedo.shapes._font_io import read_font, write_font


# Hashes of character order, dtypes, shapes and raw glyph geometry from the
# original bundled fonts, before migration. This also covers the NUL glyph.
ORIGINAL_GEOMETRY = {
    "Bongas": "1528b5e3a1a4f6ec14ad74685efd453e605730a7be3fdd43b549dce615b011f6",
    "Calco": "16fbaa4132ee57656a063a093563a742d339957c1cfdb58b259f77856da0fc19",
    "Comae": "8da58587abbd96dfc565aa82742e5057a024032b5b918dff1be844b548205793",
    "Glasgo": "92afad92d55eacbc1167916fb3971cce2f8908b1bf74178216f7559fa66610dc",
    "Kanopus": "7d568c98e43f7f9f20fc1e2d96df9c4edf5ca22d100baf50ef017faed1d9c8da",
    "Normografo": "e3a78a2f9eba9f07c28869e0486707d87feb7facb0155faebd1dd7617c014d04",
    "Quikhand": "aa4340cef616252308cfd00112d4fb7d601c927358fa45787fca3e8775ff4c05",
    "SmartCouric": "4ea51349bc39a4b6b6fdaa255d0a14e402a486fd13835182b1734f5c74c0e4ed",
    "Theemim": "0d4de65f5ece95f1d48d9f161293ed2d20cec6832cb5bbbac9e64214304262bd",
    "VictorMono": "5d1696c7c7dddf16d1b518b6e8a93424904df1e2c0fe3e82d8b779a186ca4879",
}


class Marker:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return Path.write_text, (self.path, "unpickled")


@pytest.fixture(autouse=True)
def reset_font_policy(monkeypatch):
    monkeypatch.delenv("VEDO_ALLOW_UNSAFE_PICKLE", raising=False)
    text_utils._load_font.cache_clear()
    text_utils._get_font_letter.cache_clear()
    _warn_unsafe_pickle.cache_clear()
    yield
    text_utils._load_font.cache_clear()
    text_utils._get_font_letter.cache_clear()
    _warn_unsafe_pickle.cache_clear()


@pytest.fixture
def simple_font():
    return {"A": (np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float16),
                  np.array([[0, 1, 2]], dtype=np.int16))}


@pytest.mark.parametrize("name", ORIGINAL_GEOMETRY)
def test_bundled_font_is_pickle_free_and_preserves_geometry(name, monkeypatch):
    path = Path(vedo.fonts_path) / (name + ".npz")
    original_load = np.load
    calls = []

    def safe_load(*args, **kwargs):
        calls.append(kwargs["allow_pickle"])
        assert kwargs["allow_pickle"] is False
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", safe_load)
    font = read_font(path)
    digest = hashlib.sha256()
    for character, geometry in font.items():
        digest.update(character.encode("utf-32-le"))
        for array in geometry:
            digest.update(array.dtype.str.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    assert digest.hexdigest() == ORIGINAL_GEOMETRY[name]
    assert vedo.Text3D("Az0 :alpha", font=name).npoints > 0
    assert calls


def test_safe_custom_font_roundtrip_preserves_nul_and_non_bmp(tmp_path, simple_font):
    simple_font["\0"] = simple_font["A"]
    simple_font["😀"] = simple_font["A"]
    path = tmp_path / "custom.npz"
    write_font(path, simple_font)
    restored = read_font(path)
    assert list(restored) == list(simple_font)
    for character, geometry in simple_font.items():
        for old, new in zip(geometry, restored[character]):
            np.testing.assert_array_equal(new, old)
            assert new.dtype == old.dtype
    assert vedo.Text3D("A", font=str(path)).npoints > 0


def test_custom_legacy_font_requires_opt_in(tmp_path, simple_font, monkeypatch):
    path = tmp_path / "custom.npz"
    np.savez(path, font=[simple_font])
    with pytest.raises(UnsafePickleError, match="VEDO_ALLOW_UNSAFE_PICKLE=1"):
        vedo.Text3D("A", font=str(path))
    monkeypatch.setenv("VEDO_ALLOW_UNSAFE_PICKLE", "1")
    with pytest.warns(RuntimeWarning):
        assert vedo.Text3D("A", font=str(path)).npoints > 0


def test_official_font_uses_packaged_hash_and_cached_file(tmp_path, simple_font, monkeypatch):
    path = tmp_path / "ComicMono.npz"
    np.savez(path, font=[simple_font])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(text_utils.OFFICIAL_FONT_SHA256, "ComicMono", digest)
    urls = []

    def cached_download(url, **kwargs):
        urls.append(url)
        assert kwargs["force"] is False
        return str(path)

    monkeypatch.setattr(vedo.file_io, "download", cached_download)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert vedo.Text3D("A", font="ComicMono").npoints > 0
        assert vedo.Text3D("A", font="https://vedo.embl.es/fonts/ComicMono.npz").npoints > 0
    assert urls == ["https://vedo.embl.es/fonts/ComicMono.npz"] * 2
    assert not recorded


@pytest.mark.parametrize("unsafe_enabled", [False, True])
def test_tampered_official_font_never_unpickles(tmp_path, simple_font, monkeypatch, unsafe_enabled):
    path = tmp_path / "ComicMono.npz"
    np.savez(path, font=[simple_font])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    marker = tmp_path / "marker"
    np.savez(path, font=np.array([Marker(marker)], dtype=object))
    monkeypatch.setitem(text_utils.OFFICIAL_FONT_SHA256, "ComicMono", digest)
    monkeypatch.setattr(vedo.file_io, "download", lambda *a, **k: str(path))
    if unsafe_enabled:
        monkeypatch.setenv("VEDO_ALLOW_UNSAFE_PICKLE", "1")
    with pytest.raises(NumpyIntegrityError, match="checksum mismatch"):
        vedo.Text3D("A", font="ComicMono")
    assert not marker.exists()


@pytest.mark.parametrize("font_source", ["path", "url"])
def test_official_name_does_not_trust_custom_input(tmp_path, monkeypatch, font_source):
    path = tmp_path / "ComicMono.npz"
    marker = tmp_path / "marker"
    np.savez(path, font=np.array([Marker(marker)], dtype=object))
    if font_source == "url":
        monkeypatch.setattr(vedo.file_io, "download", lambda *a, **k: str(path))
        font = "https://example.invalid/ComicMono.npz"
    else:
        font = str(path)
    with pytest.raises(UnsafePickleError):
        vedo.Text3D("A", font=font)
    assert not marker.exists()


def test_fallback_font_has_no_pickle_bypass(tmp_path, monkeypatch):
    marker = tmp_path / "marker"
    path = tmp_path / "Normografo.npz"
    np.savez(path, font=np.array([Marker(marker)], dtype=object))
    monkeypatch.setattr(text_utils, "_local_fallback_font", lambda: ("Normografo", str(path)))
    with pytest.raises(UnsafePickleError):
        text_utils._load_local_fallback_font("missing")
    assert not marker.exists()


@pytest.mark.parametrize("change", ["version", "offsets", "index", "points", "object", "empty"])
def test_invalid_font_geometry_is_rejected_before_vtk(tmp_path, simple_font, change):
    path = tmp_path / "custom.npz"
    write_font(path, simple_font)
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    if change == "version":
        data["font_format"] = np.array(99)
    elif change == "offsets":
        data["point_offsets"] = np.array([0, 100])
    elif change == "index":
        data["faces"] = np.array([[0, 1, 999]])
    elif change == "points":
        data["points"] = np.array([[np.nan, 0], [1, 0], [0, 1]])
    elif change == "empty":
        data["codepoints"] = np.array([], dtype=np.uint32)
        data["point_offsets"] = data["face_offsets"] = np.array([0])
        data["points"] = np.empty((0, 2))
        data["faces"] = np.empty((0, 3), dtype=np.int16)
    else:
        data["points"] = np.array([[Marker(tmp_path / "marker")]], dtype=object)
    np.savez(path, **data)
    with pytest.raises(ValueError):
        read_font(path)
    assert not (tmp_path / "marker").exists()
