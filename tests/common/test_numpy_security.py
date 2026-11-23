from __future__ import annotations

import gzip
import hashlib
import os
import pickle
from pathlib import Path
import subprocess
import sys
from tempfile import gettempdir
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import vedo
import vedo.vtkclasses as vtki
from vedo.file_io import numpy_io
from vedo.file_io.numpy_io import NumpyIntegrityError, UnsafePickleError, load_numpy


class Marker:
    """A harmless pickle gadget: write only a pytest temporary marker."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return Path.write_text, (self.path, "unpickled")


@pytest.fixture(autouse=True)
def disable_pickle(monkeypatch):
    monkeypatch.delenv("VEDO_ALLOW_UNSAFE_PICKLE", raising=False)
    numpy_io._warn_unsafe_pickle.cache_clear()
    yield
    numpy_io._warn_unsafe_pickle.cache_clear()


def payload_file(tmp_path, suffix, *, key="vedo_scenes"):
    marker = tmp_path / "marker"
    payload = np.array([Marker(marker)], dtype=object)
    path = tmp_path / (tmp_path.name + suffix)
    if suffix == ".npz":
        np.savez(path, **{key: payload})
    elif suffix == ".npy.gz":
        with gzip.open(path, "wb") as stream:
            np.save(stream, payload)
    elif suffix == ".pickle.npy":
        path.write_bytes(pickle.dumps(Marker(marker)))
    else:
        np.save(path, payload)
    return path, marker


@pytest.mark.parametrize("loader", [vedo.load, lambda p: vedo.Assembly(str(p)),
                                    lambda p: vedo.Mesh(str(p)), vedo.import_window])
def test_public_npy_loaders_reject_before_execution(tmp_path, loader):
    path, marker = payload_file(tmp_path, ".npy")
    with pytest.raises(UnsafePickleError, match="VEDO_ALLOW_UNSAFE_PICKLE=1"):
        loader(path)
    assert not marker.exists()


def test_npz_scene_rejects_before_member_execution(tmp_path):
    path, marker = payload_file(tmp_path, ".npz")
    with pytest.raises(UnsafePickleError):
        vedo.import_window(path)
    assert not marker.exists()


def test_gzipped_npy_rejects_before_execution(tmp_path):
    path, marker = payload_file(tmp_path, ".npy.gz")
    with pytest.raises(UnsafePickleError):
        vedo.load(path)
    assert not marker.exists()
    (Path(gettempdir()) / path.name.removesuffix(".gz")).unlink(missing_ok=True)


def test_downloaded_npy_uses_same_policy(tmp_path, monkeypatch):
    path, marker = payload_file(tmp_path, ".npy")
    monkeypatch.setattr("vedo.file_io.loaders.download", lambda *a, **k: str(path))
    with pytest.raises(UnsafePickleError):
        vedo.load("https://example.invalid/payload.npy")
    assert not marker.exists()


def test_plain_pickle_disguised_as_npy_is_rejected(tmp_path):
    path, marker = payload_file(tmp_path, ".pickle.npy")
    with pytest.raises(UnsafePickleError):
        load_numpy(path)
    assert not marker.exists()


@pytest.mark.parametrize("value", ["0", "false", "true", "yes", "01", "1 "])
def test_only_exact_one_enables_pickle(tmp_path, monkeypatch, value):
    path, marker = payload_file(tmp_path, ".npy")
    monkeypatch.setenv("VEDO_ALLOW_UNSAFE_PICKLE", value)
    with pytest.raises(UnsafePickleError):
        load_numpy(path)
    assert not marker.exists()


def test_explicit_opt_in_warns_once_and_allows_legacy_objects(tmp_path, monkeypatch):
    path, marker = payload_file(tmp_path, ".npy")
    monkeypatch.setenv("VEDO_ALLOW_UNSAFE_PICKLE", "1")
    with pytest.warns(RuntimeWarning, match="execute Python code"):
        load_numpy(path)
    assert marker.read_text() == "unpickled"
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        load_numpy(path)
    assert not recorded
    monkeypatch.delenv("VEDO_ALLOW_UNSAFE_PICKLE")
    marker.unlink()
    with pytest.raises(UnsafePickleError):
        load_numpy(path)
    assert not marker.exists()


def test_numeric_numpy_files_need_no_opt_in(tmp_path):
    array = np.arange(15, dtype=np.float32).reshape(5, 3)
    np.save(tmp_path / "points.npy", array, allow_pickle=False)
    np.savez(tmp_path / "points.npz", points=array)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        np.testing.assert_array_equal(load_numpy(tmp_path / "points.npy"), array)
        np.testing.assert_array_equal(load_numpy(tmp_path / "points.npz", key="points"), array)
    assert not recorded


def test_corrupt_numeric_data_is_not_retried_as_pickle(tmp_path, monkeypatch):
    path = tmp_path / "truncated.npy"
    np.save(path, np.arange(100), allow_pickle=False)
    path.write_bytes(path.read_bytes()[:-20])
    monkeypatch.setenv("VEDO_ALLOW_UNSAFE_PICKLE", "1")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as caught:
            load_numpy(path)
    assert not isinstance(caught.value, UnsafePickleError)
    assert not recorded


def test_hash_check_uses_same_bytes_as_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "official.npz"
    np.savez(path, font=[{"trusted": "font"}])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    malicious, marker = payload_file(tmp_path, ".npy")
    original_load = np.load

    def replace_file_after_hash(*args, **kwargs):
        path.write_bytes(malicious.read_bytes())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", replace_file_after_hash)
    assert load_numpy(path, key="font", expected_sha256=digest)[0] == {"trusted": "font"}
    assert not marker.exists()
    with pytest.raises(NumpyIntegrityError, match="checksum mismatch"):
        load_numpy(path, key="font", expected_sha256=digest)
    assert not marker.exists()


def test_legacy_assembly_roundtrip_requires_opt_in(tmp_path, monkeypatch):
    path = tmp_path / "assembly.npy"
    original = vedo.Assembly([vedo.Cube(), vedo.Sphere()]).shift(1, 2, 3)
    original.write(path)
    with pytest.raises(UnsafePickleError):
        vedo.Assembly(str(path))
    monkeypatch.setenv("VEDO_ALLOW_UNSAFE_PICKLE", "1")
    with pytest.warns(RuntimeWarning):
        restored = vedo.Assembly(str(path))
    assert len(restored.objects) == 2
    np.testing.assert_allclose(restored.transform.matrix, original.transform.matrix)
    restored_by_load = vedo.load(path)
    assert len(restored_by_load.objects) == 2


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_legacy_scene_roundtrip_requires_opt_in(tmp_path, monkeypatch, suffix):
    # Exercise real scene serialization and reconstruction without creating a
    # native window. Even offscreen Plotters probe the display; importing a
    # scene also constructs a new, on-screen Plotter by default. Neither is
    # needed to test the pickle policy and both can crash headless VTK builds.
    class ScenePlotter:
        def __init__(self, *, size=(640, 480), axes=0, title="test scene",
                     bg="white", bg2=None):
            self.objects = []
            self.size, self.axes, self.title = size, axes, title
            self.shape, self.sharecam, self.pos = (1, 1), True, (0, 0)
            self.camera = vtki.vtkCamera()
            self.renderer = SimpleNamespace(
                GetBackground=lambda: vedo.colors.get_color(bg),
                GetGradientBackground=lambda: bg2 is not None,
                GetBackground2=lambda: vedo.colors.get_color(bg2),
                GetUseDepthPeeling=lambda: False,
            )

        def add(self, objects):
            self.objects.extend(objects if isinstance(objects, list) else [objects])
            return self

        def get_actors(self, **kwargs):
            return [obj.actor for obj in self.objects]

    def forbid_render_window(*args, **kwargs):
        pytest.fail("Scene persistence tests must not create native render windows")

    monkeypatch.setattr(vtki, "vtkRenderWindow", forbid_render_window)
    monkeypatch.setattr(vedo, "Plotter", ScenePlotter)
    path = tmp_path / ("scene" + suffix)
    cube = vedo.Cube()
    plotter = vedo.Plotter()
    plotter.camera.SetPosition(1, 2, 3)
    plotter.camera.SetFocalPoint(0.5, 0.25, 0)
    plotter.add(cube)
    vedo.export_window(path, plt=plotter)
    with pytest.raises(UnsafePickleError):
        vedo.import_window(path)
    monkeypatch.setenv("VEDO_ALLOW_UNSAFE_PICKLE", "1")
    with pytest.warns(RuntimeWarning):
        restored = vedo.import_window(path)
    assert len(restored.objects) == 1
    np.testing.assert_array_equal(restored.objects[0].vertices, cube.vertices)
    np.testing.assert_array_equal(restored.objects[0].cells, cube.cells)
    np.testing.assert_allclose(restored.camera.GetPosition(), plotter.camera.GetPosition())
    np.testing.assert_allclose(restored.camera.GetFocalPoint(), plotter.camera.GetFocalPoint())
    assert restored.size == plotter.size
    assert restored.title == plotter.title


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_cli_rejects_pickle_before_execution(tmp_path, suffix):
    path, marker = payload_file(tmp_path, suffix)
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("VEDO_ALLOW_UNSAFE_PICKLE", None)
    result = subprocess.run(
        [sys.executable, str(root / "cli_entry.py"), str(path)],
        cwd=root, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert not result.stdout
    assert "loading blocked" in result.stderr
    assert "can execute code" in result.stderr
    assert "Only if you trust" in result.stderr
    assert result.stderr.count("VEDO_ALLOW_UNSAFE_PICKLE=1") == 1
    assert str(path) in result.stderr
    assert "Traceback" not in result.stderr
    assert "UnsafePickleError" not in result.stderr
    assert not marker.exists()
