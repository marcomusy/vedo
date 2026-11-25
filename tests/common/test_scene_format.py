from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest

from vedo.file_io.scene import _read_scene as read_scene, _write_scene as write_scene


class Marker:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return Path.write_text, (self.path, "unpickled")


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_scene_metadata_and_typed_arrays_roundtrip(tmp_path, monkeypatch, suffix):
    monkeypatch.delenv("VEDO_ALLOW_UNSAFE_PICKLE", raising=False)
    scene = dict(
        objects=[], shape=(1, 1), rendered_at={0, 2},
        points=np.arange(12, dtype=np.float32).reshape(4, 3),
        unicode=np.array(["alpha", "βeta"]),
        complex=np.array([1 + 2j], dtype=np.complex64),
        scalar=np.int16(7), time=np.datetime64("2026-11-23"),
        metadata={"type": "array", "name": "user metadata", "nested": [(1, 2), frozenset({3})]},
        bytes=b"\x00\xff", missing=float("nan"), limit=float("inf"),
    )
    path = tmp_path / ("scene" + suffix)
    write_scene(path, scene)
    restored = read_scene(path)
    assert restored["shape"] == (1, 1)
    assert restored["rendered_at"] == {0, 2}
    assert restored["metadata"] == scene["metadata"]
    assert restored["bytes"] == scene["bytes"]
    assert np.isnan(restored["missing"])
    assert np.isinf(restored["limit"])
    for key in ("points", "unicode", "complex", "scalar", "time"):
        np.testing.assert_array_equal(restored[key], scene[key])
        assert restored[key].dtype == scene[key].dtype


@pytest.mark.rendering
@pytest.mark.parametrize("suffix,shape", [(".npy", (1, 2)), (".npz", "1|1")])
def test_multiview_scene_roundtrip(tmp_path, suffix, shape):
    import vedo

    path = tmp_path / ("multiview" + suffix)
    with vedo.Plotter(shape=shape, size=(300, 200), sharecam=False, offscreen=True) as source:
        left, right, shared = vedo.Sphere(), vedo.Cube(), vedo.Line((0, 0, 0), (1, 0, 0))
        source.add(left, shared, at=0).add(right, shared, at=1)
        source.at(0).camera.SetPosition(1, 2, 3)
        source.at(1).camera.SetPosition(4, 5, 6)
        source.camera.ParallelProjectionOn()
        source.background("black", "red")
        background2 = source.renderer.GetBackground2()
        viewports = [renderer.GetViewport() for renderer in source.renderers]
        vedo.export_window(path, plt=source)

    with vedo.import_window(path) as restored:
        assert restored.size == (300, 200)
        assert restored.renderer is restored.renderers[1]
        assert not restored.sharecam
        assert [renderer.GetViewport() for renderer in restored.renderers] == viewports
        assert [renderer.GetActiveCamera().GetPosition() for renderer in restored.renderers] == [(1, 2, 3), (4, 5, 6)]
        assert not restored.renderers[0].GetActiveCamera().GetParallelProjection()
        assert restored.camera.GetParallelProjection()
        assert restored.renderer.GetGradientBackground()
        assert restored.renderer.GetBackground2() == background2
        assert {obj.name for obj in restored.get_meshes(at=0, include_non_pickables=True)} == {"Sphere", "Line"}
        assert {obj.name for obj in restored.get_meshes(at=1, include_non_pickables=True)} == {"Cube", "Line"}
        assert len(restored.objects) == 3
        line = next(obj for obj in restored.objects if obj.name == "Line")
        assert line.rendered_at == {0, 1}
        assert all(renderer.HasViewProp(line.actor) for renderer in restored.renderers)


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
@pytest.mark.parametrize("unsafe", [False, True])
def test_new_scene_never_unpickles_arrays_even_with_opt_in(tmp_path, monkeypatch, suffix, unsafe):
    if unsafe:
        monkeypatch.setenv("VEDO_ALLOW_UNSAFE_PICKLE", "1")
    else:
        monkeypatch.delenv("VEDO_ALLOW_UNSAFE_PICKLE", raising=False)
    path = tmp_path / ("scene" + suffix)
    write_scene(path, {"objects": [], "points": np.zeros((1, 3))})
    marker = tmp_path / "marker"
    payload = np.array([Marker(marker)], dtype=object)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
        data["array_0"] = payload
        np.savez(path, **data)
    else:
        descriptor = json.loads(np.load(path, allow_pickle=False).item())
        with io.BytesIO() as stream:
            np.save(stream, payload, allow_pickle=True)
            descriptor["arrays"]["array_0"] = base64.b64encode(stream.getvalue()).decode("ascii")
        np.save(path, np.array(json.dumps(descriptor)), allow_pickle=False)
    with pytest.raises(ValueError, match="Object arrays"):
        read_scene(path)
    assert not marker.exists()


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_scene_export_rejects_objects_without_touching_existing_file(tmp_path, monkeypatch, suffix):
    monkeypatch.setenv("VEDO_ALLOW_UNSAFE_PICKLE", "1")
    marker = tmp_path / "marker"
    path = tmp_path / ("scene" + suffix)
    path.write_bytes(b"existing scene")
    with pytest.raises(TypeError, match="object arrays"):
        write_scene(path, {"objects": [], "metadata": np.array([Marker(marker)], dtype=object)})
    assert path.read_bytes() == b"existing scene"
    assert not marker.exists()


@pytest.mark.parametrize("mutation", ["version", "reference", "node"])
def test_malformed_scene_metadata_is_rejected(tmp_path, mutation):
    path = tmp_path / "scene.npz"
    write_scene(path, {"objects": [], "points": np.zeros((1, 3))})
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    metadata = json.loads(data["vedo_scenes"].item())
    if mutation == "version":
        metadata["version"] = 999
    elif mutation == "reference":
        del data["array_0"]
    else:
        metadata["data"] = {"type": "python", "module": "os", "call": "system"}
    data["vedo_scenes"] = np.array(json.dumps(metadata))
    np.savez(path, **data)
    with pytest.raises(ValueError):
        read_scene(path)


def test_failed_scene_write_does_not_replace_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "scene.npz"
    path.write_bytes(b"existing scene")

    def fail_write(stream, **kwargs):
        stream.write(b"incomplete archive")
        raise OSError("disk error")

    monkeypatch.setattr(np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="disk error"):
        write_scene(path, {"objects": []})
    assert path.read_bytes() == b"existing scene"
    assert list(tmp_path.iterdir()) == [path]


def _mesh_geometry(**changes):
    mesh = {
        "type": "Mesh",
        "points": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "cells": np.array([3, 0, 1, 2], dtype=np.int64),
    }
    mesh.update(changes)
    return mesh


@pytest.mark.parametrize("changes", [
    {"points": np.arange(3)},
    {"points": np.zeros((3, 1))},
    {"points": np.zeros((3, 4))},
    {"points": np.array([["a", "b", "c"]])},
    {"points": np.zeros((3, 3), dtype=complex)},
    {"points": np.array([[float("nan"), 0, 0]])},
    {"points": np.array([[float("inf"), 0, 0]])},
    {"cells": np.array([3, 0, 1, 2], dtype=float)},
    {"cells": np.array([3, 0, -1, 2])},
    {"cells": np.array([3, 0, 1, 999999])},
    {"cells": np.array([4, 0, 1, 2])},
    {"cells": np.array([-1, 0, 1, 2])},
    {"cells": np.array([0])},
    {"cells": np.array([2**64 - 1, 0], dtype=np.uint64)},
    {"cells": np.array([[0, 1, 3]])},
    {"cells": np.zeros((1, 1, 3), dtype=int)},
    {"cells": [[0, 1, 2], [0, 1]]},
    {"cells": [[0, 1, 2], 0]},
    {"cells": np.array([[0, 1, 2], [0, 1, 2, 0]], dtype=object)},
    {"cells": np.array([[True, False, True]])},
    {"lines": np.array([2, 0, 1], dtype=float)},
    {"lines": np.array([2, 0, -1])},
    {"lines": np.array([2, 0, 3])},
    {"lines": np.array([1, 0])},
    {"lines": np.array([3, 0, 1])},
])
def test_invalid_geometry_is_rejected_before_mesh_construction(monkeypatch, changes):
    from vedo.file_io import loaders
    from vedo.file_io.numpy_io import InvalidGeometryError

    def forbid_mesh(*args, **kwargs):
        pytest.fail("Invalid geometry reached the VTK mesh constructor")

    monkeypatch.setattr(loaders, "Mesh", forbid_mesh)
    with pytest.raises(InvalidGeometryError, match="Invalid mesh"):
        loaders.from_numpy(_mesh_geometry(**changes))


@pytest.mark.parametrize("cells,lines", [
    (np.array([3, 0, 1, 2, 4, 0, 1, 2, 3]), np.array([4, 0, 1, 2, 3])),
    (np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint64), np.array([[0, 1], [2, 3]])),
    ([[0, 1, 2], [0, 1, 2, 3]], [[0, 1], [1, 2, 3]]),
    (None, []),
])
@pytest.mark.parametrize("dimensions", [2, 3])
def test_valid_geometry_formats_remain_supported(cells, lines, dimensions):
    from vedo.file_io.scene import _validate_mesh_geometry

    points = np.zeros((4, dimensions), dtype=np.float32)
    _validate_mesh_geometry(_mesh_geometry(points=points, cells=cells, lines=lines))


def test_empty_geometry_remains_supported():
    from vedo.file_io.scene import _validate_mesh_geometry

    _validate_mesh_geometry(_mesh_geometry(points=[], cells=[], lines=[]))
    _validate_mesh_geometry(_mesh_geometry(points=np.empty((0, 3)), cells=None))


def _replace_scene_array(path, name, array):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
        data[name] = array
        np.savez(path, **data)
    else:
        metadata = json.loads(np.load(path, allow_pickle=False).item())
        with io.BytesIO() as stream:
            np.save(stream, array, allow_pickle=False)
            metadata["arrays"][name] = base64.b64encode(stream.getvalue()).decode("ascii")
        np.save(path, np.array(json.dumps(metadata)), allow_pickle=False)


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_invalid_scene_geometry_rejected_before_plotter(tmp_path, monkeypatch, suffix, legacy, nested):
    import vedo
    from vedo.file_io import numpy_io
    from vedo.file_io.numpy_io import InvalidGeometryError

    mesh = _mesh_geometry()
    obj = {"type": "Assembly", "parts": [{"type": "Assembly", "actors": [mesh]}]} if nested else mesh
    scene = {"objects": [obj], "shape": (1, 1), "size": (640, 480)}
    path = tmp_path / ("invalid-scene" + suffix)
    if legacy:
        mesh["cells"][-1] = 999999
        if suffix == ".npz":
            np.savez(path, vedo_scenes=[scene])
        else:
            np.save(path, [scene])
        monkeypatch.setenv("VEDO_ALLOW_UNSAFE_PICKLE", "1")
        monkeypatch.setattr(numpy_io, "_warn_unsafe_pickle", lambda: None)
    else:
        monkeypatch.delenv("VEDO_ALLOW_UNSAFE_PICKLE", raising=False)
        write_scene(path, scene)
        _replace_scene_array(path, "array_1", np.array([3, 0, 1, 999999]))

    def forbid_plotter(*args, **kwargs):
        pytest.fail("Invalid scene geometry reached native window construction")

    monkeypatch.setattr(vedo, "Plotter", forbid_plotter)
    with pytest.raises(InvalidGeometryError, match="vertex indices"):
        vedo.import_window(path)


def test_invalid_geometry_does_not_replace_existing_scene(tmp_path):
    from vedo.file_io.numpy_io import InvalidGeometryError

    path = tmp_path / "existing.npz"
    path.write_bytes(b"existing scene")
    with pytest.raises(InvalidGeometryError):
        write_scene(path, {"objects": [_mesh_geometry(cells=np.array([3, 0, 1, 999999]))]})
    assert path.read_bytes() == b"existing scene"
    assert list(tmp_path.iterdir()) == [path]
