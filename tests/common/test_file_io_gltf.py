"""GLTF materials must survive the Assembly returned by load()."""
import base64
import json
import struct

import numpy as np
import pytest

import vedo
import vedo.vtkclasses as vtki
from vedo.utils import vtk2numpy


@pytest.fixture
def gltf_file(tmp_path, request):
    positions = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype="<f4")
    uv = np.array([[0, 0], [1, 0], [0.5, 1]], dtype="<f4")
    binary = positions.tobytes() + uv.tobytes()
    document = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["KHR_texture_transform"],
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"mesh": 0}, {"mesh": 0, "translation": [3, 0, 0]}],
        "meshes": [{"primitives": [
            {"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0},
            {"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 1},
        ]}],
        "buffers": [{"byteLength": len(binary), "uri":
                     "data:application/octet-stream;base64," + base64.b64encode(binary).decode()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": uv.nbytes},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
             "min": [-1, -1, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
        ],
        "materials": [
            {"pbrMetallicRoughness": {}},
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0, "extensions": {
                "KHR_texture_transform": {"offset": [0.25, 0.125], "scale": [0.5, 0.25],
                                          "rotation": np.pi}
            }}}},
        ],
        "textures": [{"source": 0}],
        "images": [{"uri": "white.png"}],
    }
    image = vtki.vtkImageData()
    image.SetDimensions(1, 1, 1)
    image.AllocateScalars(vtki.VTK_UNSIGNED_CHAR, 3)
    vtk2numpy(image.GetPointData().GetScalars())[:] = 255
    writer = vtki.new("PNGWriter")
    writer.SetFileName(str(tmp_path / "white.png"))
    writer.SetInputData(image)
    writer.Write()
    path = tmp_path / ("model" + request.param)
    if request.param == ".glb":
        document["buffers"][0].pop("uri")
    text = json.dumps(document).encode()
    if request.param == ".glb":
        text += b" " * (-len(text) % 4)
        text = (struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(binary))
                + struct.pack("<II", len(text), 0x4E4F534A) + text
                + struct.pack("<II", len(binary), 0x004E4942) + binary)
    path.write_bytes(text)
    return path


@pytest.mark.parametrize("gltf_file", [".gltf", ".glb"], indirect=True)
def test_gltf_texture_transforms_survive_grouping(gltf_file):
    model = vedo.load(gltf_file)
    assert isinstance(model, vedo.Assembly)
    assert len(model.objects) == 4
    original = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
    # GLTF origin flip, and a second material with scale/rotation/offset.
    expected = [original * [1, -1] + [0, 1],
                np.column_stack((0.25 - 0.5 * original[:, 0],
                                 0.875 + 0.25 * original[:, 1]))]
    for i, mesh in enumerate(model.objects):
        data = mesh.actor.GetMapper().GetInput()
        np.testing.assert_allclose(vtk2numpy(data.GetPointData().GetTCoords()),
                                   expected[i % 2], atol=1e-7)
        assert data is mesh.dataset
        assert not mesh.actor.GetPropertyKeys().Has(vtki.vtkProp.GENERAL_TEXTURE_TRANSFORM())
    # The same primitive appears twice. Baking one instance must not alter
    # the shared importer geometry and apply its transform a second time.
    np.testing.assert_array_equal(model.objects[0].coordinates, model.objects[2].coordinates)
    assert sorted(m.actor.GetUserMatrix().GetElement(0, 3) for m in model.objects) == [0, 0, 3, 3]


@pytest.mark.parametrize("gltf_file", [".gltf"], indirect=True)
def test_gltf_without_texture_coordinates(gltf_file):
    document = json.loads(gltf_file.read_text())
    document["materials"] = [{"pbrMetallicRoughness": {}}] * 2
    for primitive in document["meshes"][0]["primitives"]:
        primitive["attributes"].pop("TEXCOORD_0")
    gltf_file.write_text(json.dumps(document))
    model = vedo.load(gltf_file)
    assert len(model.objects) == 4
    assert all(m.dataset.GetPointData().GetTCoords() is None for m in model.objects)
