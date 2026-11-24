"""glTF geometry and scene I/O."""
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


@pytest.mark.parametrize("gltf_file", [".gltf", ".glb"], indirect=True)
def test_gltf_geometry_reader_and_multiblock_writer(gltf_file, tmp_path):
    geometry = vedo.load_gltf(gltf_file)
    assert isinstance(geometry, vtki.vtkMultiBlockDataSet)
    assert geometry.GetNumberOfPoints() == 12
    bounds = [0.0] * 6
    geometry.GetBounds(bounds)
    np.testing.assert_allclose(bounds, [-1, 4, -1, 1, 0, 0])

    output = tmp_path / "geometry.gltf"
    assert vedo.write(geometry, output) is geometry
    loaded = vedo.load_gltf(output)
    assert loaded.GetNumberOfPoints() == 12
    loaded.GetBounds(bounds)
    np.testing.assert_allclose(bounds, [-1, 4, -1, 1, 0, 0])


@pytest.mark.parametrize("extension", [".gltf", ".glb"])
def test_gltf_writer_preserves_transformed_double_precision_input(tmp_path, extension):
    if extension == ".glb" and not hasattr(vtki.vtkGLTFWriter, "GetBinary"):
        pytest.skip("VTK has no binary glTF writer")

    cube = vedo.Cube()
    coordinates = cube.coordinates.astype(np.float64)
    points = vtki.vtkPoints()
    points.SetData(vedo.utils.numpy2vtk(coordinates))
    cube.dataset.SetPoints(points)
    cube.actor.SetPosition(3, 4, 5)

    output = tmp_path / ("geometry" + extension)
    # The filename, independent of the legacy binary flag, specifies the format.
    vedo.write(cube, output, binary=False)
    if extension == ".glb":
        assert output.read_bytes()[:4] == b"glTF"
    else:
        document = json.loads(output.read_text())
        assert all(buffer["uri"].startswith("data:") for buffer in document["buffers"])

    geometry = vedo.load_gltf(output)
    assert geometry.GetNumberOfPoints() == cube.npoints
    bounds = [0.0] * 6
    geometry.GetBounds(bounds)
    np.testing.assert_allclose(bounds, [2.5, 3.5, 3.5, 4.5, 4.5, 5.5])
    assert cube.dataset.GetPoints().GetDataType() == vtki.VTK_DOUBLE
    np.testing.assert_array_equal(cube.coordinates, coordinates)
    assert cube.actor.GetPosition() == (3, 4, 5)


@pytest.mark.parametrize("gltf_file", [".gltf", ".glb"], indirect=True)
def test_gltf_scene_import_retains_renderable_materials(gltf_file):
    plotter = vedo.import_window(gltf_file)

    assert len(plotter.objects) == 4
    assert plotter.renderer.GetActors().GetNumberOfItems() == 4
    assert sum(mesh.properties.GetTexture("albedoTex") is not None for mesh in plotter.objects) == 2
    assert all(mesh.actor.retrieve_object() is mesh for mesh in plotter.objects)


def test_gltf_scene_round_trip(tmp_path):
    cube = vedo.Cube(c="red")
    sphere = vedo.Sphere(pos=(4, 0, 0), c="blue")
    output = tmp_path / "scene.gltf"
    plotter = vedo.Plotter(offscreen=True).add(cube, sphere)
    plotter.camera.SetPosition(4, 5, 6)
    try:
        assert vedo.export_window(output, plt=plotter) is plotter
    finally:
        plotter.close()

    document = json.loads(output.read_text())
    assert len(document["meshes"]) == 2
    assert all(buffer["uri"].startswith("data:") for buffer in document["buffers"])

    imported = vedo.import_window(output)
    assert len(imported.objects) == 2
    actual = sorted(tuple(mesh.bounds()) for mesh in imported.objects)
    expected = sorted(tuple(mesh.bounds()) for mesh in (cube, sphere))
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(imported.camera.GetPosition(), [4, 5, 6])
