from __future__ import annotations

"""Object writing helpers."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

import vedo
import vedo.vtkclasses as vtki
from vedo import utils
from vedo.assembly import Assembly
from vedo.mesh import Mesh

from .loaders import load
from .scene import to_numpy

__docformat__ = "google"
__all__ = ["write", "save", "read"]


def _ply_input_with_float_tcoords(obj: Any) -> Any:
    """Return a PLY-safe input object without mutating the source dataset."""
    try:
        tcoords = obj.GetPointData().GetTCoords()
    except AttributeError:
        return obj

    if not tcoords or tcoords.GetDataType() == vtki.VTK_FLOAT:
        return obj

    converted = obj.NewInstance()
    converted.ShallowCopy(obj)

    float_tcoords = utils.numpy2vtk(utils.vtk2numpy(tcoords), dtype=np.float32)
    if tcoords.GetName():
        float_tcoords.SetName(tcoords.GetName())
    converted.GetPointData().SetTCoords(float_tcoords)
    return converted


def _gltf_polydata(data, matrix=None):
    """Prepare the float arrays required by glTF without changing the input."""
    if matrix is not None and not matrix.IsIdentity():
        transform = vtki.vtkTransform()
        transform.SetMatrix(matrix)
        transform_filter = vtki.new("TransformFilter")
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(data)
        transform_filter.Update()
        data = transform_filter.GetOutput()

    if not data.GetNumberOfCells():
        glyph = vtki.new("VertexGlyphFilter")
        glyph.SetInputData(data)
        glyph.Update()
        data = glyph.GetOutput()

    result = vtki.vtkPolyData()
    result.ShallowCopy(data)
    if data.GetPoints().GetDataType() != vtki.VTK_FLOAT:
        points = vtki.vtkPoints()
        points.DeepCopy(data.GetPoints())
        result.SetPoints(points)

    point_data = result.GetPointData()
    normals = point_data.GetNormals()
    if normals is not None and normals.GetDataType() != vtki.VTK_FLOAT:
        point_data.SetNormals(utils.numpy2vtk(utils.vtk2numpy(normals), dtype=np.float32))
    tcoords = point_data.GetTCoords()
    if tcoords is not None:
        if tcoords.GetNumberOfComponents() != 2:
            point_data.SetTCoords(None)
        elif tcoords.GetDataType() != vtki.VTK_FLOAT:
            point_data.SetTCoords(utils.numpy2vtk(utils.vtk2numpy(tcoords), dtype=np.float32))
    return result


def _write_gltf(objct: Any, fileoutput: str) -> None:
    path = Path(fileoutput)
    extension = path.suffix.lower()
    writer = vtki.new("GLTFWriter")
    if extension == ".glb" and not hasattr(writer, "GetBinary"):
        raise ValueError("this VTK version only supports writing .gltf datasets")

    data = getattr(objct, "dataset", objct)
    parts = vtki.vtkMultiBlockDataSet()
    if isinstance(data, vtki.vtkPolyData):
        if data.GetNumberOfPoints():
            actor = getattr(objct, "actor", None)
            matrix = actor.GetMatrix() if actor is not None else None
            parts.SetBlock(0, _gltf_polydata(data, matrix))
    elif isinstance(data, vtki.vtkMultiBlockDataSet):
        iterator = data.NewIterator()
        iterator.InitTraversal()
        while not iterator.IsDoneWithTraversal():
            part = iterator.GetCurrentDataObject()
            if not isinstance(part, vtki.vtkPolyData):
                raise TypeError("glTF dataset writing requires polygonal geometry")
            if part.GetNumberOfPoints():
                parts.SetBlock(parts.GetNumberOfBlocks(), _gltf_polydata(part))
            iterator.GoToNextItem()
    else:
        raise TypeError("glTF dataset writing requires a mesh, point cloud or polygonal multiblock dataset")

    if not parts.GetNumberOfBlocks():
        raise ValueError("no points to write")

    # VTK expects a multiblock containing groups of polygonal parts.
    groups = vtki.vtkMultiBlockDataSet()
    groups.SetBlock(0, parts)
    writer.SetInputData(groups)
    writer.InlineDataOn()
    writer.SaveNormalOn()

    # Some VTK versions use a fixed scratch filename beside the .glb output.
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary_path = Path(directory) / ("dataset" + extension)
        writer.SetFileName(str(temporary_path))
        if not writer.Write() or writer.GetErrorCode():
            raise OSError("VTK could not write the glTF dataset")
        temporary_path.replace(path)


def write(objct: Any, fileoutput: str | os.PathLike, binary=True) -> Any:
    """
    Write object to file. Same as `save()`.

    Supported extensions are:

    - `vtk, vti, ply, obj, stl, byu, vtp, vti, mhd, xyz, xml, tif, png, bmp`
    - `gltf, glb` for meshes, point clouds and polygonal multiblock datasets.
      The extension selects JSON or binary glTF output. GLB writing requires
      a VTK version with binary glTF writer support.
    """
    fileoutput = str(fileoutput)
    fr = fileoutput.lower()

    if fr.endswith((".gltf", ".glb")):
        try:
            _write_gltf(objct, fileoutput)
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            vedo.logger.error(f"could not save {fileoutput}: {e}")
        return objct

    ###############################
    if isinstance(objct, Assembly):
        dd = to_numpy(objct)
        sdict = {"objects": [dd]}
        np.save(fileoutput, sdict)
        return objct

    ###############################
    obj = objct.dataset

    # Check if object actor has a non-identity transform and bake it before writing.
    try:
        M = objct.actor.GetMatrix()
    except AttributeError:
        M = None
    if M and not M.IsIdentity():
        objct.apply_transform_from_actor()
        obj = objct.dataset
        vedo.logger.info(
            f"object '{objct.name}' was manually moved. Writing uses current position."
        )

    if fr.endswith(".vtk"):
        writer = vtki.new("DataSetWriter")
    elif fr.endswith(".ply"):
        writer = vtki.new("PLYWriter")
        writer.AddComment("PLY file generated by vedo")
        obj = _ply_input_with_float_tcoords(obj)
        lut = objct.mapper.GetLookupTable()
        if lut:
            pscal = obj.GetPointData().GetScalars()
            if not pscal:
                pscal = obj.GetCellData().GetScalars()
            if pscal and pscal.GetName():
                writer.SetArrayName(pscal.GetName())
            writer.SetLookupTable(lut)
    elif fr.endswith(".stl"):
        writer = vtki.new("STLWriter")
    elif fr.endswith(".vtp"):
        writer = vtki.new("XMLPolyDataWriter")
    elif fr.endswith(".vtu"):
        writer = vtki.new("XMLUnstructuredGridWriter")
    elif fr.endswith(".xyz"):
        writer = vtki.new("SimplePointsWriter")
    elif fr.endswith(".facet"):
        writer = vtki.new("FacetWriter")
    elif fr.endswith(".vti"):
        writer = vtki.new("XMLImageDataWriter")
    elif fr.endswith(".vtr"):
        writer = vtki.new("XMLRectilinearGridWriter")
    elif fr.endswith(".vtm"):
        g = vtki.new("MultiBlockDataGroupFilter")
        for ob in objct:
            try:
                g.AddInputData(ob)
            except TypeError:
                vedo.logger.warning(f"cannot save object of type {type(ob)}")
        g.Update()
        mb = g.GetOutputDataObject(0)
        wri = vtki.new("vtkXMLMultiBlockDataWriter")
        wri.SetInputData(mb)
        wri.SetFileName(fileoutput)
        wri.Write()
        return objct
    elif fr.endswith(".mhd"):
        writer = vtki.new("MetaImageWriter")
    elif fr.endswith(".nii"):
        writer = vtki.new("NIFTIImageWriter")
    elif fr.endswith(".png"):
        writer = vtki.new("PNGWriter")
    elif fr.endswith(".jpg"):
        writer = vtki.new("JPEGWriter")
    elif fr.endswith(".bmp"):
        writer = vtki.new("BMPWriter")
    elif fr.endswith(".tif") or fr.endswith(".tiff"):
        writer = vtki.new("TIFFWriter")
        writer.SetFileDimensionality(len(obj.GetDimensions()))
    elif fr.endswith(".obj"):
        with open(fileoutput, "w", encoding="UTF-8") as outF:
            outF.write("# OBJ file format with ext .obj\n")
            outF.write("# File generated by vedo\n")

            for p in objct.vertices:
                outF.write("v {:.8g} {:.8g} {:.8g}\n".format(*p))

            ptxt = objct.dataset.GetPointData().GetTCoords()
            if ptxt:
                ntxt = utils.vtk2numpy(ptxt)
                for vt in ntxt:
                    outF.write("vt " + str(vt[0]) + " " + str(vt[1]) + " 0.0\n")

            if isinstance(objct, Mesh):
                for i, f in enumerate(objct.cells):
                    fs = ""
                    for fi in f:
                        if ptxt:
                            fs += f" {fi + 1}/{fi + 1}"
                        else:
                            fs += f" {fi + 1}"
                    outF.write(f"f{fs}\n")

                for l in objct.lines:
                    ls = ""
                    for li in l:
                        ls += str(li + 1) + " "
                    outF.write(f"l {ls}\n")
        return objct

    elif fr.endswith(".off"):
        with open(fileoutput, "w", encoding="UTF-8") as outF:
            outF.write("OFF\n")
            outF.write(str(objct.npoints) + " " + str(objct.ncells) + " 0\n\n")
            for p in objct.vertices:
                outF.write(" ".join([str(i) for i in p]) + "\n")
            for c in objct.cells:
                outF.write(str(len(c)) + " " + " ".join([str(i) for i in c]) + "\n")
        return objct

    else:
        vedo.logger.error(f"Unknown format {fileoutput}, file not saved")
        return objct

    try:
        if binary:
            writer.SetFileTypeToBinary()
        else:
            writer.SetFileTypeToASCII()
    except AttributeError:
        pass

    try:
        writer.SetInputData(obj)
        writer.SetFileName(fileoutput)
        writer.Write()
    except Exception as e:
        vedo.logger.error(f"could not save {fileoutput}: {e}")
    return objct


def save(obj: Any, fileoutput="out.png", binary=True) -> Any:
    """Save an object to file. Same as `write()`."""
    return write(obj, fileoutput, binary)


def read(obj: Any, unpack=True, force=False) -> Any:
    """Read an object from file. Same as `load()`."""
    return load(obj, unpack, force)
