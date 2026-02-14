from __future__ import annotations
"""Conversion helpers for optional third-party libraries."""

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk

import vedo


__all__ = [
    "vedo2trimesh",
    "trimesh2vedo",
    "vedo2meshlab",
    "meshlab2vedo",
    "open3d2vedo",
    "vedo2open3d",
    "vedo2madcad",
    "madcad2vedo",
]


def _is_sequence(obj):
    from vedo import utils

    return utils.is_sequence(obj)


def _build_polydata(points, cells=None, lines=None):
    from vedo import utils

    return utils.buildPolyData(points, cells, lines)


def _numpy2vtk(arr, dtype=None, deep=True, name="", as_image=False, dims=None):
    from vedo import utils

    return utils.numpy2vtk(arr, dtype=dtype, deep=deep, name=name, as_image=as_image, dims=dims)


def vedo2trimesh(mesh):
    """Convert `vedo.mesh.Mesh` to `trimesh.Trimesh` object."""
    if _is_sequence(mesh):
        return [vedo2trimesh(a) for a in mesh]

    try:
        from trimesh import Trimesh  # type: ignore
    except (ImportError, ModuleNotFoundError):
        vedo.logger.error("Need trimesh to run:\npip install trimesh")
        return None

    tris = mesh.cells
    ccols = mesh.celldata["CellIndividualColors"]

    points = mesh.coordinates
    vcols = mesh.pointdata["VertexColors"]

    if len(tris) == 0:
        tris = None

    return Trimesh(vertices=points, faces=tris, face_colors=ccols, vertex_colors=vcols, process=False)


def trimesh2vedo(inputobj):
    """Convert a `Trimesh` object to `vedo.Mesh` or `vedo.Assembly` object."""
    if _is_sequence(inputobj):
        return [trimesh2vedo(ob) for ob in inputobj]

    inputobj_type = str(type(inputobj))

    if "Trimesh" in inputobj_type or "primitives" in inputobj_type:
        faces = inputobj.faces
        poly = _build_polydata(inputobj.vertices, faces)
        tact = vedo.Mesh(poly)
        if inputobj.visual.kind == "face":
            trim_c = inputobj.visual.face_colors
        elif inputobj.visual.kind == "texture":
            trim_c = inputobj.visual.to_color().vertex_colors
        else:
            trim_c = inputobj.visual.vertex_colors

        if _is_sequence(trim_c) and _is_sequence(trim_c[0]):
            same_color = len(np.unique(trim_c, axis=0)) < 2
            if same_color:
                tact.c(trim_c[0, [0, 1, 2]]).alpha(trim_c[0, 3])
            elif inputobj.visual.kind == "face":
                tact.cellcolors = trim_c
        return tact

    if "PointCloud" in inputobj_type:
        vdpts = vedo.Points(inputobj.vertices, r=8, c="k")
        if hasattr(inputobj, "vertices_color"):
            vcols = (inputobj.vertices_color * 1).astype(np.uint8)
            vdpts.pointcolors = vcols
        return vdpts

    if "path" in inputobj_type:
        lines = []
        for e in inputobj.entities:
            lines.append(vedo.Line(inputobj.vertices[e.points], c="k", lw=2))
        return vedo.Assembly(lines)

    return None


def vedo2meshlab(vmesh):
    """Convert a `vedo.Mesh` to a Meshlab object."""
    try:
        import pymeshlab as mlab  # type: ignore
    except ModuleNotFoundError:
        vedo.logger.error("Need pymeshlab to run:\npip install pymeshlab")
        return None

    vertex_matrix = vmesh.vertices.astype(np.float64)

    try:
        face_matrix = np.asarray(vmesh.cells, dtype=np.float64)
    except Exception:
        print("WARNING: in vedo2meshlab(), need to triangulate mesh first!")
        face_matrix = np.array(vmesh.clone().triangulate().cells, dtype=np.float64)

    v_normals_matrix = vmesh.vertex_normals
    if not v_normals_matrix.shape[0]:
        v_normals_matrix = np.empty((0, 3), dtype=np.float64)

    f_normals_matrix = vmesh.cell_normals
    if not f_normals_matrix.shape[0]:
        f_normals_matrix = np.empty((0, 3), dtype=np.float64)

    v_color_matrix = vmesh.pointdata["RGBA"]
    if v_color_matrix is None:
        v_color_matrix = np.empty((0, 4), dtype=np.float64)
    else:
        v_color_matrix = v_color_matrix.astype(np.float64) / 255
        if v_color_matrix.shape[1] == 3:
            v_color_matrix = np.c_[v_color_matrix, np.ones(v_color_matrix.shape[0], dtype=np.float64)]

    f_color_matrix = vmesh.celldata["RGBA"]
    if f_color_matrix is None:
        f_color_matrix = np.empty((0, 4), dtype=np.float64)
    else:
        f_color_matrix = f_color_matrix.astype(np.float64) / 255
        if f_color_matrix.shape[1] == 3:
            f_color_matrix = np.c_[f_color_matrix, np.ones(f_color_matrix.shape[0], dtype=np.float64)]

    m = mlab.Mesh(
        vertex_matrix=vertex_matrix,
        face_matrix=face_matrix,
        v_normals_matrix=v_normals_matrix,
        f_normals_matrix=f_normals_matrix,
        v_color_matrix=v_color_matrix,
        f_color_matrix=f_color_matrix,
    )

    for k in vmesh.pointdata.keys():
        data = vmesh.pointdata[k]
        if data is None:
            continue
        if data.ndim == 1:
            m.add_vertex_custom_scalar_attribute(data.astype(np.float64), k)
        elif data.ndim == 2 and "tcoord" not in k.lower() and k not in ["Normals", "TextureCoordinates"]:
            m.add_vertex_custom_point_attribute(data.astype(np.float64), k)

    for k in vmesh.celldata.keys():
        data = vmesh.celldata[k]
        if data is None:
            continue
        if data.ndim == 1:
            m.add_face_custom_scalar_attribute(data.astype(np.float64), k)
        elif data.ndim == 2 and k != "Normals":
            m.add_face_custom_point_attribute(data.astype(np.float64), k)

    m.update_bounding_box()
    return m


def meshlab2vedo(mmesh, pointdata_keys=(), celldata_keys=()):
    """Convert a Meshlab object to `vedo.Mesh`."""
    inputtype = str(type(mmesh))
    if "MeshSet" in inputtype:
        mmesh = mmesh.current_mesh()

    mpoints, mcells = mmesh.vertex_matrix(), mmesh.face_matrix()
    polydata = _build_polydata(mpoints, mcells if len(mcells) > 0 else None)

    if mmesh.has_vertex_scalar():
        parr = mmesh.vertex_scalar_array()
        parr_vtk = numpy_to_vtk(parr)
        parr_vtk.SetName("MeshLabScalars")
        polydata.GetPointData().AddArray(parr_vtk)
        polydata.GetPointData().SetActiveScalars("MeshLabScalars")

    if mmesh.has_face_scalar():
        carr = mmesh.face_scalar_array()
        carr_vtk = numpy_to_vtk(carr)
        carr_vtk.SetName("MeshLabScalars")
        polydata.GetCellData().AddArray(carr_vtk)
        polydata.GetCellData().SetActiveScalars("MeshLabScalars")

    for k in pointdata_keys:
        parr = mmesh.vertex_custom_scalar_attribute_array(k)
        parr_vtk = numpy_to_vtk(parr)
        parr_vtk.SetName(k)
        polydata.GetPointData().AddArray(parr_vtk)
        polydata.GetPointData().SetActiveScalars(k)

    for k in celldata_keys:
        carr = mmesh.face_custom_scalar_attribute_array(k)
        carr_vtk = numpy_to_vtk(carr)
        carr_vtk.SetName(k)
        polydata.GetCellData().AddArray(carr_vtk)
        polydata.GetCellData().SetActiveScalars(k)

    pnorms = mmesh.vertex_normal_matrix()
    if len(pnorms) > 0:
        polydata.GetPointData().SetNormals(_numpy2vtk(pnorms, name="Normals"))

    cnorms = mmesh.face_normal_matrix()
    if len(cnorms) > 0:
        polydata.GetCellData().SetNormals(_numpy2vtk(cnorms, name="Normals"))
    return vedo.Mesh(polydata)


def open3d2vedo(o3d_mesh):
    """Convert `open3d.geometry.TriangleMesh` to a `vedo.Mesh`."""
    return vedo.Mesh([np.array(o3d_mesh.vertices), np.array(o3d_mesh.triangles)])


def vedo2open3d(vedo_mesh):
    """Return an `open3d.geometry.TriangleMesh` version of the current mesh."""
    try:
        import open3d as o3d  # type: ignore
    except RuntimeError:
        vedo.logger.error("Need open3d to run:\npip install open3d")
        return None

    return o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vedo_mesh.vertices),
        triangles=o3d.utility.Vector3iVector(vedo_mesh.cells),
    )


def vedo2madcad(vedo_mesh):
    """Convert a `vedo.Mesh` to a `madcad.Mesh`."""
    try:
        import madcad  # type: ignore
    except ModuleNotFoundError:
        vedo.logger.error("Need madcad to run:\npip install pymadcad")
        return None

    points = [madcad.vec3(*pt) for pt in vedo_mesh.vertices]
    faces = [madcad.vec3(*fc) for fc in vedo_mesh.cells]

    options = {}
    for key, val in vedo_mesh.pointdata.items():
        vec_type = f"vec{val.shape[-1]}"
        is_float = np.issubdtype(val.dtype, np.floating)
        madcad_dtype = getattr(madcad, f"f{vec_type}" if is_float else vec_type)
        options[key] = [madcad_dtype(v) for v in val]

    return madcad.Mesh(points=points, faces=faces, options=options)


def madcad2vedo(madcad_mesh):
    """Convert a `madcad.Mesh` to a `vedo.Mesh`."""
    try:
        madcad_mesh = madcad_mesh["part"]
    except Exception:
        pass

    madp = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in madcad_mesh.points])

    madf = []
    try:
        madf = np.array([[int(f[0]), int(f[1]), int(f[2])] for f in madcad_mesh.faces]).astype(np.uint16)
    except AttributeError:
        pass

    made = []
    try:
        made = np.array([[int(e[0]), int(e[1])] for e in madcad_mesh.edges]).astype(np.uint16)
    except (AttributeError, TypeError):
        pass

    try:
        line = np.array(madcad_mesh.indices).astype(np.uint16)
        made.append(line)
    except AttributeError:
        pass

    madt = []
    try:
        madt = np.array([int(t) for t in madcad_mesh.tracks]).astype(np.uint16)
    except AttributeError:
        pass

    poly = _build_polydata(madp, madf, made)
    if len(madf) == 0 and len(made) == 0:
        m = vedo.Points(poly)
    else:
        m = vedo.Mesh(poly)

    if len(madt) == len(madf):
        m.celldata["tracks"] = madt
        maxt = np.max(madt)
        m.mapper.SetScalarRange(0, np.max(madt))
        if maxt == 0:
            m.mapper.SetScalarVisibility(0)
    elif len(madt) == len(madp):
        m.pointdata["tracks"] = madt
        maxt = np.max(madt)
        m.mapper.SetScalarRange(0, maxt)
        if maxt == 0:
            m.mapper.SetScalarVisibility(0)

    try:
        m.info["madcad_groups"] = madcad_mesh.groups
    except AttributeError:
        pass

    try:
        options = dict(madcad_mesh.options)
        if "display_wire" in options and options["display_wire"]:
            m.lw(1).lc(madcad_mesh.c())
        if "display_faces" in options and not options["display_faces"]:
            m.alpha(0.2)
        if "color" in options:
            m.c(options["color"])

        for key, val in options.items():
            m.pointdata[key] = val
    except AttributeError:
        pass

    return m
