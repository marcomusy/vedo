"""
Trimesh support and interoperability module.

Install trimesh with:

.. code-block:: bash

    sudo apt install python3-rtree
    pip install rtree shapely
    conda install trimesh

Check out `trimesh <https://github.com/mikedh/trimesh>`_ github page for more info.

Check the example gallery at:
`examples/other/trimesh <https://github.com/marcomusy/vtkplotter/tree/master/examples/other/trimesh>`_
"""

from __future__ import division, print_function
import vtkplotter.utils as utils
import numpy as np


__all__ = ["vtk2trimesh", "trimesh2vtk"]


###########################################################################
def vtk2trimesh(actor):
    """
    Convert vtk ``Actor`` to ``Trimesh`` object.
    """
    from trimesh import Trimesh

    lut = actor.mapper.GetLookupTable()

    tris = actor.faces()
    carr = actor.scalars('CellColors', datatype='cell')
    ccols = None
    if carr is not None and len(carr)==len(tris):
        ccols = []
        for i in range(len(tris)):
            r,g,b,a = lut.GetTableValue(carr[i])
            ccols.append((r*255, g*255, b*255, a*255))
        ccols = np.array(ccols, dtype=np.int16)

    points = actor.coordinates()
    varr = actor.scalars('VertexColors', datatype='point')
    vcols = None
    if varr is not None and len(varr)==len(points):
        vcols = []
        for i in range(len(points)):
            r,g,b,a = lut.GetTableValue(varr[i])
            vcols.append((r*255, g*255, b*255, a*255))
        vcols = np.array(vcols, dtype=np.int16)

    if len(tris)==0:
        tris = None

    return Trimesh(vertices=points, faces=tris,
                   face_colors=ccols, vertex_colors=vcols)


def trimesh2vtk(inputobj, alphaPerCell=False):
    """
    Convert ``Trimesh`` object to ``Actor(vtkActor)`` or ``Assembly`` object.
    """
    # print('trimesh2vtk inputobj', type(inputobj))

    inputobj_type = str(type(inputobj))

    if "Trimesh" in inputobj_type or "primitives" in inputobj_type:
        from vtkplotter import Actor

        faces = inputobj.faces
        poly = utils.buildPolyData(inputobj.vertices, faces)
        tact = Actor(poly)
        if inputobj.visual.kind == "face":
            trim_c = inputobj.visual.face_colors
        else:
            trim_c = inputobj.visual.vertex_colors

        if utils.isSequence(trim_c):
            if utils.isSequence(trim_c[0]):
                trim_cc = trim_c[:, [0, 1, 2]] / 255
                trim_al = trim_c[:, 3] / 255
                if inputobj.visual.kind == "face":
                    tact.colorCellsByArray(trim_cc, trim_al, alphaPerCell)
                else:
                    tact.colorVerticesByArray(trim_cc, trim_al)
        else:
            print("trim_c not sequence?", trim_c)
        return tact

    elif "PointCloud" in inputobj_type:
        from vtkplotter.shapes import Points

        trim_cc, trim_al = "black", 1
        if hasattr(inputobj, "vertices_color"):
            trim_c = inputobj.vertices_color
            if len(trim_c):
                trim_cc = trim_c[:, [0, 1, 2]] / 255
                trim_al = trim_c[:, 3] / 255
                trim_al = np.sum(trim_al) / len(trim_al)  # just the average
        return Points(inputobj.vertices, r=8, c=trim_cc, alpha=trim_al)

    elif "path" in inputobj_type:
        from vtkplotter.shapes import Line
        from vtkplotter.actors import Assembly

        lines = []
        for e in inputobj.entities:
            # print('trimesh entity', e.to_dict())
            l = Line(inputobj.vertices[e.points], c="k", lw=2)
            lines.append(l)
        return Assembly(lines)

    return None
