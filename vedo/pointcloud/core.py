#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from weakref import ref as weak_ref_to

from typing_extensions import Self

import vedo.vtkclasses as vtki

import vedo
from vedo import colors
from vedo import utils
from vedo.core.transformations import LinearTransform
from vedo.core import PointAlgorithms
from vedo.core import input as input_utils
from vedo.visual import PointsVisual
from .transform import PointTransformMixin
from .analyze import PointAnalyzeMixin
from .reconstruct import PointReconstructMixin
from .cut import PointCutMixin

__docformat__ = "google"

__doc__ = """
Submodule to work with point clouds.

![](https://vedo.embl.es/images/basic/pca.png)
"""

__all__ = ["Point", "Points"]



def Point(pos=(0, 0, 0), r=12, c="red", alpha=1.0) -> Self:
    """Build a point at position of radius size `r`, color `c` and transparency `alpha`."""
    return Points([[0,0,0]], r=r, c=c, alpha=alpha).pos(pos)


class Points(PointsVisual, PointAlgorithms, PointTransformMixin, 
             PointAnalyzeMixin, PointReconstructMixin, PointCutMixin):
    """Work with point clouds."""

    def __init__(self, inputobj=None, r=4, c=(0.2, 0.2, 0.2), alpha=1):
        """
        Build an object made of only vertex points for a list of 2D/3D points.
        Both shapes (N, 3) or (3, N) are accepted as input, if N>3.

        Arguments:
            inputobj : (list, tuple)
            r : (int)
                Point radius in units of pixels.
            c : (str, list)
                Color name or rgb tuple.
            alpha : (float)
                Transparency in range [0,1].

        Example:
            ```python
            from vedo import *

            def fibonacci_sphere(n):
                s = np.linspace(0, n, num=n, endpoint=False)
                theta = s * 2.399963229728653
                y = 1 - s * (2/(n-1))
                r = np.sqrt(1 - y * y)
                x = np.cos(theta) * r
                z = np.sin(theta) * r
                return np._c[x,y,z]

            Points(fibonacci_sphere(1000)).show(axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/fibonacci.png)
        """
        # print("INIT POINTS")
        super().__init__()

        self.name = ""
        self.filename = ""
        self.file_size = ""

        self.info = {}
        self.time = time.time()

        self.transform = LinearTransform()

        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None

        self.actor = vtki.vtkActor()
        self.properties = self.actor.GetProperty()
        self.properties_backface = self.actor.GetBackfaceProperty()
        self.mapper = vtki.new("PolyDataMapper")
        self.dataset = vtki.vtkPolyData()

        # Create weakref so actor can access this object (eg to pick/remove):
        self.actor.retrieve_object = weak_ref_to(self)

        try:
            if vedo.settings.enable_rendering_points_as_spheres:
                self.properties.RenderPointsAsSpheresOn()
        except AttributeError:
            pass

        if inputobj is None:  ####################
            return
        ##########################################

        self.name = "Points"

        ######
        if isinstance(inputobj, vtki.vtkActor):
            self.dataset.DeepCopy(inputobj.GetMapper().GetInput())
            pr = vtki.vtkProperty()
            pr.DeepCopy(inputobj.GetProperty())
            self.actor.SetProperty(pr)
            self.properties = pr
            self.mapper.SetScalarVisibility(inputobj.GetMapper().GetScalarVisibility())

        elif isinstance(inputobj, vtki.vtkPolyData):
            self.dataset = inputobj
            if self.dataset.GetNumberOfCells() == 0:
                carr = vtki.vtkCellArray()
                for i in range(self.dataset.GetNumberOfPoints()):
                    carr.InsertNextCell(1)
                    carr.InsertCellPoint(i)
                self.dataset.SetVerts(carr)

        elif isinstance(inputobj, Points):
            self.dataset = inputobj.dataset
            self.copy_properties_from(inputobj)

        elif utils.is_sequence(inputobj):  # passing point coords
            self.dataset = utils.buildPolyData(utils.make3d(inputobj))

        elif input_utils.is_path_like(inputobj):
            inpath = input_utils.as_path(inputobj)
            verts = vedo.file_io.load(inpath)
            self.filename = str(inpath)
            self.dataset = verts.dataset

        elif "meshlib" in str(type(inputobj)):
            from meshlib import mrmeshnumpy as mn
            self.dataset = utils.buildPolyData(mn.toNumpyArray(inputobj.points))

        else:
            # try to extract the points from a generic VTK input data object
            try:
                self.dataset = input_utils.points_polydata_from_dataset(inputobj)
            except Exception as e:
                vedo.logger.error(f"cannot build Points from type {type(inputobj)}: {e}")
                raise RuntimeError() from e

        self.actor.SetMapper(self.mapper)
        self.mapper.SetInputData(self.dataset)

        self.properties.SetColor(colors.get_color(c))
        self.properties.SetOpacity(alpha)
        self.properties.SetRepresentationToPoints()
        self.properties.SetPointSize(r)
        self.properties.LightingOff()

        self.pipeline = utils.OperationNode(
            self, parents=[], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )

    def _update(self, polydata, reset_locators=True) -> Self:
        """Overwrite the polygonal dataset with a new vtkPolyData."""
        self.dataset = polydata
        self.mapper.SetInputData(self.dataset)
        self.mapper.Modified()
        if reset_locators:
            self.point_locator = None
            self.line_locator = None
            self.cell_locator = None
        return self
    def __str__(self):
        """Print a description of the Points/Mesh."""
        module = self.__class__.__module__
        name = self.__class__.__name__
        out = vedo.printc(
            f"{module}.{name} at ({hex(self.memory_address())})".ljust(75),
            c="g", bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m\x1b[32;1m"

        if self.name:
            out += "name".ljust(14) + ": " + self.name
            if "legend" in self.info.keys() and self.info["legend"]:
                out+= f", legend='{self.info['legend']}'"
            out += "\n"

        if self.filename:
            out+= "file name".ljust(14) + ": " + self.filename + "\n"

        if not self.mapper.GetScalarVisibility():
            col = utils.precision(self.properties.GetColor(), 3)
            cname = vedo.colors.get_color_name(self.properties.GetColor())
            out+= "color".ljust(14) + ": " + cname
            out+= f", rgb={col}, alpha={self.properties.GetOpacity()}\n"
            if self.actor.GetBackfaceProperty():
                bcol = self.actor.GetBackfaceProperty().GetDiffuseColor()
                cname = vedo.colors.get_color_name(bcol)
                out+= "backface color".ljust(14) + ": "
                out+= f"{cname}, rgb={utils.precision(bcol,3)}\n"

        npt = self.dataset.GetNumberOfPoints()
        npo, nln = self.dataset.GetNumberOfPolys(), self.dataset.GetNumberOfLines()
        out+= "elements".ljust(14) + f": vertices={npt:,} polygons={npo:,} lines={nln:,}"
        if self.dataset.GetNumberOfStrips():
            out+= f", strips={self.dataset.GetNumberOfStrips():,}"
        out+= "\n"
        if self.dataset.GetNumberOfPieces() > 1:
            out+= "pieces".ljust(14) + ": " + str(self.dataset.GetNumberOfPieces()) + "\n"

        out+= "position".ljust(14) + ": " + f"{utils.precision(self.pos(), 6)}\n"
        try:
            sc = self.transform.get_scale()
            out+= "scaling".ljust(14)  + ": "
            out+= utils.precision(sc, 6) + "\n"
        except AttributeError:
            pass

        if self.npoints:
            out+="size".ljust(14)+ ": average=" + utils.precision(self.average_size(),6)
            out+=", diagonal="+ utils.precision(self.diagonal_size(), 6)+ "\n"
            out+="center of mass".ljust(14) + ": " + utils.precision(self.center_of_mass(),6)+"\n"

        bnds = self.bounds()
        bx1, bx2 = utils.precision(bnds[0], 3), utils.precision(bnds[1], 3)
        by1, by2 = utils.precision(bnds[2], 3), utils.precision(bnds[3], 3)
        bz1, bz2 = utils.precision(bnds[4], 3), utils.precision(bnds[5], 3)
        out+= "bounds".ljust(14) + ":"
        out+= " x=(" + bx1 + ", " + bx2 + "),"
        out+= " y=(" + by1 + ", " + by2 + "),"
        out+= " z=(" + bz1 + ", " + bz2 + ")\n"

        for key in self.pointdata.keys():
            arr = self.pointdata[key]
            dim = arr.shape[1] if arr.ndim > 1 else 1
            mark_active = "pointdata"
            a_scalars = self.dataset.GetPointData().GetScalars()
            a_vectors = self.dataset.GetPointData().GetVectors()
            a_tensors = self.dataset.GetPointData().GetTensors()
            if   a_scalars and a_scalars.GetName() == key:
                mark_active += " *"
            elif a_vectors and a_vectors.GetName() == key:
                mark_active += " **"
            elif a_tensors and a_tensors.GetName() == key:
                mark_active += " ***"
            out += mark_active.ljust(14) + f': "{key}" ({arr.dtype}), dim={dim}'
            if dim == 1 and len(arr)>0:
                if "int" in arr.dtype.name:
                    rng = f"{arr.min()}, {arr.max()}"
                else:
                    rng = utils.precision(arr.min(), 3) + ", " + utils.precision(arr.max(), 3)
                out += f", range=({rng})\n"
            else:
                out += "\n"

        for key in self.celldata.keys():
            arr = self.celldata[key]
            dim = arr.shape[1] if arr.ndim > 1 else 1
            mark_active = "celldata"
            a_scalars = self.dataset.GetCellData().GetScalars()
            a_vectors = self.dataset.GetCellData().GetVectors()
            a_tensors = self.dataset.GetCellData().GetTensors()
            if   a_scalars and a_scalars.GetName() == key:
                mark_active += " *"
            elif a_vectors and a_vectors.GetName() == key:
                mark_active += " **"
            elif a_tensors and a_tensors.GetName() == key:
                mark_active += " ***"
            out += mark_active.ljust(14) + f': "{key}" ({arr.dtype}), dim={dim}'
            if dim == 1 and len(arr)>0:
                if "int" in arr.dtype.name:
                    rng = f"{arr.min()}, {arr.max()}"
                else:
                    rng = utils.precision(arr.min(), 3) + ", " + utils.precision(arr.max(), 3)
                out += f", range=({rng})\n"
            else:
                out += "\n"

        for key in self.metadata.keys():
            arr = self.metadata[key]
            if len(arr) > 3:
                out+= "metadata".ljust(14) + ": " + f'"{key}" ({len(arr)} values)\n'
            else:
                out+= "metadata".ljust(14) + ": " + f'"{key}" = {arr}\n'

        if self.picked3d is not None:
            idp = self.closest_point(self.picked3d, return_point_id=True)
            idc = self.closest_point(self.picked3d, return_cell_id=True)
            out+= "clicked point".ljust(14) + ": " + utils.precision(self.picked3d, 6)
            out+= f", pointID={idp}, cellID={idc}\n"

        return out.rstrip() + "\x1b[0m"

    def _repr_html_(self):
        """
        HTML representation of the Point cloud object for Jupyter Notebooks.

        Returns:
            HTML text with the image and some properties.
        """
        import io
        import base64
        from PIL import Image

        library_name = "vedo.pointcloud.Points"
        help_url = "https://vedo.embl.es/docs/vedo/pointcloud.html#Points"

        arr = self.thumbnail()
        im = Image.fromarray(arr)
        buffered = io.BytesIO()
        im.save(buffered, format="PNG", quality=100)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = "data:image/png;base64," + encoded
        image = f"<img src='{url}'></img>"

        bounds = "<br/>".join(
            [
                utils.precision(min_x, 4) + " ... " + utils.precision(max_x, 4)
                for min_x, max_x in zip(self.bounds()[::2], self.bounds()[1::2])
            ]
        )
        average_size = "{size:.3f}".format(size=self.average_size())

        help_text = ""
        if self.name:
            help_text += f"<b> {self.name}: &nbsp&nbsp</b>"
        help_text += '<b><a href="' + help_url + '" target="_blank">' + library_name + "</a></b>"
        if self.filename:
            dots = ""
            if len(self.filename) > 30:
                dots = "..."
            help_text += f"<br/><code><i>({dots}{self.filename[-30:]})</i></code>"

        pdata = ""
        if self.dataset.GetPointData().GetScalars():
            if self.dataset.GetPointData().GetScalars().GetName():
                name = self.dataset.GetPointData().GetScalars().GetName()
                pdata = "<tr><td><b> point data array </b></td><td>" + name + "</td></tr>"

        cdata = ""
        if self.dataset.GetCellData().GetScalars():
            if self.dataset.GetCellData().GetScalars().GetName():
                name = self.dataset.GetCellData().GetScalars().GetName()
                cdata = "<tr><td><b> cell data array </b></td><td>" + name + "</td></tr>"

        allt = [
            "<table>",
            "<tr>",
            "<td>",
            image,
            "</td>",
            "<td style='text-align: center; vertical-align: center;'><br/>",
            help_text,
            "<table>",
            "<tr><td><b> bounds </b> <br/> (x/y/z) </td><td>" + str(bounds) + "</td></tr>",
            "<tr><td><b> center of mass </b></td><td>"
            + utils.precision(self.center_of_mass(), 3)
            + "</td></tr>",
            "<tr><td><b> average size </b></td><td>" + str(average_size) + "</td></tr>",
            "<tr><td><b> nr. points </b></td><td>" + str(self.npoints) + "</td></tr>",
            pdata,
            cdata,
            "</table>",
            "</table>",
        ]
        return "\n".join(allt)
    def __add__(self, meshs):
        """
        Add two meshes or a list of meshes together to form an `Assembly` object.
        """
        if isinstance(meshs, list):
            alist = [self]
            for l in meshs:
                if isinstance(l, vedo.Assembly):
                    alist += l.unpack()
                else:
                    alist += l
            return vedo.assembly.Assembly(alist)

        if isinstance(meshs, vedo.Assembly):
            return meshs + self  # use Assembly.__add__

        return vedo.assembly.Assembly([self, meshs])

    def polydata(self):
        """
        Obsolete. Use property `.dataset` instead.
        Returns the underlying `vtkPolyData` object.
        """
        colors.printc(
            "WARNING: call to .polydata() is obsolete, use property .dataset instead.",
            c="y")
        return self.dataset

    def __copy__(self):
        return self.clone(deep=False)

    def __deepcopy__(self, memo):
        return self.clone(deep=memo)

    def copy(self, deep=True) -> Self:
        """Return a copy of the object. Alias of `clone()`."""
        return self.clone(deep=deep)

    def clone(self, deep=True) -> Self:
        """
        Clone a `PointCloud` or `Mesh` object to make an exact copy of it.
        Alias of `copy()`.

        Arguments:
            deep : (bool)
                if False return a shallow copy of the mesh without copying the points array.

        Examples:
            - [mirror.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mirror.py)

               ![](https://vedo.embl.es/images/basic/mirror.png)
        """
        poly = vtki.vtkPolyData()
        if deep or isinstance(deep, dict): # if a memo object is passed this checks as True
            poly.DeepCopy(self.dataset)
        else:
            poly.ShallowCopy(self.dataset)

        if isinstance(self, vedo.Mesh):
            cloned = vedo.Mesh(poly)
        else:
            cloned = Points(poly)
        # print([self], self.__class__)
        # cloned = self.__class__(poly)

        cloned.transform = self.transform.clone()

        cloned.copy_properties_from(self)

        cloned.name = str(self.name)
        cloned.filename = str(self.filename)
        cloned.info = dict(self.info)
        cloned.pipeline = utils.OperationNode("clone", parents=[self], shape="diamond", c="#edede9")

        if isinstance(deep, dict):
            deep[id(self)] = cloned

        return cloned
