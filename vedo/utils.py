#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os
import time

import numpy as np

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
from vtkmodules.util.numpy_support import vtk_to_numpy

import vedo


__docformat__ = "google"

__doc__ = "Utilities submodule."

__all__ = [
    "OperationNode",
    "ProgressBar",
    "progressbar",
    "geometry",
    "extract_cells_by_type",
    "is_sequence",
    "lin_interpolate",
    "vector",
    "mag",
    "mag2",
    "versor",
    "precision",
    "round_to_digit",
    "point_in_triangle",
    "point_line_distance",
    "grep",
    "print_info",
    "make_bands",
    "pack_spheres",
    "spher2cart",
    "cart2spher",
    "cart2cyl",
    "cyl2cart",
    "cyl2spher",
    "spher2cyl",
    "cart2pol",
    "pol2cart",
    "humansort",
    "print_histogram",
    "camera_from_quaternion",
    "camera_from_neuroglancer",
    "oriented_camera",
    "vedo2trimesh",
    "trimesh2vedo",
    "vedo2meshlab",
    "meshlab2vedo",
    "vedo2open3d",
    "open3d2vedo",
    "vtk2numpy",
    "numpy2vtk",
    "get_uv",
]


###########################################################################
class OperationNode:
    """
    Keep track of the operations which led to a final object.
    """
    # https://www.graphviz.org/doc/info/shapes.html#html
    # Mesh     #e9c46a
    # Follower #d9ed92
    # Volume, UGrid #4cc9f0
    # TetMesh  #9e2a2b
    # File     #8a817c
    # Picture  #f28482
    # Assembly #f08080

    def __init__(
        self, 
        operation,
        parents=(),
        comment="",
        shape="none",
        c="#e9c46a",
        style="filled",
    ):
        """
        Keep track of the operations which led to a final object.
        This allows to show the `pipeline` tree for any `vedo` object with e.g.:

        ```python
        from vedo import *
        sp = Sphere()
        sp.clean().subdivide()
        sp.pipeline.show()
        ```
        
        Arguments:
            operation : (str, class)
                descriptor label, if a class is passed then grab its name           
            parents : (list)
                list of the parent classes the object comes from           
            comment : (str)
                a second-line text description      
            shape : (str)
                shape of the frame, check out [this link.](https://graphviz.org/doc/info/shapes.html)     
            c : (hex)
                hex color
            style : (str)
                comma-separated list of styles

        Example:
            ```python
            from vedo.utils import OperationNode

            op_node1 = OperationNode("Operation1", c="lightblue")
            op_node2 = OperationNode("Operation2")
            op_node3 = OperationNode("Operation3", shape='diamond')
            op_node4 = OperationNode("Operation4")
            op_node5 = OperationNode("Operation5")
            op_node6 = OperationNode("Result", c="lightgreen")

            op_node3.add_parent(op_node1)
            op_node4.add_parent(op_node1)
            op_node3.add_parent(op_node2)
            op_node5.add_parent(op_node2)
            op_node6.add_parent(op_node3)
            op_node6.add_parent(op_node5)
            op_node6.add_parent(op_node1)

            op_node6.show(orientation="TB")
            ```
            ![](https://vedo.embl.es/images/feats/operation_node.png)
        """
        if not vedo.settings.enable_pipeline:
            return

        if isinstance(operation, str):
            self.operation = operation
        else:
            self.operation = operation.__class__.__name__
        self.operation_plain = str(self.operation)

        pp = [] # filter out invalid stuff
        for p in parents:
            if hasattr(p, "pipeline"):
                pp.append(p.pipeline)
        self.parents = pp

        if comment:
            self.operation = f"<{self.operation}<BR/><SUB><I>{comment}</I></SUB>>"

        self.dot = None
        self.time = time.time()
        self.shape = shape
        self.style = style
        self.color = c

    def add_parent(self, parent):
        self.parents.append(parent)

    def _build_tree(self, dot):
        dot.node(
            str(id(self)), 
            label=self.operation, 
            shape=self.shape,
            color=self.color,
            style=self.style,
        )
        for parent in self.parents:
            if parent:
                t = f"{self.time - parent.time: .1f}s"
                dot.edge(str(id(parent)), str(id(self)), label=t)
                parent._build_tree(dot)

    def __repr__(self): 
        try:      
            from treelib import Tree
        except ImportError:
            vedo.logger.error("To use this functionality please install treelib:"
                              "\n pip install treelib")
            return ""

        def _build_tree(parent):
            for par in parent.parents:
                if par:
                    op = par.operation_plain
                    tree.create_node(
                        op, 
                        op + str(par.time), 
                        parent=parent.operation_plain + str(parent.time))
                    _build_tree(par)

        tree = Tree()
        tree.create_node(
            self.operation_plain, self.operation_plain + str(self.time)
        )
        _build_tree(self)
        return tree.show(reverse=True, stdout=False)

    def show(self, orientation="LR", popup=True):
        """Show the graphviz output for the pipeline of this object"""
        if not vedo.settings.enable_pipeline:
            return
        
        try:
            from graphviz import Digraph
        except ImportError:
            vedo.logger.error("please install graphviz with command\n pip install graphviz")
            return

        # visualize the entire tree
        dot = Digraph(
            node_attr={
                'fontcolor':'#201010',
                'fontname': "Helvetica",
                'fontsize': '12',
        },
            edge_attr={ 
                'fontname': "Helvetica",
                'fontsize': '6',
                'arrowsize': '0.4',
            }
        )
        dot.attr(rankdir=orientation)
        
        self.counts = 0
        self._build_tree(dot)
        self.dot = dot
        dot.render('.vedo_pipeline_graphviz', view=popup)


###########################################################################
class ProgressBar:
    """
    Class to print a progress bar.
    """
    def __init__(
        self,
        start,
        stop,
        step=1,
        c=None,
        bold=True,
        italic=False,
        title="",
        eta=True,
        delay=0,
        width=25,
        char="\U00002501",
        char_back="\U00002500",
    ):
        """
        Class to print a progress bar with optional text message.

        Check out also function `progressbar()`.

        Example:
            ```python
            import time
            pb = ProgressBar(0,400, c='red')
            for i in pb.range():
                time.sleep(0.1)
                pb.print('some message')
            ```
            ![](https://user-images.githubusercontent.com/32848391/51858823-ed1f4880-2335-11e9-8788-2d102ace2578.png)
        """
        self.char = char
        self.char_back = char_back

        self.title = title + " "
        if title:
            self.title = " " + self.title

        self.start = start
        self.stop = stop
        self.step = step

        self.color = c
        self.bold = bold
        self.italic = italic
        self.width = width
        self.pbar = ""
        self.percent = 0.0
        self.percent_int = 0
        self.eta = eta
        self.delay = delay

        self.t0 = time.time()
        self._remaining = 1e10

        self._update(0)

        self._counts = 0
        self._oldbar = ""
        self._lentxt = 0
        self._range = np.arange(start, stop, step)

    def print(self, txt="", c=None):
        """Print the progress bar with an optional message."""
        if not c:
            c = self.color
        
        self._update(self._counts + self.step)

        if self.delay:
            if time.time() - self.t0 < self.delay:
                return

        if self.pbar != self._oldbar:
            self._oldbar = self.pbar

            if self.eta and self._counts > 1:

                tdenom = time.time() - self.t0
                if tdenom:
                    vel = self._counts / tdenom
                    self._remaining = (self.stop - self._counts) / vel
                else:
                    vel = 1
                    self._remaining = 0.0

                if self._remaining > 60:
                    mins = int(self._remaining / 60)
                    secs = self._remaining - 60 * mins
                    mins = f"{mins}m"
                    secs = f"{int(secs + 0.5)}s "
                else:
                    mins = ""
                    secs = f"{int(self._remaining + 0.5)}s "

                vel = round(vel, 1)
                eta = f"eta: {mins}{secs}({vel} it/s) "
                if self._remaining < 0.5:
                    dt = time.time() - self.t0
                    if dt > 60:
                        mins = int(dt / 60)
                        secs = dt - 60 * mins
                        mins = f"{mins}m"
                        secs = f"{int(secs + 0.5)}s "
                    else:
                        mins = ""
                        secs = f"{int(dt + 0.5)}s "
                    eta = f"elapsed: {mins}{secs}({vel} it/s)        "
                    txt = ""
            else:
                eta = ""

            eraser = " " * self._lentxt + "\b" * self._lentxt

            s = f"{self.pbar} {eraser}{eta}{txt}\r"
            vedo.printc(s, c=c, bold=self.bold, italic=self.italic, end="")
            if self.percent > 99.999:
                print("")

            self._lentxt = len(txt)

    def range(self):
        """Return the range iterator."""
        return self._range

    def _update(self, counts):
        if counts < self.start:
            counts = self.start
        elif counts > self.stop:
            counts = self.stop
        self._counts = counts

        self.percent = (self._counts - self.start) * 100.0

        delta = self.stop - self.start
        if delta:
            self.percent /= delta
        else:
            self.percent = 0.0

        self.percent_int = int(round(self.percent))
        af = self.width - 2
        nh = int(round(self.percent_int / 100 * af))
        pbar_background = "\x1b[2m" + self.char_back * (af - nh)
        self.pbar = f"{self.title}{self.char * (nh-1)}{pbar_background}"
        if self.percent < 100.0:
            ps = f" {self.percent_int}%"
        else:
            ps = ""
        self.pbar += ps

#####################################
def progressbar(
        iterable,
        c=None,
        bold=True,
        italic=False,
        title="",
        eta=True,
        width=25,
        delay=0,
    ):
    """
    Function to print a progress bar with optional text message.

    Example:
        ```python
        import time
        for i in progressbar(range(100), c='red'):
            time.sleep(0.1)
        ```
        ![](https://user-images.githubusercontent.com/32848391/51858823-ed1f4880-2335-11e9-8788-2d102ace2578.png)
    """
    try:
        if is_number(iterable):
            total = int(iterable)
            iterable = range(total)
        else:
            total = len(iterable)
    except TypeError: 
        iterable = list(iterable)
        total = len(iterable)

    pb = ProgressBar(
        0, total, 
        c=c, bold=bold, italic=italic, title=title, eta=eta, delay=delay, width=width,
    )
    for item in iterable:
        pb.print()
        yield item

###########################################################
def numpy2vtk(arr, dtype=None, deep=True, name=""):
    """
    Convert a numpy array into a `vtkDataArray`.
    Use `dtype='id'` for `vtkIdTypeArray` objects.
    """
    # https://github.com/Kitware/VTK/blob/master/Wrapping/Python/vtkmodules/util/numpy_support.py
    if arr is None:
        return None

    arr = np.ascontiguousarray(arr)

    if dtype == "id":
        varr = numpy_to_vtkIdTypeArray(arr.astype(np.int64), deep=deep)
    elif dtype:
        varr = numpy_to_vtk(arr.astype(dtype), deep=deep)
    else:
        # let numpy_to_vtk() decide what is best type based on arr type
        varr = numpy_to_vtk(arr, deep=deep)

    if name:
        varr.SetName(name)
    return varr


def vtk2numpy(varr):
    """Convert a `vtkDataArray`, `vtkIdList` or `vtTransform` into a numpy array."""
    if isinstance(varr, vtk.vtkIdList):
        return np.array([varr.GetId(i) for i in range(varr.GetNumberOfIds())])
    elif isinstance(varr, vtk.vtkBitArray):
        carr = vtk.vtkCharArray()
        carr.DeepCopy(varr)
        varr = carr
    elif isinstance(varr, vtk.vtkHomogeneousTransform):
        try:
            varr = varr.GetMatrix()
        except AttributeError:
            pass
        n = 4
        M = [[varr.GetElement(i, j) for j in range(n)] for i in range(n)]
        return np.array(M)

    return vtk_to_numpy(varr)


def make3d(pts, transpose=False):
    """
    Make an array which might be 2D to 3D.

    Array can also be in the form `[allx, ally, allz]`.
    Use `transpose` to resolve ambiguous cases (eg, shapes like `[3,3]`).
    """
    pts = np.asarray(pts)

    if pts.dtype == "object":
        raise ValueError("Cannot form a valid numpy array, input may be non-homogenous")

    if pts.shape[0] == 0:  # empty list
        return pts

    if pts.ndim == 1:
        if pts.shape[0] == 2:
            return np.hstack([pts, [0]]).astype(pts.dtype)
        elif pts.shape[0] == 3:
            return pts
        else:
            raise ValueError

    if pts.shape[1] == 3:
        return pts

    if transpose or (2 <= pts.shape[0] <= 3 and pts.shape[1] > 3):
        pts = pts.T

    if pts.shape[1] == 2:
        return np.c_[pts, np.zeros(pts.shape[0], dtype=pts.dtype)]

    if pts.shape[1] != 3:
        raise ValueError("input shape is not supported.")
    return pts


def geometry(obj, extent=None):
    """
    Apply the `vtkGeometryFilter` to the input object.
    This is a general-purpose filter to extract geometry (and associated data)
    from any type of dataset.
    This filter also may be used to convert any type of data to polygonal type.
    The conversion process may be less than satisfactory for some 3D datasets.
    For example, this filter will extract the outer surface of a volume
    or structured grid dataset.

    Returns a `vedo.Mesh` object.

    Set `extent` as the `[xmin,xmax, ymin,ymax, zmin,zmax]` bounding box to clip data.
    """
    gf = vtk.vtkGeometryFilter()
    gf.SetInputData(obj)
    if extent is not None:
        gf.SetExtent(extent)
    gf.Update()
    return vedo.Mesh(gf.GetOutput())


def extract_cells_by_type(obj, types=()):
    """
    Extract cells of a specified type from a vtk dataset.

    Given an input `vtkDataSet` and a list of cell types, produce an output
    containing only cells of the specified type(s).
    
    Find [here](https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html)
    the list of possible cell types.

    Return:
        a `vtkDataSet` object which can be wrapped.
    """
    ef = vtk.vtkExtractCellsByType()
    try:
        ef.SetInputData(obj._data)
    except:
        ef.SetInputData(obj)

    for ct in types:
        ef.AddCellType(ct)
    ef.Update()
    return ef.GetOutput()


def buildPolyData(vertices, faces=None, lines=None, index_offset=0, tetras=False):
    """
    Build a `vtkPolyData` object from a list of vertices
    where faces represents the connectivity of the polygonal mesh.

    E.g. :
        - `vertices=[[x1,y1,z1],[x2,y2,z2], ...]`
        - `faces=[[0,1,2], [1,2,3], ...]`
        - `lines=[[0,1], [1,2,3,4], ...]`

    Use `index_offset=1` if face numbering starts from 1 instead of 0.

    If `tetras=True`, interpret 4-point faces as tetrahedrons instead of surface quads.
    """
    poly = vtk.vtkPolyData()

    if len(vertices) == 0:
        return poly

    if not is_sequence(vertices[0]):
        return poly

    vertices = make3d(vertices)

    source_points = vtk.vtkPoints()
    source_points.SetData(numpy2vtk(vertices, dtype=np.float32))
    poly.SetPoints(source_points)

    if lines is not None:
        # Create a cell array to store the lines in and add the lines to it
        linesarr = vtk.vtkCellArray()
        if is_sequence(lines[0]):  # assume format [(id0,id1),..]
            for iline in lines:
                for i in range(0, len(iline) - 1):
                    i1, i2 = iline[i], iline[i + 1]
                    if i1 != i2:
                        vline = vtk.vtkLine()
                        vline.GetPointIds().SetId(0, i1)
                        vline.GetPointIds().SetId(1, i2)
                        linesarr.InsertNextCell(vline)
        else:  # assume format [id0,id1,...]
            for i in range(0, len(lines) - 1):
                vline = vtk.vtkLine()
                vline.GetPointIds().SetId(0, lines[i])
                vline.GetPointIds().SetId(1, lines[i + 1])
                linesarr.InsertNextCell(vline)
            # print('Wrong format for lines in utils.buildPolydata(), skip.')
        poly.SetLines(linesarr)

    if faces is None:
        source_vertices = vtk.vtkCellArray()
        for i in range(len(vertices)):
            source_vertices.InsertNextCell(1)
            source_vertices.InsertCellPoint(i)
        poly.SetVerts(source_vertices)
        return poly  ###################

    # faces exist
    source_polygons = vtk.vtkCellArray()

    if isinstance(faces, np.ndarray) or not is_ragged(faces):
        ##### all faces are composed of equal nr of vtxs, FAST
        faces = np.asarray(faces)
        ast = np.int32
        if vtk.vtkIdTypeArray().GetDataTypeSize() != 4:
            ast = np.int64

        if faces.ndim > 1:
            nf, nc = faces.shape
            hs = np.hstack((np.zeros(nf)[:, None] + nc, faces)).astype(ast).ravel()
            arr = numpy_to_vtkIdTypeArray(hs, deep=True)
            source_polygons.SetCells(nf, arr)

    else:
        ############################# manually add faces, SLOW

        showbar = False
        if len(faces) > 25000:
            showbar = True
            pb = ProgressBar(0, len(faces), eta=False)

        for f in faces:
            n = len(f)

            if n == 3:
                ele = vtk.vtkTriangle()
                pids = ele.GetPointIds()
                for i in range(3):
                    pids.SetId(i, f[i] - index_offset)
                source_polygons.InsertNextCell(ele)

            elif n == 4 and tetras:
                # do not use vtkTetra() because it fails
                # with dolfin faces orientation
                ele0 = vtk.vtkTriangle()
                ele1 = vtk.vtkTriangle()
                ele2 = vtk.vtkTriangle()
                ele3 = vtk.vtkTriangle()
                if index_offset:
                    for i in [0, 1, 2, 3]:
                        f[i] -= index_offset
                f0, f1, f2, f3 = f
                pid0 = ele0.GetPointIds()
                pid1 = ele1.GetPointIds()
                pid2 = ele2.GetPointIds()
                pid3 = ele3.GetPointIds()

                pid0.SetId(0, f0)
                pid0.SetId(1, f1)
                pid0.SetId(2, f2)

                pid1.SetId(0, f0)
                pid1.SetId(1, f1)
                pid1.SetId(2, f3)

                pid2.SetId(0, f1)
                pid2.SetId(1, f2)
                pid2.SetId(2, f3)

                pid3.SetId(0, f2)
                pid3.SetId(1, f3)
                pid3.SetId(2, f0)

                source_polygons.InsertNextCell(ele0)
                source_polygons.InsertNextCell(ele1)
                source_polygons.InsertNextCell(ele2)
                source_polygons.InsertNextCell(ele3)

            else:
                ele = vtk.vtkPolygon()
                pids = ele.GetPointIds()
                pids.SetNumberOfIds(n)
                for i in range(n):
                    pids.SetId(i, f[i] - index_offset)
                source_polygons.InsertNextCell(ele)
            if showbar:
                pb.print("converting mesh...    ")

    poly.SetPolys(source_polygons)
    return poly


##############################################################################
def get_font_path(font):
    """Internal use."""
    if font in vedo.settings.font_parameters.keys():
        if vedo.settings.font_parameters[font]["islocal"]:
            fl = os.path.join(vedo.fonts_path, f"{font}.ttf")
        else:
            try:
                fl = vedo.io.download(f"https://vedo.embl.es/fonts/{font}.ttf", verbose=False)
            except:
                vedo.logger.warning(f"Could not download https://vedo.embl.es/fonts/{font}.ttf")
                fl = os.path.join(vedo.fonts_path, "Normografo.ttf")
    else:
        if font.startswith("https://"):
            fl = vedo.io.download(font, verbose=False)
        elif os.path.isfile(font):
            fl = font  # assume user is passing a valid file
        else:
            if font.endswith(".ttf"):
                vedo.logger.error(
                    f"Could not set font file {font}"
                    f"-> using default: {vedo.settings.default_font}"
                )
            else:
                vedo.settings.default_font = "Normografo"
                vedo.logger.error(
                    f"Could not set font name {font}"
                    f" -> using default: Normografo\n"
                    f"Check out https://vedo.embl.es/fonts for additional fonts\n"
                    f"Type 'vedo -r fonts' to see available fonts"
                )
            fl = get_font_path(vedo.settings.default_font)
    return fl


def isSequence(arg):
    "Deprecated. Please use `is_sequence()`"
    m = "Warning! isSequence() is deprecated. Please use is_sequence()."
    print("\x1b[1m\x1b[33;1m " + m + "\x1b[0m")
    return is_sequence(arg)


def is_sequence(arg):
    """Check if the input is iterable."""
    if hasattr(arg, "strip"):
        return False
    if hasattr(arg, "__getslice__"):
        return True
    if hasattr(arg, "__iter__"):
        return True
    return False


def is_ragged(arr, deep=False):
    """
    A ragged array in Python is an array with arrays of different
    lengths as its elements. To check if an array is ragged,
    we iterate through the elements and check if their lengths are the same.

    Example:
    ```python
    arr = [[1, 2, 3], [[4, 5], [6], 1], [7, 8, 9]]
    print(is_ragged(arr, deep=True))  # output: True
    ```
    """
    if is_sequence(arr[0]):
        length = len(arr[0])
        for i in range(1, len(arr)):
            if len(arr[i]) != length or (deep and is_ragged(arr[i])):
                return True
        return False
    return False


def flatten(list_to_flatten):
    """Flatten out a list."""

    def _genflatten(lst):
        for elem in lst:
            if isinstance(elem, (list, tuple)):
                for x in flatten(elem):
                    yield x
            else:
                yield elem

    return list(_genflatten(list_to_flatten))


def humansort(alist):
    """
    Sort in place a given list the way humans expect.

    E.g. `['file11', 'file1'] -> ['file1', 'file11']`

    .. warning:: input list is modified in-place by this function.
    """
    import re

    def alphanum_key(s):
        # Turn a string into a list of string and number chunks.
        # e.g. "z23a" -> ["z", 23, "a"]
        def tryint(s):
            if s.isdigit():
                return int(s)
            return s

        return [tryint(c) for c in re.split("([0-9]+)", s)]

    alist.sort(key=alphanum_key)
    return alist  # NB: input list is modified


def sort_by_column(arr, nth, invert=False):
    """Sort a numpy array by its `n-th` column."""
    arr = np.asarray(arr)
    arr = arr[arr[:, nth].argsort()]
    if invert:
        return np.flip(arr, axis=0)
    return arr


def point_in_triangle(p, p1, p2, p3):
    """
    Return True if a point is inside (or above/below) a triangle defined by 3 points in space.
    """
    p1 = np.array(p1)
    u = p2 - p1
    v = p3 - p1
    n = np.cross(u, v)
    w = p - p1
    ln = np.dot(n, n)
    if not ln:
        return None  # degenerate triangle
    gamma = (np.dot(np.cross(u, w), n)) / ln
    if 0 < gamma < 1:
        beta = (np.dot(np.cross(w, v), n)) / ln
        if 0 < beta < 1:
            alpha = 1 - gamma - beta
            if 0 < alpha < 1:
                return True
    return False


def intersection_ray_triangle(P0, P1, V0, V1, V2):
    """
    Fast intersection between a directional ray defined by `P0,P1`
    and triangle `V0, V1, V2`.

    Returns the intersection point or
    - `None` if triangle is degenerate, or ray is  parallel to triangle plane.
    - `False` if no intersection, or ray direction points away from triangle.
    """
    # Credits: http://geomalgorithms.com/a06-_intersect-2.html
    # Get triangle edge vectors and plane normal
    # todo : this is slow should check
    # https://vtk.org/doc/nightly/html/classvtkCell.html#aa850382213d7b8693f0eeec0209c347b
    V0 = np.asarray(V0, dtype=float)
    P0 = np.asarray(P0, dtype=float)
    u = V1 - V0
    v = V2 - V0
    n = np.cross(u, v)
    if not np.abs(v).sum():  # triangle is degenerate
        return None  # do not deal with this case

    rd = P1 - P0  # ray direction vector
    w0 = P0 - V0
    a = -np.dot(n, w0)
    b = np.dot(n, rd)
    if not b:  # ray is  parallel to triangle plane
        return None

    # Get intersect point of ray with triangle plane
    r = a / b
    if r < 0.0:  # ray goes away from triangle
        return False  #  => no intersect

    # Gor a segment, also test if (r > 1.0) => no intersect
    I = P0 + r * rd  # intersect point of ray and plane

    # is I inside T?
    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    w = I - V0
    wu = np.dot(w, u)
    wv = np.dot(w, v)
    D = uv * uv - uu * vv

    # Get and test parametric coords
    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:  # I is outside T
        return False
    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:  # I is outside T
        return False
    return I  # I is in T

def triangle_solver(**input_dict):
    """
    Solve a triangle from any 3 known elements.
    (Note that there might be more than one solution or none).
    Angles are in radians.

    Example:
    ```python
    print(triangle_solver(a=3, b=4, c=5))
    print(triangle_solver(a=3, ac=0.9273, ab=1.5716))
    print(triangle_solver(a=3, b=4, ab=1.5716))
    print(triangle_solver(b=4, bc=.64, ab=1.5716))
    print(triangle_solver(c=5, ac=.9273, bc=0.6435))
    print(triangle_solver(a=3, c=5, bc=0.6435))
    print(triangle_solver(b=4, c=5, ac=0.927))
    ```
    """
    a = input_dict.get("a")
    b = input_dict.get("b")
    c = input_dict.get("c")
    ab = input_dict.get("ab")
    bc = input_dict.get("bc")
    ac = input_dict.get("ac")

    if ab and bc:
        ac = np.pi - bc - ab
    elif bc and ac:
        ab = np.pi - bc - ac
    elif ab and ac:
        bc = np.pi - ab - ac

    if a is not None and b is not None and c is not None:
        ab = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
        sinab = np.sin(ab)
        ac = np.arcsin(a / c * sinab)
        bc = np.arcsin(b / c * sinab)

    elif a is not None and b is not None and ab is not None:
        c = np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(ab))
        sinab = np.sin(ab)
        ac = np.arcsin(a / c * sinab)
        bc = np.arcsin(b / c * sinab)

    elif a is not None and ac is not None and ab is not None:
        h = a * np.sin(ac)
        b = h / np.sin(bc)
        c = b * np.cos(bc) + a * np.cos(ac)

    elif b is not None and bc is not None and ab is not None:
        h = b * np.sin(bc)
        a = h / np.sin(ac)
        c = np.sqrt(a * a + b * b)

    elif c is not None and ac is not None and bc is not None:
        h = c * np.sin(bc)
        b1 = c * np.cos(bc)
        b2 = h / np.tan(ab)
        b = b1 + b2
        a = np.sqrt(b2 * b2 + h * h)

    elif a is not None and c is not None and bc is not None:
        # double solution
        h = c * np.sin(bc)
        k = np.sqrt(a * a - h * h)
        omega = np.arcsin(k / a)
        cosbc = np.cos(bc)
        b = c * cosbc - k
        phi = np.pi / 2 - bc - omega
        ac = phi
        ab = np.pi - ac - bc
        if k:
            b2 = c * cosbc + k
            ac2 = phi + 2 * omega
            ab2 = np.pi - ac2 - bc
            return [
                {"a": a, "b": b,  "c": c, "ab": ab,  "bc": bc, "ac": ac},
                {"a": a, "b": b2, "c": c, "ab": ab2, "bc": bc, "ac": ac2},
            ]

    elif b is not None and c is not None and ac is not None:
        # double solution
        h = c * np.sin(ac)
        k = np.sqrt(b * b - h * h)
        omega = np.arcsin(k / b)
        cosac = np.cos(ac)
        a = c * cosac - k
        phi = np.pi / 2 - ac - omega
        bc = phi
        ab = np.pi - bc - ac
        if k:
            a2 = c * cosac + k
            bc2 = phi + 2 * omega
            ab2 = np.pi - ac - bc2
            return [
                {"a": a,  "b": b, "c": c, "ab": ab,  "bc": bc,  "ac": ac},
                {"a": a2, "b": b, "c": c, "ab": ab2, "bc": bc2, "ac": ac},
            ]

    else:
        vedo.logger.error(f"Case {input_dict} is not supported.")
        return []

    return [{"a": a, "b": b, "c": c, "ab": ab, "bc": bc, "ac": ac}]


def point_line_distance(p, p1, p2):
    """
    Compute the distance of a point to a line (not the segment)
    defined by `p1` and `p2`.
    """
    d = np.sqrt(vtk.vtkLine.DistanceToLine(p, p1, p2))
    return d


def linInterpolate(x, rangeX, rangeY):
    "Deprecated. Please `lin_interpolate()`"
    m = "Warning! linInterpolate() is deprecated. Please use lin_interpolate()"
    print("\x1b[1m\x1b[33;1m " + m + "\x1b[0m")
    return lin_interpolate(x, rangeX, rangeY)


def lin_interpolate(x, rangeX, rangeY):
    """
    Interpolate linearly the variable `x` in `rangeX` onto the new `rangeY`.
    If `x` is a 3D vector the linear weight is the distance to the two 3D `rangeX` vectors.

    E.g. if `x` runs in `rangeX=[x0,x1]` and I want it to run in `rangeY=[y0,y1]` then

    `y = lin_interpolate(x, rangeX, rangeY)` will interpolate `x` onto `rangeY`.

    Examples:
        - [lin_interpolate.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/lin_interpolate.py)

            ![](https://vedo.embl.es/images/basic/linInterpolate.png)
    """
    if is_sequence(x):
        x = np.asarray(x)
        x0, x1 = np.asarray(rangeX)
        y0, y1 = np.asarray(rangeY)
        dx = x1 - x0
        dxn = np.linalg.norm(dx)
        if not dxn:
            return y0
        s = np.linalg.norm(x - x0) / dxn
        t = np.linalg.norm(x - x1) / dxn
        st = s + t
        out = y0 * (t / st) + y1 * (s / st)

    else:  # faster

        x0 = rangeX[0]
        dx = rangeX[1] - x0
        if not dx:
            return rangeY[0]
        s = (x - x0) / dx
        out = rangeY[0] * (1 - s) + rangeY[1] * s
    return out


def get_uv(p, x, v):
    """
    Obtain the texture uv-coords of a point p belonging to a face that has point
    coordinates (x0, x1, x2) with the corresponding uv-coordinates v=(v0, v1, v2).
    All p and x0,x1,x2 are 3D-vectors, while v are their 2D uv-coordinates.

    Example:
        ```python
        from vedo import *

        pic = Picture(dataurl+"coloured_cube_faces.jpg")
        cb = Mesh(dataurl+"coloured_cube.obj").lighting("off").texture(pic)

        cbpts = cb.points()
        faces = cb.faces()
        uv = cb.pointdata["Material"]

        pt = [-0.2, 0.75, 2]
        pr = cb.closest_point(pt)

        idface = cb.closest_point(pt, return_cell_id=True)
        idpts = faces[idface]
        uv_face = uv[idpts]

        uv_pr = utils.get_uv(pr, cbpts[idpts], uv_face)
        print("interpolated uv =", uv_pr)

        sx, sy = pic.dimensions()
        i_interp_uv = uv_pr * [sy, sx]
        ix, iy = i_interp_uv.astype(int)
        mpic = pic.tomesh()
        rgba = mpic.pointdata["RGBA"].reshape(sy, sx, 3)
        print("color =", rgba[ix, iy])

        show(
            [[cb, Point(pr), cb.labels("Material")],
                [pic, Point(i_interp_uv)]],
            N=2, axes=1, sharecam=False,
        ).close()
        ```
        ![](https://vedo.embl.es/images/feats/utils_get_uv.png)
    """
    # Vector vp=p-x0 is representable as alpha*s + beta*t,
    # where s = x1-x0 and t = x2-x0, in matrix form
    # vp = [alpha, beta] . matrix(s,t)
    # M = matrix(s,t) is 2x3 matrix, so (alpha, beta) can be found by
    # inverting any of its minor A with non-zero determinant.
    # Once found, uv-coords of p are vt0 + alpha (vt1-v0) + beta (vt2-v0)

    p = np.asarray(p)
    x0, x1, x2 = np.asarray(x)[:3]
    vt0, vt1, vt2 = np.asarray(v)[:3]

    s = x1 - x0
    t = x2 - x0
    vs = vt1 - vt0
    vt = vt2 - vt0
    vp = p - x0

    # finding a minor with independent rows
    M = np.matrix([s, t])
    mnr = [0, 1]
    A = M[:, mnr]
    if np.abs(np.linalg.det(A)) < 0.000001:
        mnr = [0, 2]
        A = M[:, mnr]
        if np.abs(np.linalg.det(A)) < 0.000001:
            mnr = [1, 2]
            A = M[:, mnr]
    Ainv = np.linalg.inv(A)
    alpha_beta = vp[mnr].dot(Ainv)  # [alpha, beta]
    return np.asarray(vt0 + alpha_beta.dot(np.matrix([vs, vt])))[0]


def vector(x, y=None, z=0.0, dtype=np.float64):
    """
    Return a 3D numpy array representing a vector.

    If `y` is `None`, assume input is already in the form `[x,y,z]`.
    """
    if y is None:  # assume x is already [x,y,z]
        return np.asarray(x, dtype=dtype)
    return np.array([x, y, z], dtype=dtype)


def versor(x, y=None, z=0.0, dtype=np.float64):
    """Return the unit vector. Input can be a list of vectors."""
    v = vector(x, y, z, dtype)
    if isinstance(v[0], np.ndarray):
        return np.divide(v, mag(v)[:, None])
    return v / mag(v)


def mag(v):
    """Get the magnitude of a vector or array of vectors."""
    v = np.asarray(v)
    if v.ndim == 1:
        return np.linalg.norm(v)
    return np.linalg.norm(v, axis=1)


def mag2(v):
    """Get the squared magnitude of a vector or array of vectors."""
    v = np.asarray(v)
    if v.ndim == 1:
        return np.square(v).sum()
    return np.square(v).sum(axis=1)


def is_integer(n):
    """Check if input is an integer."""
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def is_number(n):
    """Check if input is a number"""
    try:
        float(n)
        return True
    except ValueError:
        return False


def round_to_digit(x, p):
    """Round a real number to the specified number of significant digits."""
    if not x:
        return 0
    r = np.round(x, p - int(np.floor(np.log10(abs(x)))) - 1)
    if int(r) == r:
        return int(r)
    return r


def packSpheres(bounds, radius):
    "Deprecated. Please use `pack_spheres()`"
    m = "Warning! packSpheres() is deprecated. Please use pack_spheres()"
    print("\x1b[1m\x1b[33;1m " + m + "\x1b[0m")
    return pack_spheres(bounds, radius)


def pack_spheres(bounds, radius):
    """
    Packing spheres into a bounding box.
    Returns a numpy array of sphere centers.
    """
    h = 0.8164965 / 2
    d = 0.8660254
    a = 0.288675135

    if is_sequence(bounds):
        x0, x1, y0, y1, z0, z1 = bounds
    else:
        x0, x1, y0, y1, z0, z1 = bounds.bounds()

    x = np.arange(x0, x1, radius)
    nul = np.zeros_like(x)
    nz = int((z1 - z0) / radius / h / 2 + 1.5)
    ny = int((y1 - y0) / radius / d + 1.5)

    pts = []
    for iz in range(nz):
        z = z0 + nul + iz * h * radius
        dx, dy, dz = [radius * 0.5, radius * a, iz * h * radius]
        for iy in range(ny):
            y = y0 + nul + iy * d * radius
            if iy % 2:
                xs = x
            else:
                xs = x + radius * 0.5
            if iz % 2:
                p = np.c_[xs, y, z] + [dx, dy, dz]
            else:
                p = np.c_[xs, y, z] + [0, 0, dz]
            pts += p.tolist()
    return np.array(pts)


def precision(x, p, vrange=None, delimiter="e"):
    """
    Returns a string representation of `x` formatted to precision `p`.

    Set `vrange` to the range in which x exists (to snap x to '0' if below precision).
    """
    # Based on the webkit javascript implementation
    # `from here <https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp>`_,
    # and implemented by `randlet <https://github.com/randlet/to-precision>`_.
    # Modified for vedo by M.Musy 2020

    if isinstance(x, str):  # do nothing
        return x

    if is_sequence(x):
        out = "("
        nn = len(x) - 1
        for i, ix in enumerate(x):

            try:
                if np.isnan(ix):
                    return "NaN"
            except:
                # cannot handle list of list
                continue

            out += precision(ix, p)
            if i < nn:
                out += ", "
        return out + ")"  ############ <--

    if np.isnan(x):
        return "NaN"

    x = float(x)

    if x == 0.0 or (vrange is not None and abs(x) < vrange / pow(10, p)):
        return "0"

    out = []
    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x / tens)

    if n < math.pow(10, p - 1):
        e = e - 1
        tens = math.pow(10, e - p + 1)
        n = math.floor(x / tens)

    if abs((n + 1.0) * tens - x) <= abs(n * tens - x):
        n = n + 1

    if n >= math.pow(10, p):
        n = n / 10.0
        e = e + 1

    m = "%.*g" % (p, n)
    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append(delimiter)
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p - 1):
        out.append(m)
    elif e >= 0:
        out.append(m[: e + 1])
        if e + 1 < len(m):
            out.append(".")
            out.extend(m[e + 1 :])
    else:
        out.append("0.")
        out.extend(["0"] * -(e + 1))
        out.append(m)
    return "".join(out)


##################################################################################
# 2d ######
def cart2pol(x, y):
    """2D Cartesian to Polar coordinates conversion."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return np.array([rho, theta])


def pol2cart(rho, theta):
    """2D Polar to Cartesian coordinates conversion."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x, y])


# 3d ######
def cart2spher(x, y, z):
    """3D Cartesian to Spherical coordinate conversion."""
    hxy = np.hypot(x, y)
    rho = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    return np.array([rho, theta, phi])


def spher2cart(rho, theta, phi):
    """3D Spherical to Cartesian coordinate conversion."""
    st = np.sin(theta)
    sp = np.sin(phi)
    ct = np.cos(theta)
    cp = np.cos(phi)
    rst = rho * st
    x = rst * cp
    y = rst * sp
    z = rho * ct
    return np.array([x, y, z])


def cart2cyl(x, y, z):
    """3D Cartesian to Cylindrical coordinate conversion."""
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return np.array([rho, theta, z])


def cyl2cart(rho, theta, z):
    """3D Cylindrical to Cartesian coordinate conversion."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x, y, z])


def cyl2spher(rho, theta, z):
    """3D Cylindrical to Spherical coordinate conversion."""
    rhos = np.sqrt(rho * rho + z * z)
    phi = np.arctan2(rho, z)
    return np.array([rhos, phi, theta])


def spher2cyl(rho, theta, phi):
    """3D Spherical to Cylindrical coordinate conversion."""
    rhoc = rho * np.sin(theta)
    z = rho * np.cos(theta)
    return np.array([rhoc, phi, z])


##################################################################################
def grep(filename, tag, first_occurrence_only=False):
    """Greps the line in a file that starts with a specific `tag` string inside the file."""
    import re

    with open(filename, "r", encoding="UTF-8") as afile:
        content = []
        for line in afile:
            if re.search(tag, line):
                c = line.split()
                c[-1] = c[-1].replace("\n", "")
                content.append(c)
                if first_occurrence_only:
                    break
    return content


def print_info(obj):
    """Print information about a `vedo` object."""

    def _print_data(poly, mapper, c):
        ptdata = poly.GetPointData()
        cldata = poly.GetCellData()
        fldata = poly.GetFieldData()
        if ptdata.GetNumberOfArrays() + cldata.GetNumberOfArrays():
            arrtypes = {}
            arrtypes[vtk.VTK_UNSIGNED_CHAR] = ("UNSIGNED_CHAR",  "np.uint8")
            arrtypes[vtk.VTK_UNSIGNED_SHORT]= ("UNSIGNED_SHORT", "np.uint16")
            arrtypes[vtk.VTK_UNSIGNED_INT]  = ("UNSIGNED_INT",   "np.uint32")
            arrtypes[vtk.VTK_UNSIGNED_LONG_LONG] = ("UNSIGNED_LONG_LONG", "np.uint64")
            arrtypes[vtk.VTK_CHAR]          = ("CHAR",           "np.int8")
            arrtypes[vtk.VTK_SHORT]         = ("SHORT",          "np.int16")
            arrtypes[vtk.VTK_INT]           = ("INT",            "np.int32")
            arrtypes[vtk.VTK_LONG]          = ("LONG",           "") # ??
            arrtypes[vtk.VTK_LONG_LONG]     = ("LONG_LONG",      "np.int64")
            arrtypes[vtk.VTK_FLOAT]         = ("FLOAT",          "np.float32")
            arrtypes[vtk.VTK_DOUBLE]        = ("DOUBLE",         "np.float64")
            arrtypes[vtk.VTK_SIGNED_CHAR]   = ("SIGNED_CHAR",    "np.int8")
            arrtypes[vtk.VTK_ID_TYPE]       = ("ID",             "np.int64")

            if ptdata.GetScalars():
                vedo.printc("active array".ljust(14)+": ", c=c, bold=True, end="")
                vedo.printc(ptdata.GetScalars().GetName(), "(pointdata)  ", c=c, bold=False)

            if cldata.GetScalars():
                vedo.printc("active array".ljust(14)+": ", c=c, bold=True, end="")
                vedo.printc(cldata.GetScalars().GetName(), "(celldata)", c=c, bold=False)

            for i in range(ptdata.GetNumberOfArrays()):
                name = ptdata.GetArrayName(i)
                if name and ptdata.GetArray(i):
                    vedo.printc("pointdata".ljust(14) + ": ", c=c, bold=True, end="")
                    try:
                        tt, _ = arrtypes[ptdata.GetArray(i).GetDataType()]
                    except:
                        tt = "VTKTYPE" + str(ptdata.GetArray(i).GetDataType())
                    ncomp = ptdata.GetArray(i).GetNumberOfComponents()
                    rng = ptdata.GetArray(i).GetRange()
                    vedo.printc(f'"{name}" ({ncomp} {tt}),', c=c, bold=False, end="")
                    vedo.printc(
                        " range=(" + precision(rng[0], 3) + "," + precision(rng[1], 3) + ")",
                        c=c,
                        bold=False,
                    )

            for i in range(cldata.GetNumberOfArrays()):
                name = cldata.GetArrayName(i)
                if name and cldata.GetArray(i):
                    vedo.printc("celldata".ljust(14) + ": ", c=c, bold=True, end="")
                    try:
                        tt, nptt = arrtypes[cldata.GetArray(i).GetDataType()]
                    except:
                        tt = cldata.GetArray(i).GetDataType()
                    ncomp = cldata.GetArray(i).GetNumberOfComponents()
                    rng = cldata.GetArray(i).GetRange()
                    vedo.printc(f'"{name}" ({ncomp} {tt}),', c=c, bold=False, end="")
                    vedo.printc(
                        " range=(" + precision(rng[0], 4) + "," + precision(rng[1], 4) + ")",
                        c=c,
                        bold=False,
                    )

            for i in range(fldata.GetNumberOfArrays()):
                name = fldata.GetArrayName(i)
                if name and fldata.GetAbstractArray(i):
                    arr = fldata.GetAbstractArray(i)
                    vedo.printc("metadata".ljust(14) + ": ", c=c, bold=True, end="")
                    ncomp = arr.GetNumberOfComponents()
                    nvals = arr.GetNumberOfValues()
                    vedo.printc(f'"{name}" ({ncomp} components, {nvals} values)', c=c, bold=False)

        else:
            vedo.printc("mesh data".ljust(14) + ":", c=c, bold=True, end=" ")
            vedo.printc("no point or cell data is present.", c=c, bold=False)

    ################################
    def _printvtkactor(actor):

        if not actor.GetPickable():
            return

        mapper = actor.GetMapper()
        if hasattr(actor, "polydata"):
            poly = actor.polydata()
        else:
            poly = mapper.GetInput()

        pro = actor.GetProperty()
        pos = actor.GetPosition()
        bnds = actor.bounds()
        col = precision(pro.GetColor(), 3)
        alpha = pro.GetOpacity()
        npt = poly.GetNumberOfPoints()
        ncl = poly.GetNumberOfCells()
        npl = poly.GetNumberOfPolys()
        nln = poly.GetNumberOfLines()

        vedo.printc("Mesh/Points".ljust(70), c="g", bold=True, invert=True, dim=1, end="")

        if hasattr(actor, "info") and "legend" in actor.info.keys() and actor.info["legend"]:
            vedo.printc("legend".ljust(14)+": ", c="g", bold=True, end="")
            vedo.printc(actor.info["legend"], c="g", bold=False)
        else:
            print()

        if hasattr(actor, "name") and actor.name:
            vedo.printc("name".ljust(14) + ": ", c="g", bold=True, end="")
            vedo.printc(actor.name, c="g", bold=False)

        if hasattr(actor, "filename") and actor.filename:
            vedo.printc("file name".ljust(14) + ": ", c="g", bold=True, end="")
            vedo.printc(actor.filename, c="g", bold=False)

        if not actor.GetMapper().GetScalarVisibility():
            vedo.printc("color".ljust(14) + ": ", c="g", bold=True, end="")
            cname = vedo.colors.get_color_name(pro.GetColor())
            vedo.printc(f"{cname}, rgb={col}, alpha={alpha}", c="g", bold=False)

            if actor.GetBackfaceProperty():
                bcol = actor.GetBackfaceProperty().GetDiffuseColor()
                cname = vedo.colors.get_color_name(bcol)
                vedo.printc("back color".ljust(14) + ": ", c="g", bold=True, end="")
                vedo.printc(f"{cname}, rgb={precision(bcol,3)}", c="g", bold=False)

        vedo.printc("points".ljust(14) + ":", npt, c="g", bold=True)
        # vedo.printc("cells".ljust(14)+":", ncl, c="g", bold=True)
        vedo.printc("polygons".ljust(14) + ":", npl, c="g", bold=True)
        if nln:
            vedo.printc("lines".ljust(14) + ":", nln, c="g", bold=True)
        vedo.printc("position".ljust(14) + ":", pos, c="g", bold=True)

        if hasattr(actor, "GetScale"):
            vedo.printc("scale".ljust(14) + ":", c="g", bold=True, end=" ")
            vedo.printc(precision(actor.GetScale(), 3), c="g", bold=False)

        if hasattr(actor, "polydata") and actor.npoints:
            vedo.printc("center of mass".ljust(14) + ":", c="g", bold=True, end=" ")
            cm = tuple(actor.center_of_mass())
            vedo.printc(precision(cm, 3), c="g", bold=False)

            vedo.printc("average size".ljust(14) + ":", c="g", bold=True, end=" ")
            vedo.printc(precision(actor.average_size(), 6), c="g", bold=False)

            vedo.printc("diagonal size".ljust(14) + ":", c="g", bold=True, end=" ")
            vedo.printc(precision(actor.diagonal_size(), 6), c="g", bold=False)

        vedo.printc("bounds".ljust(14) + ":", c="g", bold=True, end=" ")
        bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
        vedo.printc("x=(" + bx1 + ", " + bx2 + ")", c="g", bold=False, end="")
        by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
        vedo.printc(" y=(" + by1 + ", " + by2 + ")", c="g", bold=False, end="")
        bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
        vedo.printc(" z=(" + bz1 + ", " + bz2 + ")", c="g", bold=False)

        _print_data(poly, mapper, "g")

        if hasattr(actor, "picked3d") and actor.picked3d is not None:
            idpt = actor.closest_point(actor.picked3d, return_point_id=True)
            idcell = actor.closest_point(actor.picked3d, return_cell_id=True)
            vedo.printc(
                "clicked point".ljust(14) + ":",
                precision(actor.picked3d, 6),
                f"pointID={idpt}, cellID={idcell}",
                c="g",
                bold=True,
            )

    if obj is None:
        return

    if isinstance(obj, np.ndarray):
        obj = obj
        cf = "y"
        vedo.printc("Numpy Array".ljust(70), c=cf, invert=True)
        vedo.printc(obj, c=cf)
        vedo.printc("shape".ljust(8) + ":", obj.shape, c=cf)
        vedo.printc("range".ljust(8) + f": ({np.min(obj)}, {np.max(obj)})", c=cf)
        vedo.printc("mean".ljust(8) + ":", np.mean(obj), c=cf)
        vedo.printc("std_dev".ljust(8) + ":", np.std(obj), c=cf)
        if len(obj.shape) >= 2:
            vedo.printc("Axis 0".ljust(8) + ":", c=cf, italic=1)
            vedo.printc("\tmin :", np.min(obj, axis=0), c=cf)
            vedo.printc("\tmax :", np.max(obj, axis=0), c=cf)
            vedo.printc("\tmean:", np.mean(obj, axis=0), c=cf)
            if obj.shape[1] > 3:
                vedo.printc("Axis 1".ljust(8) + ":", c=cf, italic=1)
                tmin = str(np.min(obj, axis=1).tolist()[:2]).replace("]", ", ...")
                tmax = str(np.max(obj, axis=1).tolist()[:2]).replace("]", ", ...")
                tmea = str(np.mean(obj, axis=1).tolist()[:2]).replace("]", ", ...")
                vedo.printc(f"\tmin : {tmin}", c=cf)
                vedo.printc(f"\tmax : {tmax}", c=cf)
                vedo.printc(f"\tmean: {tmea}", c=cf)

    elif isinstance(obj, vedo.Points):
        _printvtkactor(obj)

    elif isinstance(obj, vedo.Assembly):
        vedo.printc("Assembly".ljust(75), c="g", bold=True, invert=True)

        pos = obj.GetPosition()
        bnds = obj.GetBounds()
        vedo.printc("position".ljust(14) + ": ", c="g", bold=True, end="")
        vedo.printc(pos, c="g", bold=False)

        vedo.printc("bounds".ljust(14) + ": ", c="g", bold=True, end="")
        bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
        vedo.printc("x=(" + bx1 + ", " + bx2 + ")", c="g", bold=False, end="")
        by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
        vedo.printc(" y=(" + by1 + ", " + by2 + ")", c="g", bold=False, end="")
        bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
        vedo.printc(" z=(" + bz1 + ", " + bz2 + ")", c="g", bold=False)

        cl = vtk.vtkPropCollection()
        obj.GetActors(cl)
        cl.InitTraversal()
        for _ in range(obj.GetNumberOfPaths()):
            act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
            if isinstance(act, vtk.vtkActor):
                _printvtkactor(act)

    elif isinstance(obj, vedo.TetMesh):
        cf = "m"
        vedo.printc("TetMesh".ljust(70), c=cf, bold=True, invert=True)
        pos = obj.GetPosition()
        bnds = obj.GetBounds()
        ug = obj.inputdata()
        vedo.printc("nr. of tetras".ljust(14) + ": ", c=cf, bold=True, end="")
        vedo.printc(ug.GetNumberOfCells(), c=cf, bold=False)
        vedo.printc("position".ljust(14) + ": ", c=cf, bold=True, end="")
        vedo.printc(pos, c=cf, bold=False)
        vedo.printc("bounds".ljust(14) + ": ", c=cf, bold=True, end="")
        bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
        vedo.printc("x=(" + bx1 + ", " + bx2 + ")", c=cf, bold=False, end="")
        by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
        vedo.printc(" y=(" + by1 + ", " + by2 + ")", c=cf, bold=False, end="")
        bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
        vedo.printc(" z=(" + bz1 + ", " + bz2 + ")", c=cf, bold=False)
        _print_data(ug, obj._mapper, cf)

    elif isinstance(obj, (vedo.volume.Volume, vedo.volume.VolumeSlice)):
        vedo.printc("Volume".ljust(70), c="b", bold=True, invert=True)

        img = obj.GetMapper().GetInput()
        vedo.printc("origin".ljust(14) + ": ", c="b", bold=True, end="")
        vedo.printc(precision(obj.origin(),6), c="b", bold=False)

        vedo.printc("center".ljust(14) + ": ", c="b", bold=True, end="")
        vedo.printc(precision(obj.center(),6), c="b", bold=False)

        vedo.printc("dimensions".ljust(14) + ": ", c="b", bold=True, end="")
        vedo.printc(img.GetDimensions(), c="b", bold=False)
        vedo.printc("spacing".ljust(14) + ": ", c="b", bold=True, end="")
        vedo.printc(precision(img.GetSpacing(),6), c="b", bold=False)
        # vedo.printc("data dimension".ljust(14) + ": ", c="b", bold=True, end="")
        # vedo.printc(img.GetDataDimension(), c="b", bold=False)

        vedo.printc("memory size".ljust(14) + ": ", c="b", bold=True, end="")
        vedo.printc(int(img.GetActualMemorySize() / 1024), "MB", c="b", bold=False)

        vedo.printc("scalar #bytes".ljust(14) + ": ", c="b", bold=True, end="")
        vedo.printc(img.GetScalarSize(), c="b", bold=False)

        bnds = obj.GetBounds()
        vedo.printc("bounds".ljust(14) + ": ", c="b", bold=True, end="")
        bx1, bx2 = precision(bnds[0], 4), precision(bnds[1], 4)
        vedo.printc("x=(" + bx1 + ", " + bx2 + ")", c="b", bold=False, end="")
        by1, by2 = precision(bnds[2], 4), precision(bnds[3], 4)
        vedo.printc(" y=(" + by1 + ", " + by2 + ")", c="b", bold=False, end="")
        bz1, bz2 = precision(bnds[4], 4), precision(bnds[5], 4)
        vedo.printc(" z=(" + bz1 + ", " + bz2 + ")", c="b", bold=False)

        vedo.printc("scalar range".ljust(14) + ": ", c="b", bold=True, end="")
        vedo.printc(img.GetScalarRange(), c="b", bold=False)

        print_histogram(obj, horizontal=True,
            logscale=True, bins=8, height=15, c="b", bold=True)

    elif isinstance(obj, vedo.Plotter) and obj.interactor:  # dumps Plotter info
        axtype = {
            0: "(no axes)",
            1: "(three customizable gray grid walls)",
            2: "(cartesian axes from origin",
            3: "(positive range of cartesian axes from origin",
            4: "(axes triad at bottom left)",
            5: "(oriented cube at bottom left)",
            6: "(mark the corners of the bounding box)",
            7: "(3D ruler at each side of the cartesian axes)",
            8: "(the vtkCubeAxesActor object)",
            9: "(the bounding box outline)",
            10: "(circles of maximum bounding box range)",
            11: "(show a large grid on the x-y plane)",
            12: "(show polar axes)",
            13: "(simple ruler at the bottom of the window)",
            14: "(the vtkCameraOrientationWidget object)",
        }
        bns, totpt = [], 0
        for a in obj.actors:
            b = a.GetBounds()
            if a.GetBounds() is not None:
                if isinstance(a, vtk.vtkActor) and a.GetMapper():
                    totpt += a.GetMapper().GetInput().GetNumberOfPoints()
                bns.append(b)
        if len(bns) == 0:
            return
        vedo.printc("Plotter".ljust(70), invert=True, dim=1, c="c")
        otit = obj.title
        if not otit:
            otit = None
        vedo.printc("window title".ljust(14) + ":", otit, bold=False, c="c")
        vedo.printc(
            "window size".ljust(14) + ":",
            obj.window.GetSize(),
            "- full screen size:",
            obj.window.GetScreenSize(),
            bold=False,
            c="c",
        )
        vedo.printc(
            "actv renderer".ljust(14) + ":",
            "nr.",
            obj.renderers.index(obj.renderer),
            f"(of {len(obj.renderers)} renderers)",
            bold=False,
            c="c",
        )
        vedo.printc(
            "nr. of actors".ljust(14) + ":", len(obj.actors), bold=False, c="c", end=""
        )
        vedo.printc(" (" + str(totpt), "vertices)", bold=False, c="c")
        max_bns = np.max(bns, axis=0)
        min_bns = np.min(bns, axis=0)
        vedo.printc("max bounds".ljust(14) + ": ", c="c", bold=False, end="")
        bx1, bx2 = precision(min_bns[0], 3), precision(max_bns[1], 3)
        vedo.printc("x=(" + bx1 + ", " + bx2 + ")", c="c", bold=False, end="")
        by1, by2 = precision(min_bns[2], 3), precision(max_bns[3], 3)
        vedo.printc(" y=(" + by1 + ", " + by2 + ")", c="c", bold=False, end="")
        bz1, bz2 = precision(min_bns[4], 3), precision(max_bns[5], 3)
        vedo.printc(" z=(" + bz1 + ", " + bz2 + ")", c="c", bold=False)
        if isinstance(obj.axes, dict):
            obj.axes = 1
        if obj.axes:
            vedo.printc(
                "axes style".ljust(14) + ":",
                obj.axes,
                axtype[obj.axes],
                bold=False,
                c="c",
            )

    elif isinstance(obj, vedo.Picture):  # dumps Picture info
        vedo.printc("Picture".ljust(70), c="y", bold=True, invert=True)
        try:
            # generate a print thumbnail
            width, height = obj.dimensions()
            w = 45
            h = int(height / width * (w - 1) * 0.5 + 0.5)
            img_arr = obj.clone().resize([w, h]).tonumpy()
            h, w = img_arr.shape[:2]
            for x in range(h):
                for y in range(w):
                    pix = img_arr[x][y]
                    r, g, b = pix[:3]
                    print(f"\x1b[48;2;{r};{g};{b}m", end=" ")
                print("\x1b[0m")
        except:
            pass

        img = obj.GetMapper().GetInput()
        pos = obj.GetPosition()
        vedo.printc("position".ljust(14) + ": ", c="y", bold=True, end="")
        vedo.printc(pos, c="y", bold=False)

        vedo.printc("dimensions".ljust(14) + ": ", c="y", bold=True, end="")
        vedo.printc(obj.shape, c="y", bold=False)

        vedo.printc("memory size".ljust(14) + ": ", c="y", bold=True, end="")
        vedo.printc(int(img.GetActualMemorySize()), "kB", c="y", bold=False)

        bnds = obj.GetBounds()
        vedo.printc("bounds".ljust(14) + ": ", c="y", bold=True, end="")
        bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
        vedo.printc("x=(" + bx1 + ", " + bx2 + ")", c="y", bold=False, end="")
        by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
        vedo.printc(" y=(" + by1 + ", " + by2 + ")", c="y", bold=False, end="")
        bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
        vedo.printc(" z=(" + bz1 + ", " + bz2 + ")", c="y", bold=False)

        vedo.printc("intensty range".ljust(14) + ": ", c="y", bold=True, end="")
        vedo.printc(img.GetScalarRange(), c="y", bold=False)
        vedo.printc("level / window".ljust(14) + ": ", c="y", bold=True, end="")
        vedo.printc(obj.level(), "/", obj.window(), c="y", bold=False)

    else:
        vedo.printc(str(type(obj)).ljust(70), invert=True)
        vedo.printc(obj)


def print_histogram(
    data,
    bins=10,
    height=10,
    logscale=False,
    minbin=0,
    horizontal=False,
    char="\u2588",
    c=None,
    bold=True,
    title="histogram",
    spacer="",
):
    """
    Ascii histogram printing.
    
    Input can be a `vedo.Volume` or `vedo.Mesh`.
    Returns the raw data before binning (useful when passing vtk objects).

    Arguments:
        bins : (int)
            number of histogram bins
        height : (int)
            height of the histogram in character units
        logscale : (bool)
            use logscale for frequencies
        minbin : (int)
            ignore bins before minbin
        horizontal : (bool)
            show histogram horizontally
        char : (str)
            character to be used
        bold : (bool)
            use boldface
        title : (str)
            histogram title
        spacer : (str)
            horizontal spacer

    Example:
        ```python
        from vedo import print_histogram
        import numpy as np
        d = np.random.normal(size=1000)
        data = print_histogram(d, c='blue', logscale=True, title='my scalars')
        data = print_histogram(d, c=1, horizontal=1)
        print(np.mean(data)) # data here is same as d
        ```
        ![](https://vedo.embl.es/images/feats/print_histogram.png)
    """
    # credits: http://pyinsci.blogspot.com/2009/10/ascii-histograms.html
    # adapted for vedo by M.Musy, 2019

    if not horizontal:  # better aspect ratio
        bins *= 2

    isimg = isinstance(data, vtk.vtkImageData)
    isvol = isinstance(data, vtk.vtkVolume)
    if isimg or isvol:
        if isvol:
            img = data.imagedata()
        else:
            img = data
        dims = img.GetDimensions()
        nvx = min(100000, dims[0] * dims[1] * dims[2])
        idxs = np.random.randint(0, min(dims), size=(nvx, 3))
        data = []
        for ix, iy, iz in idxs:
            d = img.GetScalarComponentAsFloat(ix, iy, iz, 0)
            data.append(d)
    elif isinstance(data, vtk.vtkActor):
        arr = data.polydata().GetPointData().GetScalars()
        if not arr:
            arr = data.polydata().GetCellData().GetScalars()
            if not arr:
                return None

        data = vtk2numpy(arr)

    try:
        h = np.histogram(data, bins=bins)
    except TypeError as e:
        vedo.logger.error(f"cannot compute histogram: {e}")
        return ""

    if minbin:
        hi = h[0][minbin:-1]
    else:
        hi = h[0]

    if char == "\U00002589" and horizontal:
        char = "\U00002586"

    title = title.ljust(14) + ":"
    entrs = " entries=" + str(len(data))
    if logscale:
        h0 = np.log10(hi + 1)
        maxh0 = int(max(h0) * 100) / 100
        title = title + entrs + " (logscale)"
    else:
        h0 = hi
        maxh0 = max(h0)
        title = title + entrs

    def _v():
        his = ""
        if title:
            his += title + "\n"
        bars = h0 / maxh0 * height
        for l in reversed(range(1, height + 1)):
            line = ""
            if l == height:
                line = "%s " % maxh0
            else:
                line = "   |" + " " * (len(str(maxh0)) - 3)
            for c in bars:
                if c >= np.ceil(l):
                    line += char
                else:
                    line += " "
            line += "\n"
            his += line
        his += "%.2f" % h[1][0] + "." * (bins) + "%.2f" % h[1][-1] + "\n"
        return his

    def _h():
        his = ""
        if title:
            his += title + "\n"
        xl = ["%.2f" % n for n in h[1]]
        lxl = [len(l) for l in xl]
        bars = h0 / maxh0 * height
        his += spacer + " " * int(max(bars) + 2 + max(lxl)) + "%s\n" % maxh0
        for i, c in enumerate(bars):
            line = xl[i] + " " * int(max(lxl) - lxl[i]) + "| " + char * int(c) + "\n"
            his += spacer + line
        return his

    if horizontal:
        height *= 2
        vedo.printc(_h(), c=c, bold=bold)
    else:
        vedo.printc(_v(), c=c, bold=bold)
    return data

def print_table(*columns, headers=None, c='g'):
    """
    Print lists as tables.

    Example:
    ```python
    list1 = [1, np.sqrt(2), 3]
    list2 = ["A", "B", "C"]
    list3 = [True, False, True]
    headers = ["First Column", "Second Column", "Third Column"]
    print_table(list1, list2, list3, headers=headers)
    ```
    """
    # If headers is not provided, use default header names
    corner='─'
    if headers is None:
        headers = [f"Column {i}" for i in range(1, len(columns)+1)]
    assert len(headers) == len(columns)

    # Find the maximum length of the elements in each column and header
    max_lens = [max(len(str(x)) for x in column) for column in columns]
    max_len_headers = [max(len(str(header)), max_len) for header, max_len in zip(headers, max_lens)]

    # Construct the table header
    header = "│ " + " │ ".join(header.ljust(max_len) for header, max_len in zip(headers, max_len_headers)) + " │"

    # Construct the line separator
    line1 = "┌" + corner.join("─"*(max_len+2) for max_len in max_len_headers) + "┐"
    line2 = "└" + corner.join("─"*(max_len+2) for max_len in max_len_headers) + "┘"

    # Print the table header
    vedo.printc(line1, c=c)
    vedo.printc(header, c=c)
    vedo.printc(line2, c=c)

    # Print the data rows
    for row in zip(*columns):
        row = "│ " + " │ ".join(str(col).ljust(max_len) for col, max_len in zip(row, max_len_headers)) + " │"
        vedo.printc(row, bold=False, c=c)

    # Print the line separator again to close the table
    vedo.printc(line2, c=c)


def make_bands(inputlist, n):
    """
    Group values of a list into bands of equal value, where
    `n` is the number of bands, a positive integer > 2.

    Returns a binned list of the same length as the input.
    """
    if n < 2:
        return inputlist
    vmin = np.min(inputlist)
    vmax = np.max(inputlist)
    bb = np.linspace(vmin, vmax, n, endpoint=0)
    dr = bb[1] - bb[0]
    bb += dr / 2
    tol = dr / 2 * 1.001
    newlist = []
    for s in inputlist:
        for b in bb:
            if abs(s - b) < tol:
                newlist.append(b)
                break
    return np.array(newlist)


#################################################################
# Functions adapted from:
# https://github.com/sdorkenw/MeshParty/blob/master/meshparty/trimesh_vtk.py
def camera_from_quaternion(pos, quaternion, distance=10000, ngl_correct=True):
    """
    Define a `vtkCamera` with a particular orientation.

    Arguments:
        pos: (np.array, list, tuple)
            an iterator of length 3 containing the focus point of the camera
        quaternion: (np.array, list, tuple)
            a len(4) quaternion `(x,y,z,w)` describing the rotation of the camera
            such as returned by neuroglancer `x,y,z,w` all in `[0,1]` range
        distance: (float)
            the desired distance from pos to the camera (default = 10000 nm)

    Returns:
        `vtk.vtkCamera`, a vtk camera setup according to these rules.
    """
    camera = vtk.vtkCamera()
    # define the quaternion in vtk, note the swapped order
    # w,x,y,z instead of x,y,z,w
    quat_vtk = vtk.vtkQuaterniond(quaternion[3], quaternion[0], quaternion[1], quaternion[2])
    # use this to define a rotation matrix in x,y,z
    # right handed units
    M = np.zeros((3, 3), dtype=np.float32)
    quat_vtk.ToMatrix3x3(M)
    # the default camera orientation is y up
    up = [0, 1, 0]
    # calculate default camera position is backed off in positive z
    pos = [0, 0, distance]

    # set the camera rototation by applying the rotation matrix
    camera.SetViewUp(*np.dot(M, up))
    # set the camera position by applying the rotation matrix
    camera.SetPosition(*np.dot(M, pos))
    if ngl_correct:
        # neuroglancer has positive y going down
        # so apply these azimuth and roll corrections
        # to fix orientatins
        camera.Azimuth(-180)
        camera.Roll(180)

    # shift the camera posiiton and focal position
    # to be centered on the desired location
    p = camera.GetPosition()
    p_new = np.array(p) + pos
    camera.SetPosition(*p_new)
    camera.SetFocalPoint(*pos)
    return camera


def camera_from_neuroglancer(state, zoom=300):
    """
    Define a `vtkCamera` from a neuroglancer state dictionary.

    Arguments:
        state: (dict)
            an neuroglancer state dictionary.
        zoom: (float)
            how much to multiply zoom by to get camera backoff distance
            default = 300 > ngl_zoom = 1 > 300 nm backoff distance.

    Returns:
        `vtk.vtkCamera`, a vtk camera setup that matches this state.
    """
    orient = state.get("perspectiveOrientation", [0.0, 0.0, 0.0, 1.0])
    pzoom = state.get("perspectiveZoom", 10.0)
    position = state["navigation"]["pose"]["position"]
    pos_nm = np.array(position["voxelCoordinates"]) * position["voxelSize"]
    return camera_from_quaternion(pos_nm, orient, pzoom * zoom, ngl_correct=True)


def oriented_camera(
    center=(0, 0, 0), up_vector=(0, 1, 0), backoff_vector=(0, 0, 1), backoff=1
):
    """
    Generate a `vtkCamera` pointed at a specific location,
    oriented with a given up direction, set to a backoff.
    """
    vup = np.array(up_vector)
    vup = vup / np.linalg.norm(vup)
    pt_backoff = center - backoff * np.array(backoff_vector)
    camera = vtk.vtkCamera()
    camera.SetFocalPoint(center[0], center[1], center[2])
    camera.SetViewUp(vup[0], vup[1], vup[2])
    camera.SetPosition(pt_backoff[0], pt_backoff[1], pt_backoff[2])
    return camera


def camera_from_dict(camera, modify_inplace=None):
    """
    Generate a `vtkCamera` from a dictionary.
    """
    if modify_inplace:
        vcam = modify_inplace
    else:
        vcam = vtk.vtkCamera()

    camera = dict(camera)  # make a copy so input is not emptied by pop()
    cm_pos = camera.pop("position", camera.pop("pos", None))
    cm_focal_point = camera.pop("focal_point", camera.pop("focalPoint", None))
    cm_viewup = camera.pop("viewup", None)
    cm_distance = camera.pop("distance", None)
    cm_clipping_range = camera.pop("clipping_range", camera.pop("clippingRange", None))
    cm_parallel_scale = camera.pop("parallel_scale", camera.pop("parallelScale", None))
    cm_thickness = camera.pop("thickness", None)
    cm_view_angle = camera.pop("view_angle", camera.pop("viewAngle", None))
    if len(camera.keys()):
        vedo.logger.warning(f"in camera_from_dict, key(s) not recognized: {camera.keys()}")
    if cm_pos is not None:            vcam.SetPosition(cm_pos)
    if cm_focal_point is not None:    vcam.SetFocalPoint(cm_focal_point)
    if cm_viewup is not None:         vcam.SetViewUp(cm_viewup)
    if cm_distance is not None:       vcam.SetDistance(cm_distance)
    if cm_clipping_range is not None: vcam.SetClippingRange(cm_clipping_range)
    if cm_parallel_scale is not None: vcam.SetParallelScale(cm_parallel_scale)
    if cm_thickness is not None:      vcam.SetThickness(cm_thickness)
    if cm_view_angle is not None:     vcam.SetViewAngle(cm_view_angle)
    return vcam


def vtkCameraToK3D(vtkcam):
    """
    Convert a `vtkCamera` object into a 9-element list to be used by the K3D backend.

    Output format is:
        `[posx,posy,posz, targetx,targety,targetz, upx,upy,upz]`.
    """
    cpos = np.array(vtkcam.GetPosition())
    kam = [cpos.tolist()]
    kam.append(vtkcam.GetFocalPoint())
    kam.append(vtkcam.GetViewUp())
    return np.array(kam).ravel()


def make_ticks(x0, x1, n=None, labels=None, digits=None, logscale=False, useformat=""):
    """
    Generate numeric labels for the `[x0, x1]` range.

    The format specifier could be expressed in the format:
        `:[[fill]align][sign][#][0][width][,][.precision][type]`

    where, the options are:
    ```
    fill        =  any character
    align       =  < | > | = | ^
    sign        =  + | - | " "
    width       =  integer
    precision   =  integer
    type        =  b | c | d | e | E | f | F | g | G | n | o | s | x | X | %
    ```

    E.g.: useformat=":.2f"
    """
    # Copyright M. Musy, 2021, license: MIT.
    #
    # useformat eg: ":.2f", check out:
    # https://mkaz.blog/code/python-string-format-cookbook/
    # https://www.programiz.com/python-programming/methods/built-in/format

    if x1 <= x0:
        # vedo.printc("Error in make_ticks(): x0 >= x1", x0,x1, c='r')
        return np.array([0.0, 1.0]), ["", ""]

    ticks_str, ticks_float = [], []
    baseline = (1, 2, 5, 10, 20, 50)

    if logscale:
        if x0 <= 0 or x1 <= 0:
            vedo.logger.error("make_ticks: zero or negative range with log scale.")
            raise RuntimeError
        if n is None:
            n = int(abs(np.log10(x1) - np.log10(x0))) + 1
        x0, x1 = np.log10([x0, x1])

    if not n:
        n = 5

    if labels is not None:
        # user is passing custom labels

        ticks_float.append(0)
        ticks_str.append("")
        for tp, ts in labels:
            if tp == x1:
                continue
            ticks_str.append(str(ts))
            tickn = lin_interpolate(tp, [x0, x1], [0, 1])
            ticks_float.append(tickn)

    else:
        # ..here comes one of the shortest and most painful pieces of code:
        # automatically choose the best natural axis subdivision based on multiples of 1,2,5
        dstep = (x1 - x0) / n  # desired step size, begin of the nightmare

        basestep = pow(10, np.floor(np.log10(dstep)))
        steps = np.array([basestep * i for i in baseline])
        idx = (np.abs(steps - dstep)).argmin()
        s = steps[idx]  # chosen step size

        low_bound, up_bound = 0, 0
        if x0 < 0:
            low_bound = -pow(10, np.ceil(np.log10(-x0)))
        if x1 > 0:
            up_bound = pow(10, np.ceil(np.log10(x1)))

        if low_bound < 0:
            if up_bound < 0:
                negaxis = np.arange(low_bound, int(up_bound / s) * s)
            else:
                if -low_bound / s > 1.0e06:
                    return np.array([0.0, 1.0]), ["", ""]
                negaxis = np.arange(low_bound, 0, s)
        else:
            negaxis = np.array([])

        if up_bound > 0:
            if low_bound > 0:
                posaxis = np.arange(int(low_bound / s) * s, up_bound, s)
            else:
                if up_bound / s > 1.0e06:
                    return np.array([0.0, 1.0]), ["", ""]
                posaxis = np.arange(0, up_bound, s)
        else:
            posaxis = np.array([])

        fulaxis = np.unique(np.clip(np.concatenate([negaxis, posaxis]), x0, x1))
        # end of the nightmare

        if useformat:
            sf = "{" + f"{useformat}" + "}"
            sas = ""
            for x in fulaxis:
                sas += sf.format(x) + " "
        elif digits is None:
            np.set_printoptions(suppress=True)  # avoid zero precision
            sas = str(fulaxis).replace("[", "").replace("]", "")
            sas = sas.replace(".e", "e").replace("e+0", "e+").replace("e-0", "e-")
            np.set_printoptions(suppress=None)  # set back to default
        else:
            sas = precision(fulaxis, digits, vrange=(x0, x1))
            sas = sas.replace("[", "").replace("]", "").replace(")", "").replace(",", "")

        sas2 = []
        for s in sas.split():
            if s.endswith("."):
                s = s[:-1]
            if s == "-0":
                s = "0"
            if digits is not None and "e" in s:
                s += " "  # add space to terminate modifiers
            sas2.append(s)

        for ts, tp in zip(sas2, fulaxis):
            if tp == x1:
                continue
            tickn = lin_interpolate(tp, [x0, x1], [0, 1])
            ticks_float.append(tickn)
            if logscale:
                val = np.power(10, tp)
                if useformat:
                    sf = "{" + f"{useformat}" + "}"
                    ticks_str.append(sf.format(val))
                else:
                    if val >= 10:
                        val = int(val + 0.5)
                    else:
                        val = round_to_digit(val, 2)
                    ticks_str.append(str(val))
            else:
                ticks_str.append(ts)

    ticks_str.append("")
    ticks_float.append(1)
    ticks_float = np.array(ticks_float)
    return ticks_float, ticks_str


def grid_corners(i, nm, size, margin=0, yflip=True):
    """
    Compute the 2 corners coordinates of the i-th box in a grid of shape n*m.
    The top-left square is square number 1.

    Arguments:
        i : (int)
            input index of the desired grid square (to be used in `show(..., at=...)`).
        nm : (list)
            grid shape as (n,m).
        size : (list)
            total size of the grid along x and y.
        margin : (float)
            keep a small margin between boxes.
        yflip : (bool)
            y-coordinate points downwards

    Returns:
        Two 2D points representing the bottom-left corner and the top-right corner
        of the `i`-nth box in the grid.

    Example:
        ```python
        from vedo import *
        acts=[]
        n,m = 5,7
        for i in range(1, n*m + 1):
            c1,c2 = utils.grid_corners(i, [n,m], [1,1], 0.01)
            t = Text3D(i, (c1+c2)/2, c='k', s=0.02, justify='center').z(0.01)
            r = Rectangle(c1, c2, c=i)
            acts += [t,r]
        show(acts, axes=1).close()
        ```
        ![](https://vedo.embl.es/images/feats/grid_corners.png)
    """
    i -= 1
    n, m = nm
    sx, sy = size
    dx, dy = sx / n, sy / m
    nx = i % n
    ny = int((i - nx) / n)
    if yflip:
        ny = n - ny
    c1 = (dx * nx + margin, dy * ny + margin)
    c2 = (dx * (nx + 1) - margin, dy * (ny + 1) - margin)
    return np.array(c1), np.array(c2)


############################################################################
# Trimesh support
#
# Install trimesh with:
#
#    sudo apt install python3-rtree
#    pip install rtree shapely
#    conda install trimesh
#
# Check the example gallery in: examples/other/trimesh>
###########################################################################
def vedo2trimesh(mesh):
    """
    Convert `vedo.mesh.Mesh` to `Trimesh.Mesh` object.
    """
    if is_sequence(mesh):
        tms = []
        for a in mesh:
            tms.append(vedo2trimesh(a))
        return tms

    from trimesh import Trimesh

    tris = mesh.faces()
    carr = mesh.celldata["CellIndividualColors"]
    ccols = carr

    points = mesh.points()
    varr = mesh.pointdata["VertexColors"]
    vcols = varr

    if len(tris) == 0:
        tris = None

    return Trimesh(vertices=points, faces=tris, face_colors=ccols, vertex_colors=vcols)


def trimesh2vedo(inputobj):
    """
    Convert a `Trimesh` object to `vedo.Mesh` or `vedo.Assembly` object.
    """
    if is_sequence(inputobj):
        vms = []
        for ob in inputobj:
            vms.append(trimesh2vedo(ob))
        return vms

    inputobj_type = str(type(inputobj))

    if "Trimesh" in inputobj_type or "primitives" in inputobj_type:
        faces = inputobj.faces
        poly = buildPolyData(inputobj.vertices, faces)
        tact = vedo.Mesh(poly)
        if inputobj.visual.kind == "face":
            trim_c = inputobj.visual.face_colors
        else:
            trim_c = inputobj.visual.vertex_colors

        if is_sequence(trim_c):
            if is_sequence(trim_c[0]):
                same_color = len(np.unique(trim_c, axis=0)) < 2  # all vtxs have same color

                if same_color:
                    tact.c(trim_c[0, [0, 1, 2]]).alpha(trim_c[0, 3])
                else:
                    if inputobj.visual.kind == "face":
                        tact.cell_individual_colors(trim_c)
        return tact

    if "PointCloud" in inputobj_type:

        trim_cc, trim_al = "black", 1
        if hasattr(inputobj, "vertices_color"):
            trim_c = inputobj.vertices_color
            if len(trim_c):
                trim_cc = trim_c[:, [0, 1, 2]] / 255
                trim_al = trim_c[:, 3] / 255
                trim_al = np.sum(trim_al) / len(trim_al)  # just the average
        return vedo.shapes.Points(inputobj.vertices, r=8, c=trim_cc, alpha=trim_al)

    if "path" in inputobj_type:

        lines = []
        for e in inputobj.entities:
            # print('trimesh entity', e.to_dict())
            l = vedo.shapes.Line(inputobj.vertices[e.points], c="k", lw=2)
            lines.append(l)
        return vedo.Assembly(lines)

    return None


def vedo2meshlab(vmesh):
    """Convert a `vedo.Mesh` to a Meshlab object."""
    try:
        import pymeshlab as mlab
    except RuntimeError:
        vedo.logger.error("Need pymeshlab to run:\npip install pymeshlab")

    vertex_matrix = vmesh.points().astype(np.float64)

    try:
        face_matrix = np.asarray(vmesh.faces(), dtype=np.float64)
    except:
        print("In vedo2meshlab, need to triangulate mesh first!")
        face_matrix = np.array(vmesh.clone().triangulate().faces(), dtype=np.float64)

    v_normals_matrix = vmesh.normals(cells=False, recompute=False)
    if not v_normals_matrix.shape[0]:
        v_normals_matrix = np.empty((0, 3), dtype=np.float64)

    f_normals_matrix = vmesh.normals(cells=True, recompute=False)
    if not f_normals_matrix.shape[0]:
        f_normals_matrix = np.empty((0, 3), dtype=np.float64)

    v_color_matrix = vmesh.pointdata["RGBA"]
    if v_color_matrix is None:
        v_color_matrix = np.empty((0, 4), dtype=np.float64)
    else:
        v_color_matrix = v_color_matrix.astype(np.float64) / 255
        if v_color_matrix.shape[1] == 3:
            v_color_matrix = np.c_[
                v_color_matrix, np.ones(v_color_matrix.shape[0], dtype=np.float64)
            ]

    f_color_matrix = vmesh.celldata["RGBA"]
    if f_color_matrix is None:
        f_color_matrix = np.empty((0, 4), dtype=np.float64)
    else:
        f_color_matrix = f_color_matrix.astype(np.float64) / 255
        if f_color_matrix.shape[1] == 3:
            f_color_matrix = np.c_[
                f_color_matrix, np.ones(f_color_matrix.shape[0], dtype=np.float64)
            ]

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
        if data is not None:
            if data.ndim == 1:  # scalar
                m.add_vertex_custom_scalar_attribute(data.astype(np.float64), k)
            elif data.ndim == 2:  # vectorial data
                m.add_vertex_custom_point_attribute(data.astype(np.float64), k)

    for k in vmesh.celldata.keys():
        data = vmesh.celldata[k]
        if data is not None:
            if data.ndim == 1:  # scalar
                m.add_face_custom_scalar_attribute(data.astype(np.float64), k)
            elif data.ndim == 2:  # vectorial data
                m.add_face_custom_point_attribute(data.astype(np.float64), k)

    m.update_bounding_box()
    return m


def meshlab2vedo(mmesh):
    """Convert a Meshlab object to `vedo.Mesh`."""
    inputtype = str(type(mmesh))

    if "MeshSet" in inputtype:
        mmesh = mmesh.current_mesh()

    mpoints, mcells = mmesh.vertex_matrix(), mmesh.face_matrix()
    if len(mcells):
        polydata = buildPolyData(mpoints, mcells)
    else:
        polydata = buildPolyData(mpoints, None)

    if mmesh.has_vertex_scalar():
        parr = mmesh.vertex_scalar_array()
        parr_vtk = numpy_to_vtk(parr)
        parr_vtk.SetName("MeshLabScalars")
        x0, x1 = parr_vtk.GetRange()
        if x1 - x0:
            polydata.GetPointData().AddArray(parr_vtk)
            polydata.GetPointData().SetActiveScalars("MeshLabScalars")

    if mmesh.has_face_scalar():
        carr = mmesh.face_scalar_array()
        carr_vtk = numpy_to_vtk(carr)
        carr_vtk.SetName("MeshLabScalars")
        x0, x1 = carr_vtk.GetRange()
        if x1 - x0:
            polydata.GetCellData().AddArray(carr_vtk)
            polydata.GetCellData().SetActiveScalars("MeshLabScalars")

    pnorms = mmesh.vertex_normal_matrix()
    if len(pnorms):
        polydata.GetPointData().SetNormals(numpy2vtk(pnorms))

    cnorms = mmesh.face_normal_matrix()
    if len(cnorms):
        polydata.GetCellData().SetNormals(numpy2vtk(cnorms))
    return polydata

def open3d2vedo(o3d_mesh):
    """Convert `open3d.geometry.TriangleMesh` to a `vedo.Mesh`."""
    m = vedo.Mesh([np.array(o3d_mesh.vertices), np.array(o3d_mesh.triangles)])
    # TODO: could also check whether normals and color are present in 
    # order to port with the above vertices/faces
    return m

def vedo2open3d(vedo_mesh):
    """
    Return an `open3d.geometry.TriangleMesh` version of the current mesh.
    """
    try:
        import open3d as o3d
    except RuntimeError:
        vedo.logger.error("Need open3d to run:\npip install open3d")

    # create from numpy arrays
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vedo_mesh.points()),
        triangles=o3d.utility.Vector3iVector(vedo_mesh.faces()),
    )
    # TODO: need to add some if check here in case color and normals
    #  info are not existing
    # o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vedo_mesh.pointdata["RGB"]/255)
    # o3d_mesh.vertex_normals= o3d.utility.Vector3dVector(vedo_mesh.pointdata["Normals"])
    return o3d_mesh

def vtk_version_at_least(major, minor=0, build=0):
    """
    Check the installed VTK version.

    Return `True` if the requested VTK version is greater or equal to the actual VTK version.

    Arguments:
        major : (int)
            Major version.
        minor : (int)
            Minor version.
        build : (int)
            Build version.
    """
    needed_version = 10000000000 * int(major) + 100000000 * int(minor) + int(build)
    try:
        vtk_version_number = vtk.VTK_VERSION_NUMBER
    except AttributeError:  # as error:
        ver = vtk.vtkVersion()
        vtk_version_number = (
            10000000000 * ver.GetVTKMajorVersion()
            + 100000000 * ver.GetVTKMinorVersion()
            + ver.GetVTKBuildVersion()
        )
    return vtk_version_number >= needed_version


def ctf2lut(tvobj, logscale=False):
    """Internal use."""
    # build LUT from a color transfer function for tmesh or volume
    pr = tvobj.GetProperty()
    if not isinstance(pr, vtk.vtkVolumeProperty):
        return None
    ctf = pr.GetRGBTransferFunction()
    otf = pr.GetScalarOpacity()
    x0, x1 = tvobj.inputdata().GetScalarRange()
    cols, alphas = [], []
    for x in np.linspace(x0, x1, 256):
        cols.append(ctf.GetColor(x))
        alphas.append(otf.GetValue(x))

    if logscale:
        lut = vtk.vtkLogLookupTable()
    else:
        lut = vtk.vtkLookupTable()

    lut.SetRange(x0, x1)
    lut.SetNumberOfTableValues(len(cols))
    for i, col in enumerate(cols):
        r, g, b = col
        lut.SetTableValue(i, r, g, b, alphas[i])
    lut.Build()
    return lut
