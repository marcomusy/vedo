from __future__ import division, print_function
import vtk, sys
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import numpy as np
import vedo
from vedo.colors import printc
import time
import math

__doc__ = (
    """
Utilities submodule.
"""
    + vedo.docs._defs
)

__all__ = [
    "ProgressBar",
    "geometry",
    "isSequence",
    "linInterpolate",
    "fitCircle2D",
    "vector",
    "mag",
    "mag2",
    "versor",
    "precision",
    "pointIsInTriangle",
    "pointToLineDistance",
    "grep",
    "printInfo",
    "makeBands",
    "spher2cart",
    "cart2spher",
    "cart2pol",
    "pol2cart",
    "humansort",
    "dotdict",
    "printHistogram",
    "cameraFromQuaternion",
    "cameraFromNeuroglancer",
    "orientedCamera",
    "vtkCameraToK3D",
    "vedo2trimesh",
    "trimesh2vedo",
    "resampleArrays",
]

###########################################################################
class ProgressBar:
    """
    Class to print a progress bar with optional text message.

    :Example:
        .. code-block:: python

            import time
            pb = ProgressBar(0,400, c='red')
            for i in pb.range():
                time.sleep(.1)
                pb.print('some message')

        |progbar|
    """

    def __init__(self,
                 start, stop, step=1,
                 c=None,
                 bold=True,
                 italic=False,
                 title='',
                 ETA=True,
                 width=25,
                 char=u"\U00002501",
                 char_back=u"\U00002500",
                 ):

        char_arrow = ""
        if sys.version_info[0]<3:
            char="="
            char_arrow = ''
            char_back=''
            bold=False

        self.char_back = char_back
        self.char_arrow = char_arrow

        self.char0 = ''
        self.char1 = ''
        self.title = title+' '
        if title:
            self.title = ' '+self.title

        self.start = start
        self.stop = stop
        self.step = step
        self.color = c
        self.bold = bold
        self.italic = italic
        self.width = width
        self.char = char
        self.bar = ""
        self.percent = 0
        self.clock0 = 0
        self.ETA = ETA
        self.clock0 = time.time()
        self._remt = 1e10
        self._update(0)
        self._counts = 0
        self._oldbar = ""
        self._lentxt = 0
        self._range = np.arange(start, stop, step)
        self._len = len(self._range)

    def print(self, txt="", counts=None, c=None):
        """Print the progress bar and optional message."""
        if not c:
            c=self.color
        if counts:
            self._update(counts)
        else:
            self._update(self._counts + self.step)
        if self.bar != self._oldbar:
            self._oldbar = self.bar
            eraser = [" "] * self._lentxt + ["\b"] * self._lentxt
            eraser = "".join(eraser)
            if self.ETA and self._counts>1:
                tdenom = (time.time() - self.clock0)
                if tdenom:
                    vel = self._counts / tdenom
                    self._remt = (self.stop - self._counts) / vel
                else:
                    vel = 1
                    self._remt = 0.
                if self._remt > 60:
                    mins = int(self._remt / 60)
                    secs = self._remt - 60 * mins
                    mins = str(mins) + "m"
                    secs = str(int(secs + 0.5)) + "s "
                else:
                    mins = ""
                    secs = str(int(self._remt + 0.5)) + "s "
                vel = str(round(vel, 1))
                eta = "ETA: " + mins + secs + "(" + vel + " it/s) "
                if self._remt < 1:
                    dt = time.time() - self.clock0
                    if dt > 60:
                        mins = int(dt / 60)
                        secs = dt - 60 * mins
                        mins = str(mins) + "m"
                        secs = str(int(secs + 0.5)) + "s "
                    else:
                        mins = ""
                        secs = str(int(dt + 0.5)) + "s "
                    eta = "elapsed: " + mins + secs + "(" + vel + " it/s)        "
                    txt = ""
            else:
                eta = ""
            txt = eta + str(txt)
            s = self.bar + " " + eraser + txt + "\r"
            printc(s, c=c, bold=self.bold, italic=self.italic, end="")
            sys.stdout.flush()

            if self.percent == 100:
                print("")
            self._lentxt = len(txt)

    def range(self):
        """Return the range iterator."""
        return self._range

    def len(self):
        """Return the number of steps."""
        return self._len

    def _update(self, counts):
        if counts < self.start:
            counts = self.start
        elif counts > self.stop:
            counts = self.stop
        self._counts = counts
        self.percent = (self._counts - self.start) * 100
        dd = self.stop - self.start
        if dd:
            self.percent /= self.stop - self.start
        else:
            self.percent = 0
        self.percent = int(round(self.percent))
        af = self.width - 2
        nh = int(round(self.percent / 100 * af))
        br_bk = "\x1b[2m"+self.char_back*(af-nh)
        br = "%s%s%s" % (self.char*(nh-1), self.char_arrow, br_bk)
        self.bar = self.title + self.char0 + br + self.char1
        if self.percent < 100:
            ps = " " + str(self.percent) + "%"
        else:
            ps = ""
        self.bar += ps


class dotdict(dict):
    """
    A dictionary supporting dot notation.

    Example:

        .. code-block:: python

            dd = dotdict({"a": 1,
                          "b": {"c": "hello",
                                "d": [1, 2, {"e": 123}]}
                              }
                         )
            dd.update({'k':3})
            dd.g = 7
            print("k=", dd.k)           # k= 3
            print(dd.b.c)               # hello
            print(isinstance(dd, dict)) # True
            print(dd.lookup("b.d"))     # [1, 2, {"e": 123}]
    """
    # Credits: https://stackoverflow.com/users/89391/miku
    #  https://gist.github.com/miku/dc6d06ed894bc23dfd5a364b7def5ed8

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = dotdict(v)

    def lookup(self, dotkey):
        """
        Lookup value in a nested structure with a single key, e.g. "a.b.c".
        """
        path = list(reversed(dotkey.split(".")))
        v = self
        while path:
            key = path.pop()
            if isinstance(v, dict):
                v = v[key]
            elif isinstance(v, list):
                v = v[int(key)]
            else:
                raise KeyError(key)
        return v


###########################################################
def geometry(obj, extent=None):
    """
    Apply the ``vtkGeometryFilter``.
    This is a general-purpose filter to extract geometry (and associated data)
    from any type of dataset.
    This filter also may be used to convert any type of data to polygonal type.
    The conversion process may be less than satisfactory for some 3D datasets.
    For example, this filter will extract the outer surface of a volume
    or structured grid dataset.

    Returns a ``Mesh`` object.

    :param list extent: set a `[xmin,xmax, ymin,ymax, zmin,zmax]` bounding box to clip data.
    """
    from vedo.mesh import Mesh
    gf = vtk.vtkGeometryFilter()
    gf.SetInputData(obj)
    if extent is not None:
        gf.SetExtent(extent)
    gf.Update()
    return Mesh(gf.GetOutput())


def buildPolyData(vertices, faces=None, lines=None, indexOffset=0, fast=True, tetras=False):
    """
    Build a ``vtkPolyData`` object from a list of vertices
    where faces represents the connectivity of the polygonal mesh.

    E.g. :
        - ``vertices=[[x1,y1,z1],[x2,y2,z2], ...]``
        - ``faces=[[0,1,2], [1,2,3], ...]``
        - ``lines=[[0,1], [1,2,3,4], ...]``

    Use ``indexOffset=1`` if face numbering starts from 1 instead of 0.

    If fast=False the mesh is built "manually" by setting polygons and triangles
    one by one. This is the fallback case when a mesh contains faces of
    different number of vertices.

    If tetras=True, interpret 4-point faces as tetrahedrons instead of surface quads.
    """
    poly = vtk.vtkPolyData()

    if len(vertices) == 0:
        return poly

    if not isSequence(vertices[0]):
        return poly

    if len(vertices[0]) < 3: # make sure it is 3d
        vertices = np.c_[np.array(vertices), np.zeros(len(vertices))]
        if len(vertices[0]) == 2: # make sure it was not 1d!
            vertices = np.c_[vertices, np.zeros(len(vertices))]

    sourcePoints = vtk.vtkPoints()
    sourcePoints.SetData(numpy_to_vtk(np.ascontiguousarray(vertices), deep=True))
    poly.SetPoints(sourcePoints)

    if lines is not None:
        # Create a cell array to store the lines in and add the lines to it
        linesarr = vtk.vtkCellArray()
        if isSequence(lines[0]): # assume format [(id0,id1),..]
            for iline in lines:
                for i in range(0, len(iline)-1):
                    i1, i2 =  iline[i], iline[i+1]
                    if i1 != i2:
                        vline = vtk.vtkLine()
                        vline.GetPointIds().SetId(0,i1)
                        vline.GetPointIds().SetId(1,i2)
                        linesarr.InsertNextCell(vline)
        else: # assume format [id0,id1,...]
            for i in range(0, len(lines)-1):
                vline = vtk.vtkLine()
                vline.GetPointIds().SetId(0,lines[i])
                vline.GetPointIds().SetId(1,lines[i+1])
                linesarr.InsertNextCell(vline)
            #print('Wrong format for lines in utils.buildPolydata(), skip.')
        poly.SetLines(linesarr)

    if faces is None:
        sourceVertices = vtk.vtkCellArray()
        for i in range(len(vertices)):
            sourceVertices.InsertNextCell(1)
            sourceVertices.InsertCellPoint(i)
        poly.SetVerts(sourceVertices)
        return poly ###################


    # faces exist
    sourcePolygons = vtk.vtkCellArray()
    faces = np.array(faces)
    if len(faces.shape) == 2 and indexOffset==0 and fast:
        #################### all faces are composed of equal nr of vtxs, FAST

        ast = np.int32
        if vtk.vtkIdTypeArray().GetDataTypeSize() != 4:
            ast = np.int64

        nf, nc = faces.shape
        hs = np.hstack((np.zeros(nf)[:,None] + nc, faces)).astype(ast).ravel()
        arr = numpy_to_vtkIdTypeArray(hs, deep=True)
        sourcePolygons.SetCells(nf, arr)

    else: ########################################## manually add faces, SLOW

        showbar = False
        if len(faces) > 25000:
            showbar = True
            pb = ProgressBar(0, len(faces), ETA=False)

        for f in faces:
            n = len(f)

            if n == 3:
                ele = vtk.vtkTriangle()
                pids = ele.GetPointIds()
                for i in range(3):
                    pids.SetId(i, f[i] - indexOffset)
                sourcePolygons.InsertNextCell(ele)

            elif n == 4 and tetras:
                # do not use vtkTetra() because it fails
                # with dolfin faces orientation
                ele0 = vtk.vtkTriangle()
                ele1 = vtk.vtkTriangle()
                ele2 = vtk.vtkTriangle()
                ele3 = vtk.vtkTriangle()
                if indexOffset:
                    for i in [0,1,2,3]:
                        f[i] -= indexOffset
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

                sourcePolygons.InsertNextCell(ele0)
                sourcePolygons.InsertNextCell(ele1)
                sourcePolygons.InsertNextCell(ele2)
                sourcePolygons.InsertNextCell(ele3)

            else:
                ele = vtk.vtkPolygon()
                pids = ele.GetPointIds()
                pids.SetNumberOfIds(n)
                for i in range(n):
                    pids.SetId(i, f[i] - indexOffset)
                sourcePolygons.InsertNextCell(ele)
            if showbar:
                pb.print("converting mesh...    ")

    poly.SetPolys(sourcePolygons)
    return poly

##############################################################################
def isSequence(arg):
    """Check if input is iterable."""
    if hasattr(arg, "strip"):
        return False
    if hasattr(arg, "__getslice__"):
        return True
    if hasattr(arg, "__iter__"):
        return True
    return False


def flatten(list_to_flatten):
    """Flatten out a list."""

    def genflatten(lst):
        for elem in lst:
            if isinstance(elem, (list, tuple)):
                for x in flatten(elem):
                    yield x
            else:
                yield elem

    return list(genflatten(list_to_flatten))


def humansort(l):
    """Sort in place a given list the way humans expect.

    NB: input list is modified

    E.g. ['file11', 'file1'] -> ['file1', 'file11']
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

    l.sort(key=alphanum_key)
    return l  # NB: input list is modified


def sortByColumn(array, nth):
    '''Sort a numpy array by the `n-th` column'''
    return array[array[:,nth].argsort()]


def findDistanceToLines2D(P0,P1, pts):
    """
    Consider 2 sets of points P0,P1 describing lines (2D) and a set of points pts,
    compute distance from each point j (P[j]) to each line i (P0[i],P1[i]).
    """
    #Author: Italmassov Kuanysh, at https://github.com/rougier/numpy-100
    def _distance(P0, P1, p):
        T = P1 - P0
        L = (T**2).sum(axis=1)
        U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
        U = U.reshape(len(U),1)
        D = P0 + U*T - p
        return np.sqrt((D**2).sum(axis=1))
    return [_distance(P0,P1,p_i) for p_i in pts]


def pointIsInTriangle(p, p1, p2, p3):
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
        if 0 < beta < 1 :
            alpha = 1 - gamma - beta
            if 0 < alpha < 1:
                return True
    return False


def pointToLineDistance(p, p1, p2):
    """Compute the distance of a point to a line (not the segment) defined by `p1` and `p2`."""
    d = np.sqrt(vtk.vtkLine.DistanceToLine(p, p1, p2))
    return d


def linInterpolate(x, rangeX, rangeY):
    """
    Interpolate linearly variable x in rangeX onto rangeY.
    If x is a 3D vector the linear weight is the distance to the two 3D rangeX vectors.

    E.g. if x runs in rangeX=[x0,x1] and I want it to run in rangeY=[y0,y1] then
    y = linInterpolate(x, rangeX, rangeY) will interpolate x onto rangeY.

    |linInterpolate| |linInterpolate.py|_
    """
    if isSequence(x):
        x = np.array(x)
        x0, x1 = np.array(rangeX)
        y0, y1 = np.array(rangeY)
        if len(np.unique([x.shape, x0.shape, x1.shape, y1.shape]))>1:
            printc("Error in linInterpolate(): mismatch in input shapes.", c='r')
            raise RuntimeError()
        dx = x1 - x0
        dxn = np.linalg.norm(dx)
        if not dxn:
            return y0
        s = np.linalg.norm(x - x0) / dxn
        t = np.linalg.norm(x - x1) / dxn
        st = s + t
        out = y0 * (t/st) + y1 * (s/st)
    else: #faster
        x0 = rangeX[0]
        dx = rangeX[1] - x0
        if not dx:
            return rangeY[0]
        s = (x - x0) / dx
        out = rangeY[0] * (1 - s) + rangeY[1] * s
    return out


def fitCircle2D(points):
    """
    Fits a circle through points in the plane, with a very fast non-iterative method.

    Returns the tuple (center_x, center_y, radius).

    Reference: J.F. Crawford, Nucl. Instr. Meth. 211, 1983, 223-225.

    .. note:: E.g.:

        .. code-block:: python

            from vedo import *

            x = arange(0, 3, 0.1)
            y = sin(x)

            cx, cy, R = fitCircle2D([x,y])
            Points([x, y])
            Point([cx, cy])
            Circle([cx, cy], r=R)
            show(..., axes=8)
    """
    if len(points) == 2:
        data = np.c_[points[0],points[1]]
    else:
        data = np.array(points)
    xi = data[:,0]
    yi = data[:,1]

    x   = sum(xi)
    xx  = sum(xi*xi)
    xxx = sum(xi*xi*xi)

    y   = sum(yi)
    yy  = sum(yi*yi)
    yyy = sum(yi*yi*yi)

    xy  = sum(xi*yi)
    xyy = sum(xi*yi*yi)
    xxy = sum(xi*xi*yi)

    N = len(xi)
    k = (xx+yy)/N

    a1 = xx-x*x/N
    b1 = xy-x*y/N
    c1 = 0.5*(xxx + xyy - x*k)

    a2 = xy-x*y/N
    b2 = yy-y*y/N
    c2 = 0.5*(xxy + yyy - y*k)

    x0 = (b1*c2 - b2*c1)/(a2*b1 - a1*b2)
    y0 = (c1 - a1*x0)/b1

    R = np.sqrt(x0*x0 + y0*y0 -1/N*(2*x0*x +2*y0*y -xx -yy))
    return x0, y0, R


def vector(x, y=None, z=0.0, dtype=np.float64):
    """Return a 3D numpy array representing a vector.

    If `y` is ``None``, assume input is in the form `[x,y,z]`.
    """
    if y is None:  # assume x is already [x,y,z]
        return np.array(x, dtype=dtype)
    return np.array([x, y, z], dtype=dtype)


def versor(v):
    """Return the unit vector. Input can be a list of vectors."""
    if isinstance(v[0], np.ndarray):
        return np.divide(v, mag(v)[:, None])
    else:
        return v / mag(v)


def mag(z):
    """Get the magnitude of a vector."""
    if isinstance(z[0], np.ndarray):
        return np.array(list(map(np.linalg.norm, z)))
    else:
        return np.linalg.norm(z)


def mag2(z):
    """Get the squared magnitude of a vector."""
    return np.dot(z, z)


def precision(x, p, vrange=None, delimiter='e'):
    """
    Returns a string representation of `x` formatted with precision `p`.

    :param float vrange: range in which x exists (to snap it to '0' if below precision).

    Based on the webkit javascript implementation taken
    `from here <https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp>`_,
    and implemented by `randlet <https://github.com/randlet/to-precision>`_.
    """
    #round_to_n = lambda x, n: np.around(x, -np.int(np.floor(np.log10(np.abs(x)))) + (n - 1))
    #x= 13556.783434
    #print(round_to_n(x, 3))
    #print(precision(x, 3))
    if isSequence(x):
        out = '('
        nn=len(x)-1
        for i, ix in enumerate(x):
            out += precision(ix, p)
            if i<nn: out += ', '
        return out+')' ############ <--

    x = float(x)

    if x == 0.0 or (vrange is not None and abs(x) < vrange/pow(10,p)):
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


# 2d
def cart2pol(x, y):
    """Cartesian to Polar coordinates conversion."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return rho, theta

def pol2cart(rho, theta):
    """Polar to Cartesian coordinates conversion."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

# 3d
def cart2spher(x, y, z):
    """Cartesian to Spherical coordinate conversion."""
    hxy = np.hypot(x, y)
    rho = np.hypot(hxy, z)
    #if not rho:
    #    return np.array([0,0,0])
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    return rho, theta, phi

def spher2cart(rho, theta, phi):
    """Spherical to Cartesian coordinate conversion."""
    st = np.sin(theta)
    sp = np.sin(phi)
    ct = np.cos(theta)
    cp = np.cos(phi)
    rst = rho * st
    x = rst * cp
    y = rst * sp
    z = rho * ct
    return np.array([x, y, z])

def cart2cyl(x,y,z):
    rho = np.sqrt(x*x+y*y+z*z)
    theta = np.arctan2(y, x)
    return rho, theta, z

def cyl2cart(rho, theta, z):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y, z

def cyl2spher(rho,theta,z):
    rhos = np.sqrt(rho*rho+z*z)
    phi = np.arctan2(rho, z)
    return rhos, theta, phi

def spher2cyl(rho, theta, phi):
    rhoc = rho * np.sin(phi)
    z = rho * np.cos(phi)
    return rhoc, theta, z


def grep(filename, tag, firstOccurrence=False):
    """Greps the line that starts with a specific `tag` string inside the file."""
    import re

    try:
        afile = open(filename, "r")
    except:
        print("Error in utils.grep(): cannot open file", filename)
        raise RuntimeError()
    content = None
    for line in afile:
        if re.search(tag, line):
            content = line.split()
            if firstOccurrence:
                break
    if content:
        if len(content) == 2:
            content = content[1]
        else:
            content = content[1:]
    afile.close()
    return content


def printInfo(obj):
    """Print information about a vtk object."""

    from vedo.tetmesh import TetMesh

    def printvtkactor(actor, tab=""):

        if not actor.GetPickable():
            return

        mapper = actor.GetMapper()
        if hasattr(actor, "polydata"):
            poly = actor.polydata()
        else:
            poly = mapper.GetInput()

        pro = actor.GetProperty()
        pos = actor.GetPosition()
        bnds = actor.GetBounds()
        col = pro.GetColor()
        colr = precision(col[0], 3)
        colg = precision(col[1], 3)
        colb = precision(col[2], 3)
        alpha = pro.GetOpacity()
        npt = poly.GetNumberOfPoints()
        ncl = poly.GetNumberOfCells()
        npl = poly.GetNumberOfPolys()

        print(tab, end="")
        printc("Mesh", c="g", bold=1, invert=1, dim=1, end=" ")

        if hasattr(actor, "_legend") and actor._legend:
            printc("legend: ", c="g", bold=1, end="")
            printc(actor._legend, c="g", bold=0)
        else:
            print()

        if hasattr(actor, "name") and actor.name:
            printc(tab + "           name: ", c="g", bold=1, end="")
            printc(actor.name, c="g", bold=0)

        if hasattr(actor, "filename") and actor.filename:
            printc(tab + "           file: ", c="g", bold=1, end="")
            printc(actor.filename, c="g", bold=0)

        if hasattr(actor, "_time") and actor._time:
            printc(tab + "           time: ", c="g", bold=1, end="")
            printc(actor._time, c="g", bold=0)

        if not actor.GetMapper().GetScalarVisibility():
            printc(tab + "          color: ", c="g", bold=1, end="")
            #printc("defined by point or cell data", c="g", bold=0)
        #else:
            printc(vedo.colors.getColorName(col) + ', rgb=('+colr+', '
                          + colg+', '+colb+'), alpha='+str(alpha), c='g', bold=0)

            if actor.GetBackfaceProperty():
                bcol = actor.GetBackfaceProperty().GetDiffuseColor()
                bcolr = precision(bcol[0], 3)
                bcolg = precision(bcol[1], 3)
                bcolb = precision(bcol[2], 3)
                printc(tab+'     back color: ', c='g', bold=1, end='')
                printc(vedo.colors.getColorName(bcol) + ', rgb=('+bcolr+', '
                              + bcolg+', ' + bcolb+')', c='g', bold=0)

        printc(tab + "         points: ", c="g", bold=1, end="")
        printc(npt, c="g", bold=0)

        printc(tab + "          cells: ", c="g", bold=1, end="")
        printc(ncl, c="g", bold=0)

        printc(tab + "       polygons: ", c="g", bold=1, end="")
        printc(npl, c="g", bold=0)

        printc(tab + "       position: ", c="g", bold=1, end="")
        printc(pos, c="g", bold=0)

        if hasattr(actor, "polydata") and actor.N():
            printc(tab + " center of mass: ", c="g", bold=1, end="")
            cm = tuple(actor.centerOfMass())
            printc(precision(cm, 3), c="g", bold=0)

            printc(tab + "   average size: ", c="g", bold=1, end="")
            printc(precision(actor.averageSize(), 6), c="g", bold=0)

            printc(tab + "  diagonal size: ", c="g", bold=1, end="")
            printc(precision(actor.diagonalSize(), 6), c="g", bold=0)

            if hasattr(actor, "area"):
                _area = actor.area()
                if _area:
                    printc(tab + "           area: ", c="g", bold=1, end="")
                    printc(precision(_area, 6), c="g", bold=0)

                _vol = actor.volume()
                if _vol:
                    printc(tab + "         volume: ", c="g", bold=1, end="")
                    printc(precision(_vol, 6), c="g", bold=0)

        printc(tab + "         bounds: ", c="g", bold=1, end="")
        bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
        printc("x=(" + bx1 + ", " + bx2 + ")", c="g", bold=0, end="")
        by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
        printc(" y=(" + by1 + ", " + by2 + ")", c="g", bold=0, end="")
        bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
        printc(" z=(" + bz1 + ", " + bz2 + ")", c="g", bold=0)

        if actor.picked3d is not None:
            printc(tab + "  clicked point: ", c="g", bold=1, end="")
            printc(vector(actor.picked3d), c="g", bold=0)

        ptdata = poly.GetPointData()
        cldata = poly.GetCellData()
        if ptdata.GetNumberOfArrays() + cldata.GetNumberOfArrays():

            arrtypes = dict()
            arrtypes[vtk.VTK_UNSIGNED_CHAR] = "UNSIGNED_CHAR"
            arrtypes[vtk.VTK_SIGNED_CHAR]   = "SIGNED_CHAR"
            arrtypes[vtk.VTK_UNSIGNED_INT]  = "UNSIGNED_INT"
            arrtypes[vtk.VTK_INT]           = "INT"
            arrtypes[vtk.VTK_CHAR]          = "CHAR"
            arrtypes[vtk.VTK_SHORT]         = "SHORT"
            arrtypes[vtk.VTK_LONG]          = "LONG"
            arrtypes[vtk.VTK_ID_TYPE]       = "ID"
            arrtypes[vtk.VTK_FLOAT]         = "FLOAT"
            arrtypes[vtk.VTK_DOUBLE]        = "DOUBLE"
            printc(tab + "    scalar mode:", c="g", bold=1, end=" ")
            printc(mapper.GetScalarModeAsString(),
                          '  coloring =', mapper.GetColorModeAsString(), c="g", bold=0)

            printc(tab + "   active array: ", c="g", bold=1, end="")
            if ptdata.GetScalars():
                printc(ptdata.GetScalars().GetName(), "(point data)  ", c="g", bold=0, end="")
            if cldata.GetScalars():
                printc(cldata.GetScalars().GetName(), "(cell data)", c="g", bold=0, end="")
            print()

            for i in range(ptdata.GetNumberOfArrays()):
                name = ptdata.GetArrayName(i)
                if name and ptdata.GetArray(i):
                    printc(tab + "     point data: ", c="g", bold=1, end="")
                    try:
                        tt = arrtypes[ptdata.GetArray(i).GetDataType()]
                    except:
                        tt = str(ptdata.GetArray(i).GetDataType())
                    ncomp = str(ptdata.GetArray(i).GetNumberOfComponents())
                    printc("name=" + name, "("+ncomp+" "+tt+"),", c="g", bold=0, end="")
                    rng = ptdata.GetArray(i).GetRange()
                    printc(" range=(" + precision(rng[0],4) + ',' +
                                            precision(rng[1],4) + ')', c="g", bold=0)

            for i in range(cldata.GetNumberOfArrays()):
                name = cldata.GetArrayName(i)
                if name and cldata.GetArray(i):
                    printc(tab + "      cell data: ", c="g", bold=1, end="")
                    try:
                        tt = arrtypes[cldata.GetArray(i).GetDataType()]
                    except:
                        tt = str(cldata.GetArray(i).GetDataType())
                    ncomp = str(cldata.GetArray(i).GetNumberOfComponents())
                    printc("name=" + name, "("+ncomp+" "+tt+"),", c="g", bold=0, end="")
                    rng = cldata.GetArray(i).GetRange()
                    printc(" range=(" + precision(rng[0],4) + ',' +
                                            precision(rng[1],4) + ')', c="g", bold=0)
        else:
            printc(tab + "        scalars:", c="g", bold=1, end=" ")
            printc('no point or cell scalars are present.', c="g", bold=0)


    if not obj:
        return

    elif isinstance(obj, vtk.vtkActor):
        printc("_" * 65, c="g", bold=0)
        printvtkactor(obj)

    elif isinstance(obj, vtk.vtkAssembly):
        printc("_" * 65, c="g", bold=0)
        printc("vtkAssembly", c="g", bold=1, invert=1, end=" ")
        if hasattr(obj, "_legend"):
            printc("legend: ", c="g", bold=1, end="")
            printc(obj._legend, c="g", bold=0)
        else:
            print()

        pos = obj.GetPosition()
        bnds = obj.GetBounds()
        printc("          position: ", c="g", bold=1, end="")
        printc(pos, c="g", bold=0)

        printc("            bounds: ", c="g", bold=1, end="")
        bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
        printc("x=(" + bx1 + ", " + bx2 + ")", c="g", bold=0, end="")
        by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
        printc(" y=(" + by1 + ", " + by2 + ")", c="g", bold=0, end="")
        bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
        printc(" z=(" + bz1 + ", " + bz2 + ")", c="g", bold=0)

        cl = vtk.vtkPropCollection()
        obj.GetActors(cl)
        cl.InitTraversal()
        for i in range(obj.GetNumberOfPaths()):
            act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
            if isinstance(act, vtk.vtkActor):
                printvtkactor(act, tab="     ")

    elif isinstance(obj, TetMesh):
        return # todo

    elif isinstance(obj, vtk.vtkVolume):
        printc("_" * 65, c="b", bold=0)
        printc("vtkVolume", c="b", bold=1, invert=1, end=" ")
        if hasattr(obj, "_legend") and obj._legend:
            printc("legend: ", c="b", bold=1, end="")
            printc(obj._legend, c="b", bold=0)
        else:
            print()

        pos = obj.GetPosition()
        bnds = obj.GetBounds()
        img = obj.GetMapper().GetInput()
        printc("         position: ", c="b", bold=1, end="")
        printc(pos, c="b", bold=0)

        printc("       dimensions: ", c="b", bold=1, end="")
        printc(img.GetDimensions(), c="b", bold=0)
        printc("          spacing: ", c="b", bold=1, end="")
        printc(img.GetSpacing(), c="b", bold=0)
        printc("   data dimension: ", c="b", bold=1, end="")
        printc(img.GetDataDimension(), c="b", bold=0)

        printc("      memory size: ", c="b", bold=1, end="")
        printc(int(img.GetActualMemorySize()/1024), 'Mb', c="b", bold=0)

        printc("    scalar #bytes: ", c="b", bold=1, end="")
        printc(img.GetScalarSize(), c="b", bold=0)

        printc("           bounds: ", c="b", bold=1, end="")
        bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
        printc("x=(" + bx1 + ", " + bx2 + ")", c="b", bold=0, end="")
        by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
        printc(" y=(" + by1 + ", " + by2 + ")", c="b", bold=0, end="")
        bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
        printc(" z=(" + bz1 + ", " + bz2 + ")", c="b", bold=0)

        printc("     scalar range: ", c="b", bold=1, end="")
        printc(img.GetScalarRange(), c="b", bold=0)

        printHistogram(obj, horizontal=True,
                       logscale=True, bins=8, height=15, c='b', bold=0)

    elif hasattr(obj, "interactor"):  # dumps Plotter info
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
        }
        bns, totpt = [], 0
        for a in obj.actors:
            b = a.GetBounds()
            if a.GetBounds() is not None:
                if isinstance(a, vtk.vtkActor):
                    totpt += a.GetMapper().GetInput().GetNumberOfPoints()
                bns.append(b)
        if len(bns) == 0:
            return
        acts = obj.getMeshes()
        printc("_" * 65, c="c", bold=0)
        printc("Plotter", invert=1, dim=1, c="c", end=" ")
        otit = obj.title
        if not otit:
            otit = None
        printc("   title:", otit, bold=0, c="c")
        printc(" active renderer:", obj.renderers.index(obj.renderer), bold=0, c="c")
        printc("   nr. of actors:", len(acts), bold=0, c="c", end="")
        printc(" (" + str(totpt), "vertices)", bold=0, c="c")
        max_bns = np.max(bns, axis=0)
        min_bns = np.min(bns, axis=0)
        printc("      max bounds: ", c="c", bold=0, end="")
        bx1, bx2 = precision(min_bns[0], 3), precision(max_bns[1], 3)
        printc("x=(" + bx1 + ", " + bx2 + ")", c="c", bold=0, end="")
        by1, by2 = precision(min_bns[2], 3), precision(max_bns[3], 3)
        printc(" y=(" + by1 + ", " + by2 + ")", c="c", bold=0, end="")
        bz1, bz2 = precision(min_bns[4], 3), precision(max_bns[5], 3)
        printc(" z=(" + bz1 + ", " + bz2 + ")", c="c", bold=0)
        if isinstance(obj.axes, dict): obj.axes=1
        if obj.axes:
            printc("       axes type:", obj.axes, axtype[obj.axes], bold=0, c="c")

        for a in obj.getVolumes():
            if a.GetBounds() is not None:
                img = a.GetMapper().GetDataSetInput()
                printc('_'*65, c='b', bold=0)
                printc('Volume', invert=1, dim=1, c='b')
                printc('      scalar range:',
                              np.round(img.GetScalarRange(), 4), c='b', bold=0)
                bnds = a.GetBounds()
                printc("            bounds: ", c="b", bold=0, end="")
                bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
                printc("x=(" + bx1 + ", " + bx2 + ")", c="b", bold=0, end="")
                by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
                printc(" y=(" + by1 + ", " + by2 + ")", c="b", bold=0, end="")
                bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
                printc(" z=(" + bz1 + ", " + bz2 + ")", c="b", bold=0)

        printc(" Click mesh and press i for info.", c="c")

    else:
        printc("_" * 65, c="g", bold=0)
        printc(type(obj), c="g", invert=1)



def printHistogram(data, bins=10, height=10, logscale=False, minbin=0,
                   horizontal=False, char=u"\U00002589",
                   c=None, bold=True, title='Histogram'):
    """
    Ascii histogram printing.
    Input can also be ``Volume`` or ``Mesh``.
    Returns the raw data before binning (useful when passing vtk objects).

    :param int bins: number of histogram bins
    :param int height: height of the histogram in character units
    :param bool logscale: use logscale for frequencies
    :param int minbin: ignore bins before minbin
    :param bool horizontal: show histogram horizontally
    :param str char: character to be used
    :param str,int c: ascii color
    :param bool char: use boldface
    :param str title: histogram title

    :Example:
        .. code-block:: python

            from vedo import printHistogram
            import np as np
            d = np.random.normal(size=1000)
            data = printHistogram(d, c='blue', logscale=True, title='my scalars')
            data = printHistogram(d, c=1, horizontal=1)
            print(np.mean(data)) # data here is same as d

        |printhisto|
    """
    # Adapted from http://pyinsci.blogspot.com/2009/10/ascii-histograms.html

    if not horizontal: # better aspect ratio
        bins *= 2

    isimg = isinstance(data, vtk.vtkImageData)
    isvol = isinstance(data, vtk.vtkVolume)
    if isimg or isvol:
        if isvol:
            img = data.imagedata()
        else:
            img = data
        dims = img.GetDimensions()
        nvx = min(100000, dims[0]*dims[1]*dims[2])
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
                return

        from vtk.util.numpy_support import vtk_to_numpy
        data = vtk_to_numpy(arr)

    h = np.histogram(data, bins=bins)

    if minbin:
        hi = h[0][minbin:-1]
    else:
        hi = h[0]

    if sys.version_info[0] < 3 and char == u"\U00002589":
        char = "*" # python2 hack
    if char == u"\U00002589" and horizontal:
        char = u"\U00002586"

    entrs = "\t(entries=" + str(len(data)) + ")"
    if logscale:
        h0 = np.log10(hi+1)
        maxh0 = int(max(h0)*100)/100
        title = '(logscale) ' + title + entrs
    else:
        h0 = hi
        maxh0 = max(h0)
        title = title + entrs

    def _v():
        his = ""
        if title:
            his += title +"\n"
        bars = h0 / maxh0 * height
        for l in reversed(range(1, height + 1)):
            line = ""
            if l == height:
                line = "%s " % maxh0
            else:
                line = "   |" + " " * (len(str(maxh0))-3)
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
            his += title +"\n"
        xl = ["%.2f" % n for n in h[1]]
        lxl = [len(l) for l in xl]
        bars = h0 / maxh0 * height
        his += " " * int(max(bars) + 2 + max(lxl)) + "%s\n" % maxh0
        for i, c in enumerate(bars):
            line = (xl[i] + " " * int(max(lxl) - lxl[i]) + "| " + char * int(c) + "\n")
            his += line
        return his

    if horizontal:
        height *= 2
        printc(_h(), c=c, bold=bold)
    else:
        printc(_v(), c=c, bold=bold)
    return data


def makeBands(inputlist, numberOfBands):
    """
    Group values of a list into bands of equal value.

    :param int numberOfBands: number of bands, a positive integer > 2.
    :return: a binned list of the same length as the input.
    """
    if numberOfBands < 2:
        return inputlist
    vmin = np.min(inputlist)
    vmax = np.max(inputlist)
    bb = np.linspace(vmin, vmax, numberOfBands, endpoint=0)
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
def cameraFromQuaternion(pos, quaternion, distance=10000, ngl_correct=True):
    """Define a ``vtkCamera`` with a particular orientation.

        Parameters
        ----------
        pos: np.array, list, tuple
            an iterator of length 3 containing the focus point of the camera
        quaternion: np.array, list, tuple
            a len(4) quaternion (x,y,z,w) describing the rotation of the camera
            such as returned by neuroglancer x,y,z,w all in [0,1] range
        distance: float
            the desired distance from pos to the camera (default = 10000 nm)

        Returns
        -------
        vtk.vtkCamera
            a vtk camera setup according to these rules.
    """
    camera = vtk.vtkCamera()
    # define the quaternion in vtk, note the swapped order
    # w,x,y,z instead of x,y,z,w
    quat_vtk = vtk.vtkQuaterniond(
        quaternion[3], quaternion[0], quaternion[1], quaternion[2]
    )
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


def cameraFromNeuroglancer(state, zoom=300):
    """Define a ``vtkCamera`` from a neuroglancer state dictionary.

        Parameters
        ----------
        state: dict
            an neuroglancer state dictionary.
        zoom: float
            how much to multiply zoom by to get camera backoff distance
            default = 300 > ngl_zoom = 1 > 300 nm backoff distance.

        Returns
        -------
        vtk.vtkCamera
            a vtk camera setup that matches this state.
    """
    orient = state.get("perspectiveOrientation", [0.0, 0.0, 0.0, 1.0])
    pzoom = state.get("perspectiveZoom", 10.0)
    position = state["navigation"]["pose"]["position"]
    pos_nm = np.array(position["voxelCoordinates"]) * position["voxelSize"]
    return cameraFromQuaternion(pos_nm, orient, pzoom * zoom, ngl_correct=True)


def orientedCamera(center=(0,0,0), upVector=(0,1,0), backoffVector=(0,0,1), backoff=1):
    """
    Generate a ``vtkCamera`` pointed at a specific location,
    oriented with a given up direction, set to a backoff.
    """
    vup = np.array(upVector)
    vup = vup / np.linalg.norm(vup)
    pt_backoff = center - backoff * np.array(backoffVector)
    camera = vtk.vtkCamera()
    camera.SetFocalPoint(center[0],center[1],center[2])
    camera.SetViewUp(vup[0], vup[1], vup[2])
    camera.SetPosition(pt_backoff[0], pt_backoff[1], pt_backoff[2])
    return camera


def vtkCameraToK3D(vtkcam):
    """
    Convert a ``vtkCamera`` object into a 9-element list to be used by K3D backend.

    Output format is: [posx,posy,posz, targetx,targety,targetz, upx,upy,upz]
    """
    cpos = np.array(vtkcam.GetPosition())
    kam = [cpos.tolist()]
    kam.append(vtkcam.GetFocalPoint())
    kam.append(vtkcam.GetViewUp())
    return np.array(kam).ravel()


def make_ticks(x0, x1, N, labels=None, digits=None):

    ticks_str, ticks_float = [], []

    if labels is not None:
        # user is passing custom labels
        ticks_float.append(0)
        ticks_str.append('')
        for tp, ts in labels:
            if tp == x1:
                continue
            ticks_str.append(ts)
            tickn = linInterpolate(tp, [x0,x1], [0,1])
            ticks_float.append(tickn)

    else:

        # automatic
        dstep = (x1-x0)/N  # desired step size
        if dstep <= 0:
            return [0,1], ["",""]

        if x0:
            expo = np.floor(np.log10(abs(x0)))
            lowBound = pow(10, expo)*np.sign(x0)
        else:
            lowBound = 0.0
        if np.sign(x0)<0: lowBound *= 10
        if   lowBound>0 and lowBound-dstep < 0: lowBound = 0
        elif lowBound<0 and lowBound+dstep > 0:
            lowBound = - pow(10, np.floor(np.log10(dstep)))

        if x1:
            expo = np.ceil(np.log10(abs(x1)))
            upBound = pow(10, expo)*np.sign(x1)
        else:
            upBound = 0
        if upBound<0 and lowBound+dstep > 0:
            upBound = 0
        if -1<=upBound<0:
            upBound /= 10
        if lowBound == upBound:
            if upBound<0:
                upBound /= 10
            else:
                upBound *= 10

        basestep = pow(10, np.floor(np.log10(dstep)))
        steps = np.asarray([basestep*i for i in (1,2,4,5,10,20,40,50)])
        idx = (np.abs(steps-dstep)).argmin()
        s = steps[idx]
        if not s or (upBound-lowBound)/s > 100000 or upBound == lowBound:
            return [0,1], ["",""]

        sel_axis = np.arange(lowBound, upBound, s)
        sel_axis = np.clip(sel_axis, x0, x1)
        sel_axis = np.unique(sel_axis)

        if digits is None:
            np.set_printoptions(suppress=True) # avoid zero precision
            sas = str(sel_axis).replace('[','').replace(']','')
            sas = sas.replace('.e','e').replace('e+0','e+').replace('e-0','e-')
            np.set_printoptions(suppress=None) # set back to default
        else:
            sas = precision(sel_axis, digits, vrange=(x0,x1))
            sas = sas.replace('[','').replace(']','').replace(')','').replace(',','')

        sas2 = []
        for s in sas.split():
            if s.endswith('.'):
                s = s[:-1]
            if s == '-0':
                s = '0'
            if digits is not None and 'e' in s: s+=' ' # add space to terminate modifiers
            sas2.append(s)

        for i in range(len(sel_axis)):
            ts = sas2[i]
            tp = sel_axis[i]
            if tp == x1:
                continue
            tickn = linInterpolate(tp, [x0,x1], [0,1])
            ticks_float.append(tickn)
            ticks_str.append(ts)
        #printc('------------------------------------------------', c='y')
        #printc('x0, x1, N       = ', x0, x1, N, c='y')
        #printc('bounds            ', lowBound, upBound)
        #printc('dstep, steps      ', dstep, steps)
        #printc('Result tick array\n', sel_axis, sel_axis.shape[0], c='y')

    ticks_str.append('')
    ticks_float.append(1)
    return ticks_float, ticks_str


############################################################################
#Trimesh support
#
#Install trimesh with:
#
#    sudo apt install python3-rtree
#    pip install rtree shapely
#    conda install trimesh
#
#Check the example gallery in: examples/other/trimesh>
###########################################################################

def vedo2trimesh(mesh):
    """
    Convert ``vedo.Mesh`` to ``Trimesh.Mesh`` object.
    """
    if isSequence(mesh):
        tms = []
        for a in mesh:
            tms.append(vedo2trimesh(a))
        return tms

    from trimesh import Trimesh

    lut = mesh.mapper().GetLookupTable()

    tris = mesh.faces()
    carr = mesh.getCellArray('CellIndividualColors')
    ccols = None
    if carr is not None and len(carr)==len(tris):
        ccols = []
        for i in range(len(tris)):
            r,g,b,a = lut.GetTableValue(carr[i])
            ccols.append((r*255, g*255, b*255, a*255))
        ccols = np.array(ccols, dtype=np.int16)

    points = mesh.points()
    varr = mesh.getPointArray('VertexColors')
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

def trimesh2vedo(inputobj, alphaPerCell=False):
    """
    Convert ``Trimesh`` object to ``Mesh(vtkActor)`` or ``Assembly`` object.
    """
    if isSequence(inputobj):
        vms = []
        for ob in inputobj:
            vms.append(trimesh2vedo(ob))
        return vms

    # print('trimesh2vedo inputobj', type(inputobj))

    inputobj_type = str(type(inputobj))

    if "Trimesh" in inputobj_type or "primitives" in inputobj_type:
        from vedo import Mesh

        faces = inputobj.faces
        poly = buildPolyData(inputobj.vertices, faces)
        tact = Mesh(poly)
        if inputobj.visual.kind == "face":
            trim_c = inputobj.visual.face_colors
        else:
            trim_c = inputobj.visual.vertex_colors

        if isSequence(trim_c):
            if isSequence(trim_c[0]):
                sameColor = len(np.unique(trim_c, axis=0)) < 2 # all vtxs have same color
                trim_c = trim_c/255

                if sameColor:
                    tact.c(trim_c[0, [0,1,2]]).alpha(trim_c[0, 3])
                else:
                    if inputobj.visual.kind == "face":
                        tact.cellIndividualColors(trim_c[:, [0,1,2]],
                                                  alpha=trim_c[:,3],
                                                  alphaPerCell=alphaPerCell)
        return tact

    elif "PointCloud" in inputobj_type:
        from vedo.shapes import Points

        trim_cc, trim_al = "black", 1
        if hasattr(inputobj, "vertices_color"):
            trim_c = inputobj.vertices_color
            if len(trim_c):
                trim_cc = trim_c[:, [0, 1, 2]] / 255
                trim_al = trim_c[:, 3] / 255
                trim_al = np.sum(trim_al) / len(trim_al)  # just the average
        return Points(inputobj.vertices, r=8, c=trim_cc, alpha=trim_al)

    elif "path" in inputobj_type:
        from vedo.shapes import Line
        from vedo.assembly import Assembly

        lines = []
        for e in inputobj.entities:
            # print('trimesh entity', e.to_dict())
            l = Line(inputobj.vertices[e.points], c="k", lw=2)
            lines.append(l)
        return Assembly(lines)

    return None


def vtkVersionIsAtLeast(major, minor=0, build=0):
    """
    Check the VTK version.
    Return ``True`` if the requested VTK version is greater or equal
    to the actual VTK version.

    :param major: Major version.
    :param minor: Minor version.
    :param build: Build version.
    """
    needed_version = 10000000000*int(major) +100000000*int(minor) +int(build)
    try:
        vtk_version_number = vtk.VTK_VERSION_NUMBER
    except AttributeError:  # as error:
        ver = vtk.vtkVersion()
        vtk_version_number = 10000000000 * ver.GetVTKMajorVersion() \
                             + 100000000 * ver.GetVTKMinorVersion() \
                             + ver.GetVTKBuildVersion()
    if vtk_version_number >= needed_version:
        return True
    else:
        return False

def systemReport():
    try:
        import scooby
        r = scooby.Report(additional=['vtk',
                                      'vedo',
                                      'meshio',
                                      'matplotlib',
                                      'k3d',
                                      'itkwidgets',
                                      'panel',
                                      'dolfin',
                                      'trimesh',
                                      'pymeshfix',
                                      'pygmsh',
                                      'nevergrad',
                                      'pyshtools',
                                      'cv2',
                                      ])
        printc(r)
    except:
        print('Install scooby with command: pip install scooby')
        r = ''
    return r


def ctf2lut(tvobj):
    # build LUT from a color transfer function for tmesh or volume
    pr = tvobj.GetProperty()
    if not isinstance(pr, vtk.vtkVolumeProperty):
        return None
    ctf = pr.GetRGBTransferFunction()
    otf = pr.GetScalarOpacity()
    x0,x1 = tvobj.inputdata().GetScalarRange()
    cols, alphas = [],[]
    for x in np.linspace(x0,x1, 256):
        cols.append(ctf.GetColor(x))
        alphas.append(otf.GetValue(x))
    lut = vtk.vtkLookupTable()
    lut.SetRange(x0,x1)
    lut.SetNumberOfTableValues(len(cols))
    for i, col in enumerate(cols):
        r, g, b = col
        lut.SetTableValue(i, r, g, b, alphas[i])
    lut.Build()
    return lut


def resampleArrays(source, target, tol=None):
    """Resample point and cell data of a dataset on points from another dataset.
    It takes two inputs - source and target, and samples the point and cell values
    of target onto the point locations of source.
    The output has the same structure as the source but its point data have
    the resampled values from target.

    :param float tol: set the tolerance used to compute whether
        a point in the target is in a cell of the source.
        Points without resampled values, and their cells, are be marked as blank.
    """
    rs = vtk.vtkResampleWithDataSet()
    rs.SetInputData(source.polydata())
    rs.SetSourceData(target.polydata())
    rs.SetPassPointArrays(True)
    rs.SetPassCellArrays(True)
    if tol:
        rs.SetComputeTolerance(False)
        rs.SetTolerance(tol)
    rs.Update()
    return rs.GetOutput()

