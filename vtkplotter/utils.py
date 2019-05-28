from __future__ import division, print_function
import vtk, sys
import numpy as np
import vtkplotter.colors as colors
import vtkplotter.docs as docs
import time

__doc__ = (
    """
Utilities submodule.
"""
    + docs._defs
)

__all__ = [
    "ProgressBar",
    "isSequence",
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
    "resampleArrays",
    "printHistogram",
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
                pb.print('some message') # or pb.print(counts=i)

        |progbar|
    """

    def __init__(self, start, stop, step=1, c=None, ETA=True, width=24, char=u"\U000025AC"):

        char_arrow = u"\U000025BA"
        if sys.version_info[0]<3:
            char="="
            char_arrow = '>'

        self.start = start
        self.stop = stop
        self.step = step
        self.color = c
        self.width = width
        self.char = char
        self.char_arrow = char_arrow
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

    def print(self, txt="", counts=None):
        """Print the progress bar and optional message."""
        if counts:
            self._update(counts)
        else:
            self._update(self._counts + self.step)
        if self.bar != self._oldbar:
            self._oldbar = self.bar
            eraser = [" "] * self._lentxt + ["\b"] * self._lentxt
            eraser = "".join(eraser)
            if self.ETA:
                vel = self._counts / (time.time() - self.clock0)
                self._remt = (self.stop - self._counts) / vel
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
                    eta = "Elapsed time: " + mins + secs + "(" + vel + " it/s)        "
                    txt = ""
            else:
                eta = ""
            txt = eta + str(txt)
            s = self.bar + " " + eraser + txt + "\r"
            if self.color:
                colors.printc(s, c=self.color, end="")
            else:
                sys.stdout.write(s)
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
        self.percent /= self.stop - self.start
        self.percent = int(round(self.percent))
        af = self.width - 2
        nh = int(round(self.percent / 100 * af))
        if nh == 0:
            self.bar = "["+self.char_arrow+"%s]" % (" " * (af - 1))
        elif nh == af:
            self.bar = "[%s]" % (self.char * af)
        else:
            self.bar = "[%s%s%s]" % (self.char *(nh-1), self.char_arrow, " " *(af-nh))
        if self.percent < 100:  # and self._remt > 1:
            ps = " " + str(self.percent) + "%"
        else:
            ps = ""
        self.bar += ps


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
    return None  # NB: input list is modified


def lin_interp(x, rangeX, rangeY):
    """
    Interpolate linearly variable x in rangeX onto rangeY.
    """
    s = (x - rangeX[0]) / mag(rangeX[1] - rangeX[0])
    y = rangeY[0] * (1 - s) + rangeY[1] * s
    return y


def vector(x, y=None, z=0.0):
    """Return a 3D numpy array representing a vector (of type `numpy.float64`).

    If `y` is ``None``, assume input is already in the form `[x,y,z]`.
    """
    if y is None:  # assume x is already [x,y,z]
        return np.array(x, dtype=np.float64)
    return np.array([x, y, z], dtype=np.float64)


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


def precision(x, p):
    """
    Returns a string representation of `x` formatted with precision `p`.

    Based on the webkit javascript implementation taken
    `from here <https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp>`_,
    and implemented by `randlet <https://github.com/randlet/to-precision>`_.
    """
    import math

    x = float(x)

    if x == 0.0:
        return "0." + "0" * (p - 1)

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
        out.append("e")
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


def pointIsInTriangle(p, p1, p2, p3):
    """
    Return True if a point is inside (or above/below) a triangle defined by 3 points in space.
    """
    p = np.array(p)
    u = np.array(p2) - p1
    v = np.array(p3) - p1
    n = np.cross(u, v)
    w = p - p1
    ln = np.dot(n, n)
    if not ln:
        return True  # degenerate triangle
    gamma = (np.dot(np.cross(u, w), n)) / ln
    beta = (np.dot(np.cross(w, v), n)) / ln
    alpha = 1 - gamma - beta
    if 0 < alpha < 1 and 0 < beta < 1 and 0 < gamma < 1:
        return True
    return False


def pointToLineDistance(p, p1, p2):
    """Compute the distance of a point to a line (not the segment) defined by `p1` and `p2`."""
    d = np.sqrt(vtk.vtkLine.DistanceToLine(p, p1, p2))
    return d


def spher2cart(rho, theta, phi):
    """Spherical to Cartesian coordinate conversion."""
    st = np.sin(theta)
    sp = np.sin(phi)
    ct = np.cos(theta)
    cp = np.cos(phi)
    rhost = rho * st
    x = rhost * cp
    y = rhost * sp
    z = rho * ct
    return np.array([x, y, z])


def cart2spher(x, y, z):
    """Cartesian to Spherical coordinate conversion."""
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(z, hxy)
    phi = np.arctan2(y, x)
    return r, theta, phi


def cart2pol(x, y):
    """Cartesian to Polar coordinates conversion."""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    """Polar to Cartesian coordinates conversion."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def isIdentity(M, tol=1e-06):
    """Check if vtkMatrix4x4 is Identity."""
    for i in [0, 1, 2, 3]:
        for j in [0, 1, 2, 3]:
            e = M.GetElement(i, j)
            if i == j:
                if np.abs(e - 1) > tol:
                    return False
            elif np.abs(e) > tol:
                return False
    return True


def grep(filename, tag, firstOccurrence=False):
    """Greps the line that starts with a specific `tag` string from inside a file."""
    import re

    try:
        afile = open(filename, "r")
    except:
        print("Error in utils.grep(): cannot open file", filename)
        exit()
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

    def printvtkactor(actor, tab=""):

        if not actor.GetPickable():
            return

        if hasattr(actor, "polydata"):
            poly = actor.polydata()
        else:
            poly = actor.GetMapper().GetInput()
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

        print(tab, end="")
        colors.printc("vtkActor", c="g", bold=1, invert=1, dim=1, end=" ")

        if hasattr(actor, "_legend") and actor._legend:
            colors.printc("legend: ", c="g", bold=1, end="")
            colors.printc(actor._legend, c="g", bold=0)
        else:
            print()

        if hasattr(actor, "filename") and actor.filename:
            colors.printc(tab + "           file: ", c="g", bold=1, end="")
            colors.printc(actor.filename, c="g", bold=0)

        colors.printc(tab + "          color: ", c="g", bold=1, end="")
        if actor.GetMapper().GetScalarVisibility():
            colors.printc("defined by point or cell data", c="g", bold=0)
        else:
            colors.printc(colors.getColorName(col) + ', rgb=('+colr+', '
                          + colg+', '+colb+'), alpha='+str(alpha), c='g', bold=0)

            if actor.GetBackfaceProperty():
                bcol = actor.GetBackfaceProperty().GetDiffuseColor()
                bcolr = precision(bcol[0], 3)
                bcolg = precision(bcol[1], 3)
                bcolb = precision(bcol[2], 3)
                colors.printc(tab+'     back color: ', c='g', bold=1, end='')
                colors.printc(colors.getColorName(bcol) + ', rgb=('+bcolr+', '
                              + bcolg+', ' + bcolb+')', c='g', bold=0)

        colors.printc(tab + "         points: ", c="g", bold=1, end="")
        colors.printc(npt, c="g", bold=0)

        colors.printc(tab + "          cells: ", c="g", bold=1, end="")
        colors.printc(ncl, c="g", bold=0)

        colors.printc(tab + "       position: ", c="g", bold=1, end="")
        colors.printc(pos, c="g", bold=0)

        if hasattr(actor, "polydata"):
            colors.printc(tab + "     c. of mass: ", c="g", bold=1, end="")
            colors.printc(actor.centerOfMass(), c="g", bold=0)

            colors.printc(tab + "      ave. size: ", c="g", bold=1, end="")
            colors.printc(precision(actor.averageSize(), 6), c="g", bold=0)

            colors.printc(tab + "     diag. size: ", c="g", bold=1, end="")
            colors.printc(precision(actor.diagonalSize(), 6), c="g", bold=0)

            _area = actor.area()
            if _area:
                colors.printc(tab + "           area: ", c="g", bold=1, end="")
                colors.printc(precision(_area, 6), c="g", bold=0)

            _vol = actor.volume()
            if _vol:
                colors.printc(tab + "         volume: ", c="g", bold=1, end="")
                colors.printc(precision(_vol, 6), c="g", bold=0)

        colors.printc(tab + "         bounds: ", c="g", bold=1, end="")
        bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
        colors.printc("x=(" + bx1 + ", " + bx2 + ")", c="g", bold=0, end="")
        by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
        colors.printc(" y=(" + by1 + ", " + by2 + ")", c="g", bold=0, end="")
        bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
        colors.printc(" z=(" + bz1 + ", " + bz2 + ")", c="g", bold=0)

        arrtypes = dict()
        arrtypes[vtk.VTK_UNSIGNED_CHAR] = "UNSIGNED_CHAR"
        arrtypes[vtk.VTK_UNSIGNED_INT] = "UNSIGNED_INT"
        arrtypes[vtk.VTK_FLOAT] = "FLOAT"
        arrtypes[vtk.VTK_DOUBLE] = "DOUBLE"

        if poly.GetPointData():
            ptdata = poly.GetPointData()
            for i in range(ptdata.GetNumberOfArrays()):
                name = ptdata.GetArrayName(i)
                if name:
                    colors.printc(tab + "     point data: ", c="g", bold=1, end="")
                    try:
                        tt = arrtypes[ptdata.GetArray(i).GetDataType()]
                    except:
                        tt = ptdata.GetArray(i).GetDataType()
                    colors.printc("name=" + name, "type=" + tt, c="g", bold=0, end="")
                    rng = ptdata.GetArray(i).GetRange()
                    colors.printc(" range: (" + precision(rng[0],4) + ',' +
                                            precision(rng[1],4) + ')', c="g", bold=0)

        if poly.GetCellData():
            cldata = poly.GetCellData()
            for i in range(cldata.GetNumberOfArrays()):
                name = cldata.GetArrayName(i)
                if name:
                    colors.printc(tab + "      cell data: ", c="g", bold=1, end="")
                    try:
                        tt = arrtypes[cldata.GetArray(i).GetDataType()]
                    except:
                        tt = cldata.GetArray(i).GetDataType()
                    colors.printc("name=" + name, "type=" + tt, c="g", bold=0, end="")
                    rng = cldata.GetArray(i).GetRange()
                    colors.printc(" range: (" + precision(rng[0],4) + ',' +
                                            precision(rng[1],4) + ')', c="g", bold=0)

    if not obj:
        return

    elif isinstance(obj, vtk.vtkActor):
        colors.printc("_" * 60, c="g", bold=0)
        printvtkactor(obj)

    elif isinstance(obj, vtk.vtkAssembly):
        colors.printc("_" * 60, c="g", bold=0)
        colors.printc("vtkAssembly", c="g", bold=1, invert=1, end=" ")
        if hasattr(obj, "_legend"):
            colors.printc("legend: ", c="g", bold=1, end="")
            colors.printc(obj._legend, c="g", bold=0)
        else:
            print()

        pos = obj.GetPosition()
        bnds = obj.GetBounds()
        colors.printc("          position: ", c="g", bold=1, end="")
        colors.printc(pos, c="g", bold=0)

        colors.printc("            bounds: ", c="g", bold=1, end="")
        bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
        colors.printc("x=(" + bx1 + ", " + bx2 + ")", c="g", bold=0, end="")
        by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
        colors.printc(" y=(" + by1 + ", " + by2 + ")", c="g", bold=0, end="")
        bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
        colors.printc(" z=(" + bz1 + ", " + bz2 + ")", c="g", bold=0)

        cl = vtk.vtkPropCollection()
        obj.GetActors(cl)
        cl.InitTraversal()
        for i in range(obj.GetNumberOfPaths()):
            act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
            if isinstance(act, vtk.vtkActor):
                printvtkactor(act, tab="     ")

    elif hasattr(obj, "interactor"):  # dumps Plotter info
        axtype = {
            0: "(no axes)",
            1: "(three gray grid walls)",
            2: "(cartesian axes from origin",
            3: "(positive range of cartesian axes from origin",
            4: "(axes triad at bottom left)",
            5: "(oriented cube at bottom left)",
            6: "(mark the corners of the bounding box)",
            7: "(ruler at the bottom of the window)",
            8: "(the vtkCubeAxesActor object)",
            9: "(the bounding box outline)",
            10: "(circles of maximum bounding box range)",
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
        acts = obj.getActors()
        colors.printc("_" * 60, c="c", bold=0)
        colors.printc("Plotter", invert=1, dim=1, c="c", end=" ")
        otit = obj.title
        if not otit:
            otit = None
        colors.printc("   title:", otit, bold=0, c="c")
        colors.printc(" active renderer:", obj.renderers.index(obj.renderer), bold=0, c="c")
        colors.printc("   nr. of actors:", len(acts), bold=0, c="c", end="")
        colors.printc(" (" + str(totpt), "vertices)", bold=0, c="c")
        max_bns = np.max(bns, axis=0)
        min_bns = np.min(bns, axis=0)
        colors.printc("      max bounds: ", c="c", bold=0, end="")
        bx1, bx2 = precision(min_bns[0], 3), precision(max_bns[1], 3)
        colors.printc("x=(" + bx1 + ", " + bx2 + ")", c="c", bold=0, end="")
        by1, by2 = precision(min_bns[2], 3), precision(max_bns[3], 3)
        colors.printc(" y=(" + by1 + ", " + by2 + ")", c="c", bold=0, end="")
        bz1, bz2 = precision(min_bns[4], 3), precision(max_bns[5], 3)
        colors.printc(" z=(" + bz1 + ", " + bz2 + ")", c="c", bold=0)
        colors.printc("       axes type:", obj.axes, axtype[obj.axes], bold=0, c="c")

        for a in obj.actors:
            if a.GetBounds() is not None:
                if isinstance(a, vtk.vtkVolume):  # dumps Volume info
                    img = a.GetMapper().GetDataSetInput()
                    colors.printc('_'*60, c='b', bold=0)
                    colors.printc('Volume', invert=1, dim=1, c='b')
                    colors.printc('      scalar range:',
                                  np.round(img.GetScalarRange(), 4), c='b', bold=0)
                    bnds = a.GetBounds()
                    colors.printc("            bounds: ", c="b", bold=0, end="")
                    bx1, bx2 = precision(bnds[0], 3), precision(bnds[1], 3)
                    colors.printc("x=(" + bx1 + ", " + bx2 + ")", c="b", bold=0, end="")
                    by1, by2 = precision(bnds[2], 3), precision(bnds[3], 3)
                    colors.printc(" y=(" + by1 + ", " + by2 + ")", c="b", bold=0, end="")
                    bz1, bz2 = precision(bnds[4], 3), precision(bnds[5], 3)
                    colors.printc(" z=(" + bz1 + ", " + bz2 + ")", c="b", bold=0)

        colors.printc(" Click actor and press i for Actor info.", c="c")

    else:
        colors.printc("_" * 60, c="g", bold=0)
        colors.printc(type(obj), c="g", invert=1)



def printHistogram(data, bins=10, height=10, logscale=False, minbin=0,
                   horizontal=False, char=u"\U00002589",
                   c=None, bold=True, title='Histogram'):
    """
    Ascii histogram printing.
    Input can also be ``Volume`` or ``Actor``.
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

            from vtkplotter import printHistogram
            import numpy as np
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
            img = data.image
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
        colors.printc(_h(), c=c, bold=bold)
    else:
        colors.printc(_v(), c=c, bold=bold)
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



def resampleArrays(source, target, tol=None):
        """Resample point and cell data of a dataset on points from another dataset.

        :param float tol: set the tolerance used to compute whether
            a point in the target is in a cell of the source.
            Points without resampled values, and their cells, are be marked as blank.
        """
        rs = vtk.vtkResampleWithDataSet()
        rs.SetSourceData(target.polydata())
        rs.SetInputData(source.polydata())
        rs.SetPassPointArrays(True)
        rs.SetPassCellArrays(True)
        if tol:
            rs.SetComputeTolerance(False)
            rs.SetTolerance(tol)
        rs.Update()
        return rs.GetOutput()



