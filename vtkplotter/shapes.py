from __future__ import division, print_function
import os
import vtk
import numpy as np
from vtkplotter import settings
from vtk.util.numpy_support import numpy_to_vtk
import vtkplotter.utils as utils
from vtkplotter.colors import printc, getColor, colorMap, _mapscales
from vtkplotter.mesh import Mesh
from vtkplotter.picture import Picture
import vtkplotter.docs as docs

__doc__ = ("""Submodule to generate basic geometric shapes.
"""
    + docs._defs
)

__all__ = [
    "Point",
    "Points",
    "Marker",
    "Line",
    "DashedLine",
    "Tube",
    "Lines",
    "Spline",
    "KSpline",
    "Ribbon",
    "Arrow",
    "Arrows",
    "Arrow2D",
    "Arrows2D",
    "FlatArrow",
    "Polygon",
    "Rectangle",
    "Disc",
    "Circle",
    "Arc",
    "Star",
    "Sphere",
    "Spheres",
    "Earth",
    "Ellipsoid",
    "Grid",
    "CubicGrid",
    "Plane",
    "Box",
    "Cube",
    "Spring",
    "Cylinder",
    "Cone",
    "Pyramid",
    "Torus",
    "Paraboloid",
    "Hyperboloid",
    "Text",
    "Text2D",
    "Latex",
    "Glyph",
    "Tensors",
    "ParametricShape",
]


########################################################################
def Marker(symbol, pos=(0, 0, 0), c='lb', alpha=1, s=0.1, filled=True):
    """
    Generate a marker shape.
    Can be used in association with ``Glyph``.
    """
    if isinstance(symbol, int):
        symbs = ['.', 'p','*','h','D','d','o','v','^','>','<','s', 'x', 'a']
        symbol = symbol % 14
        symbol = symbs[symbol]

    if symbol == '.':
        mesh = Polygon(nsides=24, r=s*0.75)
    elif symbol == 'p':
        mesh = Polygon(nsides=5, r=s)
    elif symbol == '*':
        mesh = Star(r1=0.65*s*1.1, r2=s*1.1, line=not filled)
    elif symbol == 'h':
        mesh = Polygon(nsides=6, r=s)
    elif symbol == 'D':
        mesh = Polygon(nsides=4, r=s)
    elif symbol == 'd':
        mesh = Polygon(nsides=4, r=s*1.1).scale([0.5,1,1])
    elif symbol == 'o':
        mesh = Polygon(nsides=24, r=s*0.75)
    elif symbol == 'v':
        mesh = Polygon(nsides=3, r=s).rotateZ(180)
    elif symbol == '^':
        mesh = Polygon(nsides=3, r=s)
    elif symbol == '>':
        mesh = Polygon(nsides=3, r=s).rotateZ(-90)
    elif symbol == '<':
        mesh = Polygon(nsides=3, r=s).rotateZ(90)
    elif symbol == 's':
        mesh = Polygon(nsides=4, r=s).rotateZ(45)
    elif symbol == 'x':
        mesh = Text('+', pos=(0,0,0), s=s*2.6, justify='center', depth=0)
        mesh.rotateZ(45)
    elif symbol == 'a':
        mesh = Text('*', pos=(0,0,0), s=s*3, justify='center', depth=0)
    else:
        mesh = Text(symbol, pos=(0,0,0), s=s*2, justify='center', depth=0)
    settings.collectable_actors.pop()
    mesh.flat().lighting('off').wireframe(not filled).c(c).alpha(alpha)
    if len(pos) == 2:
        pos = (pos[0], pos[1], 0)
    mesh.SetPosition(pos)
    mesh.name = "Marker"
    return mesh


def Point(pos=(0, 0, 0), r=12, c="red", alpha=1):
    """Create a simple point."""
    if isinstance(pos, vtk.vtkActor):
        pos = pos.GetPosition()
    pd = utils.buildPolyData([[0,0,0]])
    if len(pos)==2:
        pos = (pos[0], pos[1], 0.)

    pt = Mesh(pd, c, alpha)
    pt.SetPosition(pos)
    pt.GetProperty().SetPointSize(r)
    pt.name = "Point"
    return pt


class Points(Mesh):
    """
    Build a ``Mesh`` made of only vertex points for a list of 2D/3D points.
    Both shapes (N, 3) or (3, N) are accepted as input, if N>3.

    For very large point clouds a list of colors and alpha can be assigned to each
    point in the form `c=[(R,G,B,A), ... ]` where `0 <= R < 256, ... 0 <= A < 256`.

    :param float r: point radius.
    :param c: color name, number, or list of [R,G,B] colors of same length as plist.
    :type c: int, str, list
    :param float alpha: transparency in range [0,1].
    :param bool blur: make the point fluffy and blurred
        (works better with ``settings.useDepthPeeling=True``.)

    |manypoints.py|_ |lorenz.py|_

    |lorenz|
    """
    def __init__(self, plist, r=5, c=(0.3,0.3,0.3), alpha=1, blur=False):

        ################ interpret user input format:
        if isinstance(plist, Mesh):
            plist = plist.points()

        n = len(plist)

        if n == 3:  # assume plist is in the format [all_x, all_y, all_z]
            if utils.isSequence(plist[0]) and len(plist[0]) > 3:
                plist = np.stack((plist[0], plist[1], plist[2]), axis=1)
        elif n == 2:  # assume plist is in the format [all_x, all_y, 0]
            if utils.isSequence(plist[0]) and len(plist[0]) > 3:
                plist = np.stack((plist[0], plist[1], np.zeros(len(plist[0]))), axis=1)

        if n and len(plist[0]) == 2: # make it 3d
            plist = np.c_[np.array(plist), np.zeros(len(plist))]
        ################

        if ((utils.isSequence(c) and (len(c)>3 or (utils.isSequence(c[0]) and len(c[0])==4)))
            or utils.isSequence(alpha) ):

            cols = c

            n = len(plist)
            if n != len(cols):
                printc("Mismatch in Points() colors", n, len(cols), c=1)
                raise RuntimeError()

            src = vtk.vtkPointSource()
            src.SetNumberOfPoints(n)
            src.Update()

            vgf = vtk.vtkVertexGlyphFilter()
            vgf.SetInputData(src.GetOutput())
            vgf.Update()
            pd = vgf.GetOutput()

            pd.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(plist), deep=True))

            ucols = vtk.vtkUnsignedCharArray()
            ucols.SetNumberOfComponents(4)
            ucols.SetName("Points_RGBA")
            if utils.isSequence(alpha):
                if len(alpha) != n:
                    printc("Mismatch in Points() alphas", n, len(alpha), c=1)
                    raise RuntimeError()
                alphas = alpha
                alpha = 1
            else:
               alphas = (alpha,) * n

            if utils.isSequence(cols):
                c = None
                if len(cols[0]) == 4:
                    for i in range(n): # FAST
                        rc,gc,bc,ac = cols[i]
                        ucols.InsertNextTuple4(rc, gc, bc, ac)
                else:
                    for i in range(n): # SLOW
                        rc,gc,bc = getColor(cols[i])
                        ucols.InsertNextTuple4(rc*255, gc*255, bc*255, alphas[i]*255)
            else:
                c = cols

            pd.GetPointData().SetScalars(ucols)

            Mesh.__init__(self, pd, c, alpha)
            self.mapper().ScalarVisibilityOn()

        else:

            pd = utils.buildPolyData(plist)
            Mesh.__init__(self, pd, c, alpha)

        if blur:
            #settings.useDepthPeeling = True
            point_mapper = vtk.vtkPointGaussianMapper()
            point_mapper.SetInputData(pd)
            point_mapper.SetScaleFactor(0.0005*self.diagonalSize()*r)
            point_mapper.EmissiveOn()
            self._mapper = point_mapper
            self.SetMapper(point_mapper)
        else:
            self.GetProperty().SetPointSize(r)

        settings.collectable_actors.append(self)
        self.name = "Points"


class Glyph(Mesh):
    """
    At each vertex of a mesh, another mesh - a `'glyph'` - is shown with
    various orientation options and coloring.
    The input ``mesh`` can also be a simple list of 2D or 3D coordinates.
    Color can be specified as a colormap which maps the size of the orientation
    vectors in `orientationArray`.

    :param orientationArray: list of vectors, ``vtkAbstractArray``
        or the name of an already existing points array.
    :type orientationArray: list, str, vtkAbstractArray

    :param bool scaleByScalar: glyph mesh is scaled by the active scalars.
    :param bool scaleByVectorSize: glyph mesh is scaled by the size of the vectors.
    :param bool scaleByVectorComponents: glyph mesh is scaled by the 3 vectors components.
    :param bool colorByScalar: glyph mesh is colored based on the scalar value.
    :param bool colorByVectorSize: glyph mesh is colored based on the vector size.

    :param float tol: set a minimum separation between two close glyphs
        (not compatible with `orientationArray` being a list).

    |glyphs.py|_ |glyphs_arrows.py|_
    |glyphs| |glyphs_arrows|
    """

    def __init__(self,
                 mesh,
                 glyphObj,
                 orientationArray=None,
                 scaleByScalar=False,
                 scaleByVectorSize=False,
                 scaleByVectorComponents=False,
                 colorByScalar=False,
                 colorByVectorSize=False,
                 tol=0,
                 c='white',
                 alpha=1,
                 ):

        if utils.isSequence(mesh):
            # create a cloud of points
            poly = Points(mesh).polydata()
        elif isinstance(mesh, vtk.vtkPolyData):
            poly = mesh
        else:
            poly = mesh.polydata()

        if tol:
            cleanPolyData = vtk.vtkCleanPolyData()
            cleanPolyData.SetInputData(poly)
            cleanPolyData.SetTolerance(tol)
            cleanPolyData.Update()
            poly = cleanPolyData.GetOutput()

        if isinstance(glyphObj, Mesh):
            glyphObj = glyphObj.clean().polydata()

        cmap=''
        if c in list(_mapscales.cmap_d.keys()):
            cmap = c
            c = None
        elif utils.isSequence(c): # user passing an array of point colors
            ucols = vtk.vtkUnsignedCharArray()
            ucols.SetNumberOfComponents(3)
            ucols.SetName("glyph_RGB")
            for col in c:
                cl = getColor(col)
                ucols.InsertNextTuple3(cl[0]*255, cl[1]*255, cl[2]*255)
            poly.GetPointData().SetScalars(ucols)
            c = None

        gly = vtk.vtkGlyph3D()
        gly.SetInputData(poly)
        gly.SetSourceData(glyphObj)

        if scaleByScalar:
            gly.SetScaleModeToScaleByScalar()
        elif scaleByVectorSize:
            gly.SetScaleModeToScaleByVector()
        elif scaleByVectorComponents:
            gly.SetScaleModeToScaleByVectorComponents()
        else:
            gly.SetScaleModeToDataScalingOff()

        if colorByVectorSize:
            gly.SetVectorModeToUseVector()
            gly.SetColorModeToColorByVector()
        elif colorByScalar:
            gly.SetColorModeToColorByScalar()
        else:
            gly.SetColorModeToColorByScale()

        if orientationArray is not None:
            gly.OrientOn()
            if isinstance(orientationArray, str):
                if orientationArray.lower() == "normals":
                    gly.SetVectorModeToUseNormal()
                else:  # passing a name
                    gly.SetInputArrayToProcess(0, 0, 0, 0, orientationArray)
                    gly.SetVectorModeToUseVector()
            elif utils.isSequence(orientationArray) and not tol: # passing a list
                varr = vtk.vtkFloatArray()
                varr.SetNumberOfComponents(3)
                varr.SetName("glyph_vectors")
                for v in orientationArray:
                    varr.InsertNextTuple(v)
                poly.GetPointData().AddArray(varr)
                poly.GetPointData().SetActiveVectors("glyph_vectors")
                gly.SetInputArrayToProcess(0, 0, 0, 0, "glyph_vectors")
                gly.SetVectorModeToUseVector()

        gly.Update()

        Mesh.__init__(self, gly.GetOutput(), c, alpha)
        self.flat()

        if cmap:
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(512)
            lut.Build()
            for i in range(512):
                r, g, b = colorMap(i, cmap, 0, 512)
                lut.SetTableValue(i, r, g, b, 1)
            self.mapper().SetLookupTable(lut)
            self.mapper().ScalarVisibilityOn()
            self.mapper().SetScalarModeToUsePointData()
            if gly.GetOutput().GetPointData().GetScalars():
                rng = gly.GetOutput().GetPointData().GetScalars().GetRange()
                self.mapper().SetScalarRange(rng[0], rng[1])

        settings.collectable_actors.append(self)
        self.name = "Glyph"


class Tensors(Mesh):
    """Geometric representation of tensors defined on a domain or set of points.
    Tensors can be scaled and/or rotated according to the source at eache input point.
    Scaling and rotation is controlled by the eigenvalues/eigenvectors of the symmetrical part
    of the tensor as follows:

    For each tensor, the eigenvalues (and associated eigenvectors) are sorted
    to determine the major, medium, and minor eigenvalues/eigenvectors.
    The eigenvalue decomposition only makes sense for symmetric tensors,
    hence the need to only consider the symmetric part of the tensor,
    which is 1/2*(T+T.transposed()).

    :param str source: preset type of source shape
        ['ellipsoid', 'cylinder', 'cube' or any specified ``Mesh``]

    :param bool useEigenValues: color source glyph using the eigenvalues or by scalars.

    :param bool threeAxes: if `False` scale the source in the x-direction,
        the medium in the y-direction, and the minor in the z-direction.
        Then, the source is rotated so that the glyph's local x-axis lies
        along the major eigenvector, y-axis along the medium eigenvector, and z-axis along the minor.

        If `True` three sources are produced, each of them oriented along an eigenvector
        and scaled according to the corresponding eigenvector.

    :param bool isSymmetric: If `True` each source glyph is mirrored (2 or 6 glyphs will be produced).
        The x-axis of the source glyph will correspond to the eigenvector on output.

    :param float length: distance from the origin to the tip of the source glyph along the x-axis

    :param float scale: scaling factor of the source glyph.
    :param float maxScale: clamp scaling at this factor.

    |tensors| |tensors.py|_ |tensor_grid.py|_
    """

    def __init__(self, domain, source='ellipsoid', useEigenValues=True, isSymmetric=True,
                threeAxes=False, scale=1, maxScale=None, length=None,
                c=None, alpha=1):
        if isinstance(source, Mesh):
            src = source.normalize().polydata(False)
        else:
            if 'ellip' in source:
                src = vtk.vtkSphereSource()
                src.SetPhiResolution(24)
                src.SetThetaResolution(12)
            elif 'cyl' in source:
                src = vtk.vtkCylinderSource()
                src.SetResolution(48)
                src.CappingOn()
            elif source == 'cube':
                src = vtk.vtkCubeSource()
            src.Update()

        tg = vtk.vtkTensorGlyph()
        if isinstance(domain, vtk.vtkPolyData):
            tg.SetInputData(domain)
        else:
            tg.SetInputData(domain.GetMapper().GetInput())
        tg.SetSourceData(src.GetOutput())

        if c is None:
            tg.ColorGlyphsOn()
        else:
            tg.ColorGlyphsOff()

        tg.SetSymmetric(int(isSymmetric))

        if length is not None:
            tg.SetLength(length)
        if useEigenValues:
            tg.ExtractEigenvaluesOn()
            tg.SetColorModeToEigenvalues()
        else:
            tg.SetColorModeToScalars()
        tg.SetThreeGlyphs(threeAxes)
        tg.ScalingOn()
        tg.SetScaleFactor(scale)
        if maxScale is None:
            tg.ClampScalingOn()
            maxScale = scale*10
        tg.SetMaxScaleFactor(maxScale)
        tg.Update()
        tgn = vtk.vtkPolyDataNormals()
        tgn.SetInputData(tg.GetOutput())
        tgn.Update()
        Mesh.__init__(self, tgn.GetOutput(), c, alpha)
        self.name = "Tensors"


class Line(Mesh):
    """
    Build the line segment between points `p0` and `p1`.
    If `p0` is a list of points returns the line connecting them.
    A 2D set of coords can also be passed as p0=[x..], p1=[y..].

    :param bool closed: join last to first point
    :param c: color name, number, or list of [R,G,B] colors.
    :type c: int, str, list
    :param float alpha: transparency in range [0,1].
    :param lw: line width.
    :param int res: number of intermediate points in the segment
    """

    def __init__(self, p0, p1=None, closed=False, c="r", alpha=1, lw=1, res=None):
        if isinstance(p0, vtk.vtkActor): p0 = p0.GetPosition()
        if isinstance(p1, vtk.vtkActor): p1 = p1.GetPosition()

        self.slope = [] # used by analysis.fitLine
        self.center = []
        self.variances = []

        # detect if user is passing a 2D list of points as p0=xlist, p1=ylist:
        if len(p0) > 3:
            if not utils.isSequence(p0[0]) and not utils.isSequence(p1[0]) and len(p0)==len(p1):
                # assume input is 2D xlist, ylist
                p0 = np.stack((p0, p1), axis=1)
                p1 = None
            if len(p0[0]) == 2: # make it 3d
                p0 = np.c_[np.array(p0), np.zeros(len(p0))]
            self.base = p0[0]
            if closed:
                p0 = np.append(p0, [p0[0]], axis=0)
                self.top = p0[-2]
            else:
                self.top = p0[-1]

        # detect if user is passing a list of points:
        if utils.isSequence(p0[0]):
            ppoints = vtk.vtkPoints()  # Generate the polyline
            ppoints.SetData(numpy_to_vtk(np.ascontiguousarray(p0), deep=True))
            lines = vtk.vtkCellArray()  # Create the polyline
            npt = len(p0)
            lines.InsertNextCell(npt)
            for i in range(npt):
                lines.InsertCellPoint(i)
            poly = vtk.vtkPolyData()
            poly.SetPoints(ppoints)
            poly.SetLines(lines)
        else:  # or just 2 points to link
            lineSource = vtk.vtkLineSource()
            lineSource.SetPoint1(p0)
            lineSource.SetPoint2(p1)
            if res:
                lineSource.SetResolution(res)
            lineSource.Update()
            poly = lineSource.GetOutput()
            self.top = np.array(p1)
            self.base = np.array(p0)

        Mesh.__init__(self, poly, c, alpha)
        self.lw(lw).lighting('off')
        #if dotted: # not functional
        #    self.GetProperty().SetLineStipplePattern(0xF0F0)
        #    self.GetProperty().SetLineStippleRepeatFactor(1)
        #self.SetOrigin((self.base+self.top)/2)
        settings.collectable_actors.append(self)
        self.name = "Line"

    def length(self):
        """Calculate length of line."""
        distance = 0.
        pts = self.points()
        for i in range(1, len(pts)):
            distance += np.linalg.norm(pts[i]-pts[i-1])
        return distance


class DashedLine(Line):
    """
    Build a dashed line segment between points `p0` and `p1`.
    If `p0` is a list of points returns the line connecting them.
    A 2D set of coords can also be passed as p0=[x..], p1=[y..].

    :param bool closed: join last to first point
    :param float spacing: relative size of the dash.
    :param c: color name, number, or list of [R,G,B] colors.
    :type c: int, str, list
    :param float alpha: transparency in range [0,1].
    :param lw: line width.
    """
    def __init__(self, p0, p1=None, spacing=0.1, closed=False, c="red", alpha=1, lw=2):

        if isinstance(p0, vtk.vtkActor): p0 = p0.GetPosition()
        if isinstance(p1, vtk.vtkActor): p1 = p1.GetPosition()

        # detect if user is passing a 2D list of points as p0=xlist, p1=ylist:
        if len(p0) > 3:
            if not utils.isSequence(p0[0]) and not utils.isSequence(p1[0]) and len(p0)==len(p1):
                # assume input is 2D xlist, ylist
                p0 = np.stack((p0, p1), axis=1)
                p1 = None
            if len(p0[0]) == 2: # make it 3d
                p0 = np.c_[np.array(p0), np.zeros(len(p0))]
            if closed:
                p0 = np.append(p0, [p0[0]], axis=0)

        if p1 is not None: # assume passing p0=[x,y]
            if len(p0) == 2 and not utils.isSequence(p0[0]):
                p0 = (p0[0], p0[1], 0)
            if len(p1) == 2 and not utils.isSequence(p1[0]):
                p1 = (p1[0], p1[1], 0)

        # detect if user is passing a list of points:
        if utils.isSequence(p0[0]):
           listp = p0
        else:  # or just 2 points to link
            listp = [p0, p1]

        listp = np.array(listp)
        if listp.shape[1]==2:
            listp = np.c_[listp, np.zeros(listp.shape[0])]

        xmn = np.min(listp, axis=0)
        xmx = np.max(listp, axis=0)
        dlen = np.linalg.norm(xmx-xmn)*spacing/10
        if not dlen:
            printc("Error in DashedLine: zero dash length.", c=1)
            Mesh.__init__(self, vtk.vtkPolyData(), c, alpha)
            return

        qs = []
        for ipt in range(len(listp)-1):
            p0 = listp[ipt]
            p1 = listp[ipt+1]
            v = p1-p0
            vdist = np.linalg.norm(v)
            n1 = int(vdist/dlen)
            if not n1: continue

            res = 0
            for i in range(n1+2):
                ist = (i-0.5)/n1
                if ist<0: ist=0
                qi = p0 + v * (ist - res/vdist)
                if ist>1:
                    qi = p1
                    res = np.linalg.norm(qi-p1)
                    qs.append(qi)
                    break
                qs.append(qi)

        polylns = vtk.vtkAppendPolyData()
        for i,q1 in enumerate(qs):
            if not i%2: continue
            q0 = qs[i-1]
            lineSource = vtk.vtkLineSource()
            lineSource.SetPoint1(q0)
            lineSource.SetPoint2(q1)
            lineSource.Update()
            polylns.AddInputData(lineSource.GetOutput())
        polylns.Update()
        poly = polylns.GetOutput()

        Mesh.__init__(self, poly, c, alpha)
        self.lw(lw).lighting('off')
        self.base = listp[0]
        if closed:
            self.top = listp[-2]
        else:
            self.top = listp[-1]
        settings.collectable_actors.append(self)
        self.name = "DashedLine"



class Lines(Line):
    """
    Build the line segments between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form ``[[point1, point2], ...]``.

    :param float scale: apply a rescaling factor to the lengths.

    |lines|

    .. hint:: |fitspheres2.py|_
    """
    def __init__(self, startPoints, endPoints=None,
                 c='gray', alpha=1, lw=1, dotted=False, scale=1):

        if endPoints is not None:
            startPoints = np.stack((startPoints, endPoints), axis=1)

        polylns = vtk.vtkAppendPolyData()
        for twopts in startPoints:
            lineSource = vtk.vtkLineSource()
            lineSource.SetPoint1(twopts[0])

            if scale == 1:
                pt2 = twopts[1]
            else:
                vers = (np.array(twopts[1]) - twopts[0]) * scale
                pt2 = np.array(twopts[0]) + vers

            lineSource.SetPoint2(pt2)
            polylns.AddInputConnection(lineSource.GetOutputPort())
        polylns.Update()

        Mesh.__init__(self, polylns.GetOutput(), c, alpha)
        self.lw(lw).lighting('off')
        if dotted:
            self.GetProperty().SetLineStipplePattern(0xF0F0)
            self.GetProperty().SetLineStippleRepeatFactor(1)

        settings.collectable_actors.append(self)
        self.name = "Lines"


class Spline(Line):
    """
    Return an ``Mesh`` for a spline which does not necessarly
    passing exactly through all the input points.
    Needs to import `scypi`.

    :param float smooth: smoothing factor.

        - 0 = interpolate points exactly.
        - 1 = average point positions.

    :param int degree: degree of the spline (1<degree<5)
    :param int res: number of points on the spline

    |tutorial_spline| |tutorial.py|_
    """
    def __init__(self, points, smooth=0.5, degree=2, closed=False, s=2, res=None):

        if isinstance(points, Mesh):
            points = points.points()

        if len(points[0]) == 2: # make it 3d
            points = np.c_[np.array(points), np.zeros(len(points))]

        if closed:
            points = np.append(points, [points[0]], axis=0)

        from scipy.interpolate import splprep, splev
        if res is None:
            res = len(points)*20

        points = np.array(points)

        minx, miny, minz = np.min(points, axis=0)
        maxx, maxy, maxz = np.max(points, axis=0)
        maxb = max(maxx - minx, maxy - miny, maxz - minz)
        smooth *= maxb / 2  # must be in absolute units

        tckp, _ = splprep(points.T, task=0, s=smooth, k=degree)  # find the knots
        # evaluate spLine, including interpolated points:
        xnew, ynew, znew = splev(np.linspace(0, 1, res), tckp)

        Line.__init__(self, np.c_[xnew, ynew, znew], lw=2)
        self.lighting('off')
        settings.collectable_actors.pop()
        settings.collectable_actors.append(self)
        self.name = "Spline"


class KSpline(Line):
    """
    Return a Kochanek spline which runs exactly through all the input points.

    See: https://en.wikipedia.org/wiki/Kochanek%E2%80%93Bartels_spline

    :param float continuity: changes the sharpness in change between tangents
    :param float tension: changes the length of the tangent vector
    :param float bias: changes the direction of the tangent vector
    :param bool closed: join last to first point to produce a closed curve
    :param int res: resolution of the output line. Default is 20 times the number
        of input points.

    |kspline|
    """
    def __init__(self, points,
                 continuity=0, tension=0, bias=0,
                 closed=False, res=None):

        if isinstance(points, Mesh):
            points = points.points()

        if not res: res = len(points)*20

        if len(points[0]) == 2: # make it 3d
            points = np.c_[np.array(points), np.zeros(len(points))]

        xspline = vtk.vtkKochanekSpline()
        yspline = vtk.vtkKochanekSpline()
        zspline = vtk.vtkKochanekSpline()
        for s in [xspline, yspline, zspline]:
            if bias: s.SetDefaultBias(bias)
            if tension: s.SetDefaultTension(tension)
            if continuity: s.SetDefaultContinuity(continuity)
            s.SetClosed(closed)

        for i,p in enumerate(points):
            xspline.AddPoint(i, p[0])
            yspline.AddPoint(i, p[1])
            if len(p)>2:
                zspline.AddPoint(i, p[2])

        ln = []
        for pos in np.linspace(0, len(points), res):
            x = xspline.Evaluate(pos)
            y = yspline.Evaluate(pos)
            z=0
            if len(p)>2:
                z = zspline.Evaluate(pos)
            ln.append((x,y,z))

        Line.__init__(self, ln, lw=2, c='gray')
        settings.collectable_actors.pop()
        self.lighting('off')
        self.base = np.array(points[0])
        self.top = np.array(points[-1])
        settings.collectable_actors.append(self)
        self.name = "KSpline"


class Tube(Mesh):
    """Build a tube along the line defined by a set of points.

    :param r: constant radius or list of radii.
    :type r: float, list
    :param c: constant color or list of colors for each point.
    :type c: float, list
    :para int res: resolution, number of sides of the tube

    |ribbon.py|_ |tube.py|_

        |ribbon| |tube|
    """
    def __init__(self, points, r=1, cap=True, c=None, alpha=1, res=12):

        ppoints = vtk.vtkPoints()  # Generate the polyline
        ppoints.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(points))
        for i in range(len(points)):
            lines.InsertCellPoint(i)
        polyln = vtk.vtkPolyData()
        polyln.SetPoints(ppoints)
        polyln.SetLines(lines)

        tuf = vtk.vtkTubeFilter()
        tuf.SetCapping(cap)
        tuf.SetNumberOfSides(res)
        tuf.SetInputData(polyln)
        if utils.isSequence(r):
            arr = numpy_to_vtk(np.ascontiguousarray(r), deep=True)
            arr.SetName("TubeRadius")
            polyln.GetPointData().AddArray(arr)
            polyln.GetPointData().SetActiveScalars("TubeRadius")
            tuf.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        else:
            tuf.SetRadius(r)

        usingColScals = False
        if utils.isSequence(c) and len(c) != 3:
            usingColScals = True
            cc = vtk.vtkUnsignedCharArray()
            cc.SetName("TubeColors")
            cc.SetNumberOfComponents(3)
            cc.SetNumberOfTuples(len(c))
            for i, ic in enumerate(c):
                r, g, b = getColor(ic)
                cc.InsertTuple3(i, int(255 * r), int(255 * g), int(255 * b))
            polyln.GetPointData().AddArray(cc)
            c = None
        tuf.Update()

        Mesh.__init__(self, tuf.GetOutput(), c, alpha)
        self.phong()
        if usingColScals:
            self.mapper().SetScalarModeToUsePointFieldData()
            self.mapper().ScalarVisibilityOn()
            self.mapper().SelectColorArray("TubeColors")
            self.mapper().Modified()

        self.base = np.array(points[0])
        self.top = np.array(points[-1])
        settings.collectable_actors.append(self)
        self.name = "Tube"


class Ribbon(Mesh):
    """Connect two lines to generate the surface inbetween.
Set the mode by which to create the ruled surface.

    :param int mode: If mode=0, resample evenly the input lines (based on length)
        and generates triangle strips.
        If mode=1, use the existing points and walks around the polyline using existing points.

    :param bool closed: if True, join the last point with the first to form
        a closed surface

    :param list res: ribbon resolutions along the line and perpendicularly to it.

    |ribbon| |ribbon.py|_
    """
    def __init__(self, line1, line2, mode=0, closed=False, c="m", alpha=1, res=(200,5)):

        if isinstance(line1, Mesh):
            line1 = line1.points()
        if isinstance(line2, Mesh):
            line2 = line2.points()

        if closed:
            line1 = line1.tolist()
            line1 += [line1[0]]
            line2 = line2.tolist()
            line2 += [line2[0]]

        ppoints1 = vtk.vtkPoints()  # Generate the polyline1
        ppoints1.SetData(numpy_to_vtk(np.ascontiguousarray(line1), deep=True))
        lines1 = vtk.vtkCellArray()
        lines1.InsertNextCell(len(line1))
        for i in range(len(line1)):
            lines1.InsertCellPoint(i)
        poly1 = vtk.vtkPolyData()
        poly1.SetPoints(ppoints1)
        poly1.SetLines(lines1)

        ppoints2 = vtk.vtkPoints()  # Generate the polyline2
        ppoints2.SetData(numpy_to_vtk(np.ascontiguousarray(line2), deep=True))
        lines2 = vtk.vtkCellArray()
        lines2.InsertNextCell(len(line2))
        for i in range(len(line2)):
            lines2.InsertCellPoint(i)
        poly2 = vtk.vtkPolyData()
        poly2.SetPoints(ppoints2)
        poly2.SetLines(lines2)

        # build the lines
        lines1 = vtk.vtkCellArray()
        lines1.InsertNextCell(poly1.GetNumberOfPoints())
        for i in range(poly1.GetNumberOfPoints()):
            lines1.InsertCellPoint(i)

        polygon1 = vtk.vtkPolyData()
        polygon1.SetPoints(ppoints1)
        polygon1.SetLines(lines1)

        lines2 = vtk.vtkCellArray()
        lines2.InsertNextCell(poly2.GetNumberOfPoints())
        for i in range(poly2.GetNumberOfPoints()):
            lines2.InsertCellPoint(i)

        polygon2 = vtk.vtkPolyData()
        polygon2.SetPoints(ppoints2)
        polygon2.SetLines(lines2)

        mergedPolyData = vtk.vtkAppendPolyData()
        mergedPolyData.AddInputData(polygon1)
        mergedPolyData.AddInputData(polygon2)
        mergedPolyData.Update()

        rsf = vtk.vtkRuledSurfaceFilter()
        rsf.CloseSurfaceOff()
        rsf.SetRuledMode(mode)
        rsf.SetResolution(res[0], res[1])
        rsf.SetInputData(mergedPolyData.GetOutput())
        rsf.Update()
        Mesh.__init__(self, rsf.GetOutput(), c, alpha)
        settings.collectable_actors.append(self)
        self.name = "Ribbon"


class Arrow(Mesh):
    """
    Build a 3D arrow from `startPoint` to `endPoint` of section size `s`,
    expressed as the fraction of the window size.

    .. note:: If ``s=None`` the arrow is scaled proportionally to its length,
              otherwise it represents the fraction of the window size.

    |OrientedArrow|
    """
    def __init__(self, startPoint, endPoint, s=None, c="r", alpha=1, res=12):

        # in case user is passing meshs
        if isinstance(startPoint, vtk.vtkActor): startPoint = startPoint.GetPosition()
        if isinstance(endPoint,   vtk.vtkActor): endPoint   = endPoint.GetPosition()

        axis = np.array(endPoint) - np.array(startPoint)
        length = np.linalg.norm(axis)
        if length:
            axis = axis / length
        theta = np.arccos(axis[2])
        phi = np.arctan2(axis[1], axis[0])
        arr = vtk.vtkArrowSource()
        arr.SetShaftResolution(res)
        arr.SetTipResolution(res)
        if s:
            sz = 0.02
            arr.SetTipRadius(sz)
            arr.SetShaftRadius(sz / 1.75)
            arr.SetTipLength(sz * 15)
        arr.Update()
        t = vtk.vtkTransform()
        t.RotateZ(np.rad2deg(phi))
        t.RotateY(np.rad2deg(theta))
        t.RotateY(-90)  # put it along Z
        if s:
            sz = 800.0 * s
            t.Scale(length, sz, sz)
        else:
            t.Scale(length, length, length)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(arr.GetOutput())
        tf.SetTransform(t)
        tf.Update()

        Mesh.__init__(self, tf.GetOutput(), c, alpha)
        self.phong()
        self.SetPosition(startPoint)
        self.DragableOff()
        self.base = np.array(startPoint)
        self.top = np.array(endPoint)
        settings.collectable_actors.append(self)
        self.name = "Arrow"


def Arrows(startPoints, endPoints=None, s=None, scale=1, c=None, alpha=1, res=12):
    """
    Build arrows between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form ``[[point1, point2], ...]``.

    Color can be specified as a colormap which maps the size of the arrows.

    :param float s: fix aspect-ratio of the arrow and scale its cross section
    :param float scale: apply a rescaling factor to the length
    :param c: color or color map name.
    :param float alpha: set transparency
    :param int res: set arrow resolution

    |glyphs_arrows| |glyphs_arrows.py|_
    """
    if isinstance(startPoints, Mesh): startPoints = startPoints.points()
    if isinstance(endPoints,   Mesh): endPoints   = endPoints.points()
    startPoints = np.array(startPoints)
    if endPoints is None:
        strt = startPoints[:,0]
        endPoints = startPoints[:,1]
        startPoints = strt
    else:
         endPoints = np.array(endPoints)

    if startPoints.shape[1] == 2: # make it 3d
        startPoints = np.c_[startPoints, np.zeros(len(startPoints))]
    if endPoints.shape[1] == 2: # make it 3d
        endPoints = np.c_[np.array(endPoints), np.zeros(len(endPoints))]

    arr = vtk.vtkArrowSource()
    arr.SetShaftResolution(res)
    arr.SetTipResolution(res)
    if s:
        sz = 0.02 * s
        arr.SetTipRadius(sz*2)
        arr.SetShaftRadius(sz)
        arr.SetTipLength(sz*10)
    arr.Update()
    orients = (endPoints - startPoints) * scale
    arrg = Glyph(startPoints, arr.GetOutput(),
                 orientationArray=orients,
                 scaleByVectorSize=True,
                 colorByVectorSize=True,
                 c=c, alpha=alpha)
    arrg.flat()
    settings.collectable_actors.append(arrg)
    arrg.name = "Arrows"
    return arrg


class Arrow2D(Mesh):
    """
    Build a 2D arrow from `startPoint` to `endPoint`.

    :param float shaftLength: fractional shaft length
    :param float shaftWidth: fractional shaft width
    :param float headLength: fractional head length
    :param float headWidth: fractional head width
    :param bool fill: if False only generate the outline
    """
    def __init__(self, startPoint, endPoint,
                 shaftLength=0.8,
                 shaftWidth=0.05,
                 headLength=0.25,
                 headWidth=0.2,
                 fill=True,
                 c="r",
                 alpha=1):

        # in case user is passing meshs
        if isinstance(startPoint, vtk.vtkActor): startPoint = startPoint.GetPosition()
        if isinstance(endPoint,   vtk.vtkActor): endPoint   = endPoint.GetPosition()
        if len(startPoint) == 2:
            startPoint = [startPoint[0], startPoint[1], 0]
        if len(endPoint) == 2:
            endPoint = [endPoint[0], endPoint[1], 0]

        headBase = 1 - headLength
        if headWidth < shaftWidth:
            headWidth = shaftWidth
        if headLength is None or headBase > shaftLength:
            headBase = shaftLength

        verts = []
        verts.append([0,          -shaftWidth/2, 0])
        verts.append([shaftLength,-shaftWidth/2, 0])
        verts.append([headBase,   -headWidth/2,  0])
        verts.append([1,0,0])
        verts.append([headBase,    headWidth/2,  0])
        verts.append([shaftLength, shaftWidth/2, 0])
        verts.append([0,           shaftWidth/2, 0])
        if fill:
            faces = ((0,1,3,5,6), (5,3,4), (1,2,3))
            poly = utils.buildPolyData(verts, faces)
        else:
            lines = ((0,1,2,3,4,5,6,0))
            poly = utils.buildPolyData(verts, [], lines=lines)

        axis = np.array(endPoint) - np.array(startPoint)
        length = np.linalg.norm(axis)
        if length:
            axis = axis / length
        theta = np.arccos(axis[2])
        phi = np.arctan2(axis[1], axis[0])
        t = vtk.vtkTransform()
        t.RotateZ(np.rad2deg(phi))
        t.RotateY(np.rad2deg(theta))
        t.RotateY(-90)  # put it along Z
        t.Scale(length, length, length)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(poly)
        tf.SetTransform(t)
        tf.Update()

        Mesh.__init__(self, tf.GetOutput(), c, alpha)
        self.printInfo()
        self.SetPosition(startPoint)
        self.lighting('off')
        self.DragableOff()
        self.PickableOff()
        self.base = np.array(startPoint)
        self.top = np.array(endPoint)
        settings.collectable_actors.append(self)
        self.name = "Arrow2D"

def Arrows2D(startPoints, endPoints=None,
             shaftLength=0.8,
             shaftWidth=0.09,
             headLength=None,
             headWidth=0.2,
             fill=True,
             scale=1,
             c=None,
             cmap=None,
             alpha=1):
    """
    Build 2D arrows between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form ``[[point1, point2], ...]``.

    Color can be specified as a colormap which maps the size of the arrows.

    :param float shaftLength: fractional shaft length
    :param float shaftWidth: fractional shaft width
    :param float headLength: fractional head length
    :param float headWidth: fractional head width
    :param bool fill: if False only generate the outline

    :param float scale: apply a rescaling factor to the length
    :param c: color
    :param float alpha: set transparency

    :Example:
        .. code-block:: python

            from vtkplotter import Grid, Arrows2D
            g1 = Grid(sx=1, sy=1)
            g2 = Grid(sx=1.2, sy=1.2).rotateZ(4)
            arrs2d = Arrows2D(g1, g2, c='jet')
            arrs2d.show(axes=1, bg='white')

        |quiver|
    """
    if isinstance(startPoints, Mesh): startPoints = startPoints.points()
    if isinstance(endPoints,   Mesh): endPoints   = endPoints.points()
    startPoints = np.array(startPoints)
    if endPoints is None:
        strt = startPoints[:,0]
        endPoints = startPoints[:,1]
        startPoints = strt
    else:
        endPoints = np.array(endPoints)

    if headLength is None:
        headLength = 1 - shaftLength

    arr = Arrow2D((0,0,0), (1,0,0),
                  shaftLength, shaftWidth,
                  headLength, headWidth, fill)
    pts = Points(startPoints, r=0.001, c=c, alpha=alpha).off()

    orients = (endPoints - startPoints) * scale
    if orients.shape[1] == 2: # make it 3d
        orients = np.c_[np.array(orients), np.zeros(len(orients))]

    arrg = Glyph(pts,
                 arr.polydata(False),
                 orientationArray=orients,
                 scaleByVectorSize=True,
                 c=c, alpha=alpha).flat().lighting('off')
    if c is not None:
        arrg.color(c)

    settings.collectable_actors.append(arrg)
    arrg.name = "Arrows2D"
    return arrg


def FlatArrow(line1, line2, c="m", alpha=1, tipSize=1, tipWidth=1):
    """Build a 2D arrow in 3D space by joining two close lines.

    |flatarrow| |flatarrow.py|_
    """
    if isinstance(line1, Mesh): line1 = line1.points()
    if isinstance(line2, Mesh): line2 = line2.points()

    sm1, sm2 = np.array(line1[-1]), np.array(line2[-1])

    v = (sm1-sm2)/3*tipWidth
    p1 = sm1+v
    p2 = sm2-v
    pm1 = (sm1+sm2)/2
    pm2 = (np.array(line1[-2])+np.array(line2[-2]))/2
    pm12 = pm1-pm2
    tip = pm12/np.linalg.norm(pm12)*np.linalg.norm(v)*3*tipSize/tipWidth + pm1

    line1.append(p1)
    line1.append(tip)
    line2.append(p2)
    line2.append(tip)
    resm = max(100, len(line1))

    mesh = Ribbon(line1, line2, alpha=alpha, c=c, res=(resm, 1)).phong()
    settings.collectable_actors.pop()
    settings.collectable_actors.append(mesh)
    mesh.name = "FlatArrow"
    return mesh


class Polygon(Mesh):
    """
    Build a polygon in the `xy` plane of `nsides` of radius `r`.

    |Polygon|
    """
    def __init__(self, pos=(0, 0, 0), nsides=6, r=1, c="coral", alpha=1):

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        t = np.linspace(np.pi/2, 5/2*np.pi, num=nsides, endpoint=False)
        x, y = utils.pol2cart(np.ones_like(t)*r, t)
        faces = [list(range(nsides))]
        Mesh.__init__(self, [np.c_[x,y], faces], c, alpha)
        self.lighting('plastic')
        self.SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Polygon " + str(nsides)


class Circle(Polygon):
    """
    Build a Circle of radius `r`.
    """
    def __init__(self, pos=(0,0,0), r=1, c="grey", alpha=1, res=120):

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        Polygon.__init__(self, pos, nsides=res, r=r)
        self.alpha(alpha).c(c)
        self.name = "Circle"


class Star(Mesh):
    """
    Build a 2D star shape of `n` cusps of inner radius `r1` and outer radius `r2`.

    :param bool line: only build the outer line (no internal surface meshing).

    |extrude| |extrude.py|_
    """
    def __init__(self, pos=(0,0,0), n=5, r1=0.7, r2=1.0, line=False, c="m", alpha=1):

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        t = np.linspace(np.pi/2, 5/2*np.pi, num=n, endpoint=False)
        x, y = utils.pol2cart(np.ones_like(t)*r2, t)
        pts = np.c_[x,y, np.zeros_like(x)]

        apts=[]
        for i,p in enumerate(pts):
            apts.append(p)
            if i+1<n:
                apts.append((p+pts[i+1])/2*r1/r2)
        apts.append((pts[-1]+pts[0])/2*r1/r2)

        if line:
            apts.append(pts[0])
            poly = utils.buildPolyData(apts, lines=list(range(len(apts))))
            Mesh.__init__(self, poly, c, alpha)
            self.lw(2)
        else:
            apts.append((0,0,0))
            cells=[]
            for i in range(2*n-1):
                cell = [2*n, i, i+1]
                cells.append(cell)
            cells.append([2*n, i+1, 0])
            Mesh.__init__(self, [apts, cells], c, alpha)

        self.SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Star"


class Disc(Mesh):
    """
    Build a 2D disc of inner radius `r1` and outer radius `r2`.

    :param list res: resolution in R and Phi

    |Disk|
    """
    def __init__(self,
        pos=(0, 0, 0),
        r1=0.5,
        r2=1,
        c="coral",
        alpha=1,
        res=12,
    ):
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        if utils.isSequence(res):
            res_r, res_phi = res
        else:
            res_r, res_phi = res, 6*res
        ps = vtk.vtkDiskSource()
        ps.SetInnerRadius(r1)
        ps.SetOuterRadius(r2)
        ps.SetRadialResolution(res_r)
        ps.SetCircumferentialResolution(res_phi)
        ps.Update()
        Mesh.__init__(self, ps.GetOutput(), c, alpha)
        self.flat()
        self.SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Disc"


class Arc(Mesh):
    """
    Build a 2D circular arc between points `point1` and `point2`.
    If `normal` is specified then `center` is ignored, and
    normal vector, a starting `point1` (polar vector)
    and an angle defining the arc length need to be assigned.

    Arc spans the shortest angular sector point1 and point2,
    if invert=True, then the opposite happens.
    """
    def __init__(self,
        center,
        point1,
        point2=None,
        normal=None,
        angle=None,
        invert=False,
        c="grey",
        alpha=1,
        res=48,
    ):
        if len(point1) == 2:
            point1 = (point1[0], point1[1], 0)
        if point2 is not None and len(point2):
            point2 = (point2[0], point2[1], 0)

        ar = vtk.vtkArcSource()
        if point2 is not None:
            ar.UseNormalAndAngleOff()
            ar.SetPoint1(point1)
            ar.SetPoint2(point2)
            ar.SetCenter(center)
        elif normal is not None and angle is not None:
            ar.UseNormalAndAngleOn()
            ar.SetAngle(angle)
            ar.SetPolarVector(point1)
            ar.SetNormal(normal)
        else:
            printc("Error in Arc(): incorrect input.", c=1)
            return None
        ar.SetNegative(invert)
        ar.SetResolution(res)
        ar.Update()
        Mesh.__init__(self, ar.GetOutput(), c, alpha)
        self.flat().lw(2).lighting('off')
        settings.collectable_actors.append(self)
        self.name = "Arc"

    def length(self):
        """Calculate length of the arc."""
        distance = 0.
        pts = self.points()
        for i in range(1,len(pts)):
            distance += np.linalg.norm(pts[i]-pts[i-1])
        return distance


class Sphere(Mesh):
    """Build a sphere at position `pos` of radius `r`.

    :param r float: sphere radius
    :param int res: resolution in phi, resolution in theta is 2*res
    :param bool quads: sphere mesh will be made of quads instead of triangles

    |Sphere| |sphericgrid|
    """
    def __init__(self, pos=(0, 0, 0), r=1, c="r", alpha=1, res=24, quads=False):

        self.radius = r # used by fitSphere
        self.center = pos
        self.residue = 0

        if quads:
            if res<4: res=4
            img = vtk.vtkImageData()
            img.SetDimensions(res-1,res-1,res-1)
            rs = 1./(res-2)
            img.SetSpacing(rs,rs,rs)
            gf = vtk.vtkGeometryFilter()
            gf.SetInputData(img)
            gf.Update()
            Mesh.__init__(self, gf.GetOutput(), c, alpha)
            self.lw(0.1)

            cgpts = self.points() - (0.5,0.5,0.5)

            x, y, z = cgpts.T
            x = x*(1+x*x)/2
            y = y*(1+y*y)/2
            z = z*(1+z*z)/2
            _, theta, phi = utils.cart2spher(x, y, z)

            pts = utils.spher2cart(np.ones_like(phi)*r, theta, phi)
            self.points(pts)

        else:
            if utils.isSequence(res):
                res_t, res_phi = res
            else:
                res_t, res_phi = 2*res, res

            ss = vtk.vtkSphereSource()
            ss.SetRadius(r)
            ss.SetThetaResolution(res_t)
            ss.SetPhiResolution(res_phi)
            ss.Update()

            Mesh.__init__(self, ss.GetOutput(), c, alpha)

        self.phong()
        self.SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Sphere"


class Spheres(Mesh):
    """
    Build a (possibly large) set of spheres at `centers` of radius `r`.

    Either `c` or `r` can be a list of RGB colors or radii.

    |manyspheres| |manyspheres.py|_
    """
    def __init__(self, centers, r=1, c="r", alpha=1, res=8):

        if isinstance(centers, Mesh):
            centers = centers.points()

        cisseq = False
        if utils.isSequence(c):
            cisseq = True

        if cisseq:
            if len(centers) > len(c):
                printc("~times Mismatch in Spheres() colors", len(centers), len(c), c=1)
                raise RuntimeError()
            if len(centers) != len(c):
                printc("~lightningWarning: mismatch in Spheres() colors", len(centers), len(c))

        risseq = False
        if utils.isSequence(r):
            risseq = True

        if risseq:
            if len(centers) > len(r):
                printc("times Mismatch in Spheres() radius", len(centers), len(r), c=1)
                raise RuntimeError()
            if len(centers) != len(r):
                printc("~lightning Warning: mismatch in Spheres() radius", len(centers), len(r))
        if cisseq and risseq:
            printc("~noentry Limitation: c and r cannot be both sequences.", c=1)
            raise RuntimeError()

        src = vtk.vtkSphereSource()
        if not risseq:
            src.SetRadius(r)
        if utils.isSequence(res):
            res_t, res_phi = res
        else:
            res_t, res_phi = 2*res, res

        src.SetThetaResolution(res_t)
        src.SetPhiResolution(res_phi)
        src.Update()

        psrc = vtk.vtkPointSource()
        psrc.SetNumberOfPoints(len(centers))
        psrc.Update()
        pd = psrc.GetOutput()
        vpts = pd.GetPoints()

        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(src.GetOutputPort())

        if cisseq:
            glyph.SetColorModeToColorByScalar()
            ucols = vtk.vtkUnsignedCharArray()
            ucols.SetNumberOfComponents(3)
            ucols.SetName("colors")
            for i, p in enumerate(centers):
                cx, cy, cz = getColor(c[i])
                ucols.InsertNextTuple3(cx * 255, cy * 255, cz * 255)
            pd.GetPointData().SetScalars(ucols)
            glyph.ScalingOff()
        elif risseq:
            glyph.SetScaleModeToScaleByScalar()
            urads = numpy_to_vtk(np.ascontiguousarray(2*r).astype(float), deep=True)
            urads.SetName("radii")
            pd.GetPointData().SetScalars(urads)

        vpts.SetData(numpy_to_vtk(np.ascontiguousarray(centers), deep=True))

        glyph.SetInputData(pd)
        glyph.Update()

        Mesh.__init__(self, glyph.GetOutput(), alpha=alpha)
        self.phong()
        if cisseq:
            self.mapper().ScalarVisibilityOn()
        else:
            self.mapper().ScalarVisibilityOff()
            self.GetProperty().SetColor(getColor(c))
        settings.collectable_actors.append(self)
        self.name = "Spheres"


class Earth(Mesh):
    """Build a textured mesh representing the Earth.

    |geodesic| |geodesic.py|_
    """
    def __init__(self, pos=(0, 0, 0), r=1, style=1):
        tss = vtk.vtkTexturedSphereSource()
        tss.SetRadius(r)
        tss.SetThetaResolution(72)
        tss.SetPhiResolution(36)

        Mesh.__init__(self, tss, c="w")

        atext = vtk.vtkTexture()
        pnmReader = vtk.vtkJPEGReader()
        fn = settings.textures_path + "earth" + str(style) +".jpg"
        pnmReader.SetFileName(fn)
        atext.SetInputConnection(pnmReader.GetOutputPort())
        atext.InterpolateOn()
        self.SetTexture(atext)
        self.SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Earth"


class Ellipsoid(Mesh):
    """
    Build a 3D ellipsoid centered at position `pos`.

    .. note:: `axis1` and `axis2` are only used to define sizes and one azimuth angle.

    |projectsphere|

    |pca| |pca.py|_
    """
    def __init__(self, pos=(0, 0, 0), axis1=(1, 0, 0), axis2=(0, 2, 0), axis3=(0, 0, 3),
                 c="c", alpha=1, res=24):

        self.name = "Ellipsoid"
        self.center = pos
        self.va_error = 0
        self.vb_error = 0
        self.vc_error = 0
        self.axis1 = axis1
        self.axis2 = axis2
        self.axis3 = axis3
        self.nr_of_points = 1 # used by pcaEllipsoid

        if utils.isSequence(res):
            res_t, res_phi = res
        else:
            res_t, res_phi = 2*res, res

        elliSource = vtk.vtkSphereSource()
        elliSource.SetThetaResolution(res_t)
        elliSource.SetPhiResolution(res_phi)
        elliSource.Update()
        l1 = np.linalg.norm(axis1)
        l2 = np.linalg.norm(axis2)
        l3 = np.linalg.norm(axis3)
        self.va = l1
        self.vb = l2
        self.vc = l3
        axis1 = np.array(axis1) / l1
        axis2 = np.array(axis2) / l2
        axis3 = np.array(axis3) / l3
        angle = np.arcsin(np.dot(axis1, axis2))
        theta = np.arccos(axis3[2])
        phi = np.arctan2(axis3[1], axis3[0])

        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Scale(l1, l2, l3)
        t.RotateX(np.rad2deg(angle))
        t.RotateY(np.rad2deg(theta))
        t.RotateZ(np.rad2deg(phi))
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(elliSource.GetOutput())
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()
        self.transformation = t

        Mesh.__init__(self, pd, c, alpha)
        self.phong()
        self.GetProperty().BackfaceCullingOn()
        self.SetPosition(pos)
        self.Length = -np.array(axis1) / 2 + pos
        self.top = np.array(axis1) / 2 + pos
        settings.collectable_actors.append(self)

    def asphericity(self):
        """Return a measure of how different an ellipsoid is froma sphere.
        Values close to zero correspond to a spheric object.
        """
        a,b,c = self.va, self.vb, self.vc
        asp = ( ((a-b)/(a+b))**2
              + ((a-c)/(a+c))**2
              + ((b-c)/(b+c))**2 )/3. * 4.
        return asp

    def asphericity_error(self):
        """Calculate statistical error on the asphericity value.

        Errors on the main axes are stored in
        `Ellipsoid.va_error`
        `Ellipsoid.vb_error`
        `Ellipsoid.vc_error`
        """
        a,b,c = self.va, self.vb, self.vc
        sqrtn = np.sqrt(self.nr_of_points)
        ea, eb, ec = a/2/sqrtn, b/2/sqrtn, b/2/sqrtn

        #from sympy import *
        #init_printing(use_unicode=True)
        #a, b, c, ea, eb, ec = symbols("a b c, ea, eb,ec")
        #L = (
        #    (((a - b) / (a + b)) ** 2 + ((c - b) / (c + b)) ** 2 + ((a - c) / (a + c)) ** 2)
        #    / 3 * 4)
        #dl2 = (diff(L, a) * ea) ** 2 + (diff(L, b) * eb) ** 2 + (diff(L, c) * ec) ** 2
        #print(dl2)
        #exit()
        dL2 = (
            ea ** 2
            * (
                -8 * (a - b) ** 2 / (3 * (a + b) ** 3)
                - 8 * (a - c) ** 2 / (3 * (a + c) ** 3)
                + 4 * (2 * a - 2 * c) / (3 * (a + c) ** 2)
                + 4 * (2 * a - 2 * b) / (3 * (a + b) ** 2)
            ) ** 2
            + eb ** 2
            * (
                4 * (-2 * a + 2 * b) / (3 * (a + b) ** 2)
                - 8 * (a - b) ** 2 / (3 * (a + b) ** 3)
                - 8 * (-b + c) ** 2 / (3 * (b + c) ** 3)
                + 4 * (2 * b - 2 * c) / (3 * (b + c) ** 2)
            ) ** 2
            + ec ** 2
            * (
                4 * (-2 * a + 2 * c) / (3 * (a + c) ** 2)
                - 8 * (a - c) ** 2 / (3 * (a + c) ** 3)
                + 4 * (-2 * b + 2 * c) / (3 * (b + c) ** 2)
                - 8 * (-b + c) ** 2 / (3 * (b + c) ** 3)
            ) ** 2
        )

        err = np.sqrt(dL2)

        self.va_error = ea
        self.vb_error = eb
        self.vc_error = ec
        return err


class Grid(Mesh):
    """Return an even or uneven 2D grid at `z=0`.

    :param float,list sx: if a float is provided it is interpreted as the total size along x,
        if a list of coords is provided they are interpreted as the vertices of the grid along x.
        In this case keyword `resx` is ignored (see example below).
    :param float,list sy: see above.
    :param float lw: line width.
    :param int resx: resolution along x, e.i. the number of axis subdivisions.

    |brownian2D| |brownian2D.py|_

    :Example:
        .. code-block:: python

            from vtkplotter import *
            xcoords = arange(0, 2, 0.2)
            ycoords = arange(0, 1, 0.2)
            sqrtx = sqrt(xcoords)
            grid = Grid(sx=sqrtx, sy=ycoords)
            grid.show(axes=8)
    """
    def __init__(self,
                pos=(0, 0, 0),
                normal=(0, 0, 1),
                sx=1,
                sy=1,
                sz=(0,),
                c="gray",
                alpha=1,
                lw=1,
                resx=10,
                resy=10,
                ):

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        if utils.isSequence(sx) and utils.isSequence(sy):
            verts = []
            for y in sy:
                for x in sx:
                    verts.append([x, y, 0])
            faces = []
            n = len(sx)
            m = len(sy)
            for j in range(m-1):
                j1n = (j+1)*n
                for i in range(n-1):
                    faces.append([i+j*n, i+1+j*n, i+1+j1n, i+j1n])

            Mesh.__init__(self, [verts, faces], c, alpha)

        else:
            ps = vtk.vtkPlaneSource()
            ps.SetResolution(resx, resy)
            ps.Update()
            poly0 = ps.GetOutput()
            t0 = vtk.vtkTransform()
            t0.Scale(sx, sy, 1)
            tf0 = vtk.vtkTransformPolyDataFilter()
            tf0.SetInputData(poly0)
            tf0.SetTransform(t0)
            tf0.Update()
            poly = tf0.GetOutput()
            Mesh.__init__(self, poly, c, alpha)
            self.SetPosition(pos)

        self.orientation(normal)

        self.wireframe().lw(lw).lighting('off')
        settings.collectable_actors.append(self)
        self.name = "Grid"


class Plane(Mesh):
    """
    Draw a plane of size `sx` and `sy` oriented perpendicular to vector `normal`
    and so that it passes through point `pos`.

    |Plane|
    """
    def __init__(self, pos=(0, 0, 0), normal=(0, 0, 1), sx=1, sy=None, c="g", alpha=1):

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        self.normal = np.array(normal)
        self.center = np.array(pos)
        self.variance = 0

        if sy is None:
            sy = sx
        ps = vtk.vtkPlaneSource()
        ps.SetResolution(1, 1)
        tri = vtk.vtkTriangleFilter()
        tri.SetInputConnection(ps.GetOutputPort())
        tri.Update()
        poly = tri.GetOutput()
        axis = np.array(normal) / np.linalg.norm(normal)
        theta = np.arccos(axis[2])
        phi = np.arctan2(axis[1], axis[0])
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Scale(sx, sy, 1)
        t.RotateY(np.rad2deg(theta))
        t.RotateZ(np.rad2deg(phi))
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(poly)
        tf.SetTransform(t)
        tf.Update()
        Mesh.__init__(self, tf.GetOutput(), c, alpha)
        self.SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Plane"
        self.top = np.array(normal)
        self.bottom = np.array([0,0,0])


def Rectangle(p1=(0, 0), p2=(2, 1), c="g", alpha=1):
    """Build a rectangle in the xy plane identified by two corner points."""
    if len(p1) == 2:
        p1 = np.array([p1[0], p1[1], 0.])
    else:
        p1 = np.array(p1)
    if len(p2) == 2:
        p2 = np.array([p2[0], p2[1], 0.])
    else:
        p2 = np.array(p2)
    p1r = np.array([p2[0], p1[1], 0.])
    p2l = np.array([p1[0], p2[1], 0.])
    pts = ([0,0,0], p1r-p1 , p2-p1, p2l-p1)
    faces = [(0,1,2,3)]
    mesh = Mesh([pts, faces], c, alpha)
    mesh.SetOrigin((p1r+p1)/2)
    mesh.SetPosition(p1)
    mesh.name = "Rectangle"
    return mesh


class Box(Mesh):
    """
    Build a box of dimensions `x=length, y=width and z=height`.
    Alternatively dimensions can be defined by setting `size` keyword with a tuple.

    |aspring| |aspring.py|_
    """
    def __init__(self, pos=(0,0,0), length=1, width=2, height=3, size=(), c="g", alpha=1):

        if len(size):
            length, width, height = size
        src = vtk.vtkCubeSource()
        src.SetXLength(length)
        src.SetYLength(width)
        src.SetZLength(height)
        src.Update()
        pd = src.GetOutput()
        Mesh.__init__(self, pd, c, alpha)
        self.SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Box"

def Cube(pos=(0, 0, 0), side=1, c="g", alpha=1):
    """Build a cube of size `side`.

    |colorcubes| |colorcubes.py|_
    """
    mesh = Box(pos, side, side, side, (), c, alpha)
    mesh.name = "Cube"
    return mesh

def CubicGrid(pos=(0, 0, 0), n=(10,10,10), spacing=(), c="lightgrey", alpha=0.1):
    """Build a cubic Mesh made o `n` small quads in the 3 axis directions.

    :param list pos: position of the left bottom corner
    :param int n: number of subdivisions
    :parameter list spacing: size of the side of the single quad in the 3 directions
    """
    img = vtk.vtkImageData()
    img.SetDimensions(n[0]+1,n[1]+1,n[2]+1)
    if len(spacing)==3:
        img.SetSpacing(spacing)
    else:
        img.SetSpacing(1./n[0], 1./n[1], 1./n[2])
    mesh = utils.geometry(img).c(c).alpha(alpha)
    mesh.SetPosition(pos)
    mesh.base = np.array([0.5,0.5,0])
    mesh.top  = np.array([0.5,0.5,1])
    mesh.name = "CubicGrid"
    return mesh


class Spring(Mesh):
    """
    Build a spring of specified nr of `coils` between `startPoint` and `endPoint`.

    :param int coils: number of coils
    :param float r: radius at start point
    :param float r2: radius at end point
    :param float thickness: thickness of the coil section

    |aspring| |aspring.py|_
    """
    def __init__(self,
                startPoint=(0, 0, 0),
                endPoint=(1, 0, 0),
                coils=20,
                r=0.1,
                r2=None,
                thickness=None,
                c="grey",
                alpha=1,
    ):
        diff = endPoint - np.array(startPoint)
        length = np.linalg.norm(diff)
        if not length:
            return None
        if not r:
            r = length / 20
        trange = np.linspace(0, length, num=50 * coils)
        om = 6.283 * (coils - 0.5) / length
        if not r2:
            r2 = r
        pts = []
        for t in trange:
            f = (length - t) / length
            rd = r * f + r2 * (1 - f)
            pts.append([rd * np.cos(om * t), rd * np.sin(om * t), t])

        pts = [[0, 0, 0]] + pts + [[0, 0, length]]
        diff = diff / length
        theta = np.arccos(diff[2])
        phi = np.arctan2(diff[1], diff[0])
        sp = Line(pts).polydata(False)
        t = vtk.vtkTransform()
        t.RotateZ(np.rad2deg(phi))
        t.RotateY(np.rad2deg(theta))
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(sp)
        tf.SetTransform(t)
        tf.Update()
        tuf = vtk.vtkTubeFilter()
        tuf.SetNumberOfSides(12)
        tuf.CappingOn()
        tuf.SetInputData(tf.GetOutput())
        if not thickness:
            thickness = r / 10
        tuf.SetRadius(thickness)
        tuf.Update()
        Mesh.__init__(self, tuf.GetOutput(), c, alpha)
        self.phong()
        self.SetPosition(startPoint)
        self.base = np.array(startPoint)
        self.top = np.array(endPoint)
        settings.collectable_actors.append(self)
        self.name = "Spring"


class Cylinder(Mesh):
    """
    Build a cylinder of specified height and radius `r`, centered at `pos`.

    If `pos` is a list of 2 Points, e.g. `pos=[v1,v2]`, build a cylinder with base
    centered at `v1` and top at `v2`.

    |Cylinder|
    """
    def __init__(self, pos=(0,0,0), r=1, height=2, axis=(0,0,1), c="teal", alpha=1, res=24):

        if utils.isSequence(pos[0]):  # assume user is passing pos=[base, top]
            base = np.array(pos[0])
            top = np.array(pos[1])
            pos = (base + top) / 2
            height = np.linalg.norm(top - base)
            axis = top - base
            axis = utils.versor(axis)
        else:
            axis = utils.versor(axis)
            base = pos - axis * height / 2
            top = pos + axis * height / 2

        cyl = vtk.vtkCylinderSource()
        cyl.SetResolution(res)
        cyl.SetRadius(r)
        cyl.SetHeight(height)
        cyl.Update()

        theta = np.arccos(axis[2])
        phi = np.arctan2(axis[1], axis[0])
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.RotateX(90)  # put it along Z
        t.RotateY(np.rad2deg(theta))
        t.RotateZ(np.rad2deg(phi))
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(cyl.GetOutput())
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()

        Mesh.__init__(self, pd, c, alpha)
        self.phong()
        self.SetPosition(pos)
        self.base = base + pos
        self.top = top + pos
        settings.collectable_actors.append(self)
        self.name = "Cylinder"


class Cone(Mesh):
    """
    Build a cone of specified radius `r` and `height`, centered at `pos`.

    |Cone|
    """
    def __init__(self, pos=(0,0,0), r=1, height=3, axis=(0,0,1), c="dg", alpha=1, res=48):
        con = vtk.vtkConeSource()
        con.SetResolution(res)
        con.SetRadius(r)
        con.SetHeight(height)
        con.SetDirection(axis)
        con.Update()
        Mesh.__init__(self, con.GetOutput(), c, alpha)
        self.phong()
        self.SetPosition(pos)
        v = utils.versor(axis) * height / 2
        self.base = pos - v
        self.top = pos + v
        settings.collectable_actors.append(self)
        self.name = "Cone"

class Pyramid(Cone):
    """
    Build a pyramid of specified base size `s` and `height`, centered at `pos`.
    """
    def __init__(self, pos=(0,0,0), s=1, height=1, axis=(0,0,1), c="dg", alpha=1):
        Cone.__init__(self, pos, s, height, axis, c, alpha, 4)
        self.name = "Pyramid"


class Torus(Mesh):
    """
    Build a torus of specified outer radius `r` internal radius `thickness`, centered at `pos`.

    |gas| |gas.py|_
    """
    def __init__(self, pos=(0, 0, 0), r=1, thickness=0.2, c="khaki", alpha=1, res=30):
        rs = vtk.vtkParametricTorus()
        rs.SetRingRadius(r)
        rs.SetCrossSectionRadius(thickness)
        pfs = vtk.vtkParametricFunctionSource()
        pfs.SetParametricFunction(rs)
        if utils.isSequence(res):
            res_u, res_v = res
        else:
            res_u, res_v = 3*res, res
        pfs.SetUResolution(res_u)
        pfs.SetVResolution(res_v)
        pfs.Update()
        Mesh.__init__(self, pfs.GetOutput(), c, alpha)
        self.phong()
        self.SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Torus"

class Paraboloid(Mesh):
    """
    Build a paraboloid of specified height and radius `r`, centered at `pos`.

    .. note::
        Full volumetric expression is:
            :math:`F(x,y,z)=a_0x^2+a_1y^2+a_2z^2+a_3xy+a_4yz+a_5xz+ a_6x+a_7y+a_8z+a_9`

            |paraboloid|
    """

    def __init__(self, pos=(0,0,0), r=1, height=1, c="cyan", alpha=1, res=50):
        quadric = vtk.vtkQuadric()
        quadric.SetCoefficients(1, 1, 0, 0, 0, 0, 0, 0, height / 4, 0)
        # F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2
        #         + a3*x*y + a4*y*z + a5*x*z
        #         + a6*x   + a7*y   + a8*z  +a9
        sample = vtk.vtkSampleFunction()
        sample.SetSampleDimensions(res, res, res)
        sample.SetImplicitFunction(quadric)

        contours = vtk.vtkContourFilter()
        contours.SetInputConnection(sample.GetOutputPort())
        contours.GenerateValues(1, 0.01, 0.01)
        contours.Update()

        Mesh.__init__(self, contours.GetOutput(), c, alpha)
        self.computeNormals().phong()
        self.mapper().ScalarVisibilityOff()
        self.SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Paraboloid"


class Hyperboloid(Mesh):
    """
    Build a hyperboloid of specified aperture `a2` and `height`, centered at `pos`.

    Full volumetric expression is:
        :math:`F(x,y,z)=a_0x^2+a_1y^2+a_2z^2+a_3xy+a_4yz+a_5xz+ a_6x+a_7y+a_8z+a_9`
    """
    def __init__(self, pos=(0,0,0), a2=1, value=0.5, height=1, c="m", alpha=1, res=100):
        q = vtk.vtkQuadric()
        q.SetCoefficients(2, 2, -1 / a2, 0, 0, 0, 0, 0, 0, 0)
        # F(x,y,z) = a0*x^2 + a1*y^2 + a2*z^2
        #         + a3*x*y + a4*y*z + a5*x*z
        #         + a6*x   + a7*y   + a8*z  +a9
        sample = vtk.vtkSampleFunction()
        sample.SetSampleDimensions(res, res, res)
        sample.SetImplicitFunction(q)

        contours = vtk.vtkContourFilter()
        contours.SetInputConnection(sample.GetOutputPort())
        contours.GenerateValues(1, value, value)
        contours.Update()

        Mesh.__init__(self, contours.GetOutput(), c, alpha)
        self.computeNormals().phong()
        self.mapper().ScalarVisibilityOff()
        self.SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Hyperboloid"


class Text(Mesh):
    """
    Returns a polygonal ``Mesh`` of 3D text.

    :param list pos: position coordinates in 3D space
    :param float s: size of text.
    :param float depth: text thickness.
    :param bool,float italic: italic font type (can be a +-`float` too).
    :param str justify: text justification
        (bottom-left, bottom-right, top-left, top-right, centered).

    |markpoint| |markpoint.py|_
    """
    def __init__(self,
                txt,
                pos=(0,0,0),
                s=1,
                depth=0,
                italic=False,
                justify="bottom-left",
                c=None,
                alpha=1,
                ):

        if len(pos)==2:
            pos = (pos[0], pos[1], 0)

        if c is None: # automatic black or white
            pli = settings.plotter_instance
            if pli and pli.renderer:
                c = (0.9, 0.9, 0.9)
                if pli.renderer.GetGradientBackground():
                    bgcol = pli.renderer.GetBackground2()
                else:
                    bgcol = pli.renderer.GetBackground()
                if np.sum(bgcol) > 1.5:
                    c = (0.1, 0.1, 0.1)
            else:
                c = (0.6, 0.6, 0.6)

        # otherwise build the 3D text, fonts do not apply
        tt = vtk.vtkVectorText()
        tt.SetText(str(txt))
        tt.Update()
        tpoly = tt.GetOutput()

        bb = tpoly.GetBounds()
        dx, dy = (bb[1] - bb[0]) / 2 * s, (bb[3] - bb[2]) / 2 * s
        cm = np.array([(bb[1] + bb[0]) / 2,
                       (bb[3] + bb[2]) / 2,
                       (bb[5] + bb[4]) / 2]) * s
        shift = -cm
        if "bottom" in justify: shift += np.array([  0, dy, 0])
        if "top"    in justify: shift += np.array([  0,-dy, 0])
        if "left"   in justify: shift += np.array([ dx,  0, 0])
        if "right"  in justify: shift += np.array([-dx,  0, 0])

        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Scale(s, s, s)
        t.Translate(shift)
        if italic is True:
            italic = 1
        if italic:
            t.Concatenate([1,italic*0.28,0,0,
                           0,1,0,0,
                           0,0,1,0,
                           0,0,0,1])

        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(tpoly)
        tf.SetTransform(t)
        tf.Update()
        tpoly = tf.GetOutput()

        if depth:
            extrude = vtk.vtkLinearExtrusionFilter()
            extrude.SetInputData(tpoly)
            extrude.SetExtrusionTypeToVectorExtrusion()
            extrude.SetVector(0, 0, 1)
            extrude.SetScaleFactor(depth*dy)
            extrude.Update()
            tpoly = extrude.GetOutput()
        Mesh.__init__(self, tpoly, c, alpha)
        self.lighting('off').SetPosition(pos)
        settings.collectable_actors.append(self)
        self.name = "Text"


def Text2D(
    txt,
    pos=3,
    s=1,
    c=None,
    alpha=1,
    bg=None,
    font="Montserrat",
    justify="bottom-left",
    bold=False,
    italic=False,
):
    """Returns a ``vtkActor2D`` representing 2D text.

    :param pos: text is placed in one of the 8 positions:

            1, bottom-left
            2, bottom-right
            3, top-left
            4, top-right
            5, bottom-middle
            6, middle-right
            7, middle-left
            8, top-middle

        If a pair (x,y) is passed as input the 2D text is place at that
        position in the coordinate system of the 2D screen (with the
        origin sitting at the bottom left).

    :type pos: list, int

    :param float s: size of text.
    :param bg: background color
    :param float alpha: background opacity
    :param str justify: text justification
    :param str font: available fonts are

            - Courier
            - Times
            - Arial
            - CallingCode
            - ChineseRuler
            - Godsway
            - ImpactLabel
            - Komiko
            - Monospace
            - Montserrat
            - Overspray
            - SpecialElite

        A path to `otf` or `ttf` font-file can also be supplied as input.

    .. hint:: Examples, |fonts.py|_ |colorcubes.py|_ |annotations.py|_

        |colorcubes|

        |fonts|
    """
    if c is None: # automatic black or white
        if settings.plotter_instance and settings.plotter_instance.renderer:
            c = (0.9, 0.9, 0.9)
            if settings.plotter_instance.renderer.GetGradientBackground():
                bgcol = settings.plotter_instance.renderer.GetBackground2()
            else:
                bgcol = settings.plotter_instance.renderer.GetBackground()
            if np.sum(bgcol) > 1.5:
                c = (0.1, 0.1, 0.1)
        else:
            c = (0.5, 0.5, 0.5)

    if isinstance(pos, str): # corners
        if "top" in pos:
            if "left" in pos: pos = 3
            elif "mid" in pos: pos = 8
            elif "right" in pos: pos = 4
        elif "bottom" in pos:
            if "left" in pos: pos = 1
            elif "mid" in pos: pos = 5
            elif "right" in pos: pos = 2
        else:
            if "left" in pos: pos = 7
            elif "right" in pos: pos = 6
            else: pos = 3

    if isinstance(pos, int): # corners
        if pos > 8:
            pos = 8
        if pos < 1:
            pos = 1
        ca = vtk.vtkCornerAnnotation()
        ca.SetNonlinearFontScaleFactor(s/2.7)
        ca.SetText(pos - 1, str(txt))
        ca.PickableOff()
        cap = ca.GetTextProperty()
        cap.SetColor(getColor(c))
        if font.lower() == "courier": cap.SetFontFamilyToCourier()
        elif font.lower() == "times": cap.SetFontFamilyToTimes()
        elif font.lower() == "arial": cap.SetFontFamilyToArial()
        else:
            cap.SetFontFamily(vtk.VTK_FONT_FILE)
            cap.SetFontFile(settings.fonts_path + font +'.ttf')
        if bg:
            bgcol = getColor(bg)
            cap.SetBackgroundColor(bgcol)
            cap.SetBackgroundOpacity(alpha * 0.5)
            #cap.SetFrameColor(bgcol)
            #cap.FrameOn()
        cap.SetBold(bold)
        cap.SetItalic(italic)
        setattr(ca, 'renderedAt', set())
        settings.collectable_actors.append(ca)
        return ca

    if len(pos)!=2:
        print("Error in Text2D(): len(pos) must be 2 or integer value.")
        raise RuntimeError()

    actor2d = vtk.vtkActor2D()
    actor2d.SetPosition(pos)
    tmapper = vtk.vtkTextMapper()
    tmapper.SetInput(str(txt))
    actor2d.SetMapper(tmapper)
    tp = tmapper.GetTextProperty()
    tp.BoldOff()
    tp.SetFontSize(s*20)
    tp.SetColor(getColor(c))
    tp.SetJustificationToLeft()
    if "top" in justify:
        tp.SetVerticalJustificationToTop()
    if "bottom" in justify:
        tp.SetVerticalJustificationToBottom()
    if "cent" in justify:
        tp.SetVerticalJustificationToCentered()
        tp.SetJustificationToCentered()
    if "left" in justify:
        tp.SetJustificationToLeft()
    if "right" in justify:
        tp.SetJustificationToRight()

    if font.lower() == "courier": tp.SetFontFamilyToCourier()
    elif font.lower() == "times": tp.SetFontFamilyToTimes()
    elif font.lower() == "arial": tp.SetFontFamilyToArial()
    else:
        tp.SetFontFamily(vtk.VTK_FONT_FILE)
        if font in settings.fonts:
            tp.SetFontFile(settings.fonts_path + font + '.ttf')
        elif os.path.exists(font):
            tp.SetFontFile(font)
        else:
            printc("~sad Font", font, "not found in", settings.fonts_path, c="r")
            printc("~pin Available fonts are:", settings.fonts, c="m")
            return None
    if bg:
        bgcol = getColor(bg)
        tp.SetBackgroundColor(bgcol)
        tp.SetBackgroundOpacity(alpha * 0.5)
        tp.SetFrameColor(bgcol)
        tp.FrameOn()
    actor2d.PickableOff()
    setattr(actor2d, 'renderedAt', set())
    settings.collectable_actors.append(actor2d)
    return actor2d

class Latex(Picture):
    """
    Render Latex formulas.

    :param str formula: latex text string
    :param list pos: position coordinates in space
    :param c: face color
    :param bg: background color box
    :param int res: dpi resolution
    :param bool usetex: use latex compiler of matplotlib
    :param fromweb: retrieve the latex image from online server (codecogs)

    You can access the latex formula in `Latex.formula'`.

    |latex| |latex.py|_
    """
    def __init__(self,
        formula,
        pos=(0, 0, 0),
        c='k',
        s=1,
        bg=None,
        alpha=1,
        res=30,
        usetex=False,
        fromweb=False,
    ):
        self.formula = formula

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        try:

            def build_img_web(formula, tfile):
                import requests
                if c == 'k':
                    ct = 'Black'
                else:
                    ct = 'White'
                wsite = 'http://latex.codecogs.com/png.latex'
                try:
                    r = requests.get(wsite+'?\dpi{100} \huge \color{'+ct+'} ' + formula)
                    f = open(tfile, 'wb')
                    f.write(r.content)
                    f.close()
                except requests.exceptions.ConnectionError:
                    printc('Latex error. Web site unavailable?', wsite, c=1)

            def build_img_plt(formula, tfile):
                import matplotlib.pyplot as plt

                plt.rc('text', usetex=usetex)

                formula1 = '$'+formula+'$'
                plt.axis('off')
                col = getColor(c)
                if bg:
                    bx = dict(boxstyle="square", ec=col, fc=getColor(bg))
                else:
                    bx = None
                plt.text(0.5, 0.5, formula1,
                         size=res,
                         color=col,
                         alpha=alpha,
                         ha="center",
                         va="center",
                         bbox=bx)
                plt.savefig('_lateximg.png', format='png',
                            transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close()

            if fromweb:
                build_img_web(formula, '_lateximg.png')
            else:
                build_img_plt(formula, '_lateximg.png')

            Picture.__init__(self, '_lateximg.png')
            self.alpha(alpha)
            b = self.GetBounds()
            xm, ym = (b[1]+b[0])/200*s, (b[3]+b[2])/200*s
            self.SetOrigin(-xm, -ym, 0)
            self.SetScale(0.25/res*s, 0.25/res*s, 0.25/res*s)
            self.SetPosition(pos)
            try:
                import os
                os.unlink('_lateximg.png')
            except:
                pass

        except:
            printc('Error in Latex()\n', formula, c=1)
            printc(' latex or dvipng not installed?', c=1)
            printc(' Try: usetex=False' , c=1)
            printc(' Try: sudo apt install dvipng' , c=1)

        settings.collectable_actors.append(self)
        self.name = "Latex"


class ParametricShape(Mesh):
    """
    A set of built-in shapes mainly for illustration purposes.

    Name can be an integer or a string in this list:

        `['Boy', 'ConicSpiral', 'CrossCap', 'Dini', 'Enneper',
        'Figure8Klein', 'Klein', 'Mobius', 'RandomHills', 'Roman',
        'SuperEllipsoid', 'BohemianDome', 'Bour', 'CatalanMinimal',
        'Henneberg', 'Kuen', 'PluckerConoid', 'Pseudosphere'].`

    :Example:
        .. code-block:: python

            from vtkplotter import *
            for i in range(18):
                ps = ParametricShape(i, c=i)
                show([ps, Text(ps.name)], at=i, N=18)
            interactive()

        |paramshapes|
    """
    def __init__(self, name, c='powderblue', alpha=1, res=51):
        shapes = ['Boy', 'ConicSpiral', 'CrossCap', 'Enneper',
                  'Figure8Klein', 'Klein', 'Dini', 'Mobius', 'RandomHills', 'Roman',
                  'SuperEllipsoid', 'BohemianDome', 'Bour', 'CatalanMinimal',
                  'Henneberg', 'Kuen', 'PluckerConoid', 'Pseudosphere']

        if isinstance(name, int):
            name = name%len(shapes)
            name = shapes[name]

        if   name == 'Boy': ps = vtk.vtkParametricBoy()
        elif name == 'ConicSpiral': ps = vtk.vtkParametricConicSpiral()
        elif name == 'CrossCap': ps = vtk.vtkParametricCrossCap()
        elif name == 'Dini': ps = vtk.vtkParametricDini()
        elif name == 'Enneper': ps = vtk.vtkParametricEnneper()
        elif name == 'Figure8Klein': ps = vtk.vtkParametricFigure8Klein()
        elif name == 'Klein': ps = vtk.vtkParametricKlein()
        elif name == 'Mobius':
            ps = vtk.vtkParametricMobius()
            ps.SetRadius(2.0)
            ps.SetMinimumV(-0.5)
            ps.SetMaximumV(0.5)
        elif name == 'RandomHills':
            ps = vtk.vtkParametricRandomHills()
            ps.AllowRandomGenerationOn()
            ps.SetRandomSeed(1)
            ps.SetNumberOfHills(25)
        elif name == 'Roman': ps = vtk.vtkParametricRoman()
        elif name == 'SuperEllipsoid':
            ps = vtk.vtkParametricSuperEllipsoid()
            ps.SetN1(0.5)
            ps.SetN2(0.4)
        elif name == 'BohemianDome':
            ps = vtk.vtkParametricBohemianDome()
            ps.SetA(5.0)
            ps.SetB(1.0)
            ps.SetC(2.0)
        elif name == 'Bour': ps = vtk.vtkParametricBour()
        elif name == 'CatalanMinimal': ps = vtk.vtkParametricCatalanMinimal()
        elif name == 'Henneberg': ps = vtk.vtkParametricHenneberg()
        elif name == 'Kuen':
            ps = vtk.vtkParametricKuen()
            ps.SetDeltaV0(0.001)
        elif name == 'PluckerConoid': ps = vtk.vtkParametricPluckerConoid()
        elif name == 'Pseudosphere': ps = vtk.vtkParametricPseudosphere()
        else:
            printc("Error in ParametricShape: unknown name", name, c=1)
            printc("Available shape names:\n", shapes)
            return None

        pfs = vtk.vtkParametricFunctionSource()
        pfs.SetParametricFunction(ps)
        pfs.SetUResolution(res)
        pfs.SetVResolution(res)
        pfs.SetWResolution(res)
        pfs.SetScalarModeToZ()
        pfs.Update()

        Mesh.__init__(self, pfs.GetOutput(), c, alpha)
        if name != 'Kuen': self.normalize()
        if name == 'Dini': self.scale(0.4)
        if name == 'Enneper': self.scale(0.4)
        settings.collectable_actors.append(self)
        self.name = name
