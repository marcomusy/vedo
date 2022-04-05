#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from functools import lru_cache

import numpy as np
import vedo
import vedo.utils as utils
import vtk
from vedo import settings
from vedo.colors import cmaps_names
from vedo.colors import colorMap
from vedo.colors import getColor
from vedo.colors import printc
from vedo.mesh import merge
from vedo.mesh import Mesh
from vedo.picture import Picture
from vedo.pointcloud import Points

__doc__ = """
Submodule to generate simple and complex geometric shapes
.. image:: https://vedo.embl.es/images/basic/extrude.png
"""

__all__ = [
    "Marker",
    "Line",
    "DashedLine",
    "RoundedLine",
    "Tube",
    "Lines",
    "Spline",
    "KSpline",
    "CSpline",
    "Bezier",
    "Brace",
    "NormalLines",
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
    "GeoCircle",
    "Arc",
    "Star",
    "Star3D",
    "Cross3D",
    "Sphere",
    "Spheres",
    "Earth",
    "Ellipsoid",
    "Grid",
    "TessellatedBox",
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
    "TextBase",
    "Text3D",
    "Text2D",
    "CornerAnnotation",
    "Latex",
    "Glyph",
    "Tensors",
    "ParametricShape",
    "ConvexHull",
    "VedoLogo",
]

##############################################
_reps = [
    ("\nabla", "∇"),
    ("\infty", "∞"),
    ("\rightarrow", "→"),
    ("\lefttarrow", "←"),
    ("\partial", "∂"),
    ("\sqrt", "√"),
    ("\approx", "≈"),
    ("\neq", "≠"),
    ("\leq", "≤"),
    ("\geq", "≥"),
    ("\foreach", "∀"),
    ("\permille", "‰"),
    ("\euro", "€"),
    ("\dot", "·"),
    ("\varnothing", "∅"),
    ("\int", "∫"),
    ("\pm", "±"),
    ("\times","×"),
    ("\Gamma", "Γ"),
    ("\Delta", "Δ"),
    ("\Theta", "Θ"),
    ("\Lambda", "Λ"),
    ("\Pi", "Π"),
    ("\Sigma", "Σ"),
    ("\Phi", "Φ"),
    ("\Chi", "X"),
    ("\Xi", "Ξ"),
    ("\Psi", "Ψ"),
    ("\Omega", "Ω"),
    ("\alpha", "α"),
    ("\beta", "β"),
    ("\gamma", "γ"),
    ("\delta", "δ"),
    ("\epsilon", "ε"),
    ("\zeta", "ζ"),
    ("\eta", "η"),
    ("\theta", "θ"),
    ("\kappa", "κ"),
    ("\lambda", "λ"),
    ("\mu", "μ"),
    ("\lowerxi", "ξ"),
    ("\nu", "ν"),
    ("\pi", "π"),
    ("\rho", "ρ"),
    ("\sigma", "σ"),
    ("\tau", "τ"),
    ("\varphi", "φ"),
    ("\phi", "φ"),
    ("\chi", "χ"),
    ("\psi", "ψ"),
    ("\omega", "ω"),
    ("\circ", "°"),
    ("\onehalf", "½"),
    ("\onefourth", "¼"),
    ("\threefourths", "¾"),
    ("\^1", "¹"),
    ("\^2", "²"),
    ("\^3", "³"),
    ("\,", "~"),
]


########################################################################
class Glyph(Mesh):
    """
    At each vertex of a mesh, another mesh, i.e. a "glyph", is shown with
    various orientation options and coloring.

    The input can also be a simple list of 2D or 3D coordinates.
    Color can be specified as a colormap which maps the size of the orientation
    vectors in `orientationArray`.

    Parameters
    ----------
    orientationArray: list, str, vtkArray
        list of vectors, `vtkArray` or name of an already existing pointdata array

    scaleByScalar : bool
        glyph mesh is scaled by the active scalars

    scaleByVectorSize : bool
        glyph mesh is scaled by the size of the vectors

    scaleByVectorComponents : bool
        glyph mesh is scaled by the 3 vectors components

    colorByScalar : bool
        glyph mesh is colored based on the scalar value

    colorByVectorSize : bool
        glyph mesh is colored based on the vector size

    tol : float
        set a minimum separation between two close glyphs
        (not compatible with `orientationArray` being a list).

    .. hint:: examples/basic/glyphs.py, glyphs_arrows.py
        .. image:: https://vedo.embl.es/images/basic/glyphs.png
    """
    def __init__(
            self,
            mesh,
            glyphObj,
            orientationArray=None,
            scaleByScalar=False,
            scaleByVectorSize=False,
            scaleByVectorComponents=False,
            colorByScalar=False,
            colorByVectorSize=False,
            tol=0,
            c='k8',
            alpha=1,
        ):
        lighting = None
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

        if isinstance(glyphObj, Points):
            lighting = glyphObj.property.GetLighting()
            glyphObj = glyphObj.polydata()

        cmap=''
        if c in cmaps_names:
            cmap = c
            c = None
        elif utils.isSequence(c): # user passing an array of point colors
            ucols = vtk.vtkUnsignedCharArray()
            ucols.SetNumberOfComponents(3)
            ucols.SetName("glyph_RGB")
            for col in c:
                cl = getColor(col)
                ucols.InsertNextTuple3(cl[0]*255, cl[1]*255, cl[2]*255)
            poly.GetPointData().AddArray(ucols)
            poly.GetPointData().SetActiveScalars("glyph_RGB")
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
                    poly.GetPointData().SetActiveVectors(orientationArray)
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
        if lighting is not None:
            self.property.SetLighting(lighting)

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

        self.name = "Glyph"


class Tensors(Mesh):
    """
    Geometric representation of tensors defined on a domain or set of points.
    Tensors can be scaled and/or rotated according to the source at eache input point.
    Scaling and rotation is controlled by the eigenvalues/eigenvectors of the symmetrical part
    of the tensor as follows:

    For each tensor, the eigenvalues (and associated eigenvectors) are sorted
    to determine the major, medium, and minor eigenvalues/eigenvectors.
    The eigenvalue decomposition only makes sense for symmetric tensors,
    hence the need to only consider the symmetric part of the tensor,
    which is `1/2*(T+T.transposed())`.

    Parameters
    ----------
    source : str, Mesh
        preset type of source shape
        ['ellipsoid', 'cylinder', 'cube' or any specified ``Mesh``]

    useEigenValues : bool
        color source glyph using the eigenvalues or by scalars

    threeAxes : bool
        if `False` scale the source in the x-direction,
        the medium in the y-direction, and the minor in the z-direction.
        Then, the source is rotated so that the glyph's local x-axis lies
        along the major eigenvector, y-axis along the medium eigenvector,
        and z-axis along the minor.

        If `True` three sources are produced, each of them oriented along an eigenvector
        and scaled according to the corresponding eigenvector.

    isSymmetric : bool
        If `True` each source glyph is mirrored (2 or 6 glyphs will be produced).
        The x-axis of the source glyph will correspond to the eigenvector on output.

    length : float
        distance from the origin to the tip of the source glyph along the x-axis

    scale : float
        scaling factor of the source glyph.

    maxScale : float
        clamp scaling at this factor.

    .. hint:: examples/volumetric/tensors.py, tensor_grid.py
        .. image:: https://vedo.embl.es/images/volumetric/tensor_grid.png
    """

    def __init__(self, domain, source='ellipsoid', useEigenValues=True, isSymmetric=True,
                threeAxes=False, scale=1, maxScale=None, length=None,
                c=None, alpha=1):
        if isinstance(source, Points):
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

    If `p0` is already a list of points, return the line connecting them.

    A 2D set of coords can also be passed as `p0=[x..], p1=[y..]`.

    Parameters
    ----------
    closed : bool
        join last to first point

    res : int
        resolution, number of points along the line
        (only relevant if only 2 points are specified)

    lw : int
        line width in pixel units

    c : color, int, str, list
        color name, number, or list of [R,G,B] colors

    alpha : float
        opacity in range [0,1]
    """
    def __init__(self, p0, p1=None, closed=False, res=2, lw=1, c="k1", alpha=1):

        self.slope = [] # populated by analysis.fitLine
        self.center = []
        self.variances = []

        self.coefficients = [] # populated by pyplot.fit()
        self.covarianceMatrix = []
        self.coefficients = []
        self.coefficientErrors = []
        self.MonteCarloCoefficients = []
        self.reducedChi2 = -1
        self.ndof = 0
        self.dataSigma = 0
        self.errorLines = []
        self.errorBand = None
        self.res = res

        if isinstance(p1, Points):
            p1 = p1.GetPosition()
            if isinstance(p0, Points):
                p0 = p0.GetPosition()
        if isinstance(p0, Points):
            p0 = p0.points()

        # detect if user is passing a 2D list of points as p0=xlist, p1=ylist:
        if len(p0) > 3:
            if not utils.isSequence(p0[0]) and not utils.isSequence(p1[0]) and len(p0)==len(p1):
                # assume input is 2D xlist, ylist
                p0 = np.stack((p0, p1), axis=1)
                p1 = None
            if len(p0[0]) == 2: # make it 3d
                p0 = np.c_[np.array(p0, dtype=float), np.zeros(len(p0), dtype=float)]

        # detect if user is passing a list of points:
        if utils.isSequence(p0[0]):
            if len(p0[0]) == 2: # make it 3d
                p0 = np.c_[np.array(p0, dtype=float), np.zeros(len(p0), dtype=float)]

            ppoints = vtk.vtkPoints()  # Generate the polyline
            ppoints.SetData(utils.numpy2vtk(np.asarray(p0, dtype=float), dtype=float))
            lines = vtk.vtkCellArray()
            npt = len(p0)
            if closed:
                lines.InsertNextCell(npt+1)
            else:
                lines.InsertNextCell(npt)
            for i in range(npt):
                lines.InsertCellPoint(i)
            if closed:
                lines.InsertCellPoint(0)
            poly = vtk.vtkPolyData()
            poly.SetPoints(ppoints)
            poly.SetLines(lines)
            top  = p0[-1]
            base = p0[0]
            self.res = 2

        else:  # or just 2 points to link

            lineSource = vtk.vtkLineSource()
            if len(p0) == 2: # make it 3d
                p0 = [p0[0],p0[1],0]
            if len(p1) == 2:
                p1 = [p1[0],p1[1],0]
            lineSource.SetPoint1(p0)
            lineSource.SetPoint2(p1)
            lineSource.SetResolution(res-1)
            lineSource.Update()
            poly = lineSource.GetOutput()
            top  = np.array(p1, dtype=float)
            base = np.array(p0, dtype=float)

        Mesh.__init__(self, poly, c, alpha)
        self.lw(lw)
        self.property.LightingOff()
        self.PickableOff()
        self.DragableOff()
        self.base = base
        self.top = top
        self.name = "Line"

    def lineColor(self, c=None):
        """Assign a color to the line"""
        # overrides mesh.lineColor which would have no effect here
        return self.color(c)

    def eval(self, x):
        """
        Calculate the position of an intermediate point
        as a fraction of the length of the line,
        being x=0 the first point and x=1 the last point.
        This corresponds to an imaginary point that travels along the line
        at constant speed.

        Can be used in conjunction with `linInterpolate()`
        to map any range to the [0,1] range.
        """
        distance1 = 0.
        length = self.length()
        pts = self.points()
        for i in range(1, len(pts)):
            p0 = pts[i-1]
            p1 = pts[i]
            seg = p1-p0
            distance0 = distance1
            distance1 += np.linalg.norm(seg)
            w1 = distance1/length
            if w1 >= x:
                break
        w0 = distance0/length
        v = p0 + seg*(x-w0)/(w1-w0)
        return v

    def pattern(self, stipple, repeats=10):
        """
        Define a stipple pattern for dashing the line.
        Pass the stipple pattern as a string like `'- - -'`.
        Repeats controls the number of times the pattern repeats in a single segment.

        Examples are: `'- -', '--  -  --'`, etc.

        The resolution of the line (nr of points) can affect how pattern will show up.

        Example:
            .. code-block:: python

                from vedo import Line
                pts = [[1, 0, 0], [5, 2, 0], [3, 3, 1]]
                ln = Line(pts, c='r', lw=5).pattern('- -', repeats=10)
                ln.show(axes=1).close()
        """
        stipple = str(stipple) * int(2*repeats)
        dimension = len(stipple)

        image = vtk.vtkImageData()
        image.SetDimensions(dimension, 1, 1)
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
        image.SetExtent(0, dimension-1, 0, 0, 0, 0)
        i_dim = 0
        while i_dim < dimension:
            for i in range(dimension):
                image.SetScalarComponentFromFloat(i_dim, 0, 0, 0, 255)
                image.SetScalarComponentFromFloat(i_dim, 0, 0, 1, 255)
                image.SetScalarComponentFromFloat(i_dim, 0, 0, 2, 255)
                if stipple[i] == ' ':
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 3, 0)
                else:
                    image.SetScalarComponentFromFloat(i_dim, 0, 0, 3, 255)
                i_dim += 1

        polyData = self.polydata(False)

        # Create texture coordinates
        tcoords = vtk.vtkDoubleArray()
        tcoords.SetName("TCoordsStippledLine")
        tcoords.SetNumberOfComponents(1)
        tcoords.SetNumberOfTuples(polyData.GetNumberOfPoints())
        for i in range(polyData.GetNumberOfPoints()):
            tcoords.SetTypedTuple(i, [i/2])
        polyData.GetPointData().SetTCoords(tcoords)
        polyData.GetPointData().Modified()
        texture = vtk.vtkTexture()
        texture.SetInputData(image)
        texture.InterpolateOff()
        texture.RepeatOn()
        self.SetTexture(texture)
        return self

    def length(self):
        """Calculate length of the line."""
        distance = 0.
        pts = self.points()
        for i in range(1, len(pts)):
            distance += np.linalg.norm(pts[i]-pts[i-1])
        return distance

    def tangents(self):
        """
        Compute the tangents of a line in space.

        Example:
            .. code-block:: python

                from vedo import *
                shape = load(dataurl+"timecourse1d.npy")[58]
                pts = shape.rotateX(30).points()
                tangents = Line(pts).tangents()
                arrs = Arrows(pts, pts+tangents, c='blue9')
                show(shape.c('red5').lw(5), arrs, bg='bb', axes=1).close()
        """
        v = np.gradient(self.points())[0]
        ds_dt = np.linalg.norm(v, axis=1)
        tangent = np.array([1/ds_dt] * 3).transpose() * v
        return tangent

    def curvature(self):
        """
        Compute the signed curvature of a line in space.
        The signed is computed assuming the line is about coplanar to the xy plane.

        Example:
            .. code-block:: python

                from vedo import *
                from vedo.pyplot import plot
                shape = load(dataurl+"timecourse1d.npy")[55]
                curvs = Line(shape.points()).curvature()
                shape.cmap('coolwarm', curvs, vmin=-2,vmax=2).addScalarBar3D(c='w')
                shape.renderLinesAsTubes().lw(12)
                pp = plot(curvs, c='white', lc='yellow5')
                show(shape, pp, N=2, bg='bb', sharecam=False).close()
        """
        v = np.gradient(self.points())[0]
        a = np.gradient(v)[0]
        av = np.cross(a,v)
        mav = np.linalg.norm(av, axis=1)
        mv = utils.mag2(v)
        val = mav * np.sign(av[:,2])/ np.power(mv, 1.5)
        val[0]  = val[1]
        val[-1] = val[-2]
        return val

    def addCurvatureScalars(self):
        """Add a pointdata array named 'Curvatures' which contains the curvature value at each point."""
        # overrides mesh.addCurvatureScalars
        curvs = self.curvature()
        vmin, vmax = np.min(curvs), np.max(curvs)
        if vmin<0 and vmax>0:
            v = max(-vmin, vmax)
            self.cmap('coolwarm', curvs, vmin=-v, vmax=v, name="Curvature")
        else:
            self.cmap('coolwarm', curvs, vmin=vmin, vmax=vmax, name="Curvature")
        return self

    def sweep(self, direction=(1,0,0), res=1):
        """
        Sweep the `Line` along the specified vector direction.

        Returns a `Mesh` surface.
        Line position is updated to allow for additional sweepings.

        Example:
            .. code-block:: python

                from vedo import Line, show
                aline = Line([(0,0,0),(1,3,0),(2,4,0)])
                surf1 = aline.sweep((1,0.2,0), res=3)
                surf2 = aline.sweep((0.2,0,1))
                aline.color('r').lineWidth(4)
                show(surf1, surf2, aline, axes=1).close()
        """
        line = self.polydata()
        rows = line.GetNumberOfPoints()

        spacing = 1 / res
        surface = vtk.vtkPolyData()

        res += 1
        numberOfPoints = rows * res
        numberOfPolys = (rows - 1) * (res - 1)
        points = vtk.vtkPoints()
        points.Allocate(numberOfPoints)

        cnt = 0
        x = [0.,0.,0.]
        for row in range(rows):
            for col in range(res):
                p = [0.,0.,0.]
                line.GetPoint(row, p)
                x[0] = p[0] + direction[0] * col * spacing
                x[1] = p[1] + direction[1] * col * spacing
                x[2] = p[2] + direction[2] * col * spacing
                points.InsertPoint(cnt, x)
                cnt += 1

        # Generate the quads
        polys = vtk.vtkCellArray()
        polys.Allocate(numberOfPolys*4)
        pts = [0,0,0,0]
        for row in range(rows-1):
            for col in range(res-1):
                pts[0] = col + row * res
                pts[1] = pts[0] + 1
                pts[2] = pts[0] + res + 1
                pts[3] = pts[0] + res
                polys.InsertNextCell(4, pts)
        surface.SetPoints(points)
        surface.SetPolys(polys)
        asurface = vedo.Mesh(surface)
        prop = vtk.vtkProperty()
        prop.DeepCopy(self.GetProperty())
        asurface.SetProperty(prop)
        asurface.property = prop
        asurface.lighting('default')
        self.points(self.points()+direction)
        return asurface

    def reverse(self):
        """Reverse the points sequence order."""
        pts = np.flip(self.points(), axis=0)
        self.points(pts)
        return self


class DashedLine(Mesh):
    """
    Consider using `Line.pattern()` instead.

    Build a dashed line segment between points `p0` and `p1`.
    If `p0` is a list of points returns the line connecting them.
    A 2D set of coords can also be passed as `p0=[x..], p1=[y..]`.

    Parameters
    ----------
    closed : bool
        join last to first point

    spacing : float
        relative size of the dash

    c : color
        color name, number, or list of [R,G,B] colors

    alpha : float
        opacity in range [0,1]

    lw : int
        line width in pixels
    """
    def __init__(self, p0, p1=None, spacing=0.1, closed=False, c="k5", alpha=1, lw=2):

        if isinstance(p1, vtk.vtkActor):
            p1 = p1.GetPosition()
            if isinstance(p0, vtk.vtkActor):
                p0 = p0.GetPosition()
        if isinstance(p0, Points):
            p0 = p0.points()

        # detect if user is passing a 2D list of points as p0=xlist, p1=ylist:
        if len(p0) > 3:
            if not utils.isSequence(p0[0]) and not utils.isSequence(p1[0]) and len(p0)==len(p1):
                # assume input is 2D xlist, ylist
                p0 = np.stack((p0, p1), axis=1)
                p1 = None
            if len(p0[0]) == 2: # make it 3d
                p0 = np.c_[np.array(p0, dtype=float), np.zeros(len(p0), dtype=float)]
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
        dlen = np.linalg.norm(xmx-xmn)*np.clip(spacing, 0.01,1.0)/10
        if not dlen:
            Mesh.__init__(self, vtk.vtkPolyData(), c, alpha)
            self.name = "DashedLine (void)"
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

        Mesh.__init__(self, polylns.GetOutput(), c, alpha)
        self.lw(lw).lighting('off')
        self.base = listp[0]
        if closed:
            self.top = listp[-2]
        else:
            self.top = listp[-1]
        self.name = "DashedLine"

class RoundedLine(Mesh):
    """
    Create a 2D line of specified thickness (in absolute units) passing through
    a list of input points. Borders of the line are rounded.

    Parameters
    ----------
    pts : list
        a list of points in 2D or 3D (z will be ignored).

    lw : float
        thickness of the line.

    res : int
        resolution of the rounded regions. The default is 10.

    Example:
        .. code-block:: python

            from vedo import *
            pts = [(-4,-3),(1,1),(2,4),(4,1),(3,-1),(2,-5),(9,-3)]
            ln = Line(pts, c='r', lw=2).z(0.01)
            rl = RoundedLine(pts, 0.6)
            show(Points(pts), ln, rl, axes=1)
    """
    def __init__(self, pts, lw, res=10, c='gray4', alpha=1):
        pts = np.asarray(pts, dtype=float)
        if len(pts[0]) == 2: # make it 3d
            pts = np.c_[pts, np.zeros(len(pts))]

        def _getpts(pts, revd=False):

            if revd:
                pts = list(reversed(pts))

            if len(pts)==2:
                p0, p1 = pts
                v = p1-p0
                dv = np.linalg.norm(v)
                nv = np.cross(v, (0,0,-1))
                nv = nv/np.linalg.norm(nv)*lw
                return [p0+nv, p1+nv]

            ptsnew = []
            for k in range(len(pts)-2):
                p0 = pts[k]
                p1 = pts[k+1]
                p2 = pts[k+2]
                v = p1-p0
                u = p2-p1
                du = np.linalg.norm(u)
                dv = np.linalg.norm(v)
                nv = np.cross(v, (0,0,-1))
                nv = nv/np.linalg.norm(nv)*lw
                nu = np.cross(u, (0,0,-1))
                nu = nu/np.linalg.norm(nu)*lw
                uv = np.cross(u,v)
                if k==0:
                    ptsnew.append(p0+nv)
                if uv[2]<=0:
                    alpha = np.arccos(np.dot(u,v)/du/dv)
                    db = lw*np.tan(alpha/2)
                    p1new = p1+nv -v/dv * db
                    ptsnew.append(p1new)
                else:
                    p1a = p1+nv
                    p1b = p1+nu
                    for i in range(0,res+1):
                        pab = p1a*(res-i)/res + p1b*i/res
                        vpab = pab-p1
                        vpab = vpab/np.linalg.norm(vpab)*lw
                        ptsnew.append(p1+vpab)
                if k == len(pts)-3:
                    ptsnew.append(p2+nu)
                    if revd:
                        ptsnew.append(p2-nu)
            return ptsnew

        ptsnew = _getpts(pts) + _getpts(pts, revd=True)

        ppoints = vtk.vtkPoints()  # Generate the polyline
        ppoints.SetData(utils.numpy2vtk(np.asarray(ptsnew, dtype=float), dtype=float))
        lines = vtk.vtkCellArray()
        npt = len(ptsnew)
        lines.InsertNextCell(npt)
        for i in range(npt):
            lines.InsertCellPoint(i)
        poly = vtk.vtkPolyData()
        poly.SetPoints(ppoints)
        poly.SetLines(lines)
        vct = vtk.vtkContourTriangulator()
        vct.SetInputData(poly)
        vct.Update()
        Mesh.__init__(self, vct.GetOutput(), c, alpha)
        self.flat()
        self.property.LightingOff()
        self.name = "RoundedLine"
        self.base = ptsnew[0]
        self.top = ptsnew[-1]


class Lines(Line):
    """
    Build the line segments between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form `[[point1, point2], ...]`.

    Parameters
    ----------
    scale : float
        apply a rescaling factor to the lengths.

    c : color, int, str, list
        color name, number, or list of [R,G,B] colors

    alpha : float
        opacity in range [0,1]

    lw : int
        line width in pixel units

    res : int
        resolution, number of points along the line
        (only relevant if only 2 points are specified)

    .. hint:: examples/advanced/fitspheres2.py
        .. image:: https://user-images.githubusercontent.com/32848391/52503049-ac9cb600-2be4-11e9-86af-72a538af14ef.png
    """
    def __init__(
            self,
            startPoints,
            endPoints=None,
            dotted=False,
            res=1,
            scale=1,
            lw=1,
            c='k4',
            alpha=1,
        ):
        if isinstance(startPoints, Points):
            startPoints = startPoints.points()
        if isinstance(endPoints, Points):
            endPoints = endPoints.points()

        if endPoints is not None:
            startPoints = np.stack((startPoints, endPoints), axis=1)

        polylns = vtk.vtkAppendPolyData()
        for twopts in startPoints:
            lineSource = vtk.vtkLineSource()
            lineSource.SetResolution(res)
            if len(twopts[0])==2:
                lineSource.SetPoint1(twopts[0][0], twopts[0][1], 0.0)
            else:
                lineSource.SetPoint1(twopts[0])

            if scale == 1:
                pt2 = twopts[1]
            else:
                vers = (np.array(twopts[1]) - twopts[0]) * scale
                pt2 = np.array(twopts[0]) + vers

            if len(pt2)==2:
                lineSource.SetPoint2(pt2[0], pt2[1], 0.0)
            else:
                lineSource.SetPoint2(pt2)
            polylns.AddInputConnection(lineSource.GetOutputPort())
        polylns.Update()

        Mesh.__init__(self, polylns.GetOutput(), c, alpha)
        self.lw(lw).lighting('off')
        if dotted:
            self.GetProperty().SetLineStipplePattern(0xF0F0)
            self.GetProperty().SetLineStippleRepeatFactor(1)

        self.name = "Lines"


class Spline(Line):
    """
    Find the B-Spline curve through a set of points. This curve does not necessarly
    pass exactly through all the input points. Needs to import `scipy`.

    Parameters
    ----------
    smooth : float
        smoothing factor.
        - 0 = interpolate points exactly [default].
        - 1 = average point positions.

    degree : int
        degree of the spline (1<degree<5)

    easing : str
        control sensity of points along the spline.
        Available options are
        `[InSine, OutSine, Sine, InQuad, OutQuad, InCubic, OutCubic, InQuart, OutQuart, InCirc, OutCirc].`
        Can be used to create animations (move objects at varying speed).
        See e.g.: https://easings.net

    res : int
        number of points on the spline

    See also: ``CSpline`` and ``KSpline``.

    .. hint:: examples/simulations/spline_ease.py
        .. image:: https://vedo.embl.es/images/simulations/spline_ease.gif
    """
    def __init__(
            self,
            points,
            smooth=0,
            degree=2,
            closed=False,
            s=2,
            res=None,
            easing="",
        ):
        from scipy.interpolate import splprep, splev

        if isinstance(points, Points):
            points = points.points()

        if len(points[0]) == 2: # make it 3d
            points = np.c_[np.array(points, dtype=float), np.zeros(len(points), dtype=float)]

        per = 0
        if closed:
            points = np.append(points, [points[0]], axis=0)
            per = 1

        if res is None:
            res = len(points)*10

        points = np.array(points, dtype=float)

        minx, miny, minz = np.min(points, axis=0)
        maxx, maxy, maxz = np.max(points, axis=0)
        maxb = max(maxx - minx, maxy - miny, maxz - minz)
        smooth *= maxb / 2  # must be in absolute units

        x = np.linspace(0, 1, res)
        if easing:
            if easing=="InSine":
                x = 1 - np.cos((x * np.pi) / 2)
            elif easing=="OutSine":
                x = np.sin((x * np.pi) / 2)
            elif easing=="Sine":
                x = -(np.cos(np.pi * x) - 1) / 2
            elif easing=="InQuad":
                x = x*x
            elif easing=="OutQuad":
                x = 1 - (1 - x) * (1 - x)
            elif easing=="InCubic":
                x = x*x
            elif easing=="OutCubic":
                x = 1 - np.power(1 - x, 3)
            elif easing=="InQuart":
                x = x * x * x * x
            elif easing=="OutQuart":
                x = 1 - np.power(1 - x, 4)
            elif easing=="InCirc":
                x = 1 - np.sqrt(1 - np.power(x, 2))
            elif easing=="OutCirc":
                x = np.sqrt(1 - np.power(x - 1, 2))
            else:
                vedo.logger.error(f"unkown ease mode {easing}")

        # find the knots
        tckp, _ = splprep(points.T, task=0, s=smooth, k=degree, per=per)
        # evaluate spLine, including interpolated points:
        xnew, ynew, znew = splev(x, tckp)

        Line.__init__(self, np.c_[xnew, ynew, znew], lw=2)
        self.lighting('off')
        self.name = "Spline"


class KSpline(Line):
    """
    Return a [Kochanek spline](https://en.wikipedia.org/wiki/Kochanek%E2%80%93Bartels_spline)
    which runs exactly through all the input points.

    Parameters
    ----------
    continuity : float
        changes the sharpness in change between tangents

    tension : float
        changes the length of the tangent vector

    bias : float
        changes the direction of the tangent vector

    closed : bool
        join last to first point to produce a closed curve

    res : int
        approximate resolution of the output line.
        Default is 20 times the number of input points.

    .. image:: https://user-images.githubusercontent.com/32848391/65975805-73fd6580-e46f-11e9-8957-75eddb28fa72.png

    See also: ``Spline`` and ``CSpline``.
    """
    def __init__(
            self,
            points,
            continuity=0,
            tension=0,
            bias=0,
            closed=False,
            res=None,
        ):
        if isinstance(points, Points):
            points = points.points()

        if not res:
            res = len(points)*20

        if len(points[0]) == 2: # make it 3d
            points = np.c_[np.array(points, dtype=float),
                           np.zeros(len(points), dtype=float)]

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
            z = 0
            if len(p)>2:
                z = zspline.Evaluate(pos)
            ln.append((x,y,z))

        Line.__init__(self, ln, lw=2)
        self.clean()
        self.lighting('off')
        self.name = "KSpline"
        self.base = np.array(points[0], dtype=float)
        self.top = np.array(points[-1], dtype=float)

class CSpline(Line):
    """
    Return a Cardinal spline which runs exactly through all the input points.

    Parameters
    ----------
    closed : bool
        join last to first point to produce a closed curve

    res : int
        approximateresolution of the output line.
        Default is 20 times the number of input points.

    See also: ``Spline`` and ``KSpline``.
    """
    def __init__(self, points, closed=False, res=None):

        if isinstance(points, Points):
            points = points.points()

        if not res: res = len(points)*20

        if len(points[0]) == 2: # make it 3d
            points = np.c_[np.array(points, dtype=float), np.zeros(len(points), dtype=float)]

        xspline = vtk.vtkCardinalSpline()
        yspline = vtk.vtkCardinalSpline()
        zspline = vtk.vtkCardinalSpline()
        for s in [xspline, yspline, zspline]:
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
            z = 0
            if len(p)>2:
                z = zspline.Evaluate(pos)
            ln.append((x,y,z))

        Line.__init__(self, ln, lw=2)
        self.clean()
        self.lighting('off')
        self.name = "CSpline"
        self.base = np.array(points[0], dtype=float)
        self.top = np.array(points[-1], dtype=float)


class Bezier(Line):
    """
    Generate the Bezier line that links the first to the last point.

    Example:
        .. code-block:: python

            from vedo import *
            import numpy as np
            pts = np.random.randn(25,3)
            for i,p in enumerate(pts):
                p += [5*i, 15*sin(i/2), i*i*i/200]
            show(Points(pts), Bezier(pts), axes=1)

        .. image:: https://user-images.githubusercontent.com/32848391/90437534-dafd2a80-e0d2-11ea-9b93-9ecb3f48a3ff.png
    """
    def __init__(self, points, res=None):
        N = len(points)
        if res is None:
            res = 10 * N
        t = np.linspace(0, 1, num=res)
        bcurve = np.zeros((res, len(points[0])))

        def binom(n, k):
            b = 1
            for t in range(1, min(k, n-k)+1):
                b *= n/t
                n -= 1
            return b

        def bernstein(n, k):
            coeff = binom(n, k)
            def _bpoly(x):
                return coeff * x**k * (1-x)**(n-k)
            return _bpoly

        for ii in range(N):
            b = bernstein(N-1, ii)(t)
            bcurve += np.outer(b, points[ii])
        Line.__init__(self, bcurve, lw=2)
        self.name = "BezierLine"

class NormalLines(Lines):
    """
    Build an ``Glyph`` made of the normals at cells shown as lines.

    if `on="points"` normals are shown at mesh vertices.
    """
    def __init__(self, mesh, ratio=1, on='cells', scale=1):
        poly = mesh.clone().computeNormals().polydata()

        if 'cell' in on:
            centers = vtk.vtkCellCenters()
            centers.SetInputData(poly)
            centers.Update()
            poly = centers.GetOutput()

        maskPts = vtk.vtkMaskPoints()
        maskPts.SetInputData(poly)
        maskPts.SetOnRatio(ratio)
        maskPts.RandomModeOff()
        maskPts.Update()

        ln = vtk.vtkLineSource()
        ln.SetPoint1(0, 0, 0)
        ln.SetPoint2(1, 0, 0)
        ln.Update()
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceData(ln.GetOutput())
        glyph.SetInputData(maskPts.GetOutput())
        glyph.SetVectorModeToUseNormal()

        b = poly.GetBounds()
        sc = max([b[1] - b[0], b[3] - b[2], b[5] - b[4]]) / 50 *scale
        glyph.SetScaleFactor(sc)
        glyph.OrientOn()
        glyph.Update()
        Mesh.__init__(self, glyph.GetOutput())
        self.PickableOff()
        self.SetProperty(mesh.GetProperty())
        self.property = mesh.GetProperty()
        self.property.LightingOff()
        self.name = "NormalLines"


class Tube(Mesh):
    """
    Build a tube along the line defined by a set of points.

    Parameters
    ----------
    r :  float, list
        constant radius or list of radii.

    c : color
        constant color or list of colors for each point.

    res : int
        resolution, number of the sides of the tube

    .. hint:: examples/basic/ribbon.py, tube.py
        .. image:: https://vedo.embl.es/images/basic/tube.png
    """
    def __init__(self, points, r=1, cap=True, c=None, alpha=1, res=12):

        base = np.asarray(points[0], dtype=float)
        top = np.asarray(points[-1], dtype=float)

        vpoints = vtk.vtkPoints()
        idx = len(points)
        for p in points:
            vpoints.InsertNextPoint(p)
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(idx)
        for i in range(idx):
            line.GetPointIds().SetId(i, i)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)
        polyln = vtk.vtkPolyData()
        polyln.SetPoints(vpoints)
        polyln.SetLines(lines)

        tuf = vtk.vtkTubeFilter()
        tuf.SetCapping(cap)
        tuf.SetNumberOfSides(res)
        tuf.SetInputData(polyln)
        if utils.isSequence(r):
            arr = utils.numpy2vtk(r, dtype=float)
            arr.SetName("TubeRadius")
            polyln.GetPointData().AddArray(arr)
            polyln.GetPointData().SetActiveScalars("TubeRadius")
            tuf.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        else:
            tuf.SetRadius(r)

        usingColScals = False
        if utils.isSequence(c):
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

        self.base = base
        self.top = top
        self.name = "Tube"


class Ribbon(Mesh):
    """
    Connect two lines to generate the surface inbetween.
    Set the mode by which to create the ruled surface.

    It also works with a single line in input. In this case the ribbon
    is formed by following the local plane of the line in space.

    Parameters
    ----------
    mode : int
        If mode=0, resample evenly the input lines (based on length) and generates triangle strips.

        If mode=1, use the existing points and walks around the polyline using existing points.

    closed : bool
        if True, join the last point with the first to form a closed surface

    res : list
        ribbon resolutions along the line and perpendicularly to it.

    .. hint:: examples/basic/ribbon.py
        .. image:: https://vedo.embl.es/images/basic/ribbon.png
    """
    def __init__(self, line1, line2=None, mode=0, closed=False, width=None,
                 c="indigo3", alpha=1, res=(200,5)):

        if isinstance(line1, Points):
            line1 = line1.points()

        if isinstance(line2, Points):
            line2 = line2.points()

        elif line2 is None:
            ribbonFilter = vtk.vtkRibbonFilter()
            aline = Line(line1)
            ribbonFilter.SetInputData(aline.polydata())
            if width is None:
                width = aline.diagonalSize()/20.
            ribbonFilter.SetWidth(width)
            ribbonFilter.Update()
            Mesh.__init__(self, ribbonFilter.GetOutput(), c, alpha)
            self.name = "Ribbon"
            #######################
            return ################
            #######################

        if closed:
            line1 = line1.tolist()
            line1 += [line1[0]]
            line2 = line2.tolist()
            line2 += [line2[0]]

        if len(line1[0]) == 2:
            line1 = np.c_[np.asarray(line1), np.zeros(len(line1))]
        if len(line2[0]) == 2:
            line2 = np.c_[np.asarray(line2), np.zeros(len(line2))]

        ppoints1 = vtk.vtkPoints()  # Generate the polyline1
        ppoints1.SetData(utils.numpy2vtk(line1, dtype=float))
        lines1 = vtk.vtkCellArray()
        lines1.InsertNextCell(len(line1))
        for i in range(len(line1)):
            lines1.InsertCellPoint(i)
        poly1 = vtk.vtkPolyData()
        poly1.SetPoints(ppoints1)
        poly1.SetLines(lines1)

        ppoints2 = vtk.vtkPoints()  # Generate the polyline2
        ppoints2.SetData(utils.numpy2vtk(line2, dtype=float))
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
        self.name = "Ribbon"


class Arrow(Mesh):
    """
    Build a 3D arrow from `startPoint` to `endPoint` of section size `s`,
    expressed as the fraction of the window size.

    If c is a `float` less than 1, the arrow is rendered as a in a color scale
    from white to red.

    .. note:: If ``s=None`` the arrow is scaled proportionally to its length

    .. image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestOrientedArrow.png
    """
    def __init__(
            self,
            startPoint=(0,0,0),
            endPoint=(1,0,0),
            s=None,
            c="r4",
            alpha=1,
            res=12
        ):
        self.s = s if s is not None else 1 ## only needed by pyplot.__iadd()
        self.fill = True

        # in case user is passing meshs
        if isinstance(startPoint, vtk.vtkActor): startPoint = startPoint.GetPosition()
        if isinstance(endPoint,   vtk.vtkActor): endPoint   = endPoint.GetPosition()

        axis = np.asarray(endPoint) - np.asarray(startPoint)
        length = np.linalg.norm(axis)
        if length:
            axis = axis / length
        if len(axis)<3: # its 2d
            theta = np.pi/2
            startPoint = [startPoint[0], startPoint[1], 0.]
            endPoint = [endPoint[0], endPoint[1], 0.]
        else:
            theta = np.arccos(axis[2])
        phi = np.arctan2(axis[1], axis[0])
        self.arr = vtk.vtkArrowSource()
        self.arr.SetShaftResolution(res)
        self.arr.SetTipResolution(res)
        if s:
            sz = 0.02
            self.arr.SetTipRadius(sz)
            self.arr.SetShaftRadius(sz / 1.75)
            self.arr.SetTipLength(sz * 15)
        self.arr.Update()

        t = vtk.vtkTransform()
        t.RotateZ(np.rad2deg(phi))
        t.RotateY(np.rad2deg(theta))
        t.RotateY(-90)  # put it along Z
        if s:
            sz = 800 * s
            t.Scale(length, sz, sz)
        else:
            t.Scale(length, length, length)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(self.arr.GetOutput())
        tf.SetTransform(t)
        tf.Update()

        Mesh.__init__(self, tf.GetOutput(), c, alpha)

        self.phong().lighting('plastic')
        # self.property.LightingOff()
        self.SetPosition(startPoint)
        self.PickableOff()
        self.DragableOff()
        self.base = np.array(startPoint, dtype=float)
        self.top = np.array(endPoint, dtype=float)
        self.tipIndex = None
        self.fill = True # used by pyplot.__iadd__()
        self.name = "Arrow"

    def tipPoint(self, returnIndex=False):
        """Return the coordinates of the tip of the Arrow, or the point index."""
        if self.tipIndex is None:
            arrpts = utils.vtk2numpy(self.arr.GetOutput().GetPoints().GetData())
            self.tipIndex = np.argmax(arrpts[:,0])
        if returnIndex:
            return self.tipIndex
        else:
            return self.points()[self.tipIndex]


class Arrows(Glyph):
    """
    Build arrows between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form `[[point1, point2], ...]`.

    Color can be specified as a colormap which maps the size of the arrows.

    Parameters
    ----------
    s : float
        fix aspect-ratio of the arrow and scale its cross section

    c : color
        color or color map name

    alpha : float
        set object opacity

    res : int
        set arrow resolution

    .. hint:: examples/basic/glyphs_arrows.py
        .. image:: https://user-images.githubusercontent.com/32848391/55897850-a1a0da80-5bc1-11e9-81e0-004c8f396b43.jpg
    """
    def __init__(
            self,
            startPoints,
            endPoints=None,
            s=None,
            thickness=1,
            c=None,
            alpha=1,
            res=12,
        ):
        if isinstance(startPoints, Points): startPoints = startPoints.points()
        if isinstance(endPoints,   Points): endPoints   = endPoints.points()
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
            endPoints = np.c_[np.array(endPoints, dtype=float), np.zeros(len(endPoints), dtype=float)]

        arr = vtk.vtkArrowSource()
        arr.SetShaftResolution(res)
        arr.SetTipResolution(res)
        if s:
            sz = 0.02 * s
            arr.SetTipRadius(sz*2)
            arr.SetShaftRadius(sz*thickness)
            arr.SetTipLength(sz*10)
        arr.Update()
        out = arr.GetOutput()

        orients = endPoints - startPoints
        Glyph.__init__(self,
                       startPoints, out,
                       orientationArray=orients,
                       scaleByVectorSize=True,
                       colorByVectorSize=True,
                       c=c, alpha=alpha)
        self.flat().lighting('plastic')
        self.name = "Arrows"


class Arrow2D(Mesh):
    """
    Build a 2D arrow from `startPoint` to `endPoint`.

    Parameters
    ----------
    s : float
        a global multiplicative convenience factor controlling the arrow size

    shaftLength : float
        fractional shaft length

    shaftWidth : float
        fractional shaft width

    headLength : float
        fractional head length

    headWidth : float
        fractional head width

    fill : bool
        if False only generate the outline
    """
    def __init__(
            self,
            startPoint=(0,0,0),
            endPoint=(1,0,0),
            s=1,
            shaftLength=0.8,
            shaftWidth=0.05,
            headLength=0.225,
            headWidth=0.175,
            fill=True,
            c="r4",
            alpha=1
        ):
        self.fill = fill  ## needed by pyplot.__iadd()
        self.s = s#  # needed by pyplot.__iadd()

        if s != 1:
            shaftWidth *= s
            headWidth *= np.sqrt(s)

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
        theta = 0
        if len(axis) > 2:
            theta = np.arccos(axis[2])
        phi = np.arctan2(axis[1], axis[0])
        t = vtk.vtkTransform()
        if phi:
            t.RotateZ(np.rad2deg(phi))
        if theta:
            t.RotateY(np.rad2deg(theta))
        t.RotateY(-90)  # put it along Z
        t.Scale(length, length, length)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(poly)
        tf.SetTransform(t)
        tf.Update()

        Mesh.__init__(self, tf.GetOutput(), c, alpha)
        self.SetPosition(startPoint)
        self.lighting('off')
        self.DragableOff()
        self.PickableOff()
        self.base = np.array(startPoint, dtype=float)
        self.top = np.array(endPoint, dtype=float)
        self.name = "Arrow2D"

class Arrows2D(Glyph):
    """
    Build 2D arrows between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form `[[point1, point2], ...]`.

    Color can be specified as a colormap which maps the size of the arrows.

    Parameters
    ----------
    shaftLength : float
        fractional shaft length

    shaftWidth : float
        fractional shaft width

    headLength : float
        fractional head length

    headWidth : float
        fractional head width

    fill : bool
        if False only generate the outline

    Example:
        .. code-block:: python

            from vedo import Grid, Arrows2D
            g1 = Grid()
            g2 = Grid(s=(1.2,1.2)).rotateZ(4)
            arrs2d = Arrows2D(g1, g2, c='red5')
            arrs2d.show(axes=1)
    """
    def __init__(self,
            startPoints,
            endPoints=None,
            s=1,
            shaftLength=0.8,
            shaftWidth=0.05,
            headLength=0.225,
            headWidth=0.175,
            fill=True,
            c=None,
            alpha=1,
        ):
        if isinstance(startPoints, Points): startPoints = startPoints.points()
        if isinstance(endPoints,   Points): endPoints   = endPoints.points()
        startPoints = np.array(startPoints, dtype=float)
        if endPoints is None:
            strt = startPoints[:,0]
            endPoints = startPoints[:,1]
            startPoints = strt
        else:
            endPoints = np.array(endPoints, dtype=float)

        if headLength is None:
            headLength = 1 - shaftLength

        arr = Arrow2D((0,0,0), (1,0,0),
                      s=s,
                      shaftLength=shaftLength, shaftWidth=shaftWidth,
                      headLength=headLength, headWidth=headWidth, fill=fill)

        orients = endPoints - startPoints
        if orients.shape[1] == 2: # make it 3d
            orients = np.c_[np.array(orients, dtype=float), np.zeros(len(orients), dtype=float)]

        pts = Points(startPoints)
        Glyph.__init__(self,
                       pts,
                       arr.polydata(False),
                       orientationArray=orients,
                       scaleByVectorSize=True,
                       c=c, alpha=alpha)
        self.flat().lighting('off')
        if c is not None:
            self.color(c)
        self.name = "Arrows2D"

class FlatArrow(Ribbon):
    """Build a 2D arrow in 3D space by joining two close lines.

    .. hint:: examples/basic/flatarrow.py
        .. image:: https://vedo.embl.es/images/basic/flatarrow.png
    """
    def __init__(
            self,
            line1,
            line2,
            tipSize=1,
            tipWidth=1,
            c="r4",
            alpha=1,
        ):
        if isinstance(line1, Points): line1 = line1.points()
        if isinstance(line2, Points): line2 = line2.points()

        sm1, sm2 = np.array(line1[-1], dtype=float), np.array(line2[-1], dtype=float)

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

        Ribbon.__init__(self, line1, line2, alpha=alpha, c=c, res=(resm, 1))
        self.phong()
        self.PickableOff()
        self.DragableOff()
        self.name = "FlatArrow"


class Polygon(Mesh):
    """
    Build a polygon in the `xy` plane of `nsides` of radius `r`.

    .. image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestRegularPolygonSource.png
    """
    def __init__(self, pos=(0, 0, 0), nsides=6, r=1, c="coral", alpha=1):
        t = np.linspace(np.pi/2, 5/2*np.pi, num=nsides, endpoint=False)
        x, y = utils.pol2cart(np.ones_like(t)*r, t)
        faces = [list(range(nsides))]
        # do not use: vtkRegularPolygonSource
        Mesh.__init__(self, [np.c_[x,y], faces], c, alpha)
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.SetPosition(pos)
        self.GetProperty().LightingOff()
        self.name = "Polygon " + str(nsides)


class Circle(Polygon):
    """
    Build a Circle of radius `r`.
    """
    def __init__(self, pos=(0,0,0), r=1, res=120, c="gray5", alpha=1):
        Polygon.__init__(self, pos, nsides=res, r=r)

        self.center = [] # filled by pointcloud.pcaEllipse
        self.eigenvalues = []
        self.axis1 = []
        self.axis2 = []
        self.alpha(alpha).c(c)
        self.name = "Circle"

class GeoCircle(Polygon):
    """
    Build a Circle of radius `r` as projected on a geographic map.
    Circles near the poles will look very squashed.

    See example `vedo -r earthquake`
    """
    def __init__(self, lat, lon, r=1, c="red4", alpha=1, res=60):
        coords = []
        sinr, cosr = np.sin(r), np.cos(r)
        sinlat, coslat = np.sin(lat), np.cos(lat)
        for phi in np.linspace(0, 2*np.pi, num=res, endpoint=False):
            clat = np.arcsin(sinlat * cosr + coslat * sinr * np.cos(phi))
            clng = lon + np.arctan2(np.sin(phi) * sinr * coslat, cosr - sinlat * np.sin(clat))
            coords.append([clng/np.pi + 1, clat*2/np.pi + 1, 0])

        Polygon.__init__(self, nsides=res, c=c, alpha=alpha)
        self.points(coords)  # warp polygon points to match geo projection
        self.name = "Circle"


class Star(Mesh):
    """
    Build a 2D star shape of `n` cusps of inner radius `r1` and outer radius `r2`.

    :param bool line: only build the outer line (no internal surface meshing).

    .. hint:: examples/basic/extrude.py
    """
    def __init__(self, pos=(0,0,0), n=5, r1=0.7, r2=1.0, line=False, c="blue6", alpha=1):

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

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.SetPosition(pos)
        self.property.LightingOff()
        self.name = "Star"


class Disc(Mesh):
    """
    Build a 2D disc of inner radius `r1` and outer radius `r2`.

    Set `res` as the resolution in R and Phi (can be a list).

    .. image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestDisk.png
    """
    def __init__(self,
            pos=(0, 0, 0),
            r1=0.5,
            r2=1,
            c="gray4",
            alpha=1,
            res=(2,120),
        ):
        if utils.isSequence(res):
            res_r, res_phi = res
        else:
            res_r, res_phi = res, 12*res
        ps = vtk.vtkDiskSource()
        ps.SetInnerRadius(r1)
        ps.SetOuterRadius(r2)
        ps.SetRadialResolution(res_r)
        ps.SetCircumferentialResolution(res_phi)
        ps.Update()
        Mesh.__init__(self, ps.GetOutput(), c, alpha)
        self.flat()
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.SetPosition(pos)
        self.name = "Disc"


class Arc(Mesh):
    """
    Build a 2D circular arc between points `point1` and `point2`.
    If `normal` is specified then `center` is ignored, and
    normal vector, a starting `point1` (polar vector)
    and an angle defining the arc length need to be assigned.

    Arc spans the shortest angular sector point1 and point2,
    if `invert=True`, then the opposite happens.
    """
    def __init__(self,
            center,
            point1,
            point2=None,
            normal=None,
            angle=None,
            invert=False,
            c="gray4",
            alpha=1,
            res=48,
        ):
        if len(point1) == 2:
            point1 = (point1[0], point1[1], 0)
        if point2 is not None and len(point2) == 2:
            point2 = (point2[0], point2[1], 0)

        self.base = point1
        self.top = point2

        ar = vtk.vtkArcSource()
        if point2 is not None:
            self.top = point2
            point2 = point2 - np.asarray(point1)
            ar.UseNormalAndAngleOff()
            ar.SetPoint1([0,0,0])
            ar.SetPoint2(point2)
            ar.SetCenter(center)
        elif normal is not None and angle is not None:
            ar.UseNormalAndAngleOn()
            ar.SetAngle(angle)
            ar.SetPolarVector(point1)
            ar.SetNormal(normal)
        else:
            vedo.logger.error("incorrect input combination")
            return None
        ar.SetNegative(invert)
        ar.SetResolution(res)
        ar.Update()
        Mesh.__init__(self, ar.GetOutput(), c, alpha)
        self.SetPosition(self.base)
        self.lw(2).lighting('off')
        self.name = "Arc"


class Sphere(Mesh):
    """
    Build a sphere at position `pos` of radius `r`.

    Parameters
    ----------
    r : float
        sphere radius

    res : int, list
        resolution in phi, resolution in theta is 2*res

    quads : bool
        sphere mesh will be made of quads instead of triangles

    .. image:: https://user-images.githubusercontent.com/32848391/72433092-f0a31e00-3798-11ea-85f7-b2f5fcc31568.png
    """
    def __init__(self, pos=(0, 0, 0), r=1, c="r5", alpha=1, res=24, quads=False):

        if len(pos) == 2:
            pos = np.asarray([pos[0], pos[1], 0])

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
        self.name = "Sphere"


class Spheres(Mesh):
    """
    Build a (possibly large) set of spheres at `centers` of radius `r`.

    Either `c` or `r` can be a list of RGB colors or radii.

    .. hint:: examples/basic/manyspheres.py
        .. image:: https://vedo.embl.es/images/basic/manyspheres.jpg
    """
    def __init__(self, centers, r=1, c="r5", alpha=1, res=8):

        if isinstance(centers, Points):
            centers = centers.points()
        centers = np.asarray(centers, dtype=float)
        base = centers[0]

        cisseq = False
        if utils.isSequence(c):
            cisseq = True

        if cisseq:
            if len(centers) != len(c):
                vedo.logger.error(f"mismatch #centers {len(centers)} != {len(c)} #colors")
                raise RuntimeError()

        risseq = False
        if utils.isSequence(r):
            risseq = True

        if risseq:
            if len(centers) != len(r):
                vedo.logger.error(f"mismatch #centers {len(centers)} != {len(r)} #radii")
                raise RuntimeError()
        if cisseq and risseq:
            vedo.logger.error("Limitation: c and r cannot be both sequences.")
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
            ucols.SetName("Colors")
            for acol in c:
                cx, cy, cz = getColor(acol)
                ucols.InsertNextTuple3(cx * 255, cy * 255, cz * 255)
            pd.GetPointData().AddArray(ucols)
            pd.GetPointData().SetActiveScalars("Colors")
            glyph.ScalingOff()
        elif risseq:
            glyph.SetScaleModeToScaleByScalar()
            urads = utils.numpy2vtk(2*np.ascontiguousarray(r), dtype=float)
            urads.SetName("Radii")
            pd.GetPointData().AddArray(urads)
            pd.GetPointData().SetActiveScalars("Radii")

        vpts.SetData(utils.numpy2vtk(centers-base, dtype=float))

        glyph.SetInputData(pd)
        glyph.Update()

        Mesh.__init__(self, glyph.GetOutput(), alpha=alpha)
        self.SetPosition(base)
        self.base = base
        self.top = centers[-1]
        self.phong()
        if cisseq:
            self.mapper().ScalarVisibilityOn()
        else:
            self.mapper().ScalarVisibilityOff()
            self.GetProperty().SetColor(getColor(c))
        self.name = "Spheres"


class Earth(Mesh):
    """
    Build a textured mesh representing the Earth.

    .. hint:: examples/advanced/geodesic.py
        .. image:: https://vedo.embl.es/images/advanced/geodesic.png
    """
    def __init__(self, style=1, r=1):
        tss = vtk.vtkTexturedSphereSource()
        tss.SetRadius(r)
        tss.SetThetaResolution(72)
        tss.SetPhiResolution(36)
        Mesh.__init__(self, tss, c="w")
        atext = vtk.vtkTexture()
        pnmReader = vtk.vtkJPEGReader()
        fn = vedo.io.download(vedo.dataurl + f"textures/earth{style}.jpg", verbose=False)
        pnmReader.SetFileName(fn)
        atext.SetInputConnection(pnmReader.GetOutputPort())
        atext.InterpolateOn()
        self.SetTexture(atext)
        self.name = "Earth"


class Ellipsoid(Mesh):
    """
    Build a 3D ellipsoid centered at position `pos`.

    .. note:: `axis1` and `axis2` are only used to define sizes and one azimuth angle.

    Parameters
    ----------
    axis1 : list
        First axis

    axis2 : list
        Second axis

    axis3 : list
        Third axis
    """
    def __init__(self, pos=(0, 0, 0), axis1=(1, 0, 0), axis2=(0, 2, 0), axis3=(0, 0, 3),
                 c="cyan4", alpha=1, res=24):

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
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.SetPosition(pos)
        self.name = "Ellipsoid"

    def asphericity(self):
        """
        Return a measure of how different an ellipsoid is froma sphere.
        Values close to zero correspond to a spheric object.
        """
        a,b,c = self.va, self.vb, self.vc
        asp = ( ((a-b)/(a+b))**2
              + ((a-c)/(a+c))**2
              + ((b-c)/(b+c))**2 )/3. * 4.
        return asp

    def asphericity_error(self):
        """
        Calculate statistical error on the asphericity value.

        Errors on the main axes are stored in
        *Ellipsoid.va_error, Ellipsoid.vb_error and Ellipsoid.vc_error*.
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
    """
    Return an even or uneven 2D grid.

    Parameters
    ----------
    s : float, list
        if a float is provided it is interpreted as the total size along x and y,
        if a list of coords is provided they are interpreted as the vertices of the grid along x and y.
        In this case keyword `res` is ignored (see example below).

    sx : float, list
        deprecated, please use s=(sx, sy)

    res : list
        resolutions along x and y, e.i. the number of subdivisions

    resx : int
        deprecated, please use res=(n,m)

    lw : int
        line width

    Example:
        .. code-block:: python

            from vedo import *
            import numpy as np
            xcoords = np.arange(0, 2, 0.2)
            ycoords = np.arange(0, 1, 0.2)
            sqrtx = sqrt(xcoords)
            grid = Grid(s=(sqrtx, ycoords))
            grid.show(axes=8)

            # can also create a grid from np.mgrid:
            X, Y = np.mgrid[-12:12:1000*1j, 0:15:1000*1j]
            vgrid = Grid(s=(X[:,0], Y[0]))
            vgrid.show(axes=8)
    """
    def __init__(
            self,
            pos=(0, 0, 0),
            normal=(0, 0, 1),
            sx=1, # softly deprecated
            sy=1, # softly deprecated
            s=(),
            c="k3",
            alpha=1,
            lw=1,
            resx=10, # softly deprecated
            resy=10, # softly deprecated
            res=(),
        ):
        if len(res)==2:
            resx, resy = res
        if len(s)==2:
            sx, sy = s

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

            verts = np.array(verts)
            Mesh.__init__(self, [verts-verts[0], faces], c, alpha)
            self.SetPosition(verts[0])

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

        self.wireframe().lw(lw)
        self.GetProperty().LightingOff()
        self.name = "Grid"


class Plane(Mesh):
    """
    Draw a plane of size `s=(xsize, ysize)` oriented perpendicular to vector `normal`
    and so that it passes through point `pos`.

    Parameters
    ----------
    normal : list
        normal vector to the plane

    sx : int
        deprecated, please use s to set the size.

    .. image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestPlane.png
    """
    def __init__(self, pos=(0, 0, 0), normal=(0, 0, 1), s=(), sx=1, sy=1, c="gray6", alpha=1):

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        if len(s)==2:
            sx, sy = s

        self.normal = np.array(normal, dtype=float)
        self.center = np.array(pos, dtype=float)
        self.variance = 0

        ps = vtk.vtkPlaneSource()
        ps.SetResolution(1, 1)
        tri = vtk.vtkTriangleFilter()
        tri.SetInputConnection(ps.GetOutputPort())
        tri.Update()
        poly = tri.GetOutput()
        axis = self.normal / np.linalg.norm(normal)
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
        self.lighting('ambient')
        self.SetPosition(pos)
        self.name = "Plane"
        self.top = self.normal
        self.bottom = np.array([0.,0.,0.])

    def contains(self, points):
        """
        Check if each of the provided point lies on this plane.

        points is an array with shape ( , 3).
        """
        points = np.array(points, dtype=float)
        bounds = self.points()

        mask = np.isclose(np.dot(points - self.center, self.normal), 0)

        for i in [1, 3]:
            AB = bounds[i] - bounds[0]
            AP = points - bounds[0]
            mask_l = np.less_equal(np.dot(AP, AB), np.linalg.norm(AB))
            mask_g = np.greater_equal(np.dot(AP, AB), 0)
            mask = np.logical_and(mask, mask_l)
            mask = np.logical_and(mask, mask_g)
        return mask


class Rectangle(Mesh):
    """
    Build a rectangle in the xy plane identified by any two corner points.

    Parameters
    ----------
    p1 : list
        bottom-left position of the corner

    p2 : list
        top-right position of the corner

    radius : float, list
        smoothing radius of the corner in world units.
        A list can be passed with 4 individual values.
    """
    def __init__(self, p1=(0, 0), p2=(1, 1), radius=None, res=12, c="gray5", alpha=1):
        if len(p1) == 2:
            p1 = np.array([p1[0], p1[1], 0.])
        else:
            p1 = np.array(p1, dtype=float)
        if len(p2) == 2:
            p2 = np.array([p2[0], p2[1], 0.])
        else:
            p2 = np.array(p2, dtype=float)

        self.corner1 = p1
        self.corner2 = p2

        color = c
        smoothr = False
        risseq = False
        if utils.isSequence(radius):
            risseq = True
            smoothr= True
            if max(radius) == 0:
                smoothr = False
        elif radius:
            smoothr = True

        if not smoothr:
            radius = None
        self.radius= radius


        if smoothr:
            r = radius
            if not risseq:
                r = [r,r,r,r]
            rd, ra, rb, rc = r

            if p1[0] > p2[0]: # flip p1 - p2
                ptmp = p1
                p1 = p2
                p2 = ptmp
            if p1[1] > p2[1]: # flip p1y - p2y
                ptmp = p1[1]
                p1[1] = p2[1]
                p2[1] = ptmp

            px, py, _ = p2 - p1
            k = min(px/2, py/2)
            ra = min(abs(ra), k)
            rb = min(abs(rb), k)
            rc = min(abs(rc), k)
            rd = min(abs(rd), k)
            beta = np.linspace(0, 2*np.pi, num=res*4, endpoint=False)
            betas = np.split(beta, 4)
            rrx = np.cos(betas)
            rry = np.sin(betas)

            q1 = (rd, 0)
            # q2 = (px-ra, 0)
            q3 = (px, ra)
            # q4 = (px, py-rb)
            q5 = (px-rb, py)
            # q6 = (rc, py)
            q7 = (0, py-rc)
            # q8 = (0, rd)
            a = np.c_[rrx[3], rry[3]]*ra + [px-ra, ra]    if ra else np.array([])
            b = np.c_[rrx[0], rry[0]]*rb + [px-rb, py-rb] if rb else np.array([])
            c = np.c_[rrx[1], rry[1]]*rc + [rc, py-rc]    if rc else np.array([])
            d = np.c_[rrx[2], rry[2]]*rd + [rd, rd]       if rd else np.array([])

            pts = [q1, *a.tolist(), q3, *b.tolist(), q5, *c.tolist(), q7, *d.tolist()]
            faces = [list(range(len(pts)))]
        else:
            p1r = np.array([p2[0], p1[1], 0.])
            p2l = np.array([p1[0], p2[1], 0.])
            pts = ([0.,0.,0.], p1r-p1 , p2-p1, p2l-p1)
            faces = [(0,1,2,3)]

        Mesh.__init__(self, [pts, faces], color, alpha)
        self.SetPosition(p1)
        self.property.LightingOff()
        self.name = "Rectangle"


class Box(Mesh):
    """
    Build a box of dimensions `x=length, y=width and z=height`.
    Alternatively dimensions can be defined by setting `size` keyword with a tuple.
    If `size` is a list of 6 numbers, this will be interpreted as the bounding box:
    `[xmin,xmax, ymin,ymax, zmin,zmax]`

    .. hint:: examples/simulations/aspring.py
        .. image:: https://vedo.embl.es/images/simulations/50738955-7e891800-11d9-11e9-85cd-02bd4f3f13ea.gif
    """
    def __init__(self, pos=(0,0,0), length=1, width=2, height=3, size=(), c="g4", alpha=1):

        if len(size)==6:
            bounds = size
            length = bounds[1]-bounds[0]
            width  = bounds[3]-bounds[2]
            height = bounds[5]-bounds[4]
            xp = (bounds[1]+bounds[0])/2
            yp = (bounds[3]+bounds[2])/2
            zp = (bounds[5]+bounds[4])/2
            pos = (xp, yp, zp)
        elif len(size)==3:
            length, width, height = size

        src = vtk.vtkCubeSource()
        src.SetXLength(length)
        src.SetYLength(width)
        src.SetZLength(height)
        src.Update()
        pd = src.GetOutput()

        tc = [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
        ]
        vtc = utils.numpy2vtk(tc)
        pd.GetPointData().SetTCoords(vtc)
        Mesh.__init__(self, pd, c, alpha)
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.SetPosition(pos)
        self.name = "Box"

class Cube(Box):
    """Build a cube of size `side`."""
    def __init__(self, pos=(0, 0, 0), side=1, c="g4", alpha=1):
        Box.__init__(self, pos, side, side, side, (), c, alpha)
        self.name = "Cube"

class TessellatedBox(Mesh):
    """
    Build a cubic `Mesh` made o `n` small quads in the 3 axis directions.

    Parameters
    ----------
    pos : list
        position of the left bottom corner

    n : int, list
        number of subdivisions along each side

    spacing : float
        size of the side of the single quad in the 3 directions
    """
    def __init__(self, pos=(0, 0, 0), n=10, spacing=(1,1,1), c="k5", alpha=0.5):
        if utils.isSequence(n): # slow
            img = vtk.vtkImageData()
            img.SetDimensions(n[0]+1, n[1]+1, n[2]+1)
            img.SetSpacing(spacing)
            gf = vtk.vtkGeometryFilter()
            gf.SetInputData(img)
            gf.Update()
            poly = gf.GetOutput()
        else:  # fast
            n -= 1
            boxSource = vtk.vtkTessellatedBoxSource()
            boxSource.SetLevel(n)
            boxSource.QuadsOn()
            boxSource.SetBounds(0,n*spacing[0], 0,n*spacing[1], 0,n*spacing[2])
            boxSource.SetOutputPointsPrecision(vtk.vtkAlgorithm.SINGLE_PRECISION)
            boxSource.Update()
            poly = boxSource.GetOutput()
        Mesh.__init__(self, poly, c=c, alpha=alpha)
        self.SetPosition(pos)
        self.lw(1)
        self.base = np.array([0.5,0.5,0.0])
        self.top  = np.array([0.5,0.5,1.0])
        self.name = "TessellatedBox"


class Spring(Mesh):
    """
    Build a spring of specified nr of `coils` between `startPoint` and `endPoint`.

    Parameters
    ----------
    coils : int
        number of coils

    r : float
        radius at start point

    r2 : float
        radius at end point

    thickness : float
        thickness of the coil section
    """
    def __init__(
            self,
            startPoint=(0, 0, 0),
            endPoint=(1, 0, 0),
            coils=20,
            r=0.1,
            r2=None,
            thickness=None,
            c="gray5",
            alpha=1,
        ):
        diff = endPoint - np.array(startPoint, dtype=float)
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
        self.base = np.array(startPoint, dtype=float)
        self.top = np.array(endPoint, dtype=float)
        self.name = "Spring"


class Cylinder(Mesh):
    """
    Build a cylinder of specified height and radius `r`, centered at `pos`.

    If `pos` is a list of 2 points, e.g. `pos=[v1,v2]`, build a cylinder with base
    centered at `v1` and top at `v2`.

    .. image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestCylinder.png
    """
    def __init__(self, pos=(0,0,0), r=1, height=2, axis=(0,0,1),
                 cap=True, res=24, c="teal3", alpha=1):

        if utils.isSequence(pos[0]):  # assume user is passing pos=[base, top]
            base = np.array(pos[0], dtype=float)
            top = np.array(pos[1], dtype=float)
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
        cyl.SetCapping(cap)
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
        self.name = "Cylinder"


class Cone(Mesh):
    """Build a cone of specified radius `r` and `height`, centered at `pos`."""
    def __init__(self, pos=(0,0,0), r=1, height=3, axis=(0,0,1), res=48, c="green3", alpha=1):
        con = vtk.vtkConeSource()
        con.SetResolution(res)
        con.SetRadius(r)
        con.SetHeight(height)
        con.SetDirection(axis)
        con.Update()
        Mesh.__init__(self, con.GetOutput(), c, alpha)
        self.phong()
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.SetPosition(pos)
        v = utils.versor(axis) * height / 2
        self.base = pos - v
        self.top = pos + v
        self.name = "Cone"

class Pyramid(Cone):
    """Build a pyramid of specified base size `s` and `height`, centered at `pos`."""
    def __init__(self, pos=(0,0,0), s=1, height=1, axis=(0,0,1), c="green3", alpha=1):
        Cone.__init__(self, pos, s, height, axis, 4, c, alpha)
        self.name = "Pyramid"


class Torus(Mesh):
    """Build a torus of specified outer radius `r` internal radius `thickness`, centered at `pos`."""
    def __init__(self, pos=(0, 0, 0), r=1, thickness=0.2, res=30, c="yellow3", alpha=1):
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
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.SetPosition(pos)
        self.name = "Torus"

class Paraboloid(Mesh):
    """
    Build a paraboloid of specified height and radius `r`, centered at `pos`.

    Full volumetric expression is:
        `F(x,y,z)=a_0x^2+a_1y^2+a_2z^2+a_3xy+a_4yz+a_5xz+ a_6x+a_7y+a_8z+a_9`

    .. image:: https://user-images.githubusercontent.com/32848391/51211547-260ef480-1916-11e9-95f6-4a677e37e355.png
    """

    def __init__(self, pos=(0,0,0), r=1, height=1, res=50, c="cyan5", alpha=1):
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
        self.name = "Paraboloid"

class Hyperboloid(Mesh):
    """
    Build a hyperboloid of specified aperture `a2` and `height`, centered at `pos`.

    Full volumetric expression is:
        `F(x,y,z)=a_0x^2+a_1y^2+a_2z^2+a_3xy+a_4yz+a_5xz+ a_6x+a_7y+a_8z+a_9`
    """
    def __init__(self, pos=(0,0,0), a2=1, value=0.5, height=1, res=100, c="pink4", alpha=1):
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
        self.name = "Hyperboloid"


def Marker(symbol, pos=(0, 0, 0), c='k', alpha=1, s=0.1, filled=True):
    """
    Generate a marker shape.
    Can be used in association with ``Glyph``.
    """
    if isinstance(symbol, Mesh):
        return symbol.c(c).alpha(alpha).lighting('off')

    if isinstance(symbol, int):
        symbs = ['.','o','O', '0', 'p','*','h','D','d','v','^','>','<','s', 'x', 'a']
        symbol = symbol % len(symbs)
        symbol = symbs[symbol]

    if symbol == '.':
        mesh = Polygon(nsides=24, r=s*0.6)
    elif symbol == 'o':
        mesh = Polygon(nsides=24, r=s*0.75)
    elif symbol == 'O':
            mesh = Disc(r1=s*0.6, r2=s*0.75, res=(1,24))
    elif symbol == '0':
            m1 = Disc(r1=s*0.6, r2=s*0.75, res=(1,24))
            m2 = Circle(r=s*0.36).reverse()
            mesh = merge(m1,m2)
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
    elif symbol == 'v':
        mesh = Polygon(nsides=3, r=s).rotateZ(180)
    elif symbol == '^':
        mesh = Polygon(nsides=3, r=s)
    elif symbol == '>':
        mesh = Polygon(nsides=3, r=s).rotateZ(-90)
    elif symbol == '<':
        mesh = Polygon(nsides=3, r=s).rotateZ(90)
    elif symbol == 's':
        mesh = Mesh([[[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]],
                     [[0,1,2,3]]]).scale(s/1.8)
    elif symbol == 'x':
        mesh = Text3D('+', pos=(0,0,0), s=s*2.6, justify='center', depth=0)
        # mesh.rotateZ(45)
    elif symbol == 'a':
        mesh = Text3D('*', pos=(0,0,0), s=s*2.6, justify='center', depth=0)
    else:
        mesh = Text3D(symbol, pos=(0,0,0), s=s*2, justify='center', depth=0)
    mesh.flat().lighting('off').wireframe(not filled).c(c).alpha(alpha)
    if len(pos) == 2:
        pos = (pos[0], pos[1], 0)
    mesh.SetPosition(pos)
    mesh.name = "Marker"
    return mesh

class Brace(Mesh):
    """
    Create a brace (bracket) shape which spans from point q1 to point q2.

    Parameters
    ----------
    q1 : list
        point 1.

    q2 : list
        point 2.

    style : str
        style of the bracket, eg. `{}, [], (), <>`.

    padding1 : float
        padding space in percent form the input points.

    font : str
        font type.

    comment : str
        additional text to appear next to the brace symbol.

    justify : str
        specify the anchor point to justify text comment, e.g. "top-left".

    italic, float
        italicness of the text comment (can be a positive or negative number)

    angle : float
        rotation angle of text. Use `None` to keep it horizontal.

    padding2 : float
        padding space in percent form brace to text comment.

    s : float
        scale factor for the comment

    .. hint:: examples/pyplot/scatter3.py
        .. image:: https://vedo.embl.es/images/pyplot/scatter3.png
    """
    def __init__(
            self,
            q1,
            q2,
            style='}',
            padding1=0,
            font='Theemim',
            comment='',
            justify=None,
            angle=0,
            padding2=0.2,
            s=1,
            italic=0,
            c='k1',
            alpha=1,
        ):
        if isinstance(q1, vtk.vtkActor):
            q1 = q1.GetPosition()
        if isinstance(q2, vtk.vtkActor):
            q2 = q2.GetPosition()
        if len(q1)==2:
            q1 = [q1[0],q1[1],0.0]
        if len(q2)==2:
            q2 = [q2[0],q2[1],0.0]
        q1 = np.array(q1, dtype=float)
        q2 = np.array(q2, dtype=float)
        mq = (q1+q2)/2
        q1 = q1 - mq
        q2 = q2 - mq
        d = np.linalg.norm(q2-q1)
        q2[2] = q1[2]

        if style not in '{}[]()<>|I':
            vedo.logger.error(f"unknown style {style}." + "Use {}[]()<>|I")
            style = '}'

        flip = False
        if style in ['{','[','(','<']:
            flip = True
            i = ['{','[','(','<'].index(style)
            style = ['}',']',')','>'][i]

        br = Text3D(style, font='Theemim', justify='center-left')
        br.scale([0.4, 1, 1])

        angler = np.arctan2(q2[1], q2[0])*180/np.pi - 90
        if flip:
            angler += 180

        x0,x1, y0,y1, _,_ = br.GetBounds()
        if comment:
            just = 'center-top'
            if angle is None:
                angle= -angler + 90
                if not flip:
                    angle += 180

            if flip:
                angle += 180
                just = 'center-bottom'
            if justify is not None:
                just = justify
            cmt = Text3D(comment, font=font, justify=just, italic=italic)
            cx0,cx1, cy0,cy1, _,_ = cmt.bounds()
            cmt.rotateZ(90 + angle)
            cmt.scale(1/(cx1-cx0) * s * len(comment)/5)
            cmt.shift(x1*(1 + padding2), 0, 0)
            poly = merge(br, cmt).polydata()

        else:
            poly = br.polydata()

        tr = vtk.vtkTransform()
        tr.RotateZ(angler)
        tr.Translate(padding1*d, 0, 0)
        pscale = 1
        tr.Scale(pscale/(y1-y0)*d, pscale/(y1-y0)*d, 1)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(poly)
        tf.SetTransform(tr)
        tf.Update()
        poly = tf.GetOutput()

        Mesh.__init__(self, poly, c, alpha)
        self.SetPosition(mq)
        self.name = "Brace"
        self.base = q1
        self.top = q2

class Star3D(Mesh):
    """
    Build a 3D star shape of 5 cusps, mainly useful as a 3D marker.
    """
    def __init__(self, pos=(0,0,0), r=1.0, thickness=0.1, c="blue4", alpha=1):

        pts = ((1.34, 0., -0.37), (5.75e-3, -0.588, thickness/10), (0.377, 0.,-0.38),
               (0.0116, 0., -1.35), (-0.366, 0., -0.384), (-1.33, 0., -0.385),
               (-0.600, 0., 0.321), (-0.829, 0., 1.19), (-1.17e-3, 0., 0.761),
               (0.824, 0., 1.20), (0.602, 0., 0.328), (6.07e-3, 0.588, thickness/10))
        fcs = [[0, 1, 2], [0, 11,10], [2, 1, 3], [2, 11, 0], [3, 1, 4], [3, 11, 2],
               [4, 1, 5], [4, 11, 3], [5, 1, 6], [5, 11, 4], [6, 1, 7], [6, 11, 5],
               [7, 1, 8], [7, 11, 6], [8, 1, 9], [8, 11, 7], [9, 1,10], [9, 11, 8],
               [10,1, 0],[10,11, 9]]

        Mesh.__init__(self, [pts, fcs], c, alpha)
        self.RotateX(90)
        self.scale(r).lighting('shiny')

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.SetPosition(pos)
        self.name = "Star3D"

class Cross3D(Mesh):
    """
    Build a 3D cross shape, mainly useful as a 3D marker.
    """
    def __init__(self, pos=(0,0,0), s=1.0, thickness=0.3, c="b", alpha=1):
        c1 = Cylinder(r=thickness*s, height=2*s)
        c2 = Cylinder(r=thickness*s, height=2*s).rotateX(90)
        c3 = Cylinder(r=thickness*s, height=2*s).rotateY(90)
        poly = merge(c1,c2,c3).color(c).alpha(alpha).polydata(False)
        Mesh.__init__(self, poly, c, alpha)

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.SetPosition(pos)
        self.name = "Cross3D"


class ParametricShape(Mesh):
    """
    A set of built-in shapes mainly for illustration purposes.

    Name can be an integer or a string in this list:
        `['Boy', 'ConicSpiral', 'CrossCap', 'Dini', 'Enneper',
        'Figure8Klein', 'Klein', 'Mobius', 'RandomHills', 'Roman',
        'SuperEllipsoid', 'BohemianDome', 'Bour', 'CatalanMinimal',
        'Henneberg', 'Kuen', 'PluckerConoid', 'Pseudosphere']`.

    Example:
        .. code-block:: python

            from vedo import *
            for i in range(18):
                ps = ParametricShape(i, c=i)
                show([ps, ps.name], at=i, N=18)
            interactive()

        .. image:: https://user-images.githubusercontent.com/32848391/69181075-bb6aae80-0b0e-11ea-92f7-d0cd3b9087bf.png
    """
    def __init__(self, name, res=51, n=25, seed=1):
        shapes = ['Boy', 'ConicSpiral', 'CrossCap', 'Enneper',
                  'Figure8Klein', 'Klein', 'Dini', 'Mobius', 'RandomHills', 'Roman',
                  'SuperEllipsoid', 'BohemianDome', 'Bour', 'CatalanMinimal',
                  'Henneberg', 'Kuen', 'PluckerConoid', 'Pseudosphere']

        if isinstance(name, int):
            name = name%len(shapes)
            name = shapes[name]

        if   name == 'Boy':
            ps = vtk.vtkParametricBoy()
        elif name == 'ConicSpiral':
            ps = vtk.vtkParametricConicSpiral()
        elif name == 'CrossCap':
            ps = vtk.vtkParametricCrossCap()
        elif name == 'Dini':
            ps = vtk.vtkParametricDini()
        elif name == 'Enneper':
            ps = vtk.vtkParametricEnneper()
        elif name == 'Figure8Klein':
            ps = vtk.vtkParametricFigure8Klein()
        elif name == 'Klein':
            ps = vtk.vtkParametricKlein()
        elif name == 'Mobius':
            ps = vtk.vtkParametricMobius()
            ps.SetRadius(2.0)
            ps.SetMinimumV(-0.5)
            ps.SetMaximumV(0.5)
        elif name == 'RandomHills':
            ps = vtk.vtkParametricRandomHills()
            ps.AllowRandomGenerationOn()
            ps.SetRandomSeed(seed)
            ps.SetNumberOfHills(n)
        elif name == 'Roman':
            ps = vtk.vtkParametricRoman()
        elif name == 'SuperEllipsoid':
            ps = vtk.vtkParametricSuperEllipsoid()
            ps.SetN1(0.5)
            ps.SetN2(0.4)
        elif name == 'BohemianDome':
            ps = vtk.vtkParametricBohemianDome()
            ps.SetA(5.0)
            ps.SetB(1.0)
            ps.SetC(2.0)
        elif name == 'Bour':
            ps = vtk.vtkParametricBour()
        elif name == 'CatalanMinimal':
            ps = vtk.vtkParametricCatalanMinimal()
        elif name == 'Henneberg':
            ps = vtk.vtkParametricHenneberg()
        elif name == 'Kuen':
            ps = vtk.vtkParametricKuen()
            ps.SetDeltaV0(0.001)
        elif name == 'PluckerConoid':
            ps = vtk.vtkParametricPluckerConoid()
        elif name == 'Pseudosphere':
            ps = vtk.vtkParametricPseudosphere()
        else:
            vedo.logger.error(f"unknown ParametricShape {name}")
            return None

        pfs = vtk.vtkParametricFunctionSource()
        pfs.SetParametricFunction(ps)
        pfs.SetUResolution(res)
        pfs.SetVResolution(res)
        pfs.SetWResolution(res)
        pfs.SetScalarModeToZ()
        pfs.Update()

        Mesh.__init__(self, pfs.GetOutput())

        if name != 'Kuen': self.normalize()
        if name == 'Dini': self.scale(0.4)
        if name == 'Enneper': self.scale(0.4)
        if name == 'ConicSpiral': self.bc('tomato')
        self.name = name


@lru_cache(None)
def _load_font(font):
    # print('_load_font', font)

    if font not in settings.font_parameters.keys():
        printc("Unknown font:", font, c='r')
        printc("Avaliable 3D fonts are:", list(settings.font_parameters.keys()), c='y')
        printc("Using font Normografo instead.", c='y')
        font = "Normografo"

    if not settings.font_parameters[font]['islocal']:
        font = "https://vedo.embl.es/fonts/"+font+".npz"

    # some other fonts are downloadable from the vedo website
    if font.startswith('https'): # user passed URL link, make it a path
        try:
            font = vedo.io.download(font, verbose=False, force=False)
        except:
            vedo.logger.warning(f"font {font} not found")
            font = "Normografo"

    if font.endswith('.npz'):    # user passed font as a local path
        fontfile = font
        font = os.path.basename(font).split('.')[0]
    else:                        # user passed font by its name
        fontfile = os.path.join(vedo.fonts_path, font + '.npz')

    try:
        #printc('loading', font, fontfile)
        font_meshes = np.load(fontfile, allow_pickle=True)['font'][0]
    except:
        vedo.logger.error(f"font name {font} not found.")
        raise RuntimeError
    return font_meshes

@lru_cache(None)
def _get_font_letter(font, letter):
    # print("_get_font_letter", font, letter)
    font_meshes = _load_font(font)

    if letter in font_meshes.keys():
        pts, faces = font_meshes[letter]
        return utils.buildPolyData(pts, faces)
    return None


class Text3D(Mesh):
    """
    Generate a 3D polygonal ``Mesh`` representing a text string.

    Can render strings like `3.7 10^9` or `H_2 O` with subscripts and superscripts.
    Most Latex symbols are also supported.

    Symbols ~ ^ _ are reserved modifiers:

        use ~ to add a short space, 1/4 of the default empty space,

        use ^ and _ to start up/sub scripting, a space terminates their effect.

    Monospaced fonts are: `Calco, Glasgo, SmartCouric, VictorMono, Justino`.

    More fonts at: https://vedo.embl.es/fonts/

    Parameters
    ----------
    pos : list
        position coordinates in 3D space

    s : float
        size of the text

    depth : float
        text thickness (along z)

    italic : bool, float
        italic font type (can be a signed float too)

    justify : str
        text justification as centering of the bounding box
        (bottom-left, bottom-right, top-left, top-right, centered)

    font : str, int
        some of the available 3D-polygonized fonts are:
        Bongas, Calco, Comae, Kanopus, Glasgo, Ubuntu,
        LogoType, Normografo, Quikhand, SmartCouric, Theemim, VictorMono, VTK,
        Capsmall, Cartoons123, Vega, Justino, Spears, Meson.

        Check for more at https://vedo.embl.es/fonts/

        Or type in your terminal `vedo --run fonts`.

        Default is Normografo, which can be changed using `settings.defaultFont`.

    hspacing : float
        horizontal spacing of the font

    vspacing : float
        vertical spacing of the font for multiple lines text

    literal : bool
        if set to True will ignore modifiers like _ or ^

    .. note:: Type ``vedo -r fonts`` for a demo.

    .. hint:: examples/pyplot/markpoint.py, fonts.py, caption.py
        .. image:: https://vedo.embl.es/images/pyplot/fonts3d.png
    """
    def __init__(
            self,
            txt,
            pos=(0,0,0),
            s=1,
            font='',
            hspacing=1.15,
            vspacing=2.15,
            depth=0,
            italic=False,
            justify="bottom-left",
            literal=False,
            c=None,
            alpha=1,
        ):
        if not font:
            font = settings.defaultFont

        if len(pos)==2:
            pos = (pos[0], pos[1], 0)

        if c is None: # automatic black or white
            pli = vedo.plotter_instance
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

        txt = str(txt)

        if font == "VTK": #######################################
            vtt = vtk.vtkVectorText()
            vtt.SetText(txt)
            vtt.Update()
            tpoly = vtt.GetOutput()

        else: ###################################################

            stxt = set(txt) # check here if null or only spaces
            if not txt or (len(stxt)==1 and " " in stxt):
                Mesh.__init__(self, vtk.vtkPolyData(), c, alpha)
                self.name = "Text3D"
                #######################
                return ################
                #######################

            if italic is True:
                italic = 1

            if isinstance(font, int):
                lfonts = list(settings.font_parameters.keys())
                font = font%len(lfonts)
                font = lfonts[font]

            if font not in settings.font_parameters.keys():
                fpars = settings.font_parameters["Normografo"]
            else:
                fpars = settings.font_parameters[font]

            # ad hoc adjustments
            mono = fpars['mono']
            lspacing = fpars['lspacing']
            hspacing *=  fpars['hspacing']
            fscale = fpars['fscale']
            dotsep = fpars['dotsep']

            # replacements
            if "\\" in repr(txt):
                for r in _reps:
                    txt = txt.replace(r[0], r[1])

            if not literal:
                reps2 = [
                            ("\_", "┭"), # trick to protect ~ _ and ^ chars
                            ("\^", "┮"), #
                            ("\~", "┯"), #
                            ("**", "^"), # order matters
                            ("e+0", dotsep+"10^"), ("e-0", dotsep+"10^-"),
                            ("E+0", dotsep+"10^"), ("E-0", dotsep+"10^-"),
                            ("e+" , dotsep+"10^"), ("e-" , dotsep+"10^-"),
                            ("E+" , dotsep+"10^"), ("E-" , dotsep+"10^-"),
                ]
                for r in reps2:
                    txt = txt.replace(r[0], r[1])

            xmax, ymax, yshift, scale = 0, 0, 0, 1
            save_xmax = 0

            notfounds = set()
            polyletters = []
            ntxt = len(txt)
            for i, t in enumerate(txt):
                ##########
                if t=='┭':
                    t="_"
                elif t=='┮':
                    t="^"
                elif t=='┯':
                    t="~"
                elif t=='^' and not literal:
                    if yshift<0:
                        xmax = save_xmax
                    yshift = 0.9*fscale
                    scale = 0.5
                    continue
                elif t=='_' and not literal:
                    if yshift>0:
                        xmax = save_xmax
                    yshift = -0.3*fscale
                    scale = 0.5
                    continue
                elif (t==' ' or t=="\n") and yshift:
                    yshift = 0
                    scale = 1
                    save_xmax = xmax
                    if t==' ': continue
                elif t=='~':
                    if i<ntxt-1 and txt[i+1]=='_':
                        continue
                    xmax += hspacing*scale*fscale / 4
                    continue

                ############
                if t==" ":
                    xmax += hspacing*scale*fscale

                elif t=="\n":
                    xmax = 0
                    save_xmax = 0
                    ymax -= vspacing

                else:
                    poly = _get_font_letter(font, t)
                    if not poly:
                        notfounds.add(t)
                        xmax += hspacing*scale*fscale
                        continue

                    tr = vtk.vtkTransform()
                    tr.Translate(xmax, ymax+yshift, 0)
                    pscale = scale*fscale / 1000
                    tr.Scale(pscale, pscale, pscale)
                    if italic:
                        tr.Concatenate([1,italic*0.15,0,0,
                                        0,1,0,0,
                                        0,0,1,0,
                                        0,0,0,1])
                    tf = vtk.vtkTransformPolyDataFilter()
                    tf.SetInputData(poly)
                    tf.SetTransform(tr)
                    tf.Update()
                    poly = tf.GetOutput()
                    polyletters.append(poly)

                    bx = poly.GetBounds()
                    if mono:
                        xmax += hspacing*scale*fscale
                    else:
                        xmax += bx[1]-bx[0] + hspacing*scale*fscale*lspacing
                    if yshift==0:
                        save_xmax = xmax

            if len(polyletters) == 1:
                tpoly = polyletters[0]
            else:
                polyapp = vtk.vtkAppendPolyData()
                for polyd in polyletters:
                    polyapp.AddInputData(polyd)
                polyapp.Update()
                tpoly = polyapp.GetOutput()

            if notfounds:
                printc("These characters are not available in font name", font+": ", c='y', end='')
                printc(notfounds, c='y')
                printc('Type "vedo -r fonts" for a demo.', c='y')

        bb = tpoly.GetBounds()
        dx, dy = (bb[1] - bb[0]) / 2 * s, (bb[3] - bb[2]) / 2 * s
        shift = -np.array([(bb[1] + bb[0]), (bb[3] + bb[2]), (bb[5] + bb[4])]) * s /2
        if "bottom" in justify: shift += np.array([  0, dy, 0.])
        if "top"    in justify: shift += np.array([  0,-dy, 0.])
        if "left"   in justify: shift += np.array([ dx,  0, 0.])
        if "right"  in justify: shift += np.array([-dx,  0, 0.])

        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Scale(s, s, s)
        t.Translate(shift)
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
        self.lighting('off')
        self.SetPosition(pos)
        self.PickableOff()
        self.DragableOff()
        self.name = "Text3D"
        self.text = txt



class TextBase:
    "Do not instantiate this base class."
    def __init__(self):

        self.renderedAt = set()

        if isinstance(settings.defaultFont, int):
            lfonts = list(settings.font_parameters.keys())
            font = settings.defaultFont%len(lfonts)
            self.fontname = lfonts[font]
        else:
            self.fontname = settings.defaultFont
        self.name = "Text"

    def angle(self, a):
        """Orientation angle in degrees"""
        self.property.SetOrientation(a)
        return self

    def lineSpacing(self, ls):
        """Set the extra spacing between lines, expressed as a text height multiplication factor."""
        self.property.SetLineSpacing(ls)
        return self

    def lineOffset(self, lo):
        """Set/Get the vertical offset (measured in pixels)."""
        self.property.SetLineOffset(lo)
        return self

    def bold(self, value=True):
        self.property.SetBold(value)
        return self

    def italic(self, value=True):
        self.property.SetItalic(value)
        return self

    def shadow(self, offset=(1,-1)):
        """Text shadowing. Set to ``None`` to disable it."""
        if offset is None:
            self.property.ShadowOff()
        else:
            self.property.ShadowOn()
            self.property.SetShadowOffset(offset)
        return self

    def color(self, c):
        self.property.SetColor(getColor(c))
        return self

    def c(self, color):
        return self.color(color)

    def alpha(self, value):
        self.property.SetBackgroundOpacity(value)
        return self

    def background(self, color="k9", alpha=1):
        """Text background. Set to ``None`` to disable it."""
        bg = getColor(color)
        if color is None:
            self.property.SetBackgroundOpacity(0)
        else:
            self.property.SetBackgroundColor(bg)
            if alpha:
                self.property.SetBackgroundOpacity(alpha)
        return self

    def frame(self, color='k1', lw=2):
        if color is None:
            self.property.FrameOff()
        else:
            c = getColor(color)
            self.property.FrameOn()
            self.property.SetFrameColor(c)
            self.property.SetFrameWidth(lw)
        return self

    def font(self, font):

        if isinstance(font, int):
            lfonts = list(settings.font_parameters.keys())
            n = font%len(lfonts)
            font = lfonts[n]
            self.fontname = font

        if not font:                   # use default font
            font = self.fontname
            fpath = os.path.join(vedo.fonts_path, font +'.ttf')
        elif font.startswith('https'): # user passed URL link, make it a path
            fpath = vedo.io.download(font, verbose=False, force=False)
        elif font.endswith('.ttf'):    # user passing a local path to font file
            fpath = font
        else:                          # user passing name of preset font
            fpath = os.path.join(vedo.fonts_path, font +'.ttf')

        if   font == "Courier": self.property.SetFontFamilyToCourier()
        elif font == "Times":   self.property.SetFontFamilyToTimes()
        elif font == "Arial":   self.property.SetFontFamilyToArial()
        else:
            fpath = utils.getFontPath(font)
            self.property.SetFontFamily(vtk.VTK_FONT_FILE)
            self.property.SetFontFile(fpath)

        self.fontname = font  # io.toNumpy() uses it
        return self


class Text2D(vtk.vtkActor2D, TextBase):
    """
    Returns a 2D text object.

    All properties of the text, and the text itself, can be changed after creation
    (which is expecially useful in loops).

    Parameters
    ----------
    pos : str
        text is placed in one of the 8 positions

        bottom-left
        bottom-right
        top-left
        top-right
        bottom-middle
        middle-right
        middle-left
        top-middle

        If a pair (x,y) is passed as input the 2D text is place at that
        position in the coordinate system of the 2D screen (with the
        origin sitting at the bottom left).

    s : float
        size of text

    bg : color
        background color

    alpha : float
        background opacity

    justify : str
        text justification

    font : str
        predefined available fonts are

        - Arial
        - Bongas
        - Calco
        - Comae
        - Courier
        - Glasgo
        - Kanopus
        - LogoType
        - Normografo
        - Quikhand
        - SmartCouric
        - Theemim
        - Times
        - VictorMono
        - More fonts at: https://vedo.embl.es/fonts/

        A path to a `.otf` or `.ttf` font-file can also be supplied as input.

    .. hint:: examples/pyplot/fonts.py, caption.py, examples/basic/colorcubes.py
        .. image:: https://vedo.embl.es/images/basic/colorcubes.png
    """
    def __init__(
            self,
            txt="",
            pos="top-left",
            s=1,
            bg=None,
            font="",
            justify="",
            bold=False,
            italic=False,
            c=None,
            alpha=0.2,
        ):
        vtk.vtkActor2D.__init__(self)
        TextBase.__init__(self)

        self._mapper = vtk.vtkTextMapper()
        self.SetMapper(self._mapper)

        self.property = self._mapper.GetTextProperty()

        self.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()

        # automatic black or white
        if c is None:
            c = (0.1, 0.1, 0.1)
            if vedo.plotter_instance and vedo.plotter_instance.renderer:
                if vedo.plotter_instance.renderer.GetGradientBackground():
                    bgcol = vedo.plotter_instance.renderer.GetBackground2()
                else:
                    bgcol = vedo.plotter_instance.renderer.GetBackground()
                c = (0.9, 0.9, 0.9)
                if np.sum(bgcol) > 1.5:
                    c = (0.1, 0.1, 0.1)

        self.font(font).color(c).background(bg, alpha).bold(bold).italic(italic)
        self.pos(pos, justify).size(s).text(txt).lineSpacing(1.2).lineOffset(5)
        self.PickableOff()

    def pos(self, pos="top-left", justify=""):
        """
        Set position of the text to draw. Keyword ``pos`` can be a string
        or 2D coordinates in the range [0,1], being (0,0) the bottom left corner.
        """
        ajustify="top-left" # autojustify
        if isinstance(pos, str): # corners
            ajustify = pos
            if "top" in pos:
                if "left" in pos:
                    pos = (0.008, 0.994)
                elif "right" in pos:
                    pos = (0.994, 0.994)
                elif "mid" in pos or "cent" in pos:
                    pos = (0.5, 0.994)
            elif "bottom" in pos:
                if "left" in pos:
                    pos = (0.008, 0.008)
                elif "right" in pos:
                    pos = (0.994, 0.008)
                elif "mid" in pos or "cent" in pos:
                    pos = (0.5, 0.008)
            elif "mid" in pos or "cent" in pos:
                if "left" in pos:
                    pos = (0.008, 0.5)
                elif "right" in pos:
                    pos = (0.994, 0.5)
                else:
                    pos = (0.5, 0.5)

            else:
                vedo.logger.warning(f"cannot understand text position {pos}")
                pos = (0.008, 0.994)
                ajustify = "top-left"

        elif len(pos)!=2:
            vedo.logger.error("pos must be of length 2 or integer value or string")
            raise RuntimeError()

        if not justify:
            justify = ajustify

        self.property.SetJustificationToLeft()
        if "top" in justify:
            self.property.SetVerticalJustificationToTop()
        if "bottom" in justify:
            self.property.SetVerticalJustificationToBottom()
        if "cent" in justify or "mid" in justify:
            self.property.SetJustificationToCentered()
        if "left" in justify:
            self.property.SetJustificationToLeft()
        if "right" in justify:
            self.property.SetJustificationToRight()

        self.SetPosition(pos)
        return self

    def text(self, txt=None):
        """Set/get the input text string"""

        if txt is None:
            return self._mapper.GetInput()

        if "\\" in repr(txt):
            for r in _reps:
                txt = txt.replace(r[0], r[1])
        else:
            txt = str(txt)

        self._mapper.SetInput(txt)
        return self

    def size(self, s):
        self.property.SetFontSize(int(s * 22.5))
        return self



class CornerAnnotation(vtk.vtkCornerAnnotation, TextBase):
    # PROBABLY USELEES given that Text2D does pretty much the same ...
    """
    Annotate the window corner with 2D text.

    See ``Text2D`` description as the basic functionality is very similar.

    The added value of this class is the possibility to manage with one single
    object the all corner annotations (instead of creating 4 ``Text2D`` instances).

    .. hint:: examples/advanced/timer_callback2.py
    """
    def __init__(
            self,
            s=1,
            c=None,
            alpha=0.15,
            bg=None,
            font="",
        ):
        vtk.vtkCornerAnnotation.__init__(self)
        TextBase.__init__(self)

        self.property = self.GetTextProperty()

        self.font(font)

        # automatic black or white
        if c is None:
            if vedo.plotter_instance and vedo.plotter_instance.renderer:
                c = (0.9, 0.9, 0.9)
                if vedo.plotter_instance.renderer.GetGradientBackground():
                    bgcol = vedo.plotter_instance.renderer.GetBackground2()
                else:
                    bgcol = vedo.plotter_instance.renderer.GetBackground()
                if np.sum(bgcol) > 1.5:
                    c = (0.1, 0.1, 0.1)
            else:
                c = (0.5, 0.5, 0.5)

        self.SetNonlinearFontScaleFactor(1/2.75)
        self.PickableOff()
        self.property.SetColor(getColor(c))
        self.property.SetBold(False)
        self.property.SetItalic(False)

    def size(self, s, linear=False):
        """
        The font size is calculated as the largest possible value such that the annotations
        for the given viewport do not overlap.

        This font size can be scaled non-linearly with the viewport size, to maintain an
        acceptable readable size at larger viewport sizes, without being too big.
        f' = linearScale * pow(f,nonlinearScale)
        """
        if linear:
            self.SetLinearFontScaleFactor(s*5.5)
        else:
            self.SetNonlinearFontScaleFactor(s/2.75)
        return self

    def text(self, txt, pos=2):
        """Set text at the assigned position"""

        if isinstance(pos, str): # corners
            if "top" in pos:
                if "left" in pos: pos = 2
                elif "right" in pos: pos = 3
                elif "mid" in pos or "cent" in pos: pos = 7
            elif "bottom" in pos:
                if "left" in pos: pos = 0
                elif "right" in pos: pos = 1
                elif "mid" in pos or "cent" in pos: pos = 4
            else:
                if "left" in pos: pos = 6
                elif "right" in pos: pos = 5
                else: pos = 2

        if "\\" in repr(txt):
            for r in _reps:
                txt = txt.replace(r[0], r[1])
        else:
            txt = str(txt)

        self.SetText(pos, txt)
        return self

    def clear(self):
        self.ClearAllTexts()
        return self


class Latex(Picture):
    """
    Render Latex formulas.

    Parameters
    ----------
    formula : str
        latex text string

    pos : list
        position coordinates in space

    bg : color
        background color box

    res : int
        dpi resolution

    usetex : bool
        use latex compiler of matplotlib if available

    You can access the latex formula in *Latex.formula*.

    .. hint:: examples/pyplot/latex.py
        .. image:: https://vedo.embl.es/images/pyplot/latex.png
    """
    def __init__(
            self,
            formula,
            pos=(0, 0, 0),
            s=1,
            bg=None,
            res=150,
            usetex=False,
            c='k',
            alpha=1,
        ):
        self.formula = formula

        try:
            from tempfile import NamedTemporaryFile
            import matplotlib.pyplot as mpltib

            def build_img_plt(formula, tfile):

                mpltib.rc('text', usetex=usetex)

                formula1 = '$'+formula+'$'
                mpltib.axis('off')
                col = getColor(c)
                if bg:
                    bx = dict(boxstyle="square", ec=col, fc=getColor(bg))
                else:
                    bx = None
                mpltib.text(
                    0.5, 0.5, formula1,
                    size=res,
                    color=col,
                    alpha=alpha,
                    ha="center",
                    va="center",
                    bbox=bx,
                )
                mpltib.savefig(
                    tfile, format='png',
                    transparent=True, bbox_inches='tight', pad_inches=0)
                mpltib.close()

            if len(pos) == 2:
                pos = (pos[0], pos[1], 0)

            tmp_file = NamedTemporaryFile(delete=True)
            tmp_file.name = tmp_file.name + ".png"

            build_img_plt(formula, tmp_file.name)

            Picture.__init__(self, tmp_file.name, channels=4)
            self.alpha(alpha)
            self.SetScale(0.25/res*s, 0.25/res*s, 0.25/res*s)
            self.SetPosition(pos)
            self.name = "Latex"

        except:
            printc('Error in Latex()\n', formula, c='r')
            printc(' latex or dvipng not installed?', c='r')
            printc(' Try: usetex=False' , c='r')
            printc(' Try: sudo apt install dvipng' , c='r')



class ConvexHull(Mesh):
    """
    Create the 2D/3D convex hull of a set of input points or input Mesh.

    .. hint:: examples/advanced/convexHull.py
        .. image:: https://vedo.embl.es/images/advanced/convexHull.png
    """
    def __init__(self, pts):
        if utils.isSequence(pts):
            if len(pts[0]) == 2: # make it 3d
                pts = np.c_[np.array(pts, dtype=float), np.zeros(len(pts), dtype=float)]
            mesh = Points(pts)
        else:
            mesh = pts
        apoly = mesh.clean().polydata()

        # Create the convex hull of the pointcloud
        z0,z1 = mesh.zbounds()
        d = mesh.diagonalSize()
        if (z1-z0)/d > 0.001:
            delaunay = vtk.vtkDelaunay3D()
        else:
            delaunay = vtk.vtkDelaunay2D()

        delaunay.SetInputData(apoly)
        delaunay.Update()

        surfaceFilter = vtk.vtkDataSetSurfaceFilter()
        surfaceFilter.SetInputConnection(delaunay.GetOutputPort())
        surfaceFilter.Update()
        Mesh.__init__(self, surfaceFilter.GetOutput(), alpha=0.75)
        self.flat()
        self.name = "ConvexHull"


def VedoLogo(distance=0, c=None, bc='t', version=False, frame=True):
    """
    Create the 3D vedo logo.

    Parameters
    ----------
    distance : float
        send back logo by this distance from camera

    version : bool
        add version text to the right end of the logo

    bc : color
        text back face color
    """
    if c is None:
        c = (0,0,0)
        if vedo.plotter_instance:
            if sum(getColor(vedo.plotter_instance.backgrcol))>1.5:
                c=[0,0,0]
            else:
                c='linen'

    font = 'Comae'
    vlogo = Text3D('vэdo', font=font, s=1350, depth=0.2, c=c, hspacing=0.8)
    vlogo.scale([1,.95,1]).x(-2525).pickable(False).bc(bc)
    vlogo.GetProperty().LightingOn()

    vr, rul = None, None
    if version:
        vr = Text3D(vedo.__version__, font=font,
                  s=165, depth=0.2, c=c, hspacing=1).scale([1,.7,1])
        vr.RotateZ(90)
        vr.pos(2450,50,80).bc(bc).pickable(False)
    elif frame:
        rul = vedo.RulerAxes((-2600,2110, 0,1650, 0,0),
                             xlabel='European Molecular Biology Laboratory',
                             ylabel=vedo.__version__,
                             font=font,
                             xpadding=0.09, ypadding=0.04,
                            )
    fakept = vedo.Point((0,500, distance*1725), alpha=0, c=c, r=1).pickable(0)
    return vedo.Assembly([vlogo, vr, fakept, rul]).scale(1/1725)
