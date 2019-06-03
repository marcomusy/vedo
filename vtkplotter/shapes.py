from __future__ import division, print_function
import vtk
import numpy as np
from vtkplotter import settings
from vtk.util.numpy_support import numpy_to_vtk
import vtkplotter.utils as utils
import vtkplotter.colors as colors
from vtkplotter.actors import Actor, Assembly
import vtkplotter.docs as docs

__doc__ = (
    """
Submodule to generate basic geometric shapes.
"""
    + docs._defs
)

__all__ = [
    "Point",
    "Points",
    "Line",
    "Tube",
    "Lines",
    "Ribbon",
    "Arrow",
    "Arrows",
    "FlatArrow",
    "Polygon",
    "Rectangle",
    "Disc",
    "Sphere",
    "Spheres",
    "Earth",
    "Ellipsoid",
    "Grid",
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
    "Latex",
    "Glyph",
]


########################################################################
def Point(pos=(0, 0, 0), r=12, c="red", alpha=1):
    """Create a simple point actor."""
    if len(pos) == 2:
        pos = (pos[0], pos[1], 0)
    actor = Points([pos], r, c, alpha)
    return actor


def Points(plist, r=5, c="gray", alpha=1):
    """
    Build a point ``Actor`` for a list of points.

    :param float r: point radius.
    :param c: color name, number, or list of [R,G,B] colors of same length as plist.
    :type c: int, str, list
    :param float alpha: transparency in range [0,1].

    .. hint:: |lorenz| |lorenz.py|_
    """

    def _colorPoints(plist, cols, r, alpha):
        n = len(plist)
        if n > len(cols):
            colors.printc("~times Error: mismatch in colorPoints()", n, len(cols), c=1)
            raise RuntimeError()
        if n != len(cols):
            colors.printc("~lightning Warning: mismatch in colorPoints()", n, len(cols))
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(n)
        src.Update()
        vgf = vtk.vtkVertexGlyphFilter()
        vgf.SetInputData(src.GetOutput())
        vgf.Update()
        pd = vgf.GetOutput()
        ucols = vtk.vtkUnsignedCharArray()
        ucols.SetNumberOfComponents(3)
        ucols.SetName("pointsRGB")
        for i in range(len(plist)):
            c = np.array(colors.getColor(cols[i])) * 255
            ucols.InsertNextTuple3(c[0], c[1], c[2])
        pd.GetPoints().SetData(numpy_to_vtk(plist, deep=True))
        pd.GetPointData().SetScalars(ucols)
        actor = Actor(pd, c, alpha)
        actor.mapper.ScalarVisibilityOn()
        actor.GetProperty().SetInterpolationToFlat()
        actor.GetProperty().SetPointSize(r)
        settings.collectable_actors.append(actor)
        return actor

    n = len(plist)
    if n == 0:
        return None
    elif n == 3:  # assume plist is in the format [all_x, all_y, all_z]
        if utils.isSequence(plist[0]) and len(plist[0]) > 3:
            plist = list(zip(plist[0], plist[1], plist[2]))
    elif n == 2:  # assume plist is in the format [all_x, all_y, 0]
        if utils.isSequence(plist[0]) and len(plist[0]) > 3:
            plist = list(zip(plist[0], plist[1], [0] * len(plist[0])))

    if utils.isSequence(c) and len(c) > 3:
        actor = _colorPoints(plist, c, r, alpha)
        settings.collectable_actors.append(actor)
        return actor
    ################

    n = len(plist)  # refresh
    sourcePoints = vtk.vtkPoints()
    sourceVertices = vtk.vtkCellArray()
    is3d = len(plist[0]) > 2
    if is3d:  # its faster
        for pt in plist:
            aid = sourcePoints.InsertNextPoint(pt)
            sourceVertices.InsertNextCell(1)
            sourceVertices.InsertCellPoint(aid)
    else:
        for pt in plist:
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], 0)
            sourceVertices.InsertNextCell(1)
            sourceVertices.InsertCellPoint(aid)

    pd = vtk.vtkPolyData()
    pd.SetPoints(sourcePoints)
    pd.SetVerts(sourceVertices)
    if n == 1:  # passing just one point
        pd.GetPoints().SetPoint(0, [0, 0, 0])
    else:
        pd.GetPoints().SetData(numpy_to_vtk(plist, deep=True))
    actor = Actor(pd, c, alpha)
    actor.GetProperty().SetPointSize(r)
    if n == 1:
        actor.SetPosition(plist[0])

    settings.collectable_actors.append(actor)
    return actor


def Glyph(actor, glyphObj, orientationArray=None,
          scaleByVectorSize=False, c=None, alpha=1):
    """
    At each vertex of a mesh, another mesh - a `'glyph'` - is shown with
    various orientation options and coloring.

    Color can be specfied as a colormap which maps the size of the orientation
    vectors in `orientationArray`.

    :param orientationArray: list of vectors, ``vtkAbstractArray``
        or the name of an already existing points array.
    :type orientationArray: list, str, vtkAbstractArray
    :param bool scaleByVectorSize: glyph mesh is scaled by the size of
        the vectors.

    .. hint:: |glyphs.py|_ |glyphs_arrows.py|_

        |glyphs| |glyphs_arrows|
    """
    cmap = None
    # user passing a color map to map orientationArray sizes
    if c in list(colors._mapscales.cmap_d.keys()):
        cmap = c
        c = None

    # user is passing an array of point colors
    if utils.isSequence(c) and len(c) > 3:
        ucols = vtk.vtkUnsignedCharArray()
        ucols.SetNumberOfComponents(3)
        ucols.SetName("glyphRGB")
        for col in c:
            cl = colors.getColor(col)
            ucols.InsertNextTuple3(cl[0]*255, cl[1]*255, cl[2]*255)
        actor.polydata().GetPointData().SetScalars(ucols)
        c = None

    if isinstance(glyphObj, Actor):
        glyphObj = glyphObj.clean().polydata()

    gly = vtk.vtkGlyph3D()
    gly.SetInputData(actor.polydata())
    gly.SetSourceData(glyphObj)
    gly.SetColorModeToColorByScalar()

    if orientationArray is not None:
        gly.OrientOn()
        gly.SetScaleFactor(1)

        if scaleByVectorSize:
            gly.SetScaleModeToScaleByVector()
        else:
            gly.SetScaleModeToDataScalingOff()

        if isinstance(orientationArray, str):
            if orientationArray.lower() == "normals":
                gly.SetVectorModeToUseNormal()
            else:  # passing a name
                gly.SetInputArrayToProcess(0, 0, 0, 0, orientationArray)
                gly.SetVectorModeToUseVector()
        elif isinstance(orientationArray, vtk.vtkAbstractArray):
            actor.GetMapper().GetInput().GetPointData().AddArray(orientationArray)
            actor.GetMapper().GetInput().GetPointData().SetActiveVectors("glyph_vectors")
            gly.SetInputArrayToProcess(0, 0, 0, 0, "glyph_vectors")
            gly.SetVectorModeToUseVector()
        elif utils.isSequence(orientationArray):  # passing a list
            actor.addPointVectors(orientationArray, "glyph_vectors")
            gly.SetInputArrayToProcess(0, 0, 0, 0, "glyph_vectors")

        if cmap:
            gly.SetColorModeToColorByVector()
        else:
            gly.SetColorModeToColorByScalar()


    gly.Update()
    pd = gly.GetOutput()

    actor = Actor(pd, c, alpha)

    if cmap:
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(512)
        lut.Build()
        for i in range(512):
            r, g, b = colors.colorMap(i, cmap, 0, 512)
            lut.SetTableValue(i, r, g, b, 1)
        actor.mapper.SetLookupTable(lut)
        actor.mapper.ScalarVisibilityOn()
        actor.mapper.SetScalarModeToUsePointData()
        rng = pd.GetPointData().GetScalars().GetRange()
        actor.mapper.SetScalarRange(rng[0], rng[1])

    actor.GetProperty().SetInterpolationToFlat()
    settings.collectable_actors.append(actor)
    return actor


def Line(p0, p1=None, c="r", alpha=1, lw=1, dotted=False, res=None):
    """
    Build the line segment between points `p0` and `p1`.
    If `p0` is a list of points returns the line connecting them.
    A 2D set of coords can also be passed as p0=[x..], p1=[y..].

    :param c: color name, number, or list of [R,G,B] colors.
    :type c: int, str, list
    :param float alpha: transparency in range [0,1].
    :param lw: line width.
    :param bool dotted: draw a dotted line
    :param int res: number of intermediate points in the segment
    """
    # detect if user is passing a 2D ist of points as p0=xlist, p1=ylist:
    if len(p0) > 3:
        if not utils.isSequence(p0[0]) and not utils.isSequence(p1[0]) and len(p0)==len(p1):
            # assume input is 2D xlist, ylist
            p0 = list(zip(p0, p1))
            p1 = None

    # detect if user is passing a list of points:
    if utils.isSequence(p0[0]):
        ppoints = vtk.vtkPoints()  # Generate the polyline
        dim = len((p0[0]))
        if dim == 2:
            for i, p in enumerate(p0):
                ppoints.InsertPoint(i, p[0], p[1], 0)
        else:
            ppoints.SetData(numpy_to_vtk(p0, deep=True))
        lines = vtk.vtkCellArray()  # Create the polyline.
        lines.InsertNextCell(len(p0))
        for i in range(len(p0)):
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

    actor = Actor(poly, c, alpha)
    actor.GetProperty().SetLineWidth(lw)
    if dotted:
        actor.GetProperty().SetLineStipplePattern(0xF0F0)
        actor.GetProperty().SetLineStippleRepeatFactor(1)
    actor.base = np.array(p0)
    actor.top = np.array(p1)
    settings.collectable_actors.append(actor)
    return actor


def Lines(startPoints, endPoints=None, c=None, alpha=1, lw=1, dotted=False, scale=1):
    """
    Build the line segments between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form ``[[point1, point2], ...]``.

    :param float scale: apply a rescaling factor to the length

    |lines|

    .. hint:: |fitspheres2.py|_
    """
    if endPoints is not None:
        startPoints = list(zip(startPoints, endPoints))

    polylns = vtk.vtkAppendPolyData()
    for twopts in startPoints:
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(twopts[0])

        if scale != 1:
            vers = (np.array(twopts[1]) - twopts[0]) * scale
            pt2 = np.array(twopts[0]) + vers
        else:
            pt2 = twopts[1]

        lineSource.SetPoint2(pt2)
        polylns.AddInputConnection(lineSource.GetOutputPort())
    polylns.Update()

    actor = Actor(polylns.GetOutput(), c, alpha)
    actor.GetProperty().SetLineWidth(lw)
    if dotted:
        actor.GetProperty().SetLineStipplePattern(0xF0F0)
        actor.GetProperty().SetLineStippleRepeatFactor(1)

    settings.collectable_actors.append(actor)
    return actor


def Tube(points, r=1, c="r", alpha=1, res=12):
    """Build a tube along the line defined by a set of points.

    :param r: constant radius or list of radii.
    :type r: float, list
    :param c: constant color or list of colors for each point.
    :type c: float, list

    .. hint:: |ribbon.py|_ |tube.py|_

        |ribbon| |tube|
    """
    ppoints = vtk.vtkPoints()  # Generate the polyline
    ppoints.SetData(numpy_to_vtk(points, deep=True))
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(points))
    for i in range(len(points)):
        lines.InsertCellPoint(i)
    polyln = vtk.vtkPolyData()
    polyln.SetPoints(ppoints)
    polyln.SetLines(lines)

    tuf = vtk.vtkTubeFilter()
    tuf.CappingOn()
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
            r, g, b = colors.getColor(ic)
            cc.InsertTuple3(i, int(255 * r), int(255 * g), int(255 * b))
        polyln.GetPointData().AddArray(cc)
        c = None

    tuf.Update()
    polytu = tuf.GetOutput()

    actor = Actor(polytu, c=c, alpha=alpha, computeNormals=0)
    actor.phong()
    if usingColScals:
        actor.mapper.SetScalarModeToUsePointFieldData()
        actor.mapper.ScalarVisibilityOn()
        actor.mapper.SelectColorArray("TubeColors")
        actor.mapper.Modified()

    actor.base = np.array(points[0])
    actor.top = np.array(points[-1])
    settings.collectable_actors.append(actor)
    return actor


def Ribbon(line1, line2, c="m", alpha=1, res=(200, 5)):
    """Connect two lines to generate the surface inbetween.

    .. hint:: |ribbon| |ribbon.py|_
    """
    if isinstance(line1, Actor):
        line1 = line1.coordinates()
    if isinstance(line2, Actor):
        line2 = line2.coordinates()

    ppoints1 = vtk.vtkPoints()  # Generate the polyline1
    ppoints1.SetData(numpy_to_vtk(line1, deep=True))
    lines1 = vtk.vtkCellArray()
    lines1.InsertNextCell(len(line1))
    for i in range(len(line1)):
        lines1.InsertCellPoint(i)
    poly1 = vtk.vtkPolyData()
    poly1.SetPoints(ppoints1)
    poly1.SetLines(lines1)

    ppoints2 = vtk.vtkPoints()  # Generate the polyline2
    ppoints2.SetData(numpy_to_vtk(line2, deep=True))
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
    rsf.SetRuledModeToResample()
    rsf.SetResolution(res[0], res[1])
    rsf.SetInputData(mergedPolyData.GetOutput())
    rsf.Update()
    actor = Actor(rsf.GetOutput(), c=c, alpha=alpha)
    settings.collectable_actors.append(actor)
    return actor


def FlatArrow(line1, line2, c="m", alpha=1, tipSize=1, tipWidth=1):
    """Build a 2D arrow in 3D space by joining two close lines.

    .. hint:: |flatarrow| |flatarrow.py|_
    """
    if isinstance(line1, Actor):
        line1 = line1.coordinates()
    if isinstance(line2, Actor):
        line2 = line2.coordinates()

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

    actor = Ribbon(line1, line2, alpha=alpha, c=c, res=(resm, 1)).phong()
    settings.collectable_actors.pop()
    settings.collectable_actors.append(actor)
    return actor


def Arrow(startPoint, endPoint, s=None, c="r", alpha=1, res=12):
    """
    Build a 3D arrow from `startPoint` to `endPoint` of section size `s`,
    expressed as the fraction of the window size.

    .. note:: If ``s=None`` the arrow is scaled proportionally to its length,
              otherwise it represents the fraction of the window size.

    |OrientedArrow|
    """
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

    actor = Actor(tf.GetOutput(), c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(startPoint)
    actor.DragableOff()
    actor.base = np.array(startPoint)
    actor.top = np.array(endPoint)
    settings.collectable_actors.append(actor)
    return actor


def Arrows(startPoints, endPoints=None, s=None, scale=1, c="r", alpha=1, res=12):
    """
    Build arrows between two lists of points `startPoints` and `endPoints`.
    `startPoints` can be also passed in the form ``[[point1, point2], ...]``.

    Color can be specfied as a colormap which maps the size of the arrows.

    :param float s: fix aspect-ratio of the arrow and scale its cross section
    :param float scale: apply a rescaling factor to the length
    :param c: color or array of colors
    :param str cmap: color arrows by size using this color map
    :param float alpha: set transparency
    :param int res: set arrow resolution

    .. hint:: |glyphs_arrows| |glyphs_arrows.py|_
    """
    startPoints = np.array(startPoints)
    if endPoints is None:
        strt = startPoints[:,0]
        endPoints = startPoints[:,1]
        startPoints = strt

    arr = vtk.vtkArrowSource()
    arr.SetShaftResolution(res)
    arr.SetTipResolution(res)
    if s:
        sz = 0.02 * s
        arr.SetTipRadius(sz*2)
        arr.SetShaftRadius(sz)
        arr.SetTipLength(sz * 10)
    arr.Update()
    pts = Points(startPoints)
    orients = (endPoints - startPoints) * scale
    arrg = Glyph(pts, arr.GetOutput(),
                 orientationArray=orients, scaleByVectorSize=True,
                 c=c, alpha=alpha)
    settings.collectable_actors.append(arrg)
    return arrg


def Polygon(pos=(0, 0, 0), nsides=6, r=1, c="coral", alpha=1):
    """
    Build a 2D polygon of `nsides` of radius `r` oriented as `normal`.

    |Polygon|
    """
    ps = vtk.vtkRegularPolygonSource()
    ps.SetNumberOfSides(nsides)
    ps.SetRadius(r)
    ps.Update()
    actor = Actor(ps.GetOutput(), c, alpha)
    actor.SetPosition(pos)
    settings.collectable_actors.append(actor)
    return actor


def Rectangle(p1=(0, 0, 0), p2=(2, 1, 0), c="k", lw=1, alpha=1):
    """Build a rectangle in the xy plane identified by two corner points."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    pos = (p1 + p2) / 2
    length = abs(p2[0] - p1[0])
    height = abs(p2[1] - p1[1])
    return Plane(pos, [0, 0, -1], length, height, c, alpha)


def Disc(
    pos=(0, 0, 0),
    r1=0.5,
    r2=1,
    c="coral",
    alpha=1,
    res=12,
    resphi=None,
):
    """
    Build a 2D disc of internal radius `r1` and outer radius `r2`,
    oriented perpendicular to `normal`.

    |Disk|
    """
    ps = vtk.vtkDiskSource()
    ps.SetInnerRadius(r1)
    ps.SetOuterRadius(r2)
    ps.SetRadialResolution(res)
    if not resphi:
        resphi = 6 * res
    ps.SetCircumferentialResolution(resphi)
    ps.Update()
    actor = Actor(ps.GetOutput(), c, alpha).flat()
    actor.SetPosition(pos)
    settings.collectable_actors.append(actor)
    return actor


def Sphere(pos=(0, 0, 0), r=1, c="r", alpha=1, res=24):
    """Build a sphere at position `pos` of radius `r`.

    |Sphere|
    """
    ss = vtk.vtkSphereSource()
    ss.SetRadius(r)
    ss.SetThetaResolution(2 * res)
    ss.SetPhiResolution(res)
    ss.Update()
    pd = ss.GetOutput()
    actor = Actor(pd, c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    settings.collectable_actors.append(actor)
    return actor


def Spheres(centers, r=1, c="r", alpha=1, res=8):
    """
    Build a (possibly large) set of spheres at `centers` of radius `r`.

    Either `c` or `r` can be a list of RGB colors or radii.

    .. hint:: |manyspheres| |manyspheres.py|_
    """

    cisseq = False
    if utils.isSequence(c):
        cisseq = True

    if cisseq:
        if len(centers) > len(c):
            colors.printc("~times Mismatch in Spheres() colors", len(centers), len(c), c=1)
            raise RuntimeError()
        if len(centers) != len(c):
            colors.printc("~lightningWarning: mismatch in Spheres() colors", len(centers), len(c))

    risseq = False
    if utils.isSequence(r):
        risseq = True

    if risseq:
        if len(centers) > len(r):
            colors.printc("times Mismatch in Spheres() radius", len(centers), len(r), c=1)
            raise RuntimeError()
        if len(centers) != len(r):
            colors.printc("~lightning Warning: mismatch in Spheres() radius", len(centers), len(r))
    if cisseq and risseq:
        colors.printc("~noentry Limitation: c and r cannot be both sequences.", c=1)
        raise RuntimeError()

    src = vtk.vtkSphereSource()
    if not risseq:
        src.SetRadius(r)
    src.SetPhiResolution(res)
    src.SetThetaResolution(2 * res)
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
            vpts.SetPoint(i, p)
            cx, cy, cz = colors.getColor(c[i])
            ucols.InsertNextTuple3(cx * 255, cy * 255, cz * 255)
        pd.GetPointData().SetScalars(ucols)
        glyph.ScalingOff()
    elif risseq:
        glyph.SetScaleModeToScaleByScalar()
        urads = vtk.vtkFloatArray()
        urads.SetName("scales")
        for i, p in enumerate(centers):
            vpts.SetPoint(i, p)
            urads.InsertNextValue(r[i])
        pd.GetPointData().SetScalars(urads)
    else:
        for i, p in enumerate(centers):
            vpts.SetPoint(i, p)

    glyph.SetInputData(pd)
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(glyph.GetOutput())

    actor = Actor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetOpacity(alpha)
    if cisseq:
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()
        actor.GetProperty().SetColor(colors.getColor(c))
    settings.collectable_actors.append(actor)
    return actor


def Earth(pos=(0, 0, 0), r=1, lw=1):
    """Build a textured actor representing the Earth.

    .. hint:: |geodesic| |geodesic.py|_
    """
    import os

    tss = vtk.vtkTexturedSphereSource()
    tss.SetRadius(r)
    tss.SetThetaResolution(72)
    tss.SetPhiResolution(36)
    earthMapper = vtk.vtkPolyDataMapper()
    earthMapper.SetInputConnection(tss.GetOutputPort())
    earthActor = Actor(c="w")
    earthActor.SetMapper(earthMapper)
    atext = vtk.vtkTexture()
    pnmReader = vtk.vtkPNMReader()
    cdir = os.path.dirname(__file__)
    if cdir == "":
        cdir = "."
    fn = settings.textures_path + "earth.ppm"
    pnmReader.SetFileName(fn)
    atext.SetInputConnection(pnmReader.GetOutputPort())
    atext.InterpolateOn()
    earthActor.SetTexture(atext)
    if not lw:
        earthActor.SetPosition(pos)
        return earthActor
    es = vtk.vtkEarthSource()
    es.SetRadius(r / 0.995)
    earth2Mapper = vtk.vtkPolyDataMapper()
    earth2Mapper.SetInputConnection(es.GetOutputPort())
    earth2Actor = Actor()
    earth2Actor.SetMapper(earth2Mapper)
    earth2Mapper.ScalarVisibilityOff()
    earth2Actor.GetProperty().SetLineWidth(lw)
    ass = Assembly([earthActor, earth2Actor])
    ass.SetPosition(pos)
    settings.collectable_actors.append(ass)
    return ass


def Ellipsoid(pos=(0, 0, 0), axis1=(1, 0, 0), axis2=(0, 2, 0), axis3=(0, 0, 3),
              c="c", alpha=1, res=24):
    """
    Build a 3D ellipsoid centered at position `pos`.

    .. note:: `axis1` and `axis2` are only used to define sizes and one azimuth angle.

    |projectsphere|
    """
    elliSource = vtk.vtkSphereSource()
    elliSource.SetThetaResolution(res)
    elliSource.SetPhiResolution(res)
    elliSource.Update()
    l1 = np.linalg.norm(axis1)
    l2 = np.linalg.norm(axis2)
    l3 = np.linalg.norm(axis3)
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

    actor = Actor(pd, c=c, alpha=alpha)
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    actor.base = -np.array(axis1) / 2 + pos
    actor.top = np.array(axis1) / 2 + pos
    settings.collectable_actors.append(actor)
    return actor


def Grid(
    pos=(0, 0, 0),
    normal=(0, 0, 1),
    sx=1,
    sy=1,
    c="g",
    alpha=1,
    lw=1,
    resx=10,
    resy=10,
):
    """Return a grid plane.

    .. hint:: |brownian2D| |brownian2D.py|_
    """
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
    axis = np.array(normal) / np.linalg.norm(normal)
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(np.rad2deg(theta))
    t.RotateZ(np.rad2deg(phi))
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(poly)
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()
    actor = Actor(pd, c, alpha)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(lw)
    actor.SetPosition(pos)
    settings.collectable_actors.append(actor)
    return actor


def Plane(pos=(0, 0, 0), normal=(0, 0, 1), sx=1, sy=None, c="g",
          alpha=1, texture=None):
    """
    Draw a plane of size `sx` and `sy` oriented perpendicular to vector `normal`
    and so that it passes through point `pos`.

    |Plane|
    """
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
    pd = tf.GetOutput()
    actor = Actor(pd, c, alpha, texture=texture)
    actor.SetPosition(pos)
    settings.collectable_actors.append(actor)
    return actor


def Box(pos=(0, 0, 0), length=1, width=2, height=3, c="g", alpha=1):
    """
    Build a box of dimensions `x=length, y=width and z=height` oriented along vector `normal`.

    .. hint:: |aspring| |aspring.py|_
    """
    src = vtk.vtkCubeSource()
    src.SetXLength(length)
    src.SetYLength(width)
    src.SetZLength(height)
    src.Update()
    pd = src.GetOutput()
    actor = Actor(pd, c, alpha)
    actor.SetPosition(pos)
    settings.collectable_actors.append(actor)
    return actor


def Cube(pos=(0, 0, 0), side=1, c="g", alpha=1):
    """Build a cube of size `side` oriented along vector `normal`.

    .. hint:: |colorcubes| |colorcubes.py|_
    """
    return Box(pos, side, side, side, c, alpha)


def Spring(
    startPoint=(0, 0, 0),
    endPoint=(1, 0, 0),
    coils=20,
    r=0.1,
    r2=None,
    thickness=None,
    c="grey",
    alpha=1,
):
    """
    Build a spring of specified nr of `coils` between `startPoint` and `endPoint`.

    :param int coils: number of coils
    :param float r: radius at start point
    :param float r2: radius at end point
    :param float thickness: thickness of the coil section

    .. hint:: |aspring| |aspring.py|_
    """
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
    poly = tuf.GetOutput()
    actor = Actor(poly, c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(startPoint)
    actor.base = np.array(startPoint)
    actor.top = np.array(endPoint)
    settings.collectable_actors.append(actor)
    return actor


def Cylinder(pos=(0, 0, 0), r=1, height=1, axis=(0, 0, 1), c="teal", alpha=1, res=24):
    """
    Build a cylinder of specified height and radius `r`, centered at `pos`.

    If `pos` is a list of 2 Points, e.g. `pos=[v1,v2]`, build a cylinder with base
    centered at `v1` and top at `v2`.

    |Cylinder|
    """

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

    actor = Actor(pd, c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    actor.base = base + pos
    actor.top = top + pos
    settings.collectable_actors.append(actor)
    return actor


def Cone(pos=(0, 0, 0), r=1, height=3, axis=(0, 0, 1), c="dg", alpha=1, res=48):
    """
    Build a cone of specified radius `r` and `height`, centered at `pos`.

    |Cone|
    """
    con = vtk.vtkConeSource()
    con.SetResolution(res)
    con.SetRadius(r)
    con.SetHeight(height)
    con.SetDirection(axis)
    con.Update()
    actor = Actor(con.GetOutput(), c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    v = utils.versor(axis) * height / 2
    actor.base = pos - v
    actor.top = pos + v
    settings.collectable_actors.append(actor)
    return actor


def Pyramid(pos=(0, 0, 0), s=1, height=1, axis=(0, 0, 1), c="dg", alpha=1):
    """
    Build a pyramid of specified base size `s` and `height`, centered at `pos`.
    """
    return Cone(pos, s, height, axis, c, alpha, 4)


def Torus(pos=(0, 0, 0), r=1, thickness=0.1, axis=(0, 0, 1), c="khaki", alpha=1, res=30):
    """
    Build a torus of specified outer radius `r` internal radius `thickness`, centered at `pos`.

    .. hint:: |gas| |gas.py|_
    """
    rs = vtk.vtkParametricTorus()
    rs.SetRingRadius(r)
    rs.SetCrossSectionRadius(thickness)
    pfs = vtk.vtkParametricFunctionSource()
    pfs.SetParametricFunction(rs)
    pfs.SetUResolution(res * 3)
    pfs.SetVResolution(res)
    pfs.Update()

    nax = np.linalg.norm(axis)
    if nax:
        axis = np.array(axis) / nax
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(np.rad2deg(theta))
    t.RotateZ(np.rad2deg(phi))
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(pfs.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = Actor(pd, c, alpha)
    actor.GetProperty().SetInterpolationToPhong()
    actor.SetPosition(pos)
    settings.collectable_actors.append(actor)
    return actor


def Paraboloid(pos=(0, 0, 0), r=1, height=1, axis=(0, 0, 1), c="cyan", alpha=1, res=50):
    """
    Build a paraboloid of specified height and radius `r`, centered at `pos`.

    .. note::
        Full volumetric expression is:
            :math:`F(x,y,z)=a_0x^2+a_1y^2+a_2z^2+a_3xy+a_4yz+a_5xz+ a_6x+a_7y+a_8z+a_9`

            |paraboloid|
    """
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

    axis = np.array(axis) / np.linalg.norm(axis)
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(np.rad2deg(theta))
    t.RotateZ(np.rad2deg(phi))
    t.Scale(r, r, r)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(contours.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = Actor(pd, c, alpha).flipNormals()
    actor.GetProperty().SetInterpolationToPhong()
    actor.mapper.ScalarVisibilityOff()
    actor.SetPosition(pos)
    settings.collectable_actors.append(actor)
    return actor


def Hyperboloid(pos=(0, 0, 0), a2=1, value=0.5, height=1, axis=(0, 0, 1),
                c="magenta", alpha=1, res=100):
    """
    Build a hyperboloid of specified aperture `a2` and `height`, centered at `pos`.

    Full volumetric expression is:
        :math:`F(x,y,z)=a_0x^2+a_1y^2+a_2z^2+a_3xy+a_4yz+a_5xz+ a_6x+a_7y+a_8z+a_9`

    .. hint:: |mesh_bands| |mesh_bands.py|_
    """
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

    axis = np.array(axis) / np.linalg.norm(axis)
    theta = np.arccos(axis[2])
    phi = np.arctan2(axis[1], axis[0])
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.RotateY(np.rad2deg(theta))
    t.RotateZ(np.rad2deg(phi))
    t.Scale(1, 1, height)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(contours.GetOutput())
    tf.SetTransform(t)
    tf.Update()
    pd = tf.GetOutput()

    actor = Actor(pd, c, alpha).flipNormals()
    actor.GetProperty().SetInterpolationToPhong()
    actor.mapper.ScalarVisibilityOff()
    actor.SetPosition(pos)
    settings.collectable_actors.append(actor)
    return actor


def Text(
    txt,
    pos=3,
    s=1,
    depth=0.1,
    justify="bottom-left",
    c=None,
    alpha=1,
    bc=None,
    bg=None,
    font="courier",
    followcam=False,
):
    """
    Returns an ``Actor`` that shows a 2D/3D text.

    :param pos: position in 3D space,
                if an integer is passed [1,8],
                a 2D text is placed in one of the 4 corners:

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
    :param float depth: text thickness.
    :param str justify: text justification
        (bottom-left, bottom-right, top-left, top-right, centered).
    :param bg: background color of corner annotations. Only applies of `pos` is ``int``.
    :param str font: additional available fonts are:

            - Ageo
            - Aldora
            - CallingCode
            - Godsway
            - Gula
            - ImpactLabel
            - Komiko
            - Lamborgini
            - MidnightDrive
            - Militech
            - MonaShark
            - Montserrat
            - MyDisplaySt
            - PointedLaidSt
            - SchoolTeacher
            - SpecialElite

        Font choice does not apply for 3D text.
        A path to `otf` or `ttf` font-file can also be supplied as input.

        All fonts are free for personal use.
        Check out conditions in `vtkplotter/fonts/licenses` for commercial use
        and: https://www.1001freefonts.com

    :param followcam: if `True` the text will auto-orient itself to the active camera.
        A ``vtkCamera`` object can also be passed.
    :type followcam: bool, vtkCamera

    .. hint:: Examples, |fonts.py|_ |colorcubes.py|_ |markpoint.py|_ |annotations.py|_

        |colorcubes| |markpoint|

        |fonts|
    """
    if c is None: # automatic black or white
        if settings.plotter_instance and settings.plotter_instance.renderer:
            c = (0.9, 0.9, 0.9)
            if np.sum(settings.plotter_instance.renderer.GetBackground()) > 1.5:
                c = (0.1, 0.1, 0.1)
        else:
            c = (0.6, 0.6, 0.6)

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
        cap.SetColor(colors.getColor(c))
        if font.lower() == "courier": cap.SetFontFamilyToCourier()
        elif font.lower() == "times": cap.SetFontFamilyToTimes()
        elif font.lower() == "arial": cap.SetFontFamilyToArial()
        else:
            cap.SetFontFamily(vtk.VTK_FONT_FILE)
            cap.SetFontFile(settings.fonts_path+font+'.ttf')
        if bg:
            bgcol = colors.getColor(bg)
            cap.SetBackgroundColor(bgcol)
            cap.SetBackgroundOpacity(alpha * 0.5)
            cap.SetFrameColor(bgcol)
            cap.FrameOn()
        setattr(ca, 'renderedAt', set())
        settings.collectable_actors.append(ca)
        return ca

    elif len(pos)==2: # passing (x,y) coords
        actor2d = vtk.vtkActor2D()
        actor2d.SetPosition(pos)
        tmapper = vtk.vtkTextMapper()
        actor2d.SetMapper(tmapper)
        tp = tmapper.GetTextProperty()
        tp.BoldOff()
        tp.SetFontSize(s*20)
        tp.SetColor(colors.getColor(c))
        tp.SetJustificationToLeft()
        tp.SetVerticalJustificationToBottom()
        if font.lower() == "courier": tp.SetFontFamilyToCourier()
        elif font.lower() == "times": tp.SetFontFamilyToTimes()
        elif font.lower() == "arial": tp.SetFontFamilyToArial()
        else:
            tp.SetFontFamily(vtk.VTK_FONT_FILE)
            import os
            if font in settings.fonts:
                tp.SetFontFile(settings.fonts_path + font + '.ttf')
            elif os.path.exists(font):
                tp.SetFontFile(font)
            else:
                colors.printc("~sad Font", font, "not found in", settings.fonts_path, c="r")
                colors.printc("~pin Available fonts are:", settings.fonts, c="m")
                return None
        if bg:
            bgcol = colors.getColor(bg)
            tp.SetBackgroundColor(bgcol)
            tp.SetBackgroundOpacity(alpha * 0.5)
            tp.SetFrameColor(bgcol)
            tp.FrameOn()
        tmapper.SetInput(str(txt))
        actor2d.PickableOff()
        setattr(actor2d, 'renderedAt', set())
        settings.collectable_actors.append(actor2d)
        return actor2d

    else:
        # otherwise build the 3D text, fonts do not apply
        tt = vtk.vtkVectorText()
        tt.SetText(str(txt))
        tt.Update()
        tpoly = tt.GetOutput()

        bb = tpoly.GetBounds()
        dx, dy = (bb[1] - bb[0]) / 2 * s, (bb[3] - bb[2]) / 2 * s
        cm = np.array([(bb[1] + bb[0]) / 2, (bb[3] + bb[2]) / 2, (bb[5] + bb[4]) / 2]) * s
        shift = -cm
        if "bottom" in justify: shift += np.array([  0, dy, 0])
        if "top"    in justify: shift += np.array([  0,-dy, 0])
        if "left"   in justify: shift += np.array([ dx,  0, 0])
        if "right"  in justify: shift += np.array([-dx,  0, 0])

        t = vtk.vtkTransform()
        t.Translate(shift)
        t.Scale(s, s, s)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(tpoly)
        tf.SetTransform(t)
        tf.Update()
        tpoly = tf.GetOutput()

        if followcam:
            ttactor = vtk.vtkFollower()
            ttactor.GetProperty().SetOpacity(alpha)
            ttactor.GetProperty().SetColor(colors.getColor(c))
            if isinstance(followcam, vtk.vtkCamera):
                ttactor.SetCamera(followcam)
            else:
                ttactor.SetCamera(settings.plotter_instance.camera)
        else:
            if depth:
                extrude = vtk.vtkLinearExtrusionFilter()
                extrude.SetInputData(tpoly)
                extrude.SetExtrusionTypeToVectorExtrusion()
                extrude.SetVector(0, 0, 1)
                extrude.SetScaleFactor(depth*dy)
                extrude.Update()
                tpoly = extrude.GetOutput()
            ttactor = Actor(tpoly, c, alpha, bc=bc)

        ttactor.SetPosition(pos)
        settings.collectable_actors.append(ttactor)
        return ttactor


def Latex(
    formula,
    pos=(0, 0, 0),
    normal=(0, 0, 1),
    c='k',
    s=1,
    bg=None,
    alpha=1,
    res=30,
    usetex=False,
    fromweb=False,
):
    """
    Render Latex formulas.

    :param str formula: latex text string
    :param list pos: position coordinates in space
    :param list normal: normal to the plane of the image
    :param c: face color
    :param bg: background color box
    :param int res: dpi resolution
    :param bool usetex: use latex compiler of matplotlib
    :param fromweb: retrieve the latex image from online server (codecogs)

    You can access the latex formula from the `Actor` object with `actor.info['formula']`.

    .. hint:: |latex| |latex.py|_
    """
    vactor = None
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
                colors.printc('Latex error. Web site unavailable?', wsite, c=1)
                return None

        def build_img_plt(formula, tfile):
            import matplotlib.pyplot as plt

            plt.rc('text', usetex=usetex)

            formula1 = '$'+formula+'$'
            plt.axis('off')
            col = colors.getColor(c)
            if bg:
                bx = dict(boxstyle="square", ec=col, fc=colors.getColor(bg))
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

        from vtkplotter.actors import Image

        picr = vtk.vtkPNGReader()
        picr.SetFileName('_lateximg.png')
        picr.Update()
        vactor = Image()
        vactor.SetInputData(picr.GetOutput())
        vactor.info['formula'] = formula
        vactor.alpha(alpha)
        b = vactor.GetBounds()
        xm, ym = (b[1]+b[0])/200*s, (b[3]+b[2])/200*s
        vactor.SetOrigin(-xm, -ym, 0)
        nax = np.linalg.norm(normal)
        if nax:
            normal = np.array(normal) / nax
        theta = np.arccos(normal[2])
        phi = np.arctan2(normal[1], normal[0])
        vactor.SetScale(0.25/res*s, 0.25/res*s, 0.25/res*s)
        vactor.RotateZ(np.rad2deg(phi))
        vactor.RotateY(np.rad2deg(theta))
        vactor.SetPosition(pos)
        try:
            import os
            os.unlink('_lateximg.png')
        except:
            pass

    except:
        colors.printc('Error in Latex()\n', formula, c=1)
        colors.printc(' latex or dvipng not installed?', c=1)
        colors.printc(' Try: usetex=False' , c=1)
        colors.printc(' Try: sudo apt install dvipng' , c=1)

    return vactor
