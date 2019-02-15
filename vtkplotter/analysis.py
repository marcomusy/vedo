from __future__ import division, print_function
import vtkplotter.docs as docs
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk

import vtkplotter.utils as vu
import vtkplotter.colors as vc
import vtkplotter.vtkio as vio
import vtkplotter.shapes as vs
from vtkplotter.actors import Actor, Assembly

__doc__ = """
Defines methods useful to analyse 3D meshes.
"""+docs._defs


__all__ = [
    'spline',
    'xyplot',
    'fxy',
    'histogram',
    'histogram2D',
    'delaunay2D',
    'normals',
    'boundaries',
    'extractLargestRegion',
    'align',
    'procrustes',
    'fitLine',
    'fitPlane',
    'fitSphere',
    'pcaEllipsoid',
    'smoothMLS3D',
    'smoothMLS2D',
    'smoothMLS1D',
    'booleanOperation',
    'surfaceIntersection',
    'probeLine',
    'probePlane',
    'imageOperation',
    'recoSurface',
    'cluster',
    'removeOutliers',
    'thinPlateSpline',
    'meshQuality',
    'pointSampler',
    'geodesic',
    'convexHull',
    'actor2ImageData',
    'splitByConnectivity',
    'projectSphereFilter',
]


def spline(points, smooth=0.5, degree=2,
           s=2, c='b', alpha=1., nodes=False, legend=None, res=20):
    '''
    Return an ``Actor`` for a spline so that it does not necessarly pass exactly throught all points.

    :param float smooth: smoothing factor. 0 = interpolate points exactly. 1 = average point positions.
    :param int degree: degree of the spline (1<degree<5)
    :param bool nodes: if `True`, show also the input points.

    .. hint:: |tutorial_spline| |tutorial.py|_
    '''
    try:
        from scipy.interpolate import splprep, splev
    except ImportError:
        vc.printc('Warning: ..scipy not installed, using vtkCardinalSpline instead.', c=5)
        return _vtkspline(points, s, c, alpha, nodes, legend, res)

    Nout = len(points)*res  # Number of points on the spline
    points = np.array(points)

    minx, miny, minz = np.min(points, axis=0)
    maxx, maxy, maxz = np.max(points, axis=0)
    maxb = max(maxx-minx, maxy-miny, maxz-minz)
    smooth *= maxb/2  # must be in absolute units

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    tckp, _ = splprep([x, y, z], task=0, s=smooth, k=degree)  # find the knots
    # evaluate spLine, including interpolated points:
    xnew, ynew, znew = splev(np.linspace(0, 1, Nout), tckp)

    ppoints = vtk.vtkPoints()  # Generate the polyline for the spline
    profileData = vtk.vtkPolyData()
    ppoints.SetData(numpy_to_vtk(list(zip(xnew, ynew, znew)), deep=True))
    lines = vtk.vtkCellArray()  # Create the polyline
    lines.InsertNextCell(Nout)
    for i in range(Nout):
        lines.InsertCellPoint(i)
    profileData.SetPoints(ppoints)
    profileData.SetLines(lines)
    actline = Actor(profileData, c=c, alpha=alpha, legend=legend)
    actline.GetProperty().SetLineWidth(s)
    if nodes:
        actnodes = vs.Points(points, r=5, c=c, alpha=alpha)
        ass = Assembly([actline, actnodes], legend=legend)
        return ass
    else:
        return actline


def _vtkspline(points, s, c, alpha, nodes, legend, res):
    numberOfOutputPoints = len(points)*res  # Number of points on the spline
    numberOfInputPoints = len(points)  # One spline for each direction.
    aSplineX = vtk.vtkCardinalSpLine()  # interpolate the x values
    aSplineY = vtk.vtkCardinalSpLine()  # interpolate the y values
    aSplineZ = vtk.vtkCardinalSpLine()  # interpolate the z values

    inputPoints = vtk.vtkPoints()
    for i in range(0, numberOfInputPoints):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        aSplineX.AddPoint(i, x)
        aSplineY.AddPoint(i, y)
        aSplineZ.AddPoint(i, z)
        inputPoints.InsertPoint(i, x, y, z)

    inputData = vtk.vtkPolyData()
    inputData.SetPoints(inputPoints)
    points = vtk.vtkPoints()
    profileData = vtk.vtkPolyData()
    for i in range(numberOfOutputPoints):
        t = (numberOfInputPoints-1.)/(numberOfOutputPoints-1.)*i
        x, y, z = aSplineX.Evaluate(
            t), aSplineY.Evaluate(t), aSplineZ.Evaluate(t)
        points.InsertPoint(i, x, y, z)

    lines = vtk.vtkCellArray()  # Create the polyline.
    lines.InsertNextCell(numberOfOutputPoints)
    for i in range(numberOfOutputPoints):
        lines.InsertCellPoint(i)

    profileData.SetPoints(points)
    profileData.SetLines(lines)
    actline = Actor(profileData, c=c, alpha=alpha, legend=legend)
    actline.GetProperty().SetLineWidth(s)
    actline.GetProperty().SetInterpolationToPhong()
    return actline


def xyplot(points, title='', c='b', corner=1, lines=False):
    """
    Return a ``vtkXYPlotActor`` that is a plot of `x` versus `y`, 
    where `points` is a list of `(x,y)` points.

    :param int corner: assign position:  

        - 1, topleft,

        - 2, topright,

        - 3, bottomleft,

        - 4, bottomright.

    .. hint:: Example: |fitspheres1.py|_
    """
    c = vc.getColor(c)  # allow different codings
    array_x = vtk.vtkFloatArray()
    array_y = vtk.vtkFloatArray()
    array_x.SetNumberOfTuples(len(points))
    array_y.SetNumberOfTuples(len(points))
    for i, p in enumerate(points):
        array_x.InsertValue(i, p[0])
        array_y.InsertValue(i, p[1])
    field = vtk.vtkFieldData()
    field.AddArray(array_x)
    field.AddArray(array_y)
    data = vtk.vtkDataObject()
    data.SetFieldData(field)
    plot = vtk.vtkXYPlotActor()
    plot.AddDataObjectInput(data)
    plot.SetDataObjectXComponent(0, 0)
    plot.SetDataObjectYComponent(0, 1)
    plot.SetXValuesToValue()
    plot.SetXTitle(title)
    plot.SetYTitle('')
    plot.ExchangeAxesOff()
    plot.PlotPointsOn()
    if not lines:
        plot.PlotLinesOff()
    plot.GetProperty().SetPointSize(5)
    plot.GetProperty().SetLineWidth(2)
    plot.SetNumberOfXLabels(3)  # not working
    plot.GetProperty().SetColor(0, 0, 0)
    plot.GetProperty().SetOpacity(0.7)
    plot.SetPlotColor(0, c[0], c[1], c[2])
    tprop = plot.GetAxisLabelTextProperty()
    tprop.SetColor(0, 0, 0)
    tprop.SetOpacity(0.7)
    tprop.SetFontFamily(0)
    tprop.BoldOff()
    tprop.ItalicOff()
    tprop.ShadowOff()
    tprop.SetFontSize(3)  # not working
    plot.SetAxisTitleTextProperty(tprop)
    plot.SetAxisLabelTextProperty(tprop)
    plot.SetTitleTextProperty(tprop)
    if corner == 1:
        plot.GetPositionCoordinate().SetValue(.0, .8, 0)
    if corner == 2:
        plot.GetPositionCoordinate().SetValue(.7, .8, 0)
    if corner == 3:
        plot.GetPositionCoordinate().SetValue(.0, .0, 0)
    if corner == 4:
        plot.GetPositionCoordinate().SetValue(.7, .0, 0)
    plot.GetPosition2Coordinate().SetValue(.3, .2, 0)
    return plot


def histogram(values, bins=10, vrange=None,
              title='', c='g', corner=1, lines=True):
    '''
    Build a 2D histogram from a list of values in n bins.

    Use *vrange* to restrict the range of the histogram.

    Use *corner* to assign its position:
        - 1, topleft,
        - 2, topright,
        - 3, bottomleft,
        - 4, bottomright.

    .. hint:: Example: |fitplanes.py|_
    '''
    fs, edges = np.histogram(values, bins=bins, range=vrange)
    pts = []
    for i in range(len(fs)):
        pts.append([(edges[i]+edges[i+1])/2, fs[i]])
    return xyplot(pts, title, c, corner, lines)


def fxy(z='sin(3*x)*log(x-y)/3', x=(0, 3), y=(0, 3),
        zlimits=(None, None), showNan=True, zlevels=10, wire=False,
        c='b', bc='aqua', alpha=1, legend=True, texture='paper', res=100):
    '''
    Build a surface representing the function :math:`f(x,y)` specified as a string
    or as a reference to an external function.

    :param float x: x range of values.
    :param float y: y range of values.
    :param float zlimits: limit the z range of the independent variable.
    :param int zlevels: will draw the specified number of z-levels contour lines.
    :param bool showNan: show where the function does not exist as red points.
    :param bool wire: show surface as wireframe.

    .. hint:: |fxy| |fxy.py|_

        Function is: :math:`f(x,y)=\sin(3x) \cdot \log(x-y)/3` in range :math:`x=[0,3], y=[0,3]`.
    '''
    if isinstance(z, str):
        try:
            z = z.replace('math.', '').replace('np.', '')
            namespace = locals()
            code = "from math import*\ndef zfunc(x,y): return "+z
            exec(code, namespace)
            z = namespace['zfunc']
        except:
            vc.printc('Syntax Error in fxy()', c=1)
            return None

    ps = vtk.vtkPlaneSource()
    ps.SetResolution(res, res)
    ps.SetNormal([0, 0, 1])
    ps.Update()
    poly = ps.GetOutput()
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    todel, nans = [], []

    if zlevels:
        tf = vtk.vtkTriangleFilter()
        tf.SetInputData(poly)
        tf.Update()
        poly = tf.GetOutput()

    for i in range(poly.GetNumberOfPoints()):
        px, py, _ = poly.GetPoint(i)
        xv = (px+.5)*dx+x[0]
        yv = (py+.5)*dy+y[0]
        try:
            zv = z(xv, yv)
            poly.GetPoints().SetPoint(i, [xv, yv, zv])
        except:
            todel.append(i)
            nans.append([xv, yv, 0])

    if len(todel):
        cellIds = vtk.vtkIdList()
        poly.BuildLinks()

        for i in todel:
            poly.GetPointCells(i, cellIds)
            for j in range(cellIds.GetNumberOfIds()):
                poly.DeleteCell(cellIds.GetId(j))  # flag cell

        poly.RemoveDeletedCells()
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(poly)
        cl.Update()
        poly = cl.GetOutput()

    if not poly.GetNumberOfPoints():
        vc.printc('Function is not real in the domain', c=1)
        return None

    if zlimits[0]:
        tmpact1 = Actor(poly)
        a = tmpact1.cutWithPlane((0, 0, zlimits[0]), (0, 0, 1))
        poly = a.polydata()
    if zlimits[1]:
        tmpact2 = Actor(poly)
        a = tmpact2.cutWithPlane((0, 0, zlimits[1]), (0, 0, -1))
        poly = a.polydata()

    if c is None:
        elev = vtk.vtkElevationFilter()
        elev.SetInputData(poly)
        elev.Update()
        poly = elev.GetOutput()

    actor = Actor(poly, c=c, bc=bc, alpha=alpha, wire=wire,
                  legend=legend, texture=texture)
    acts = [actor]
    if zlevels:
        elevation = vtk.vtkElevationFilter()
        elevation.SetInputData(poly)
        bounds = poly.GetBounds()
        elevation.SetLowPoint(0, 0, bounds[4])
        elevation.SetHighPoint(0, 0, bounds[5])
        elevation.Update()
        bcf = vtk.vtkBandedPolyDataContourFilter()
        bcf.SetInputData(elevation.GetOutput())
        bcf.SetScalarModeToValue()
        bcf.GenerateContourEdgesOn()
        bcf.GenerateValues(zlevels, elevation.GetScalarRange())
        bcf.Update()
        zpoly = bcf.GetContourEdgesOutput()
        zbandsact = Actor(zpoly, c='k', alpha=alpha)
        zbandsact.GetProperty().SetLineWidth(1.5)
        acts.append(zbandsact)

    if showNan and len(todel):
        bb = actor.GetBounds()
        zm = (bb[4]+bb[5])/2
        nans = np.array(nans)+[0, 0, zm]
        nansact = vs.Points(nans, c='red', alpha=alpha/2)
        acts.append(nansact)

    if len(acts) > 1:
        asse = Assembly(acts)
        return asse
    else:
        return actor


def histogram2D(xvalues, yvalues, bins=12, norm=1, c='g', alpha=1, fill=False):
    '''
    Build a 2D hexagonal histogram from a list of x and y values.

    :param bool bins: nr of bins for the smaller range in x or y.
    :param float norm: sets a scaling factor for the z axis.
    :param bool fill: draw solid hexagons.

    .. hint:: |histo2D| |histo2D.py|_
    '''
    xmin, xmax = np.min(xvalues), np.max(xvalues)
    ymin, ymax = np.min(yvalues), np.max(yvalues)
    dx, dy = xmax-xmin, ymax-ymin

    if xmax-xmin < ymax - ymin:
        n = bins
        m = np.rint(dy/dx*n/1.2+.5).astype(int)
    else:
        m = bins
        n = np.rint(dx/dy*m*1.2+.5).astype(int)

    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(len(xvalues))
    src.Update()
    pointsPolydata = src.GetOutput()

    values = list(zip(xvalues, yvalues))
    zs = [[0.0]]*len(values)
    values = np.append(values, zs, axis=1)

    pointsPolydata.GetPoints().SetData(numpy_to_vtk(values, deep=True))
    cloud = Actor(pointsPolydata)

    c1 = vc.getColor(c)
    c2 = np.array(c1)*.7
    r = 0.47/n*1.2*dx

    hexs, binmax = [], 0
    for i in range(n+3):
        for j in range(m+2):
            cyl = vtk.vtkCylinderSource()
            cyl.SetResolution(6)
            cyl.CappingOn()
            cyl.SetRadius(0.5)
            cyl.SetHeight(0.1)
            cyl.Update()
            t = vtk.vtkTransform()
            if not i % 2:
                p = (i/1.33, j/1.12, 0)
                c = c1
            else:
                p = (i/1.33, j/1.12+0.443, 0)
                c = c2
            q = (p[0]/n*1.2*dx+xmin, p[1]/m*dy+ymin, 0)
            ids = cloud.closestPoint(q, radius=r, returnIds=True)
            ne = len(ids)
            if fill:
                t.Translate(p[0], p[1], ne/2)
                t.Scale(1, 1, ne*5)
            else:
                t.Translate(p[0], p[1], ne)
            t.RotateX(90)  # put it along Z
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(cyl.GetOutput())
            tf.SetTransform(t)
            tf.Update()
            h = Actor(tf.GetOutput(), c=c, alpha=alpha)
            h.PickableOff()
            hexs.append(h)
            if ne > binmax:
                binmax = ne

    asse = Assembly(hexs)
    # asse.PickableOff()
    asse.SetScale(1/n*1.2*dx, 1/m*dy, norm/binmax*(dx+dy)/4)
    asse.SetPosition(xmin, ymin, 0)
    return asse


def delaunay2D(plist, tol=None):
    '''
    Create a mesh from points in the XY plane.

    .. hint:: |delaunay2d| |delaunay2d.py|_
    '''
    pd = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(plist, deep=True))
    pd.SetPoints(vpts)
    delny = vtk.vtkDelaunay2D()
    delny.SetInputData(pd)
    if tol:
        delny.SetTolerance(tol)
    delny.Update()
    return Actor(delny.GetOutput())


def normals(actor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8):
    '''
    Build a ``vtkActor`` made of the normals at vertices shown as lines.

    .. hint:: Examples: |tutorial.py|_

        |fatlimb|
    '''
    maskPts = vtk.vtkMaskPoints()
    maskPts.SetOnRatio(ratio)
    maskPts.RandomModeOff()
    actor = actor.computeNormals()
    src = actor.polydata()
    maskPts.SetInputData(src)
    arrow = vtk.vtkLineSource()
    arrow.SetPoint1(0, 0, 0)
    arrow.SetPoint2(.75, 0, 0)
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetInputConnection(maskPts.GetOutputPort())
    glyph.SetVectorModeToUseNormal()
    b = src.GetBounds()
    sc = max([b[1]-b[0], b[3]-b[2], b[5]-b[4]])/20.
    glyph.SetScaleFactor(sc)
    glyph.OrientOn()
    glyph.Update()
    glyphMapper = vtk.vtkPolyDataMapper()
    glyphMapper.SetInputConnection(glyph.GetOutputPort())
    glyphMapper.SetScalarModeToUsePointFieldData()
    glyphActor = vtk.vtkActor()
    glyphActor.SetMapper(glyphMapper)
    glyphActor.GetProperty().SetColor(vc.getColor(c))
    glyphActor.GetProperty().SetOpacity(alpha)
    glyphActor.PickableOff()
    aactor = Assembly([actor, glyphActor])
    return aactor


def boundaries(actor, c='p', lw=5):
    '''Build an ``Actor`` that shows the boundary lines of an input mesh.'''

    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(actor.GetMapper().GetInput())
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOn()
    fe.ManifoldEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.ColoringOff()
    fe.Update()
    bactor = Actor(fe.GetOutput(), c=c, alpha=1)
    bactor.GetProperty().SetLineWidth(lw)
    return bactor


def extractLargestRegion(actor):
    '''Keep only the largest connected part of a mesh and discard all the smaller pieces.

    .. hint:: |largestregion.py|_
    '''
    conn = vtk.vtkConnectivityFilter()
    conn.SetExtractionModeToLargestRegion()
    conn.ScalarConnectivityOff()
    poly = actor.GetMapper().GetInput()
    conn.SetInputData(poly)
    conn.Update()
    epoly = conn.GetOutput()
    eact = Actor(epoly)
    pr = vtk.vtkProperty()
    pr.DeepCopy(actor.GetProperty())
    eact.SetProperty(pr)
    return eact


def align(source, target, iters=100, rigid=False):
    '''
    Return a copy of source actor which is aligned to
    target actor through the `Iterative Closest Point` algorithm.

    The core of the algorithm is to match each vertex in one surface with
    the closest surface point on the other, then apply the transformation
    that modify one surface to best match the other (in the least-square sense).

    .. hint:: |align1| |align1.py|_

        |align2| |align2.py|_
    '''
    if isinstance(source, Actor):
        source = source.polydata()
    if isinstance(target, Actor):
        target = target.polydata()

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.SetMaximumNumberOfIterations(iters)
    if rigid:
        icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.StartByMatchingCentroidsOn()
    icp.Update()
    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(source)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()
    poly = icpTransformFilter.GetOutput()
    actor = Actor(poly)

    # actor.info['transform'] = icp.GetLandmarkTransform() # not working!
    # do it manually...
    sourcePoints = vtk.vtkPoints()
    targetPoints = vtk.vtkPoints()
    for i in range(10):
        p1 = [0, 0, 0]
        source.GetPoints().GetPoint(i, p1)
        sourcePoints.InsertNextPoint(p1)
        p2 = [0, 0, 0]
        poly.GetPoints().GetPoint(i, p2)
        targetPoints.InsertNextPoint(p2)

    # Setup the transform
    landmarkTransform = vtk.vtkLandmarkTransform()
    landmarkTransform.SetSourceLandmarks(sourcePoints)
    landmarkTransform.SetTargetLandmarks(targetPoints)
    if rigid:
        landmarkTransform.SetModeToRigidBody()
    actor.info['transform'] = landmarkTransform

    return actor


def procrustes(sources, rigid=False):
    '''
    Return an ``Assembly`` of aligned source actors with
    the `Procrustes` algorithm. The output ``Assembly`` is normalized in size.

    `Procrustes` algorithm takes N set of points and aligns them in a least-squares sense
    to their mutual mean. The algorithm is iterated until convergence,
    as the mean must be recomputed after each alignment.

    :param bool rigid: if `True` scaling is disabled.

    .. hint:: |align3| |align3.py|_
    '''
    group = vtk.vtkMultiBlockDataGroupFilter()
    for source in sources:
        if sources[0].N() != source.N():
            vc.printc('Procrustes error in align():', c=1)
            vc.printc(' sources have different nr of points', c=1)
            exit(0)
        group.AddInputData(source.polydata())
    procrustes = vtk.vtkProcrustesAlignmentFilter()
    procrustes.StartFromCentroidOn()
    procrustes.SetInputConnection(group.GetOutputPort())
    if rigid:
        procrustes.GetLandmarkTransform().SetModeToRigidBody()
    procrustes.Update()

    acts = []
    for i, s in enumerate(sources):
        poly = procrustes.GetOutput().GetBlock(i)
        actor = Actor(poly)
        actor.SetProperty(s.GetProperty())
        acts.append(actor)
    assem = Assembly(acts)
    assem.info['transform'] = procrustes.GetLandmarkTransform()
    return assem


################# working with point clouds

def fitLine(points, c='orange', lw=1):
    '''
    Fits a line through points.

    Extra info is stored in ``actor.info['slope']``, ``actor.info['center']``, ``actor.info['variances']``.

    .. hint:: |fitline| |fitline.py|_
    '''
    data = np.array(points)
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    vv = vv[0]/np.linalg.norm(vv[0])
    # vv contains the first principal component, i.e. the direction
    # vector of the best fit line in the least squares sense.
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    a = np.linalg.norm(xyz_min - datamean)
    b = np.linalg.norm(xyz_max - datamean)
    p1 = datamean - a*vv
    p2 = datamean + b*vv
    l = vs.Line(p1, p2, c=c, lw=lw, alpha=1)
    l.info['slope'] = vv
    l.info['center'] = datamean
    l.info['variances'] = dd
    return l


def fitPlane(points, c='g', bc='darkgreen'):
    '''
    Fits a plane to a set of points.

    Extra info is stored in ``actor.info['normal']``, ``actor.info['center']``, ``actor.info['variance']``.

    .. hint:: Example: |fitplanes.py|_
    '''
    data = np.array(points)
    datamean = data.mean(axis=0)
    res = np.linalg.svd(data - datamean)
    dd, vv = res[1], res[2]
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    s = np.linalg.norm(xyz_max - xyz_min)
    n = np.cross(vv[0], vv[1])
    pla = vs.Plane(datamean, n, s, s, c, bc)
    pla.info['normal'] = n
    pla.info['center'] = datamean
    pla.info['variance'] = dd[2]
    return pla


def fitSphere(coords):
    '''
    Fits a sphere to a set of points.

    Extra info is stored in ``actor.info['radius']``, ``actor.info['center']``, ``actor.info['residue']``.

    .. hint:: Example: |fitspheres1.py|_

        |fitspheres2| |fitspheres2.py|_
    '''
    coords = np.array(coords)
    n = len(coords)
    A = np.zeros((n, 4))
    A[:, :-1] = coords*2
    A[:,  3] = 1
    f = np.zeros((n, 1))
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    f[:, 0] = x*x + y*y + z*z
    C, residue, rank, sv = np.linalg.lstsq(A, f)  # solve AC=f
    if rank < 4:
        return None
    t = (C[0]*C[0]) + (C[1]*C[1]) + (C[2]*C[2]) + C[3]
    radius = np.sqrt(t)[0]
    center = np.array([C[0][0], C[1][0], C[2][0]])
    if len(residue):
        residue = np.sqrt(residue[0])/n
    else:
        residue = 0
    s = vs.Sphere(center, radius, c='r', alpha=1).wire(1)
    s.info['radius'] = radius
    s.info['center'] = center
    s.info['residue'] = residue
    return s


def pcaEllipsoid(points, pvalue=.95, pcaAxes=False):
    '''
    Show the oriented PCA ellipsoid that contains fraction `pvalue` of points.

    :param float pvalue: ellypsoid will contain the specified fraction of points.
    :param bool pcaAxes: if `True`, show the 3 PCA semi axes.

    Extra info is stored in ``actor.info['sphericity']``, 
    ``actor.info['va']``, ``actor.info['vb']``, ``actor.info['vc']``
    (sphericity is equal to 0 for a perfect sphere).

    .. hint:: |pca| |pca.py|_

        |cell_main| |cell_main.py|_
    '''
    try:
        from scipy.stats import f
    except:
        vc.printc("Error in Ellipsoid(): scipy not installed. Skip.", c=1)
        return None
    if isinstance(points, vtk.vtkActor):
        points = points.coordinates()
    if len(points) == 0:
        return None
    P = np.array(points, ndmin=2, dtype=float)
    cov = np.cov(P, rowvar=0)      # covariance matrix
    U, s, R = np.linalg.svd(cov)   # singular value decomposition
    p, n = s.size, P.shape[0]
    fppf = f.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p)  # f % point function
    ua, ub, uc = np.sqrt(s*fppf)*2  # semi-axes (largest first)
    center = np.mean(P, axis=0)    # centroid of the hyperellipsoid
    sphericity = (  ((ua-ub)/(ua+ub))**2
                  + ((ua-uc)/(ua+uc))**2
                  + ((ub-uc)/(ub+uc))**2)/3. * 4.
    elliSource = vtk.vtkSphereSource()
    elliSource.SetThetaResolution(48)
    elliSource.SetPhiResolution(48)
    matri = vtk.vtkMatrix4x4()
    matri.DeepCopy((R[0][0] * ua, R[1][0] * ub, R[2][0] * uc, center[0],
                    R[0][1] * ua, R[1][1] * ub, R[2][1] * uc, center[1],
                    R[0][2] * ua, R[1][2] * ub, R[2][2] * uc, center[2], 0, 0, 0, 1))
    vtra = vtk.vtkTransform()
    vtra.SetMatrix(matri)
    ftra = vtk.vtkTransformFilter()
    ftra.SetTransform(vtra)
    ftra.SetInputConnection(elliSource.GetOutputPort())
    ftra.Update()
    actor_elli = Actor(ftra.GetOutput(), 'c', 0.5)
    actor_elli.GetProperty().BackfaceCullingOn()
    actor_elli.GetProperty().SetInterpolationToPhong()
    if pcaAxes:
        axs = []
        for ax in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            l = vtk.vtkLineSource()
            l.SetPoint1([0, 0, 0])
            l.SetPoint2(ax)
            l.Update()
            t = vtk.vtkTransformFilter()
            t.SetTransform(vtra)
            t.SetInputData(l.GetOutput())
            t.Update()
            axs.append(Actor(t.GetOutput(), 'c', 0.5).lineWidth(3))
        finact = Assembly([actor_elli]+axs)
    else:
        finact = actor_elli
    finact.info['sphericity'] = sphericity
    finact.info['va'] = ua
    finact.info['vb'] = ub
    finact.info['vc'] = uc
    return finact


def smoothMLS3D(actors, neighbours=10):
    '''
    A time sequence of actors is being smoothed in 4D
    using a `MLS (Moving Least Squares)` variant.
    The time associated to an actor must be specified in advance with ``actor.time()`` method.
    Data itself can suggest a meaningful time separation based on the spatial
    distribution of points.

    :param int neighbours: fixed nr. of neighbours in space-time to take into account in the fit.

    .. hint:: |moving_least_squares3D| |moving_least_squares3D.py|_
    '''
    from scipy.spatial import KDTree

    coords4d = []
    for a in actors:  # build the list of 4d coordinates
        coords3d = a.coordinates()
        n = len(coords3d)
        pttimes = [[a.time()]]*n
        coords4d += np.append(coords3d, pttimes, axis=1).tolist()

    avedt = float(actors[-1].time()-actors[0].time())/len(actors)
    print("Average time separation between actors dt =", round(avedt, 3))

    coords4d = np.array(coords4d)
    newcoords4d = []
    kd = KDTree(coords4d, leafsize=neighbours)
    suggest = ''

    pb = vio.ProgressBar(0, len(coords4d))
    for i in pb.range():
        mypt = coords4d[i]

        #dr = np.sqrt(3*dx**2+dt**2)
        #iclosest = kd.query_ball_Point(mypt, r=dr)
        #dists, iclosest = kd.query(mypt, k=None, distance_upper_bound=dr)
        dists, iclosest = kd.query(mypt, k=neighbours)
        closest = coords4d[iclosest]

        nc = len(closest)
        if nc >= neighbours and nc > 5:
            m = np.linalg.lstsq(closest, [1.]*nc)[0]  # needs python3
            vers = m/np.linalg.norm(m)
            hpcenter = np.mean(closest, axis=0)  # hyperplane center
            dist = np.dot(mypt-hpcenter, vers)
            projpt = mypt - dist*vers
            newcoords4d.append(projpt)

            if not i % 1000:  # work out some stats
                v = np.std(closest, axis=0)
                vx = round((v[0]+v[1]+v[2])/3, 3)
                suggest = 'data suggest dt='+str(vx)

        pb.print(suggest)
    newcoords4d = np.array(newcoords4d)

    ctimes = newcoords4d[:, 3]
    ccoords3d = np.delete(newcoords4d, 3, axis=1)  # get rid of time
    act = vs.Points(ccoords3d)
    act.pointColors(ctimes, cmap='jet') # use a colormap to associate a color to time
    return act


def smoothMLS2D(actor, f=0.2, decimate=1, recursive=0, showNPlanes=0):
    '''
    Smooth actor or points with a `Moving Least Squares` variant.
    The list ``actor.info['variances']`` contains the residue calculated for each point.
    Input actor's polydata is modified.

    :param f: smoothing factor - typical range is [0,2].
    :param decimate: decimation factor (an integer number).
    :param recursive: move points while algorithm proceedes.
    :param showNPlanes: build an actor showing the fitting plane for N random points.

    .. hint:: |mesh_smoothers| 
        |mesh_smoothers.py|_

        |moving_least_squares2D| |moving_least_squares2D.py|_

        |recosurface| |recosurface.py|_
    '''
    coords = actor.coordinates()
    ncoords = len(coords)
    Ncp = int(ncoords*f/100)
    nshow = int(ncoords/decimate)
    if showNPlanes:
        ndiv = int(nshow/showNPlanes*decimate)

    if Ncp < 5:
        vc.printc('Please choose a higher fraction than '+str(f), c=1)
        Ncp = 5
    print('smoothMLS: Searching #neighbours, #pt:', Ncp, ncoords)

    poly = actor.GetMapper().GetInput()
    vpts = poly.GetPoints()
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    vtklist = vtk.vtkIdList()
    variances, newsurf, acts = [], [], []
    pb = vio.ProgressBar(0, ncoords)
    for i, p in enumerate(coords):
        pb.print('smoothing...')
        if i % decimate:
            continue

        locator.FindClosestNPoints(Ncp, p, vtklist)
        points = []
        for j in range(vtklist.GetNumberOfIds()):
            trgp = [0, 0, 0]
            vpts.GetPoint(vtklist.GetId(j), trgp)
            points.append(trgp)
        if len(points) < 5:
            continue

        points = np.array(points)
        pointsmean = points.mean(axis=0)  # plane center
        uu, dd, vv = np.linalg.svd(points-pointsmean)
        a, b, c = np.cross(vv[0], vv[1])  # normal
        d, e, f = pointsmean  # plane center
        x, y, z = p
        t = (a*d - a*x + b*e - b*y + c*f - c*z)  # /(a*a+b*b+c*c)
        newp = [x+t*a, y+t*b, z+t*c]
        variances.append(dd[2])
        newsurf.append(newp)
        if recursive:
            vpts.SetPoint(i, newp)

        if showNPlanes and not i % ndiv:
            plane = fitPlane(points).alpha(0.3)  # fitting plane
            iapts = vs.Points(points)  # blue points
            acts += [plane, iapts]

    if decimate == 1 and not recursive:
        for i in range(ncoords):
            vpts.SetPoint(i, newsurf[i])

    actor.info['variances'] = np.array(variances)

    if showNPlanes:
        apts = vs.Points(newsurf, c='r 0.6', r=2)
        ass = Assembly([apts]+acts)
        return ass  # NB: a demo actor is returned

    return actor  # NB: original actor is modified


def smoothMLS1D(actor, f=0.2, showNLines=0):
    '''
    Smooth actor or points with a `Moving Least Squares` variant.
    The list ``actor.info['variances']`` contain the residue calculated for each point.
    Input actor's polydata is modified.

    :param float f: smoothing factor - typical range is [0,2].
    :param int showNLines: build an actor showing the fitting line for N random points.

    .. hint:: |moving_least_squares1D| |moving_least_squares1D.py|_

        |skeletonize| |skeletonize.py|_
    '''
    coords = actor.coordinates()
    ncoords = len(coords)
    Ncp = int(ncoords*f/10)
    nshow = int(ncoords)
    if showNLines:
        ndiv = int(nshow/showNLines)

    if Ncp < 3:
        vc.printc('Please choose a higher fraction than '+str(f), c=1)
        Ncp = 3

    poly = actor.GetMapper().GetInput()
    vpts = poly.GetPoints()
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    vtklist = vtk.vtkIdList()
    variances, newline, acts = [], [], []
    for i, p in enumerate(coords):

        locator.FindClosestNPoints(Ncp, p, vtklist)
        points = []
        for j in range(vtklist.GetNumberOfIds()):
            trgp = [0, 0, 0]
            vpts.GetPoint(vtklist.GetId(j), trgp)
            points.append(trgp)
        if len(points) < 2:
            continue

        points = np.array(points)
        pointsmean = points.mean(axis=0)  # plane center
        uu, dd, vv = np.linalg.svd(points-pointsmean)
        newp = np.dot(p-pointsmean, vv[0])*vv[0] + pointsmean
        variances.append(dd[1]+dd[2])
        newline.append(newp)

        if showNLines and not i % ndiv:
            fline = fitLine(points, lw=4)  # fitting plane
            iapts = vs.Points(points)  # blue points
            acts += [fline, iapts]

    for i in range(ncoords):
        vpts.SetPoint(i, newline[i])

    if showNLines:
        apts = vs.Points(newline, c='r 0.6', r=2)
        ass = Assembly([apts]+acts)
        return ass  # NB: a demo actor is returned

    actor.info['variances'] = np.array(variances)
    return actor  # NB: original actor is modified


def booleanOperation(actor1, operation, actor2, c=None, alpha=1,
                     wire=False, bc=None, legend=None, texture=None):
    '''Volumetric union, intersection and subtraction of surfaces.

    :param str operation: allowed operations: ``'plus'``, ``'intersect'``, ``'minus'``.

    .. hint:: |boolean| |boolean.py|_
    '''
    bf = vtk.vtkBooleanOperationPolyDataFilter()
    poly1 = actor1.computeNormals().polydata()
    poly2 = actor2.computeNormals().polydata()
    if operation.lower() == 'plus' or operation.lower() == '+':
        bf.SetOperationToUnion()
    elif operation.lower() == 'intersect':
        bf.SetOperationToIntersection()
    elif operation.lower() == 'minus' or operation.lower() == '-':
        bf.SetOperationToDifference()
        bf.ReorientDifferenceCellsOn()
    bf.SetInputData(0, poly1)
    bf.SetInputData(1, poly2)
    bf.Update()
    actor = Actor(bf.GetOutput(), c, alpha, wire, bc, legend, texture)
    return actor


def surfaceIntersection(actor1, actor2, tol=1e-06, lw=3):
    '''Intersect 2 surfaces and return a line actor.

    .. hint:: |surfIntersect.py|_
    '''
    try:
        bf = vtk.vtkIntersectionPolyDataFilter()
    except AttributeError:
        vc.printc('surfaceIntersection only possible for vtk version > 6', c='r')
        return None
    poly1 = actor1.GetMapper().GetInput()
    poly2 = actor2.GetMapper().GetInput()
    bf.SetInputData(0, poly1)
    bf.SetInputData(1, poly2)
    bf.Update()
    actor = Actor(bf.GetOutput(), 'k', 1, 0)
    actor.GetProperty().SetLineWidth(lw)
    return actor


def probeLine(img, p1, p2, res=100):
    '''
    Takes a ``vtkImageData`` and probes its scalars along a line defined by 2 points `p1` and `p2`.

    .. hint:: |probeLine| |probeLine.py|_
    '''
    line = vtk.vtkLineSource()
    line.SetResolution(res)
    line.SetPoint1(p1)
    line.SetPoint2(p2)
    probeFilter = vtk.vtkProbeFilter()
    probeFilter.SetSourceData(img)
    probeFilter.SetInputConnection(line.GetOutputPort())
    probeFilter.Update()

    lact = Actor(probeFilter.GetOutput(), c=None)  # ScalarVisibilityOn
    lact.mapper.SetScalarRange(img.GetScalarRange())
    return lact


def probePlane(img, origin=(0, 0, 0), normal=(1, 0, 0)):
    '''
    Takes a ``vtkImageData`` and probes its scalars on a plane.

    .. hint:: |probePlane| |probePlane.py|_
    '''
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    planeCut = vtk.vtkCutter()
    planeCut.SetInputData(img)
    planeCut.SetCutFunction(plane)
    planeCut.Update()
    cutActor = Actor(planeCut.GetOutput(), c=None)  # ScalarVisibilityOn
    cutActor.mapper.SetScalarRange(img.GetPointData().GetScalars().GetRange())
    return cutActor


def imageOperation(image1, operation='+', image2=None):
    '''
    Perform operations with ``vtkImageData`` objects. 

    `image2` can be a constant value.

    Possible operations are: ``+``, ``-``, ``/``, ``1/x``, ``sin``, ``cos``, ``exp``, ``log``, 
    ``abs``, ``**2``, ``sqrt``, ``min``, ``max``, ``atan``, ``atan2``, ``median``,
    ``mag``, ``dot``, ``gradient``, ``divergence``, ``laplacian``.

    .. hint:: |imageOperations| |imageOperations.py|_
    '''
    op = operation.lower()

    if op in ['median']:
        mf = vtk.vtkImageMedian3D()
        mf.SetInputData(image1)
        mf.Update()
        return mf.GetOutput()
    elif op in ['mag']:
        mf = vtk.vtkImageMagnitude()
        mf.SetInputData(image1)
        mf.Update()
        return mf.GetOutput()
    elif op in ['dot', 'dotproduct']:
        mf = vtk.vtkImageDotProduct()
        mf.SetInput1Data(image1)
        mf.SetInput2Data(image2)
        mf.Update()
        return mf.GetOutput()
    elif op in ['grad', 'gradient']:
        mf = vtk.vtkImageGradient()
        mf.SetDimensionality(3)
        mf.SetInputData(image1)
        mf.Update()
        return mf.GetOutput()
    elif op in ['div', 'divergence']:
        mf = vtk.vtkImageDivergence()
        mf.SetInputData(image1)
        mf.Update()
        return mf.GetOutput()
    elif op in ['laplacian']:
        mf = vtk.vtkImageLaplacian()
        mf.SetDimensionality(3)
        mf.SetInputData(image1)
        mf.Update()
        return mf.GetOutput()

    mat = vtk.vtkImageMathematics()
    mat.SetInput1Data(image1)
    K = None
    if image2:
        if isinstance(image2, vtk.vtkImageData):
            mat.SetInput2Data(image2)
        else:  # assume image2 is a constant value
            K = image2
            mat.SetConstantK(K)
            mat.SetConstantC(K)

    if op in ['+', 'add', 'plus']:
        if K:
            mat.SetOperationToAddConstant()
        else:
            mat.SetOperationToAdd()

    elif op in ['-', 'subtract', 'minus']:
        if K:
            mat.SetConstantC(-K)
            mat.SetOperationToAddConstant()
        else:
            mat.SetOperationToSubtract()

    elif op in ['*', 'multiply', 'times']:
        if K:
            mat.SetOperationToMultiplyByK()
        else:
            mat.SetOperationToMultiply()

    elif op in ['/', 'divide']:
        if K:
            mat.SetConstantK(1.0/K)
            mat.SetOperationToMultiplyByK()
        else:
            mat.SetOperationToDivide()

    elif op in ['1/x', 'invert']:
        mat.SetOperationToInvert()
    elif op in ['sin']:
        mat.SetOperationToSin()
    elif op in ['cos']:
        mat.SetOperationToCos()
    elif op in ['exp']:
        mat.SetOperationToExp()
    elif op in ['log']:
        mat.SetOperationToLog()
    elif op in ['abs']:
        mat.SetOperationToAbsoluteValue()
    elif op in ['**2', 'square']:
        mat.SetOperationToSquare()
    elif op in ['sqrt', 'sqr']:
        mat.SetOperationToSquareRoot()
    elif op in ['min']:
        mat.SetOperationToMin()
    elif op in ['max']:
        mat.SetOperationToMax()
    elif op in ['atan']:
        mat.SetOperationToATAN()
    elif op in ['atan2']:
        mat.SetOperationToATAN2()
    else:
        vc.printc('Error in imageOperation: unknown operation', operation, c=1)
        exit()
    mat.Update()
    return mat.GetOutput()


def recoSurface(points, bins=256):
    '''
    Surface reconstruction from a scattered cloud of points.

    :param int bins: number of voxels in x, y and z.

    .. hint:: |recosurface| |recosurface.py|_
    '''

    if isinstance(points, vtk.vtkActor):
        points = points.coordinates()
    N = len(points)
    if N < 50:
        print('recoSurface: Use at least 50 points.')
        return None
    points = np.array(points)

    ptsSource = vtk.vtkPointSource()
    ptsSource.SetNumberOfPoints(N)
    ptsSource.Update()
    vpts = ptsSource.GetOutput().GetPoints()
    for i, p in enumerate(points):
        vpts.SetPoint(i, p)
    polyData = ptsSource.GetOutput()

    distance = vtk.vtkSignedDistance()
    f = 0.1
    x0, x1, y0, y1, z0, z1 = polyData.GetBounds()
    distance.SetBounds(x0-(x1-x0)*f, x1+(x1-x0)*f,
                       y0-(y1-y0)*f, y1+(y1-y0)*f,
                       z0-(z1-z0)*f, z1+(z1-z0)*f)
    if polyData.GetPointData().GetNormals():
        distance.SetInputData(polyData)
    else:
        normals = vtk.vtkPCANormalEstimation()
        normals.SetInputData(polyData)
        normals.SetSampleSize(int(N/50))
        normals.SetNormalOrientationToGraphTraversal()
        distance.SetInputConnection(normals.GetOutputPort())
        print('Recalculating normals for', N,
              'Points, sample size=', int(N/50))

    b = polyData.GetBounds()
    diagsize = np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)
    radius = diagsize/bins*5
    distance.SetRadius(radius)
    distance.SetDimensions(bins, bins, bins)
    distance.Update()

    print('Calculating mesh from points with R =', radius)
    surface = vtk.vtkExtractSurface()
    surface.SetRadius(radius * .99)
    surface.HoleFillingOn()
    surface.ComputeNormalsOff()
    surface.ComputeGradientsOff()
    surface.SetInputConnection(distance.GetOutputPort())
    surface.Update()
    return Actor(surface.GetOutput(), 'gold', 1, 0, 'tomato')


def cluster(points, radius):
    '''
    Clustering of points in space.

    `radius` is the radius of local search.
    Individual subsets can be accessed through ``actor.clusters``.

    .. hint:: |clustering| |clustering.py|_
    '''
    if isinstance(points, vtk.vtkActor):
        poly = points.GetMapper().GetInput()
    else:
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(points))
        src.Update()
        vpts = src.GetOutput().GetPoints()
        for i, p in enumerate(points):
            vpts.SetPoint(i, p)
        poly = src.GetOutput()

    cluster = vtk.vtkEuclideanClusterExtraction()
    cluster.SetInputData(poly)
    cluster.SetExtractionModeToAllClusters()
    cluster.SetRadius(radius)
    cluster.ColorClustersOn()
    cluster.Update()

    idsarr = cluster.GetOutput().GetPointData().GetArray('ClusterId')
    Nc = cluster.GetNumberOfExtractedClusters()

    sets = [[] for i in range(Nc)]
    for i, p in enumerate(points):
        sets[idsarr.GetValue(i)].append(p)

    acts = []
    for i, aset in enumerate(sets):
        acts.append(vs.Points(aset, c=i))

    actor = Assembly(acts)

    actor.info['clusters'] = sets
    print('Nr. of extracted clusters', Nc)
    if Nc > 10:
        print('First ten:')
    for i in range(Nc):
        if i > 9:
            print('...')
            break
        print('Cluster #'+str(i)+',  N =', len(sets[i]))
    print('Access individual clusters through attribute: actor.cluster')
    return actor


def removeOutliers(points, radius):
    '''
    Remove outliers from a cloud of points within the specified `radius` search.

    .. hint:: |clustering| |clustering.py|_
    '''
    isactor = False
    if isinstance(points, vtk.vtkActor):
        isactor = True
        poly = points.GetMapper().GetInput()
    else:
        src = vtk.vtkPointSource()
        src.SetNumberOfPoints(len(points))
        src.Update()
        vpts = src.GetOutput().GetPoints()
        for i, p in enumerate(points):
            vpts.SetPoint(i, p)
        poly = src.GetOutput()

    removal = vtk.vtkRadiusOutlierRemoval()
    removal.SetInputData(poly)
    removal.SetRadius(radius)
    removal.SetNumberOfNeighbors(5)
    removal.GenerateOutliersOff()
    removal.Update()
    rpoly = removal.GetOutput()
    print("# of removed outlier points: ",
          removal.GetNumberOfPointsRemoved(), '/', poly.GetNumberOfPoints())
    outpts = []
    for i in range(rpoly.GetNumberOfPoints()):
        outpts.append(list(rpoly.GetPoint(i)))
    outpts = np.array(outpts)
    if not isactor:
        return outpts

    actor = vs.Points(outpts)
    return actor  # return same obj for concatenation


def thinPlateSpline(actor, sourcePts, targetPts, userFunctions=(None, None)):
    '''
    `Thin Plate Spline` transformations describe a nonlinear warp transform defined by a set 
    of source and target landmarks. Any point on the mesh close to a source landmark will 
    be moved to a place close to the corresponding target landmark. 
    The points in between are interpolated smoothly using Bookstein's Thin Plate Spline algorithm.

    Transformation object is saved in ``actor.info['transform']``.

    :param userFunctions: You must supply both the function 
        and its derivative with respect to r.

    .. hint:: |thinplate| |thinplate.py|_

        |thinplate_grid| |thinplate_grid.py|_

        |thinplate_morphing| |thinplate_morphing.py|_

        |interpolateField| |interpolateField.py|_
    '''
    ns = len(sourcePts)
    ptsou = vtk.vtkPoints()
    ptsou.SetNumberOfPoints(ns)
    for i in range(ns):
        ptsou.SetPoint(i, sourcePts[i])

    nt = len(sourcePts)
    if ns != nt:
        vc.printc('thinPlateSpline Error: #source != #target points', ns, nt, c=1)
        exit()

    pttar = vtk.vtkPoints()
    pttar.SetNumberOfPoints(nt)
    for i in range(ns):
        pttar.SetPoint(i, targetPts[i])

    transform = vtk.vtkThinPlateSplineTransform()
    transform.SetBasisToR()
    if userFunctions[0]:
        transform.SetBasisFunction(userFunctions[0])
        transform.SetBasisDerivative(userFunctions[1])
    transform.SetSigma(1)
    transform.SetSourceLandmarks(ptsou)
    transform.SetTargetLandmarks(pttar)

    tfa = transformFilter(actor.polydata(), transform)
    tfa.info['transform'] = transform
    return tfa


def transformFilter(actor, transformation):
    '''
    Transform a ``vtkActor`` and return a new object.
    '''
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetTransform(transformation)
    prop = None
    if isinstance(actor, vtk.vtkPolyData):
        tf.SetInputData(actor)
    else:
        tf.SetInputData(actor.polydata())
        prop = vtk.vtkProperty()
        prop.DeepCopy(actor.GetProperty())
    tf.Update()

    tfa = Actor(tf.GetOutput())
    if prop:
        tfa.SetProperty(prop)
    return tfa


def meshQuality(actor, measure=6):
    '''
    Calculate functions of quality of the elements of a triangular mesh.
    See class `vtkMeshQuality <https://vtk.org/doc/nightly/html/classvtkMeshQuality.html>`_
    for explaination.

    :param int measure: type of estimator

        - EDGE_RATIO, 0
        - ASPECT_RATIO, 1
        - RADIUS_RATIO, 2
        - ASPECT_FROBENIUS, 3
        - MED_ASPECT_FROBENIUS, 4
        - MAX_ASPECT_FROBENIUS, 5
        - MIN_ANGLE, 6
        - COLLAPSE_RATIO, 7
        - MAX_ANGLE, 8
        - CONDITION, 9
        - SCALED_JACOBIAN, 10
        - SHEAR, 11
        - RELATIVE_SIZE_SQUARED, 12
        - SHAPE, 13
        - SHAPE_AND_SIZE, 14
        - DISTORTION, 15
        - MAX_EDGE_RATIO, 16
        - SKEW, 17
        - TAPER, 18
        - VOLUME, 19
        - STRETCH, 20
        - DIAGONAL, 21
        - DIMENSION, 22
        - ODDY, 23
        - SHEAR_AND_SIZE, 24
        - JACOBIAN, 25
        - WARPAGE, 26
        - ASPECT_GAMMA, 27
        - AREA, 28
        - ASPECT_BETA, 29

    .. hint:: |meshquality| |meshquality.py|_
    '''

    mesh = actor.GetMapper().GetInput()

    qf = vtk.vtkMeshQuality()
    qf.SetInputData(mesh)
    qf.SetTriangleQualityMeasure(measure)
    qf.SaveCellQualityOn()
    qf.Update()

    pd = vtk.vtkPolyData()
    pd.ShallowCopy(qf.GetOutput())

    qactor = Actor(pd, c=None, alpha=1)
    qactor.mapper.SetScalarRange(pd.GetScalarRange())
    return qactor


def splitByConnectivity(actor, maxdepth=100):
    '''
    Split a mesh by connectivity and order the pieces by increasing area.

    :param int maxdepth: only consider this number of mesh parts.

    .. hint:: |splitmesh| |splitmesh.py|_   
    '''
    actor.addIDs()
    pd = actor.polydata()
    cf = vtk.vtkConnectivityFilter()
    cf.SetInputData(pd)
    cf.SetExtractionModeToAllRegions()
    cf.ColorRegionsOn()
    cf.Update()
    cpd = cf.GetOutput()
    a = Actor(cpd)
    alist = []

    for t in range(max(a.scalars('RegionId'))-1):
        if t == maxdepth:
            break
        suba = a.clone().threshold('RegionId', t-0.1, t+0.1)
        area = suba.area()
        alist.append([suba, area])

    alist.sort(key=lambda x: x[1])
    alist.reverse()
    blist = []
    for i, l in enumerate(alist):
        l[0].color(i+1)
        l[0].mapper.ScalarVisibilityOff()
        blist.append(l[0])
    return blist



def pointSampler(actor, distance=None):
    '''
    Algorithm to generate points the specified distance apart. 
    '''
    poly = actor.polydata(True)

    pointSampler = vtk.vtkPolyDataPointSampler()
    if not distance:
        distance = actor.diagonalSize()/100.
    pointSampler.SetDistance(distance)
    #    pointSampler.GenerateVertexPointsOff()
    #    pointSampler.GenerateEdgePointsOff()
    #    pointSampler.GenerateVerticesOn()
    #    pointSampler.GenerateInteriorPointsOn()
    pointSampler.SetInputData(poly)
    pointSampler.Update()

    uactor = Actor(pointSampler.GetOutput())
    prop = vtk.vtkProperty()
    prop.DeepCopy(actor.GetProperty())
    uactor.SetProperty(prop)
    return uactor


def geodesic(actor, start, end):
    '''
    Dijkstra algorithm to compute the graph geodesic.
    Takes as input a polygonal mesh and performs a single source shortest path calculation.

    :param start: start vertex index or close point `[x,y,z]`
    :type start: int, list
    :param end: end vertex index or close point `[x,y,z]`
    :type start: int, list

    .. hint:: |geodesic| |geodesic.py|_    
    '''

    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()

    if vu.isSequence(start):
        cc = actor.coordinates()
        pa = vs.Points(cc)
        start = pa.closestPoint(start, returnIds=True)
        end = pa.closestPoint(end, returnIds=True)
        dijkstra.SetInputData(pa.polydata())
    else:
        dijkstra.SetInputData(actor.polydata())

    dijkstra.SetStartVertex(start)
    dijkstra.SetEndVertex(end)
    dijkstra.Update()

    weights = vtk.vtkDoubleArray()
    dijkstra.GetCumulativeWeights(weights)

    length = weights.GetMaxId()+1
    arr = np.zeros(length)
    for i in range(length):
        arr[i] = weights.GetTuple(i)[0]

    dactor = Actor(dijkstra.GetOutput())
    prop = vtk.vtkProperty()
    prop.DeepCopy(actor.GetProperty())
    prop.SetLineWidth(3)
    prop.SetOpacity(1)
    dactor.SetProperty(prop)
    dactor.info['CumulativeWeights'] = arr
    return dactor


def convexHull(actor_or_list, alphaConstant=0):
    '''
    Create a 3D Delaunay triangulation of input points.

    :param actor_or_list: can be either an ``Actor`` or a list of 3D points.
    :param float alphaConstant: For a non-zero alpha value, only verts, edges, faces, 
        or tetra contained within the circumsphere (of radius alpha) will be output. 
        Otherwise, only tetrahedra will be output. 

    .. hint:: |convexHull| |convexHull.py|_    
    '''
    if vu.isSequence(actor_or_list):
        actor = vs.Points(actor_or_list)
    else:
        actor = actor_or_list
    apoly = actor.clean().polydata()

    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputData(apoly)
    triangleFilter.Update()
    poly = triangleFilter.GetOutput()

    delaunay = vtk.vtkDelaunay3D()  # Create the convex hull of the pointcloud
    if alphaConstant:
        delaunay.SetAlpha(alphaConstant)
    delaunay.SetInputData(poly)
    delaunay.Update()

    surfaceFilter = vtk.vtkDataSetSurfaceFilter()
    surfaceFilter.SetInputConnection(delaunay.GetOutputPort())
    surfaceFilter.Update()

    chuact = Actor(surfaceFilter.GetOutput())
    return chuact


def actor2ImageData(actor, spacing=(1, 1, 1)):
    '''
    Convert a mesh it into volume representation as ``vtkImageData`` 
    where the foreground voxels are 1 and the background voxels are 0.
    Internally the ``vtkPolyDataToImageStencil`` class is used.

    .. hint:: |mesh2volume| |mesh2volume.py|_    
    '''
    # https://vtk.org/Wiki/VTK/Examples/Cxx/PolyData/PolyDataToImageData
    pd = actor.polydata()

    whiteImage = vtk.vtkImageData()
    bounds = pd.GetBounds()

    whiteImage.SetSpacing(spacing)

    # compute dimensions
    dim = [0, 0, 0]
    for i in [0, 1, 2]:
        dim[i] = int(np.ceil((bounds[i*2+1] - bounds[i*2]) / spacing[i]))
    whiteImage.SetDimensions(dim)
    whiteImage.SetExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1)

    origin = [0, 0, 0]
    origin[0] = bounds[0] + spacing[0] / 2
    origin[1] = bounds[2] + spacing[1] / 2
    origin[2] = bounds[4] + spacing[2] / 2
    whiteImage.SetOrigin(origin)
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # fill the image with foreground voxels:
    inval = 255
    count = whiteImage.GetNumberOfPoints()
    for i in range(count):
        whiteImage.GetPointData().GetScalars().SetTuple1(i, inval)

    # polygonal data --> image stencil:
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(pd)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
    pol2stenc.Update()

    # cut the corresponding white image and set the background:
    outval = 0
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImage)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    return imgstenc.GetOutput()


def projectSphereFilter(actor):
    '''
    Project a spherical-like object onto a plane.

    .. hint:: |projectsphere| |projectsphere.py|_    
    '''
    poly = actor.polydata()
    psf = vtk.vtkProjectSphereFilter()
    psf.SetInputData(poly)
    psf.Update()

    a = Actor(psf.GetOutput())
    return a

    
    
