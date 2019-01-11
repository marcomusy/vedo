"""
Defines methods useful to analise 3D meshes.
"""


from __future__ import division, print_function


__all__ = [
    'spline',
    'xyplot',
    'fxy',
    'histogram2D',
    'delaunay2D',
    'normals',
    'curvature',
    'boundaries',
    'extractLargestRegion',
    'align',
    'procrustes',
    'fitLine',
    'fitPlane',
    'fitSphere',
    'pca',
    'smoothLaplacian',
    'smoothWSinc',
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
]

import vtk
import numpy as np

import vtkplotter.colors as vc
import vtkplotter.vtkio as vio
import vtkplotter.shapes as vs
from vtkplotter.actors import Actor, Assembly
from vtk.util.numpy_support import numpy_to_vtk


def spline(points, smooth=0.5, degree=2,
           s=2, c='b', alpha=1., nodes=False, legend=None, res=20):
    '''
    Return a vtkActor for a spline that doesnt necessarly pass exactly throught all points.

    Options:

        smooth, smoothing factor:
                0 = interpolate points exactly,
                1 = average point positions

        degree = degree of the spline (1<degree<5)

        nodes = True shows also original the points

    `tutorial.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py>`_

    .. image:: https://user-images.githubusercontent.com/32848391/50738978-d889dd80-11d9-11e9-90f1-485dc8212760.jpg
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
    # evaluate spline, including interpolated points:
    xnew, ynew, znew = splev(np.linspace(0, 1, Nout), tckp)

    ppoints = vtk.vtkPoints()  # Generate the polyline for the spline
    profileData = vtk.vtkPolyData()
    ppoints.SetData(numpy_to_vtk( list(zip(xnew, ynew, znew)), deep=True))
    lines = vtk.vtkCellArray()  # Create the polyline
    lines.InsertNextCell(Nout)
    for i in range(Nout):
        lines.InsertCellPoint(i)
    profileData.SetPoints(ppoints)
    profileData.SetLines(lines)
    actline = Actor(profileData, c=c, alpha=alpha, legend=legend)
    actline.GetProperty().SetLineWidth(s)
    if nodes:
        actnodes = vs.points(points, r=5, c=c, alpha=alpha)
        ass = Assembly([actline, actnodes], legend=legend)
        return ass
    else:
        return actline


def _vtkspline(points, s, c, alpha, nodes, legend, res):
    numberOfOutputPoints = len(points)*res  # Number of points on the spline
    numberOfInputPoints = len(points)  # One spline for each direction.
    aSplineX = vtk.vtkCardinalSpline()  # interpolate the x values
    aSplineY = vtk.vtkCardinalSpline()  # interpolate the y values
    aSplineZ = vtk.vtkCardinalSpline()  # interpolate the z values

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
    Return a vtkActor that is a plot of 2D points in x and y.

    Use corner to assign its position:

        1=topleft,

        2=topright,

        3=bottomleft,

        4=bottomright.

    `tutorial.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py>`_
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


def fxy(z='sin(3*x)*log(x-y)/3', x=[0, 3], y=[0, 3],
        zlimits=[None, None], showNan=True, zlevels=10, wire=False,
        c='b', bc='aqua', alpha=1, legend=True, texture=None, res=100):
    '''
    Build a surface representing the 3D function specified as a string
    or as a reference to an external function.
    Red points indicate where the function does not exist (showNan).

    zlevels will draw the specified number of z-levels contour lines.

    `fxy.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/fxy.py>`_

    .. image:: https://user-images.githubusercontent.com/32848391/36611824-fd524fac-18d4-11e8-8c76-d3d1b1bb3954.png
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
        a = tmpact1.cutPlane((0, 0, zlimits[0]), (0, 0, 1))
        poly = a.polydata()
    if zlimits[1]:
        tmpact2 = Actor(poly)
        a = tmpact2.cutPlane((0, 0, zlimits[1]), (0, 0, -1))
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
        elevation.SetLowPoint( 0, 0, bounds[4])
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
        nansact = vs.points(nans, c='red', alpha=alpha/2)
        acts.append(nansact)

    if len(acts) > 1:
        asse = Assembly(acts)
        return asse
    else:
        return actor


def histogram2D(xvalues, yvalues, bins=12, norm=1, c='g', alpha=1, fill=False):
    '''
    Build a 2D hexagonal histogram from a list of x and y values.

    bins, nr of bins for the smaller range in x or y

    norm, sets a scaling factor for the z axis

    fill, draw solid hexagons
    
    `histo2D.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/histo2D.py>`_    

    .. image:: https://user-images.githubusercontent.com/32848391/50738861-bfccf800-11d8-11e9-9698-c0b9dccdba4d.jpg    
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
            if not i%2:
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
    asse.PickableOff()
    asse.SetScale(1/n*1.2*dx, 1/m*dy, norm/binmax*(dx+dy)/4)
    asse.SetPosition(xmin,ymin,0)
    return asse


def delaunay2D(plist, tol=None, c='gold', alpha=0.5, wire=False, bc=None,
               legend=None, texture=None):
    '''
    Create a mesh from points in the XY plane.

    `delaunay2d.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/delaunay2d.py>`_
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738865-c0658e80-11d8-11e9-8616-b77363aa4695.jpg
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
    return Actor(delny.GetOutput(), c, alpha, wire, bc, legend, texture)


def normals(actor, ratio=5, c=(0.6, 0.6, 0.6), alpha=0.8, legend=None):
    '''
    Build a vtkActor made of the normals at vertices shown as arrows

    `tutorial.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py>`_

    `fatlimb.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fatlimb.py>`_
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738945-7335ec80-11d9-11e9-9d3f-c6c19df8f10d.jpg
    '''
    maskPts = vtk.vtkMaskPoints()
    maskPts.SetOnRatio(ratio)
    maskPts.RandomModeOff()
    src = actor.polydata()
    maskPts.SetInputData(src)
    arrow = vtk.vtkLineSource()
    arrow.SetPoint1(0,0,0)
    arrow.SetPoint2(.75,0,0)
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetInputConnection(maskPts.GetOutputPort())
    glyph.SetVectorModeToUseNormal()
    b = src.GetBounds()
    sc = max([b[1]-b[0], b[3]-b[2], b[5]-b[4]])/20.
    glyph.SetScaleFactor(sc)
    glyph.SetColorModeToColorByVector()
    glyph.SetScaleModeToScaleByVector()
    glyph.OrientOn()
    glyph.Update()
    glyphMapper = vtk.vtkPolyDataMapper()
    glyphMapper.SetInputConnection(glyph.GetOutputPort())
    glyphMapper.SetScalarModeToUsePointFieldData()
    glyphMapper.SetColorModeToMapScalars()
    glyphMapper.ScalarVisibilityOn()
    glyphMapper.SelectColorArray("Elevation")
    glyphActor = vtk.vtkActor()
    glyphActor.SetMapper(glyphMapper)
    glyphActor.GetProperty().EdgeVisibilityOff()
    glyphActor.GetProperty().SetColor(vc.getColor(c))
    # check if color string contains a float, in this case ignore alpha
    al = vc.getAlpha(c)
    if al:
        alpha = al
    glyphActor.GetProperty().SetOpacity(alpha)
    glyphActor.PickableOff()
    aactor = Assembly([actor, glyphActor], legend=legend)
    return aactor


def curvature(actor, method=1, r=1, alpha=1, lut=None, legend=None):
    '''
    Build a copy of vtkActor that contains the color coded surface
    curvature following four different ways to calculate it:
    method =  0-gaussian, 1-mean, 2-max, 3-min
    '''
    poly = actor.polydata()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    curve = vtk.vtkCurvatures()
    curve.SetInputConnection(cleaner.GetOutputPort())
    curve.SetCurvatureType(method)
    curve.InvertMeanCurvatureOn()
    curve.Update()
    print('CurvatureType set to:', method)
    if not lut:
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)
        lut.SetHueRange(0.15, 1)
        lut.SetSaturationRange(1, 1)
        lut.SetValueRange(1, 1)
        lut.SetAlphaRange(alpha, 1)
        b = poly.GetBounds()
        sc = max([b[1]-b[0], b[3]-b[2], b[5]-b[4]])
        lut.SetRange(-0.01/sc*r, 0.01/sc*r)
    cmapper = vtk.vtkPolyDataMapper()
    cmapper.SetInputConnection(curve.GetOutputPort())
    cmapper.SetLookupTable(lut)
    cmapper.SetUseLookupTableScalarRange(1)
    cactor = vtk.vtkActor()
    cactor.SetMapper(cmapper)
    return cactor


def boundaries(actor, c='p', lw=5, legend=None):
    '''Build a copy of actor that shows the boundary lines of its surface. '''
 
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(actor.polydata())
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOn()
    fe.ManifoldEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.ColoringOff()
    fe.Update()
    bactor = Actor(fe.GetOutput(), c=c, alpha=1, legend=legend)
    bactor.GetProperty().SetLineWidth(lw)
    return bactor


def extractLargestRegion(actor, legend=None):
    '''Keep only the largest connected part of a mesh and discard all the smaller pieces.

    `largestregion.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/largestregion.py>`_
    '''
    conn = vtk.vtkConnectivityFilter()
    conn.SetExtractionModeToLargestRegion()
    conn.ScalarConnectivityOff()
    poly = actor.polydata(True)
    conn.SetInputData(poly)
    conn.Update()
    epoly = conn.GetOutput()
    if legend is True:
        legend = actor.legend
    eact = Actor(epoly, legend)
    pr = vtk.vtkProperty()
    pr.DeepCopy(actor.GetProperty())
    eact.SetProperty(pr)
    return eact


def align(source, target, iters=100, rigid=False, legend=None):
    '''
    Return a copy of source actor which is aligned to
    target actor through vtkIterativeClosestPointTransform class.

    The core of the algorithm is to match each vertex in one surface with
    the closest surface point on the other, then apply the transformation
    that modify one surface to best match the other (in the least-square sense).

    `align1.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/align1.py>`_
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738875-c196bb80-11d8-11e9-8bdc-b80fd01a928d.jpg
    
    `align2.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/align2.py>`_
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738874-c196bb80-11d8-11e9-9587-2177d1680b70.jpg
    '''
    if isinstance(source, Actor): source = source.polydata()
    if isinstance(target, Actor): target = target.polydata()

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
    actor = Actor(poly, legend=legend)
    actor.info['transform'] = icp.GetLandmarkTransform()
    return actor


def procrustes(sources, rigid=False, legend=None):
    '''
    Return an Assembly of aligned source actors with
    the vtkProcrustesAlignmentFilter class. Assembly is normalized in space.

    Takes N set of points and aligns them in a least-squares sense
    to their mutual mean. The algorithm is iterated until convergence,
    as the mean must be recomputed after each alignment.
    
    `align3.py`_
    
    .. _align3.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/align3.py
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738873-c196bb80-11d8-11e9-8653-a41108a5f02d.png
    '''
    group = vtk.vtkMultiBlockDataGroupFilter()
    for source in sources:
        if sources[0].N() != source.N():
            vc.printc('Procrustes error in align():' , c=1)
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
    for i in range(len(sources)):
        poly = procrustes.GetOutput().GetBlock(i)
        actor = Actor(poly)
        actor.SetProperty(sources[i].GetProperty())
        acts.append(actor)
    assem = Assembly(acts, legend=legend)
    assem.info['transform'] = procrustes.GetLandmarkTransform()
    return assem


################# working with point clouds

def fitLine(points, c='orange', lw=1, alpha=0.6, legend=None):
    '''
    Fits a line through points.

    Extra info is stored in actor.slope, actor.center, actor.variances

    `fitline.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitline.py>`_
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
    l = vs.line(p1, p2, c=c, lw=lw, alpha=alpha)
    l.info['slope'] = vv
    l.info['center'] = datamean
    l.info['variances'] = dd
    return l


def fitPlane(points, c='g', bc='darkgreen', alpha=0.8, legend=None):
    '''
    Fits a plane to a set of points.

    Extra info is stored in actor.normal, actor.center, actor.variance

    `fitline.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitline.py>`_
    
    `fitplanes.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitplanes.py>`_
    '''
    data = np.array(points)
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    s = np.linalg.norm(xyz_max - xyz_min)
    n = np.cross(vv[0], vv[1])
    pla = vs.plane(datamean, n, s, s, c, bc, alpha, legend, None)
    pla.info['normal'] = n
    pla.info['center'] = datamean
    pla.info['variance'] =  dd[2]
    return pla


def fitSphere(coords, c='r', alpha=1, wire=1, legend=None):
    '''
    Fits a sphere to a set of points.

    Extra info is stored in actor.radius, actor.center, actor.residue

    `fitspheres1.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres1.py>`_
    
    `fitspheres2.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres2.py>`_
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738943-687b5780-11d9-11e9-87a6-054e0fe76241.jpg
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
    s = vs.sphere(center, radius, c, alpha, wire=wire, legend=legend)
    s.info['radius'] = radius
    s.info['center'] = center
    s.info['residue'] = residue
    return s


def pca(points, pvalue=.95, c='c', alpha=0.5, pcaAxes=False, legend=None):
    '''
    Show the oriented PCA ellipsoid that contains fraction pvalue of points.

    axes = True, show the 3 PCA semi axes

    Extra info is stored in actor.sphericity, actor.va, actor.vb, actor.vc
    (sphericity = 1 for a perfect sphere)

    `tutorial.py <https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py>`_
    
    `cell_main.py`_
    
    .. _cell_main.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/cell_main.py
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738947-7335ec80-11d9-11e9-9a45-6053b4eaf9f9.jpg
    '''
    try:
        from scipy.stats import f
    except:
        vc.printc("Error in ellipsoid(): scipy not installed. Skip.", c=1)
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
    ua, ub, uc = np.sqrt(s*fppf)*2 # semi-axes (largest first)
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
    actor_elli = Actor(ftra.GetOutput(), c, alpha, legend=legend)
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
            axs.append(Actor(t.GetOutput(), c, alpha).lineWidth(3))
        finact = Assembly([actor_elli]+axs, legend=legend)
    else:
        finact = actor_elli
    finact.info['sphericity'] = sphericity
    finact.info['va'] = ua
    finact.info['vb'] = ub
    finact.info['vc'] = uc
    return finact


def smoothLaplacian(actor, niter=15, relaxfact=0.1, edgeAngle=15, featureAngle=60):
    '''
    Adjust mesh point positions using Laplacian smoothing.

    `mesh_smoothers.py`_
    
    .. _mesh_smoothers.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/mesh_smoothers.py
    '''
    poly = actor.polydata()
    cl = vtk.vtkCleanPolyData()
    cl.SetInputData(poly)
    cl.Update()
    poly = cl.GetOutput()  # removes the boudaries duplication
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(poly)
    smoothFilter.SetNumberOfIterations(niter)
    smoothFilter.SetRelaxationFactor(relaxfact)
    smoothFilter.SetEdgeAngle(edgeAngle)
    smoothFilter.SetFeatureAngle(featureAngle)
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.FeatureEdgeSmoothingOn()
    smoothFilter.GenerateErrorScalarsOn()
    smoothFilter.Update()
    return align(smoothFilter.GetOutput(), poly)


def smoothWSinc(actor, niter=15, passBand=0.1, edgeAngle=15, featureAngle=60):
    '''
    Adjust mesh point positions using the windowed sinc function interpolation kernel.

    `mesh_smoothers.py`_
    
    .. _mesh_smoothers.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/mesh_smoothers.py
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738939-67e2c100-11d9-11e9-90cb-716ff3f03f67.jpg
    '''
    poly = actor.polydata()
    cl = vtk.vtkCleanPolyData()
    cl.SetInputData(poly)
    cl.Update()
    poly = cl.GetOutput()  # removes the boudaries duplication
    smoothFilter = vtk.vtkWindowedSincPolyDataFilter()
    smoothFilter.SetInputData(poly)
    smoothFilter.SetNumberOfIterations(niter)
    smoothFilter.SetEdgeAngle(edgeAngle)
    smoothFilter.SetFeatureAngle(featureAngle)
    smoothFilter.SetPassBand(passBand)
    smoothFilter.NormalizeCoordinatesOn()
    smoothFilter.NonManifoldSmoothingOn()
    smoothFilter.FeatureEdgeSmoothingOn()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()
    return align(smoothFilter.GetOutput(), poly)


def smoothMLS3D(actors, neighbours=10):
    '''
    A time sequence of actors is being smoothed in 4D
    using a MLS (Moving Least Squares) variant.
    Time assciated to an actor must be specified in advance with actor.time(t).
    Data itself can suggest a meaningful time separation based on the spatial
    distribution of points.

    neighbours, fixed nr of neighbours in space-time to take into account in fit.
    
    `moving_least_squares3D.py`_
    
    .. _moving_least_squares3D.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares3D.py
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738935-61544980-11d9-11e9-9c20-f2ce944d2238.jpg
    '''
    from scipy.spatial import KDTree

    coords4d = []
    for a in actors: # build the list of 4d coordinates
        coords3d = a.coordinates()
        n = len(coords3d)
        pttimes = [[a.time()]]*n
        coords4d += np.append(coords3d, pttimes, axis=1).tolist()

    avedt = float(actors[-1].time()-actors[0].time())/len(actors)
    print("Average time separation between actors dt =", round(avedt, 3))

    coords4d = np.array(coords4d)
    newcoords4d = []
    kd = KDTree(coords4d, leafsize=neighbours)
    suggest=''

    pb = vio.ProgressBar(0, len(coords4d))
    for i in pb.range():
        mypt = coords4d[i]

        #dr = np.sqrt(3*dx**2+dt**2)
        #iclosest = kd.query_ball_point(mypt, r=dr)
        #dists, iclosest = kd.query(mypt, k=None, distance_upper_bound=dr)
        dists, iclosest = kd.query(mypt, k=neighbours)
        closest = coords4d[iclosest]

        nc = len(closest)
        if nc >= neighbours and nc > 5:
            m = np.linalg.lstsq(closest, [1.]*nc, rcond=None)[0]
            vers = m/np.linalg.norm(m)
            hpcenter = np.mean(closest, axis=0)  # hyperplane center
            dist = np.dot(mypt-hpcenter, vers)
            projpt = mypt - dist*vers
            newcoords4d.append(projpt)

            if not i%1000: # work out some stats
                v = np.std(closest, axis=0)
                vx = round((v[0]+v[1]+v[2])/3, 3)
                suggest='data suggest dt='+str(vx)

        pb.print(suggest)
    newcoords4d = np.array(newcoords4d)

    ctimes = newcoords4d[:, 3]
    ccoords3d = np.delete(newcoords4d, 3, axis=1) # get rid of time
    act = vs.points(ccoords3d)
    act.pointColors(ctimes, cmap='jet') # use a colormap to associate a color to time
    return act


def smoothMLS2D(actor, f=0.2, decimate=1, recursive=0, showNPlanes=0):
    '''
    Smooth actor or points with a Moving Least Squares variant.
    The list actor.variances contain the residue calculated for each point.
    Input actor's polydata is modified.

    Options:

        f, smoothing factor - typical range s [0,2]

        decimate, decimation factor (an integer number)

        recursive, move points while algorithm proceedes

        showNPlanes, build an actor showing the fitting plane for N random points

    `mesh_smoothers.py`_
    
    .. _mesh_smoothers.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/mesh_smoothers.py
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738939-67e2c100-11d9-11e9-90cb-716ff3f03f67.jpg
        
    `moving_least_squares2D.py`_
    
    .. _moving_least_squares2D.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738936-61544980-11d9-11e9-9efb-e2a923762b72.jpg

    `recosurface.py`_
    
    .. _recosurface.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/recosurface.py

    .. image:: https://user-images.githubusercontent.com/32848391/46817107-b3263880-cd7e-11e8-985d-f5d158992f0c.png
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

    poly = actor.polydata(True)
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
            plane = fitPlane(points, alpha=0.3)  # fitting plane
            iapts = vs.points(points)  # blue points
            acts += [plane, iapts]

    if decimate == 1 and not recursive:
        for i in range(ncoords):
            vpts.SetPoint(i, newsurf[i])

    actor.info['variances'] = np.array(variances)

    if showNPlanes:
        apts = vs.points(newsurf, c='r 0.6', r=2)
        ass = Assembly([apts]+acts)
        return ass  # NB: a demo actor is returned

    return actor  # NB: original actor is modified


def smoothMLS1D(actor, f=0.2, showNLines=0):
    '''
    Smooth actor or points with a Moving Least Squares variant.
    The list actor.variances contain the residue calculated for each point.
    Input actor's polydata is modified.

    Options:

        f, smoothing factor - typical range is [0,2]

        showNLines, build an actor showing the fitting line for N random points

    `moving_least_squares1D.py`_
    
    .. _moving_least_squares1D.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares1D.py
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738937-61544980-11d9-11e9-8be8-8826032b8baf.jpg
    
    `skeletonize.py`_
    
    .. _skeletonize.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/skeletonize.py

    .. image:: https://user-images.githubusercontent.com/32848391/46820954-c5f13b00-cd87-11e8-87aa-286528a09de8.png
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

    poly = actor.polydata(True)
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
            fline = fitLine(points, lw=4, alpha=1)  # fitting plane
            iapts = vs.points(points)  # blue points
            acts += [fline, iapts]

    for i in range(ncoords):
        vpts.SetPoint(i, newline[i])

    if showNLines:
        apts = vs.points(newline, c='r 0.6', r=2)
        ass = Assembly([apts]+acts)
        return ass  # NB: a demo actor is returned

    actor.info['variances'] = np.array(variances)
    return actor  # NB: original actor is modified


def booleanOperation(actor1, actor2, operation='plus', c=None, alpha=1,
                     wire=False, bc=None, legend=None, texture=None):
    '''Volumetric union, intersection and subtraction of surfaces.

    `boolean.py`_
    
    .. _boolean.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/boolean.py
    
    .. image:: https://user-images.githubusercontent.com/32848391/50738871-c0fe2500-11d8-11e9-8812-442b69be6db9.png
    '''
    try:
        bf = vtk.vtkBooleanOperationPolyDataFilter()
    except AttributeError:
        vc.printc('Boolean operation only possible for vtk version >= 8', c='r')
        return None
    poly1 = actor1.polydata(True)
    poly2 = actor2.polydata(True)
    if operation.lower() == 'plus':
        bf.SetOperationToUnion()
    elif operation.lower() == 'intersect':
        bf.SetOperationToIntersection()
    elif operation.lower() == 'minus':
        bf.SetOperationToDifference()
        bf.ReorientDifferenceCellsOn()
    bf.SetInputData(0, poly1)
    bf.SetInputData(1, poly2)
    bf.Update()
    actor = Actor(bf.GetOutput(), c, alpha, wire, bc, legend, texture)
    return actor


def surfaceIntersection(actor1, actor2, tol=1e-06, lw=3,
                        c=None, alpha=1, legend=None):
    '''Intersect 2 surfaces and return a line actor.

    `surfIntersect.py`_
    
    .. _surfIntersect.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/surfIntersect.py
    '''
    try:
        bf = vtk.vtkIntersectionPolyDataFilter()
    except AttributeError:
        vc.printc('surfaceIntersection only possible for vtk version > 6', c='r')
        return None
    poly1 = actor1.polydata(True)
    poly2 = actor2.polydata(True)
    bf.SetInputData(0, poly1)
    bf.SetInputData(1, poly2)
    bf.Update()
    if c is None:
        c = actor1.GetProperty().GetColor()
    actor = Actor(bf.GetOutput(), c, alpha, 0, legend=legend)
    actor.GetProperty().SetLineWidth(lw)
    return actor


def probeLine(img, p1, p2, res=100):
    '''
    Takes a vtkImageData and probes its scalars along a line defined by 2 points.

    `probeLine.py`_

    .. _probeLine.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/probeLine.py

    .. image:: https://user-images.githubusercontent.com/32848391/48198460-3aa0a080-e359-11e8-982d-23fadf4de66f.jpg
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
    mapper = lact.GetMapper()
    mapper.SetScalarRange(img.GetScalarRange())
    return lact


def probePlane(img, origin=(0, 0, 0), normal=(1, 0, 0)):
    '''
    Takes a vtkImageData and probes its scalars on a plane.

    `probePlane.py`_
    
    .. _probePlane.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/probePlane.py

    .. image:: https://user-images.githubusercontent.com/32848391/48198461-3aa0a080-e359-11e8-8c29-18f287f105e6.jpg
    '''
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    planeCut = vtk.vtkCutter()
    planeCut.SetInputData(img)
    planeCut.SetCutFunction(plane)
    planeCut.Update()
    cutActor = Actor(planeCut.GetOutput(), c=None)  # ScalarVisibilityOn
    cutMapper = cutActor.GetMapper()
    cutMapper.SetScalarRange(img.GetPointData().GetScalars().GetRange())
    return cutActor


def imageOperation(image1, operation='+', image2=None):
    '''
    Perform operations with vtkImageData objects. Image2 can contain a constant value.
    Possible operations are: +, -, /, 1/x, sin, cos, exp, log, abs, ``**2``, sqrt, min,
    max, atan, atan2, median, mag, dot, gradient, divergence, laplacian.

    `imageOperations.py`_

    .. _imageOperations.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/imageOperations.py

    .. image:: https://user-images.githubusercontent.com/32848391/48198940-d1ba2800-e35a-11e8-96a7-ffbff797f165.jpg

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


def recoSurface(points, bins=256,
                c='gold', alpha=1, wire=False, bc='t', legend=None):
    '''
    Surface reconstruction from sparse points.

    `recosurface.py`_
    
    .. _recosurface.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/recosurface.py

    .. image:: https://user-images.githubusercontent.com/32848391/46817107-b3263880-cd7e-11e8-985d-f5d158992f0c.png
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
              'points, sample size=', int(N/50))

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
    return Actor(surface.GetOutput(), c, alpha, wire, bc, legend)


def cluster(points, radius, legend=None):
    '''
    Clustering of points in space.
    radius, is the radius of local search.
    Individual subsets can be accessed through actor.clusters

    `clustering.py`_ 
    
    .. _clustering.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/clustering.py

    .. image:: https://user-images.githubusercontent.com/32848391/46817286-2039ce00-cd7f-11e8-8b29-42925e03c974.png
    '''
    if isinstance(points, vtk.vtkActor):
        poly = points.polydata()
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
        acts.append(vs.points(aset, c=i))

    actor = Assembly(acts, legend=legend)

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


def removeOutliers(points, radius, c='k', alpha=1, legend=None):
    '''
    Remove outliers from a cloud of points within radius search

    `clustering.py`_

    .. _clustering.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/clustering.py

    .. image:: https://user-images.githubusercontent.com/32848391/46817286-2039ce00-cd7f-11e8-8b29-42925e03c974.png
    '''
    isactor = False
    if isinstance(points, vtk.vtkActor):
        isactor = True
        poly = points.polydata()
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

    actor = vs.points(outpts, c=c, alpha=alpha, legend=legend)
    return actor  # return same obj for concatenation
