from __future__ import division, print_function
import vtkplotter.docs as docs
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import vtkplotter.utils as utils
import vtkplotter.colors as colors
import vtkplotter.shapes as shapes
from vtkplotter.assembly import Assembly
from vtkplotter.mesh import Mesh
from vtkplotter.volume import Volume

__doc__ = (
    """
Defines methods useful to analyse 3D meshes.
"""
    + docs._defs
)


__all__ = [
    "delaunay2D",
    "normalLines",
    "alignLandmarks",
    "alignICP",
    "alignProcrustes",
    "fitLine",
    "fitPlane",
    "fitSphere",
    "pcaEllipsoid",
    "smoothMLS3D",
    "booleanOperation",
    "surfaceIntersection",
    "probePoints",
    "probeLine",
    "probePlane",
    "resampleArrays",
    "recoSurface",
    "cluster",
    "removeOutliers",
    "pointSampler",
    "geodesic",
    "mesh2Volume",
    "projectSphereToPlane",
    "voronoi3D",
    "connectedPoints",
    "interpolateToVolume",
    "interpolateToStructuredGrid",
    "streamLines",
    "densifyCloud",
    "implicitModeller",
    "signedDistanceFromPointCloud",
    "volumeFromMesh",
    "pointDensity",
    "pointCloudFrom",
    "pointDensity",
    "visiblePoints",
]


def delaunay2D(plist, mode='scipy', tol=None):
    """
    Create a mesh from points in the XY plane.
    If `mode='fit'` then the filter computes a best fitting
    plane and projects the points onto it.

    |delaunay2d| |delaunay2d.py|_
    """
    plist = np.ascontiguousarray(plist)

    if mode == 'scipy':
        try:
            from scipy.spatial import Delaunay as scipy_Delaunay
            tri = scipy_Delaunay(plist[:, 0:2])
            return Mesh([plist, tri.simplices])

        except:
            mode='xy'

    pd = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(np.ascontiguousarray(plist), deep=True))
    pd.SetPoints(vpts)

    if plist.shape[1] == 2: # make it 3d
        plist = np.c_[plist, np.zeros(len(plist))]
    delny = vtk.vtkDelaunay2D()
    delny.SetInputData(pd)
    if tol:
        delny.SetTolerance(tol)

    if mode=='fit':
        delny.SetProjectionPlaneMode(vtk.VTK_BEST_FITTING_PLANE)
    delny.Update()
    return Mesh(delny.GetOutput())

def normalLines(mesh, ratio=1, atCells=True, scale=1):
    """
    Build an ``Mesh`` made of the normals at cells shown as lines.

    if `atCells` is `False` normals are shown at vertices.
    """
    poly = mesh.computeNormals().polydata()

    if atCells:
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
    glyphActor = Mesh(glyph.GetOutput())
    glyphActor.mapper().SetScalarModeToUsePointFieldData()
    glyphActor.PickableOff()
    prop = vtk.vtkProperty()
    prop.DeepCopy(mesh.GetProperty())
    glyphActor.SetProperty(prop)
    return glyphActor


def alignLandmarks(source, target, rigid=False):
    """
    Find best matching of source points towards target
    in the mean least square sense, in one single step.
    """
    lmt = vtk.vtkLandmarkTransform()
    ss = source.polydata().GetPoints()
    st = target.polydata().GetPoints()
    if source.N() != target.N():
        colors.printc('~times Error in alignLandmarks(): Source and Target with != nr of points!',
                  source.N(), target.N(), c=1)
        raise RuntimeError()
    lmt.SetSourceLandmarks(ss)
    lmt.SetTargetLandmarks(st)
    if rigid:
        lmt.SetModeToRigidBody()
    lmt.Update()
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(source.polydata())
    tf.SetTransform(lmt)
    tf.Update()
    mesh = Mesh(tf.GetOutput())
    mesh.transform = lmt
    pr = vtk.vtkProperty()
    pr.DeepCopy(source.GetProperty())
    mesh.SetProperty(pr)
    return mesh


def alignICP(source, target, iters=100, rigid=False):
    """
    Return a copy of source mesh which is aligned to
    target mesh through the `Iterative Closest Point` algorithm.

    The core of the algorithm is to match each vertex in one surface with
    the closest surface point on the other, then apply the transformation
    that modify one surface to best match the other (in the least-square sense).

    .. hint:: |align1.py|_ |align2.py|_

         |align1| |align2|
    """
    prop = None
    if isinstance(source, Mesh):
        prop = vtk.vtkProperty()
        prop.DeepCopy(source.GetProperty())
        source = source.polydata()
    if isinstance(target, Mesh):
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
    mesh = Mesh(poly)
    if prop: mesh.SetProperty(prop)

    # mesh.info['transform'] = icp.GetLandmarkTransform() # not working!
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
    mesh.transform = landmarkTransform

    return mesh


def alignProcrustes(sources, rigid=False):
    """
    Return an ``Assembly`` of aligned source meshes with
    the `Procrustes` algorithm. The output ``Assembly`` is normalized in size.

    `Procrustes` algorithm takes N set of points and aligns them in a least-squares sense
    to their mutual mean. The algorithm is iterated until convergence,
    as the mean must be recomputed after each alignment.

    The set of average points generated by the algorithm can be accessed with
    ``algoutput.info['mean']`` as a numpy array.

    :param bool rigid: if `True` scaling is disabled.

    |align3| |align3.py|_
    """
    group = vtk.vtkMultiBlockDataGroupFilter()
    for source in sources:
        if sources[0].N() != source.N():
            colors.printc("~times Procrustes error in align():", c=1)
            colors.printc(" sources have different nr of points", c=1)
            raise RuntimeError()
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
        mesh = Mesh(poly)
        mesh.SetProperty(s.GetProperty())
        if hasattr(s, 'name'):
            mesh.name = s.name
            mesh.flagText = s.flagText
        acts.append(mesh)
    assem = Assembly(acts)
    assem.transform = procrustes.GetLandmarkTransform()
    assem.info['mean'] = vtk_to_numpy(procrustes.GetMeanPoints().GetData())
    return assem


################################################### working with point clouds
def fitLine(points):
    """
    Fits a line through points.

    Extra info is stored in ``Line.slope``, ``Line.center``, ``Line.variances``.

    |fitline| |fitline.py|_
    """
    if isinstance(points, Mesh):
        points = points.points()
    data = np.array(points)
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    vv = vv[0] / np.linalg.norm(vv[0])
    # vv contains the first principal component, i.e. the direction
    # vector of the best fit line in the least squares sense.
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    a = np.linalg.norm(xyz_min - datamean)
    b = np.linalg.norm(xyz_max - datamean)
    p1 = datamean - a * vv
    p2 = datamean + b * vv
    l = shapes.Line(p1, p2, lw=1)
    l.slope = vv
    l.center = datamean
    l.variances = dd
    return l


def fitPlane(points):
    """
    Fits a plane to a set of points.

    Extra info is stored in ``Plane.normal``, ``Plane.center``, ``Plane.variance``.

    .. hint:: Example: |fitplanes.py|_
    """
    if isinstance(points, Mesh):
        points = points.points()
    data = np.array(points)
    datamean = data.mean(axis=0)
    res = np.linalg.svd(data - datamean)
    dd, vv = res[1], res[2]
    n = np.cross(vv[0], vv[1])
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    s = np.linalg.norm(xyz_max - xyz_min)
    pla = shapes.Plane(datamean, n, s, s)
    pla.normal = n
    pla.center = datamean
    pla.variance = dd[2]
    pla.name = "fitPlane"
    return pla


def fitSphere(coords):
    """
    Fits a sphere to a set of points.

    Extra info is stored in ``Sphere.radius``, ``Sphere.center``, ``Sphere.residue``.

    .. hint:: Example: |fitspheres1.py|_

        |fitspheres2| |fitspheres2.py|_
    """
    if isinstance(coords, Mesh):
        coords = coords.points()
    coords = np.array(coords)
    n = len(coords)
    A = np.zeros((n, 4))
    A[:, :-1] = coords * 2
    A[:, 3] = 1
    f = np.zeros((n, 1))
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    f[:, 0] = x * x + y * y + z * z
    C, residue, rank, sv = np.linalg.lstsq(A, f)  # solve AC=f
    if rank < 4:
        return None
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = np.sqrt(t)[0]
    center = np.array([C[0][0], C[1][0], C[2][0]])
    if len(residue):
        residue = np.sqrt(residue[0]) / n
    else:
        residue = 0
    s = shapes.Sphere(center, radius, c=(1,0,0)).wireframe(1)
    s.radius = radius # used by fitSphere
    s.center = center
    s.residue = residue
    s.name = "fitSphere"
    return s


def pcaEllipsoid(points, pvalue=0.95):
    """
    Show the oriented PCA ellipsoid that contains fraction `pvalue` of points.

    :param float pvalue: ellypsoid will contain the specified fraction of points.

    Extra can be calculated with ``mesh.asphericity()``, ``mesh.asphericity_error()``
    (asphericity is equal to 0 for a perfect sphere).

    Axes can be accessed in ``mesh.va``, ``mesh.vb``, ``mesh.vc``.
    End point of the axes are stored in ``mesh.axis1``, ``mesh.axis12`` and ``mesh.axis3``.

    .. hint:: Examples: |pca.py|_  |cell_colony.py|_

         |pca| |cell_colony|
    """
    from scipy.stats import f

    if isinstance(points, Mesh):
        coords = points.points()
    else:
        coords = points
    if len(coords) < 4:
        colors.printc("Warning in pcaEllipsoid(): not enough points!", c='y')
        return None

    P = np.array(coords, ndmin=2, dtype=float)
    cov = np.cov(P, rowvar=0)     # covariance matrix
    U, s, R = np.linalg.svd(cov)  # singular value decomposition
    p, n = s.size, P.shape[0]
    fppf = f.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p)  # f % point function
    cfac = 1 + 6/(n-1)            # correction factor for low statistics
    ua, ub, uc = np.sqrt(s*fppf)/cfac  # semi-axes (largest first)
    center = np.mean(P, axis=0)   # centroid of the hyperellipsoid

    elli = shapes.Ellipsoid((0,0,0), (1,0,0), (0,1,0), (0,0,1), alpha=0.2)

    matri = vtk.vtkMatrix4x4()
    matri.DeepCopy((R[0][0] * ua*2, R[1][0] * ub*2, R[2][0] * uc*2, center[0],
                    R[0][1] * ua*2, R[1][1] * ub*2, R[2][1] * uc*2, center[1],
                    R[0][2] * ua*2, R[1][2] * ub*2, R[2][2] * uc*2, center[2],
                    0, 0, 0, 1))
    vtra = vtk.vtkTransform()
    vtra.SetMatrix(matri)
    # assign the transformation
    elli.SetScale(vtra.GetScale())
    elli.SetOrientation(vtra.GetOrientation())
    elli.SetPosition(vtra.GetPosition())

    elli.GetProperty().BackfaceCullingOn()

    elli.nr_of_points = n
    elli.va = ua
    elli.vb = ub
    elli.vc = uc
    elli.axis1 = vtra.TransformPoint([1,0,0])
    elli.axis2 = vtra.TransformPoint([0,1,0])
    elli.axis3 = vtra.TransformPoint([0,0,1])
    elli.transformation = vtra
    elli.name = "pcaEllipsoid"
    return elli


def smoothMLS3D(meshs, neighbours=10):
    """
    A time sequence of point clouds (Mesh) is being smoothed in 4D (3D + time)
    using a `MLS (Moving Least Squares)` algorithm variant.
    The time associated to an mesh must be specified in advance with ``mesh.time()`` method.
    Data itself can suggest a meaningful time separation based on the spatial
    distribution of points.

    :param int neighbours: fixed nr. of neighbours in space-time to take into account in the fit.

    |moving_least_squares3D| |moving_least_squares3D.py|_
    """
    from scipy.spatial import KDTree

    coords4d = []
    for a in meshs:  # build the list of 4d coordinates
        coords3d = a.points()
        n = len(coords3d)
        pttimes = [[a.time()]] * n
        coords4d += np.append(coords3d, pttimes, axis=1).tolist()

    avedt = float(meshs[-1].time() - meshs[0].time()) / len(meshs)
    print("Average time separation between meshes dt =", round(avedt, 3))

    coords4d = np.array(coords4d)
    newcoords4d = []
    kd = KDTree(coords4d, leafsize=neighbours)
    suggest = ""

    pb = utils.ProgressBar(0, len(coords4d))
    for i in pb.range():
        mypt = coords4d[i]

        # dr = np.sqrt(3*dx**2+dt**2)
        # iclosest = kd.query_ball_Point(mypt, r=dr)
        # dists, iclosest = kd.query(mypt, k=None, distance_upper_bound=dr)
        dists, iclosest = kd.query(mypt, k=neighbours)
        closest = coords4d[iclosest]

        nc = len(closest)
        if nc >= neighbours and nc > 5:
            m = np.linalg.lstsq(closest, [1.0] * nc)[0]  # needs python3
            vers = m / np.linalg.norm(m)
            hpcenter = np.mean(closest, axis=0)  # hyperplane center
            dist = np.dot(mypt - hpcenter, vers)
            projpt = mypt - dist * vers
            newcoords4d.append(projpt)

            if not i % 1000:  # work out some stats
                v = np.std(closest, axis=0)
                vx = round((v[0] + v[1] + v[2]) / 3, 3)
                suggest = "data suggest dt=" + str(vx)

        pb.print(suggest)
    newcoords4d = np.array(newcoords4d)

    ctimes = newcoords4d[:, 3]
    ccoords3d = np.delete(newcoords4d, 3, axis=1)  # get rid of time
    act = shapes.Points(ccoords3d)
    act.pointColors(ctimes, cmap="jet")  # use a colormap to associate a color to time
    return act


def extractSurface(volume, radius=0.5):
    """Generate the zero-crossing isosurface from truncated signed distance volume in input.
    Output is an ``Mesh`` object.
    """
    img = _getinput(volume)
    fe = vtk.vtkExtractSurface()
    fe.SetInputData(img)
    fe.SetRadius(radius)
    fe.Update()
    return Mesh(fe.GetOutput())


def recoSurface(pts, dims=(250,250,250), radius=None,
                sampleSize=None, holeFilling=True, bounds=(), pad=0.1):
    """
    Surface reconstruction from a scattered cloud of points.

    :param int dims: number of voxels in x, y and z to control precision.
    :param float radius: radius of influence of each point.
        Smaller values generally improve performance markedly.
        Note that after the signed distance function is computed,
        any voxel taking on the value >= radius
        is presumed to be "unseen" or uninitialized.
    :param int sampleSize: if normals are not present
        they will be calculated using this sample size per point.
    :param bool holeFilling: enables hole filling, this generates
        separating surfaces between the empty and unseen portions of the volume.
    :param list bounds: region in space in which to perform the sampling
        in format (xmin,xmax, ymin,ymax, zim, zmax)
    :param float pad: increase by this fraction the bounding box

    |recosurface| |recosurface.py|_
    """
    if not utils.isSequence(dims):
        dims = (dims,dims,dims)

    if isinstance(pts, Mesh):
        polyData = pts.polydata()
    else:
        polyData = shapes.Points(pts).polydata()

    sdf = vtk.vtkSignedDistance()

    if len(bounds)==6:
        sdf.SetBounds(bounds)
    else:
        x0, x1, y0, y1, z0, z1 = polyData.GetBounds()
        sdf.SetBounds(x0-(x1-x0)*pad, x1+(x1-x0)*pad,
                      y0-(y1-y0)*pad, y1+(y1-y0)*pad,
                      z0-(z1-z0)*pad, z1+(z1-z0)*pad)

    if polyData.GetPointData().GetNormals():
        sdf.SetInputData(polyData)
    else:
        normals = vtk.vtkPCANormalEstimation()
        normals.SetInputData(polyData)
        if not sampleSize:
            sampleSize = int(polyData.GetNumberOfPoints()/50)
        normals.SetSampleSize(sampleSize)
        normals.SetNormalOrientationToGraphTraversal()
        sdf.SetInputConnection(normals.GetOutputPort())
        #print("Recalculating normals with sample size =", sampleSize)

    if radius is None:
        b = polyData.GetBounds()
        diagsize = np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)
        radius = diagsize / (sum(dims)/3) * 5
        #print("Calculating mesh from points with radius =", radius)

    sdf.SetRadius(radius)
    sdf.SetDimensions(dims)
    sdf.Update()

    surface = vtk.vtkExtractSurface()
    surface.SetRadius(radius * 0.99)
    surface.SetHoleFilling(holeFilling)
    surface.ComputeNormalsOff()
    surface.ComputeGradientsOff()
    surface.SetInputConnection(sdf.GetOutputPort())
    surface.Update()
    return Mesh(surface.GetOutput())


def cluster(points, radius):
    """
    Clustering of points in space.

    `radius` is the radius of local search.
    Individual subsets can be accessed through ``mesh.clusters``.

    |clustering| |clustering.py|_
    """
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

    idsarr = cluster.GetOutput().GetPointData().GetArray("ClusterId")
    Nc = cluster.GetNumberOfExtractedClusters()

    sets = [[] for i in range(Nc)]
    for i, p in enumerate(points):
        sets[idsarr.GetValue(i)].append(p)

    acts = []
    for i, aset in enumerate(sets):
        acts.append(shapes.Points(aset, c=i))

    asse = Assembly(acts)

    asse.info["clusters"] = sets
    print("Nr. of extracted clusters", Nc)
    if Nc > 10:
        print("First ten:")
    for i in range(Nc):
        if i > 9:
            print("...")
            break
        print("Cluster #" + str(i) + ",  N =", len(sets[i]))
    print("Access individual clusters through attribute: obj.info['cluster']")
    return asse


def removeOutliers(points, radius):
    """
    Remove outliers from a cloud of points within the specified `radius` search.

    |clustering| |clustering.py|_
    """
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

    return shapes.Points(outpts)


def booleanOperation(mesh1, operation, mesh2):
    """Volumetric union, intersection and subtraction of surfaces.

    :param str operation: allowed operations: ``'plus'``, ``'intersect'``, ``'minus'``.

    |boolean| |boolean.py|_
    """
    bf = vtk.vtkBooleanOperationPolyDataFilter()
    poly1 = mesh1.computeNormals().polydata()
    poly2 = mesh2.computeNormals().polydata()
    if operation.lower() == "plus" or operation.lower() == "+":
        bf.SetOperationToUnion()
    elif operation.lower() == "intersect":
        bf.SetOperationToIntersection()
    elif operation.lower() == "minus" or operation.lower() == "-":
        bf.SetOperationToDifference()
    #bf.ReorientDifferenceCellsOn()
    bf.SetInputData(0, poly1)
    bf.SetInputData(1, poly2)
    bf.Update()
    mesh = Mesh(bf.GetOutput(), c=None)
    mesh.name = mesh1.name+operation+mesh2.name
    return mesh


def surfaceIntersection(mesh1, mesh2, tol=1e-06):
    """Intersect 2 surfaces and return a line mesh.

    .. hint:: |surfIntersect.py|_
    """
    bf = vtk.vtkIntersectionPolyDataFilter()
    if isinstance(mesh1, Mesh):
        poly1 = mesh1.polydata()
    else:
        poly1 = mesh1.GetMapper().GetInput()
    if isinstance(mesh2, Mesh):
        poly2 = mesh2.polydata()
    else:
        poly2 = mesh2.GetMapper().GetInput()
    bf.SetInputData(0, poly1)
    bf.SetInputData(1, poly2)
    bf.Update()
    mesh = Mesh(bf.GetOutput(), "k", 1).lighting('off')
    mesh.GetProperty().SetLineWidth(3)
    mesh.name = "surfaceIntersection"
    return mesh


def _getinput(obj):
    if isinstance(obj, (vtk.vtkVolume, vtk.vtkActor)):
        return obj.GetMapper().GetInput()
    else:
        return obj


def probePoints(vol, pts):
    """
    Takes a ``Volume`` (or any other vtk data set)
    and probes its scalars at the specified points in space.

    Note that a mask is also output with valid/invalid points which can be accessed
    with `mesh.getPointArray('vtkValidPointMask')`.
    """
    if isinstance(pts, Mesh):
        pts = pts.points()

    def readPoints():
        output = src.GetPolyDataOutput()
        points = vtk.vtkPoints()
        for p in pts:
            x, y, z = p
            points.InsertNextPoint(x, y, z)
        output.SetPoints(points)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(len(pts))
        for i in range(len(pts)):
            cells.InsertCellPoint(i)
        output.SetVerts(cells)

    src = vtk.vtkProgrammableSource()
    src.SetExecuteMethod(readPoints)
    src.Update()
    img = _getinput(vol)
    probeFilter = vtk.vtkProbeFilter()
    probeFilter.SetSourceData(img)
    probeFilter.SetInputConnection(src.GetOutputPort())
    probeFilter.Update()
    poly = probeFilter.GetOutput()
    pm = Mesh(poly)
    pm.name = 'probePoints'
    return pm

def probeLine(vol, p1, p2, res=100):
    """
    Takes a ``Volume``  (or any other vtk data set)
    and probes its scalars along a line defined by 2 points `p1` and `p2`.

    Note that a mask is also output with valid/invalid points which can be accessed
    with `mesh.getPointArray('vtkValidPointMask')`.

    :param int res: nr of points along the line

    |probeLine1| |probeLine1.py|_ |probeLine2.py|_
    """
    line = vtk.vtkLineSource()
    line.SetResolution(res)
    line.SetPoint1(p1)
    line.SetPoint2(p2)
    img = _getinput(vol)
    probeFilter = vtk.vtkProbeFilter()
    probeFilter.SetSourceData(img)
    probeFilter.SetInputConnection(line.GetOutputPort())
    probeFilter.Update()
    poly = probeFilter.GetOutput()
    lnn = Mesh(poly)
    lnn.name = 'probeLine'
    return lnn

def probePlane(vol, origin=(0, 0, 0), normal=(1, 0, 0)):
    """
    Takes a ``Volume`` (or any other vtk data set)
    and probes its scalars on a plane defined by a point and a normal.
    """
    img = _getinput(vol)
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)
    planeCut = vtk.vtkCutter()
    planeCut.SetInputData(img)
    planeCut.SetCutFunction(plane)
    planeCut.Update()
    poly = planeCut.GetOutput()
    cutmesh = Mesh(poly)
    cutmesh.name = 'probePlane'
    return cutmesh


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


def connectedPoints(mesh, radius, mode=0, regions=(), vrange=(0,1), seeds=(), angle=0):
    """
    Extracts and/or segments points from a point cloud based on geometric distance measures
    (e.g., proximity, normal alignments, etc.) and optional measures such as scalar range.
    The default operation is to segment the points into "connected" regions where the connection
    is determined by an appropriate distance measure. Each region is given a region id.

    Optionally, the filter can output the largest connected region of points; a particular region
    (via id specification); those regions that are seeded using a list of input point ids;
    or the region of points closest to a specified position.

    The key parameter of this filter is the radius defining a sphere around each point which defines
    a local neighborhood: any other points in the local neighborhood are assumed connected to the point.
    Note that the radius is defined in absolute terms.

    Other parameters are used to further qualify what it means to be a neighboring point.
    For example, scalar range and/or point normals can be used to further constrain the neighborhood.
    Also the extraction mode defines how the filter operates.
    By default, all regions are extracted but it is possible to extract particular regions;
    the region closest to a seed point; seeded regions; or the largest region found while processing.
    By default, all regions are extracted.

    On output, all points are labeled with a region number.
    However note that the number of input and output points may not be the same:
    if not extracting all regions then the output size may be less than the input size.

    :param float radius: radius variable specifying a local sphere used to define local point neighborhood
    :param int mode:

        - 0,  Extract all regions
        - 1,  Extract point seeded regions
        - 2,  Extract largest region
        - 3,  Test specified regions
        - 4,  Extract all regions with scalar connectivity
        - 5,  Extract point seeded regions

    :param list regions: a list of non-negative regions id to extract
    :param list vrange: scalar range to use to extract points based on scalar connectivity
    :param list seeds: a list of non-negative point seed ids
    :param list angle: points are connected if the angle between their normals is
        within this angle threshold (expressed in degrees).
    """
    # https://vtk.org/doc/nightly/html/classvtkConnectedPointsFilter.html
    cpf = vtk.vtkConnectedPointsFilter()
    cpf.SetInputData(mesh.polydata())
    cpf.SetRadius(radius)
    if   mode == 0: # Extract all regions
        pass

    elif mode == 1: # Extract point seeded regions
        cpf.SetExtractionModeToPointSeededRegions()
        for s in seeds:
            cpf.AddSeed(s)

    elif mode == 2: # Test largest region
        cpf.SetExtractionModeToLargestRegion()

    elif mode == 3: # Test specified regions
        cpf.SetExtractionModeToSpecifiedRegions()
        for r in regions:
            cpf.AddSpecifiedRegion(r)

    elif mode == 4: # Extract all regions with scalar connectivity
        cpf.SetExtractionModeToLargestRegion()
        cpf.ScalarConnectivityOn()
        cpf.SetScalarRange(vrange[0], vrange[1])

    elif mode == 5: # Extract point seeded regions
        cpf.SetExtractionModeToLargestRegion()
        cpf.ScalarConnectivityOn()
        cpf.SetScalarRange(vrange[0], vrange[1])
        cpf.AlignedNormalsOn()
        cpf.SetNormalAngle(angle)

    cpf.Update()
    m = Mesh(cpf.GetOutput())
    m.name = "connectedPoints"
    return m


def pointSampler(mesh, distance=None):
    """Generate a cloud of points the specified distance apart from the input."""
    poly = mesh.polydata()

    pointSampler = vtk.vtkPolyDataPointSampler()
    if not distance:
        distance = mesh.diagonalSize() / 100.0
    pointSampler.SetDistance(distance)
    #    pointSampler.GenerateVertexPointsOff()
    #    pointSampler.GenerateEdgePointsOff()
    #    pointSampler.GenerateVerticesOn()
    #    pointSampler.GenerateInteriorPointsOn()
    pointSampler.SetInputData(poly)
    pointSampler.Update()

    umesh = Mesh(pointSampler.GetOutput())
    prop = vtk.vtkProperty()
    prop.DeepCopy(mesh.GetProperty())
    umesh.SetProperty(prop)
    umesh.name = 'pointSampler'
    return umesh


def geodesic(mesh, start, end):
    """Dijkstra algorithm to compute the graph geodesic.
    Takes as input a polygonal mesh and performs a single source
    shortest path calculation.

    :param start: start vertex index or close point `[x,y,z]`
    :type start: int, list
    :param end: end vertex index or close point `[x,y,z]`
    :type start: int, list

    |geodesic| |geodesic.py|_
    """

    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()

    if utils.isSequence(start):
        cc = mesh.points()
        pa = shapes.Points(cc)
        start = pa.closestPoint(start, returnIds=True)
        end = pa.closestPoint(end, returnIds=True)
        dijkstra.SetInputData(pa.polydata())
    else:
        dijkstra.SetInputData(mesh.polydata())

    dijkstra.SetStartVertex(start)
    dijkstra.SetEndVertex(end)
    dijkstra.Update()

    weights = vtk.vtkDoubleArray()
    dijkstra.GetCumulativeWeights(weights)

    length = weights.GetMaxId() + 1
    arr = np.zeros(length)
    for i in range(length):
        arr[i] = weights.GetTuple(i)[0]

    dmesh = Mesh(dijkstra.GetOutput())
    prop = vtk.vtkProperty()
    prop.DeepCopy(mesh.GetProperty())
    prop.SetLineWidth(3)
    prop.SetOpacity(1)
    dmesh.SetProperty(prop)
    dmesh.info["CumulativeWeights"] = arr
    dmesh.name = "geodesicLine"
    return dmesh


def mesh2Volume(mesh, spacing=(1, 1, 1)):
    """
    Convert a mesh it into a ``Volume``
    where the foreground (exterior) voxels value is 1 and the background
    (interior) voxels value is 0.
    Internally the ``vtkPolyDataToImageStencil`` class is used.

    |mesh2volume| |mesh2volume.py|_
    """
    # https://vtk.org/Wiki/VTK/Examples/Cxx/PolyData/PolyDataToImageData
    pd = mesh.polydata()

    whiteImage = vtk.vtkImageData()
    bounds = pd.GetBounds()

    whiteImage.SetSpacing(spacing)

    # compute dimensions
    dim = [0, 0, 0]
    for i in [0, 1, 2]:
        dim[i] = int(np.ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i]))
    whiteImage.SetDimensions(dim)
    whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)

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
    return Volume(imgstenc.GetOutput())

def projectSphereToPlane(mesh):
    """
    Project a spherical-like object onto a plane.

    |projectsphere| |projectsphere.py|_
    """
    poly = mesh.polydata()
    psf = vtk.vtkProjectSphereFilter()
    psf.SetInputData(poly)
    psf.Update()
    m = Mesh(psf.GetOutput()).flat()
    m.name = "projectSphereToPlane"
    return m


def voronoi3D(nuclei, bbfactor=1, tol=None):
    """Generate 3D Voronio tasselization with the `Voro++ <http://math.lbl.gov/voro++/>`_ package.

    |voronoi3d| |voronoi3d.py|_
    """
    from vtkplotter import settings
    import os

    # run voro++
    if os.path.isfile(settings.voro_path+'/voro++') or settings.voro_path=='':
        outF = open('voronoi3d.txt', "w")
        for i,p in enumerate(nuclei):
            outF.write(str(i)+' '+str(p[0])+' '+str(p[1])+' '+str(p[2])+'\n')
        outF.close()
        ncl = shapes.Points(nuclei)
        b = np.array(ncl.GetBounds())*bbfactor
        bbstr = str(b[0])+' '+str(b[1])+' '+str(b[2])+' '+str(b[3])+' '+str(b[4])+' '+str(b[5])
        print('Using Voro++ in           :', settings.voro_path)
        os.system(settings.voro_path+'/voro++ -c "%F %v %w %P %t" -v '+bbstr+' voronoi3d.txt')
        f = open('voronoi3d.txt.vol', "r")
        lines = f.readlines()
        f.close()
    else:
        print('Cannot find Voro++ installation in:', settings.voro_path)
        print('Download and install Voro++ from http://math.lbl.gov/voro++/download')
        print('Then add:')
        print('from vtkplotter import settings"')
        print('settings.voro_path="path_to_voro++_executable"')
        raise RuntimeError()

    # build polydata
    sourcePoints = vtk.vtkPoints()
    sourcePolygons = vtk.vtkCellArray()
    cells, areas, volumes = [], [], []
    for l in lines: # each line corresponds to an input point
        ls = l.split()
        area = float(ls[0])
        volu = float(ls[1])
        n = int(ls[2])
        ids = []
        for i in range(3, n+3):
            p = tuple(map(float, ls[i][1:-1].split(',')))
            aid = sourcePoints.InsertNextPoint(p[0], p[1], p[2])
            if tol:
                bp = np.array([p[0]-b[0], p[0]-b[1],
                               p[1]-b[2], p[1]-b[3],
                               p[2]-b[4], p[2]-b[5]])
                bp = np.abs(bp) < tol
                if np.any(bp):
                    ids.append(None)
                else:
                    ids.append(aid)
            else:
                ids.append(aid)

        # fill polygon elements
        if None in ids:
            continue

        faces = []
        for j in range(n+3, len(ls)):
            face = tuple(map(int, ls[j][1:-1].split(',')))
            ele = vtk.vtkPolygon()
            ele.GetPointIds().SetNumberOfIds(len(face))
            elems = []
            for k,f in enumerate(face):
                ele.GetPointIds().SetId(k, ids[f])
                elems.append(ids[f])
            sourcePolygons.InsertNextCell(ele)
            faces.append(elems)
        cells.append(faces)
        areas.append(area)
        volumes.append(volu)

    poly = vtk.vtkPolyData()
    poly.SetPoints(sourcePoints)
    poly.SetPolys(sourcePolygons)
    voro = Mesh(poly).alpha(0.5)
    voro.info['cells'] = cells
    voro.info['areas'] = areas
    voro.info['volumes'] = volumes
    return voro



def pointCloudFrom(obj, useCellData=False):
    """Build a `Mesh` object from any VTK dataset as a point cloud.

    :param bool useCellData: if True cell data is interpolated at point positions.
    """
    from vtk.numpy_interface import dataset_adapter

    if useCellData:
        c2p = vtk.vtkCellDataToPointData()
        c2p.SetInputData(obj)
        c2p.Update()
        obj = c2p.GetOutput()

    wrapped = dataset_adapter.WrapDataObject(obj)
    ptdatanames = wrapped.PointData.keys()

    vpts = obj.GetPoints()
    poly = vtk.vtkPolyData()
    poly.SetPoints(vpts)

    for name in ptdatanames:
        arr = obj.GetPointData().GetArray(name)
        poly.GetPointData().AddArray(arr)

    m = Mesh(poly, c=None)
    m.name = "pointCloud"
    return m


def interpolateToVolume(mesh, kernel='shepard', radius=None,
                       bounds=None, nullValue=None,
                       dims=(20,20,20)):
    """
    Generate a ``Volume`` by interpolating a scalar
    or vector field which is only known on a scattered set of points or mesh.
    Available interpolation kernels are: shepard, gaussian, voronoi, linear.

    :param str kernel: interpolation kernel type [shepard]
    :param float radius: radius of the local search
    :param list bounds: bounding box of the output Volume object
    :param list dims: dimensions of the output Volume object
    :param float nullValue: value to be assigned to invalid points

    |interpolateVolume| |interpolateVolume.py|_
    """
    if isinstance(mesh, vtk.vtkPolyData):
        output = mesh
    else:
        output = mesh.polydata()

    # Create a probe volume
    probe = vtk.vtkImageData()
    probe.SetDimensions(dims)
    if bounds is None:
        bounds = output.GetBounds()
    probe.SetOrigin(bounds[0],bounds[2],bounds[4])
    probe.SetSpacing((bounds[1]-bounds[0])/dims[0],
                     (bounds[3]-bounds[2])/dims[1],
                     (bounds[5]-bounds[4])/dims[2])

    if radius is None:
        radius = min(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])/3
#    print(radius,bounds,dims)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(output)
    locator.BuildLocator()

    if kernel == 'shepard':
        kern = vtk.vtkShepardKernel()
        kern.SetPowerParameter(2)
        kern.SetRadius(radius)
    elif kernel == 'gaussian':
        kern = vtk.vtkGaussianKernel()
        kern.SetRadius(radius)
    elif kernel == 'voronoi':
        kern = vtk.vtkVoronoiKernel()
    elif kernel == 'linear':
        kern = vtk.vtkLinearKernel()
        kern.SetRadius(radius)
    else:
        print('Error in interpolateToVolume, available kernels are:')
        print(' [shepard, gaussian, voronoi, linear]')
        raise RuntimeError()

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(probe)
    interpolator.SetSourceData(output)
    interpolator.SetKernel(kern)
    interpolator.SetLocator(locator)
    if nullValue is not None:
        interpolator.SetNullValue(nullValue)
    else:
        interpolator.SetNullPointsStrategyToClosestPoint()
    interpolator.Update()
    return Volume(interpolator.GetOutput())


def interpolateToStructuredGrid(mesh, kernel=None, radius=None,
                               bounds=None, nullValue=None, dims=None):
    """
    Generate a volumetric dataset (vtkStructuredData) by interpolating a scalar
    or vector field which is only known on a scattered set of points or mesh.
    Available interpolation kernels are: shepard, gaussian, voronoi, linear.

    :param str kernel: interpolation kernel type [shepard]
    :param float radius: radius of the local search
    :param list bounds: bounding box of the output vtkStructuredGrid object
    :param list dims: dimensions of the output vtkStructuredGrid object
    :param float nullValue: value to be assigned to invalid points
    """
    if isinstance(mesh, vtk.vtkPolyData):
        output = mesh
    else:
        output = mesh.polydata()

    if dims is None:
        dims = (20,20,20)

    if bounds is None:
        bounds = output.GetBounds()

    # Create a probe volume
    probe = vtk.vtkStructuredGrid()
    probe.SetDimensions(dims)

    points = vtk.vtkPoints()
    points.Allocate(dims[0] * dims[1] * dims[2])
    deltaZ = (bounds[5]-bounds[4]) / (dims[2] - 1)
    deltaY = (bounds[3]-bounds[2]) / (dims[1] - 1)
    deltaX = (bounds[1]-bounds[0]) / (dims[0] - 1)
    for k in range(dims[2]):
        z = bounds[4] + k * deltaZ
        kOffset = k * dims[0] * dims[1]
        for j in range(dims[1]):
            y = bounds[2] + j * deltaY
            jOffset = j * dims[0]
            for i  in range(dims[0]):
                x = bounds[0] + i * deltaX
                offset = i + jOffset + kOffset
                points.InsertPoint(offset, [x,y,z])
    probe.SetPoints(points)

    if radius is None:
        radius = min(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])/3

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(output)
    locator.BuildLocator()

    if kernel == 'gaussian':
        kern = vtk.vtkGaussianKernel()
        kern.SetRadius(radius)
    elif kernel == 'voronoi':
        kern = vtk.vtkVoronoiKernel()
    elif kernel == 'linear':
        kern = vtk.vtkLinearKernel()
        kern.SetRadius(radius)
    else:
        kern = vtk.vtkShepardKernel()
        kern.SetPowerParameter(2)
        kern.SetRadius(radius)

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(probe)
    interpolator.SetSourceData(output)
    interpolator.SetKernel(kern)
    interpolator.SetLocator(locator)
    if nullValue is not None:
        interpolator.SetNullValue(nullValue)
    else:
        interpolator.SetNullPointsStrategyToClosestPoint()
    interpolator.Update()
    return interpolator.GetOutput()


def streamLines(domain, probe,
                activeVectors='',
                integrator='rk4',
                direction='forward',
                initialStepSize=None,
                maxPropagation=None,
                maxSteps=10000,
                stepLength=None,
                extrapolateToBoundingBox=(),
                surfaceConstrain=False,
                computeVorticity=True,
                ribbons=None,
                tubes={},
                scalarRange=None,
                lw=None,
    ):
    """
    Integrate a vector field on a domain (a Mesh or other vtk datasets types)
    to generate streamlines.

    The integration is performed using a specified integrator (Runge-Kutta).
    The length of a streamline is governed by specifying a maximum value either
    in physical arc length or in (local) cell length.
    Otherwise, the integration terminates upon exiting the field domain.

    :param domain: the vtk object that contains the vector field
    :param str activeVectors: name of the vector array
    :param Mesh,list probe: the Mesh that probes the domain. Its coordinates will
        be the seeds for the streamlines, can also be an array of positions.
    :param str integrator: Runge-Kutta integrator, either 'rk2', 'rk4' of 'rk45'
    :param float initialStepSize: initial step size of integration
    :param float maxPropagation: maximum physical length of the streamline
    :param int maxSteps: maximum nr of steps allowed
    :param float stepLength: length of step integration.
    :param dict extrapolateToBoundingBox:
        Vectors defined on a surface are extrapolated to the entire volume defined by its bounding box

        - kernel, (str) - interpolation kernel type [shepard]
        - radius (float)- radius of the local search
        - bounds, (list) - bounding box of the output Volume
        - dims, (list) - dimensions of the output Volume object
        - nullValue, (float) - value to be assigned to invalid points

    :param bool surfaceConstrain: force streamlines to be computed on a surface
    :param bool computeVorticity: Turn on/off vorticity computation at streamline points
        (necessary for generating proper stream-ribbons)
    :param int ribbons: render lines as ribbons by joining them.
        An integer value represent the ratio of joining (e.g.: ribbons=2 groups lines 2 by 2)
    :param dict tubes: dictionary containing the parameters for the tube representation:

            - ratio, (int) - draws tube as longitudinal stripes
            - res, (int) - tube resolution (nr. of sides, 12 by default)
            - maxRadiusFactor (float) - max tube radius as a multiple of the min radius
            - varyRadius, (int) - radius varies based on the scalar or vector magnitude:

                - 0 - do not vary radius
                - 1 - vary radius by scalar
                - 2 - vary radius by vector
                - 3 - vary radius by absolute value of scalar

    :param list scalarRange: specify the scalar range for coloring

    .. hint:: Examples: |streamlines1.py|_ |streamribbons.py|_ |office.py|_ |streamlines2.py|_

        |streamlines2| |office| |streamribbons| |streamlines1|
    """

    if isinstance(domain, vtk.vtkActor):
        if len(extrapolateToBoundingBox):
            grid = interpolateToStructuredGrid(domain, **extrapolateToBoundingBox)
        else:
            grid = domain.polydata()
    else:
        grid = domain

    if activeVectors:
        grid.GetPointData().SetActiveVectors(activeVectors)

    b = grid.GetBounds()
    size = (b[5]-b[4] + b[3]-b[2] + b[1]-b[0])/3
    if initialStepSize is None:
        initialStepSize = size/100.
    if maxPropagation is None:
        maxPropagation = size

    if utils.isSequence(probe):
        pts = np.array(probe)
        if pts.shape[1] == 2: # make it 3d
            pts = np.c_[pts, np.zeros(len(pts))]
    else:
        pts = probe.clean().points()
    src = vtk.vtkProgrammableSource()
    def readPoints():
        output = src.GetPolyDataOutput()
        points = vtk.vtkPoints()
        for x, y, z in pts:
            points.InsertNextPoint(x, y, z)
        output.SetPoints(points)
    src.SetExecuteMethod(readPoints)
    src.Update()

    st = vtk.vtkStreamTracer()
    st.SetInputDataObject(grid)
    st.SetSourceConnection(src.GetOutputPort())

    st.SetInitialIntegrationStep(initialStepSize)
    st.SetComputeVorticity(computeVorticity)
    st.SetMaximumNumberOfSteps(maxSteps)
    st.SetMaximumPropagation(maxPropagation)
    st.SetSurfaceStreamlines(surfaceConstrain)
    if stepLength:
        st.SetMaximumIntegrationStep(stepLength)

    if 'f' in direction:
        st.SetIntegrationDirectionToForward()
    elif 'back' in direction:
        st.SetIntegrationDirectionToBackward()
    elif 'both' in direction:
        st.SetIntegrationDirectionToBoth()

    if integrator == 'rk2':
        st.SetIntegratorTypeToRungeKutta2()
    elif integrator == 'rk4':
        st.SetIntegratorTypeToRungeKutta4()
    elif integrator == 'rk45':
        st.SetIntegratorTypeToRungeKutta45()
    else:
        colors.printc("Error in streamlines, unknown integrator", integrator, c=1)

    st.Update()
    output = st.GetOutput()

    if ribbons:
        scalarSurface = vtk.vtkRuledSurfaceFilter()
        scalarSurface.SetInputConnection(st.GetOutputPort())
        scalarSurface.SetOnRatio(int(ribbons))
        scalarSurface.SetRuledModeToPointWalk()
        scalarSurface.Update()
        output = scalarSurface.GetOutput()

    if len(tubes):
        streamTube = vtk.vtkTubeFilter()
        streamTube.SetNumberOfSides(12)
        streamTube.SetRadius(tubes['radius'])

        if 'res' in tubes:
            streamTube.SetNumberOfSides(tubes['res'])

        # max tube radius as a multiple of the min radius
        streamTube.SetRadiusFactor(50)
        if 'maxRadiusFactor' in tubes:
            streamTube.SetRadius(tubes['maxRadiusFactor'])

        if 'ratio' in tubes:
            streamTube.SetOnRatio(int(tubes['ratio']))

        if 'varyRadius' in tubes:
            streamTube.SetVaryRadius(int(tubes['varyRadius']))

        streamTube.SetInputData(output)
        vname = grid.GetPointData().GetVectors().GetName()
        streamTube.SetInputArrayToProcess(1, 0, 0,
                                          vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                          vname)
        streamTube.Update()
        sta = Mesh(streamTube.GetOutput(), c=None)

        scals = grid.GetPointData().GetScalars()
        if scals:
            sta.mapper().SetScalarRange(scals.GetRange())
        if scalarRange is not None:
            sta.mapper().SetScalarRange(scalarRange)

        sta.GetProperty().BackfaceCullingOn()
        sta.phong()
        return sta

    sta = Mesh(output, c=None)

    if lw is not None and len(tubes)==0 and not ribbons:
        sta.lw(lw)
        sta._mapper.SetResolveCoincidentTopologyToPolygonOffset()
        sta.lighting('off')

    scals = grid.GetPointData().GetScalars()
    if scals:
        sta.mapper().SetScalarRange(scals.GetRange())
    if scalarRange is not None:
        sta.mapper().SetScalarRange(scalarRange)
    return sta


def densifyCloud(mesh, targetDistance, closestN=6, radius=0, maxIter=None, maxN=None):
    """Adds new points to an input point cloud.
    The new points are created in such a way that all points in any local neighborhood are
    within a target distance of one another.

    The algorithm works as follows. For each input point, the distance to all points
    in its neighborhood is computed. If any of its neighbors is further than the target distance,
    the edge connecting the point and its neighbor is bisected and a new point is inserted at the
    bisection point. A single pass is completed once all the input points are visited.
    Then the process repeats to the limit of the maximum number of iterations.

    .. note:: Points will be created in an iterative fashion until all points in their
        local neighborhood are the target distance apart or less.
        Note that the process may terminate early due to the limit on the
        maximum number of iterations. By default the target distance is set to 0.5.
        Note that the TargetDistance should be less than the Radius or nothing will change on output.

    .. warning:: This class can generate a lot of points very quickly.
        The maximum number of iterations is by default set to =1.0 for this reason.
        Increase the number of iterations very carefully.
        Also, `maxN` can be set to limit the explosion of points.
        It is also recommended that a N closest neighborhood is used.
    """
    src = vtk.vtkProgrammableSource()
    def readPoints():
        output = src.GetPolyDataOutput()
        points = vtk.vtkPoints()
        pts = mesh.points()
        for p in pts:
            x, y, z = p
            points.InsertNextPoint(x, y, z)
        output.SetPoints(points)
    src.SetExecuteMethod(readPoints)

    dens = vtk.vtkDensifyPointCloudFilter()
    dens.SetInputConnection(src.GetOutputPort())
    dens.InterpolateAttributeDataOn()
    dens.SetTargetDistance(targetDistance)
    if maxIter: dens.SetMaximumNumberOfIterations(maxIter)
    if maxN: dens.SetMaximumNumberOfPoints(maxN)

    if radius:
        dens.SetNeighborhoodTypeToRadius()
        dens.SetRadius(radius)
    elif closestN:
        dens.SetNeighborhoodTypeToNClosest()
        dens.SetNumberOfClosestPoints(closestN)
    else:
        colors.printc("Error in densifyCloud: set either radius or closestN", c=1)
        raise RuntimeError()
    dens.Update()
    pts = vtk_to_numpy(dens.GetOutput().GetPoints().GetData())
    cld = shapes.Points(pts, c=None).pointSize(3)
    cld.name = "densifiedCloud"
    return cld


def implicitModeller(mesh, distance=0.05, res=(110,40,20), bounds=(), maxdist=None, outer=True):
    """Finds the surface at the specified distance from the input one"""

    if not len(bounds):
        bounds = mesh.bounds()

    if not maxdist:
        maxdist = mesh.diagonalSize()/2

    imp = vtk.vtkImplicitModeller()
    imp.SetInputData(mesh.polydata())
    imp.SetSampleDimensions(res)
    imp.SetMaximumDistance(maxdist)
    imp.SetModelBounds(bounds)
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(imp.GetOutputPort())
    contour.SetValue(0, distance)
    contour.Update()
    poly = contour.GetOutput()
    if outer:
        return Mesh(poly).extractLargestRegion().c('lb')
    return Mesh(poly, c='lb')


def signedDistanceFromPointCloud(mesh, maxradius=None, bounds=None, dims=(20,20,20)):
    """
    Compute signed distances over a volume from an input point cloud.
    The output is a ``Volume`` object whose voxels contains the signed distance from
    the cloud.

    :param float maxradius: how far out to propagate distance calculation
    :param list bounds: volume bounds.
    :param list dims: dimensions (nr. of voxels) of the output volume.
    """
    if bounds is None:
        bounds = mesh.GetBounds()
    if maxradius is None:
        maxradius = mesh.diagonalSize()/10.
    dist = vtk.vtkSignedDistance()
    dist.SetInputData(mesh.polydata(True))
    dist.SetRadius(maxradius)
    dist.SetBounds(bounds)
    dist.SetDimensions(dims)
    dist.Update()
    vol = Volume(dist.GetOutput())
    vol.name = "signedDistanceVolume"
    return vol


def volumeFromMesh(mesh, bounds=None, dims=(20,20,20), signed=True, negate=False):
    """
    Compute signed distances over a volume from an input mesh.
    The output is a ``Volume`` object whose voxels contains the signed distance from
    the mesh.

    :param list bounds: bounds of the output volume.
    :param list dims: dimensions (nr. of voxels) of the output volume.

    See example script: |volumeFromMesh.py|_
    """
    if bounds is None:
        bounds = mesh.GetBounds()
    sx = (bounds[1]-bounds[0])/dims[0]
    sy = (bounds[3]-bounds[2])/dims[1]
    sz = (bounds[5]-bounds[4])/dims[2]

    img = vtk.vtkImageData()
    img.SetDimensions(dims)
    img.SetSpacing(sx, sy, sz)
    img.SetOrigin(bounds[0], bounds[2], bounds[4])
    img.AllocateScalars(vtk.VTK_FLOAT, 1)

    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh.polydata())
    b4 = bounds[4]
    r2 = range(dims[2])

    for i in range(dims[0]):
        x = i*sx+bounds[0]
        for j in range(dims[1]):
            y = j*sy+bounds[2]
            for k in r2:
                v = imp.EvaluateFunction((x, y, k*sz+b4))
                if signed:
                    if negate:
                        v = -v
                else:
                    v = abs(v)
                img.SetScalarComponentFromFloat(i,j,k, 0, v)

    vol = Volume(img)
    vol.name = "VolumeFromMesh"
    return vol


def pointDensity(mesh, dims=(40,40,40), bounds=None, radius=None, computeGradient=False):
    """Generate a density field from a point cloud. Output is a ``Volume``.
    The local neighborhood is specified as a `radius` around each sample position (each voxel).
    The density is normalized to the upper value of the scalar range.

    See example script: |pointDensity.py|_
    """
    if not utils.isSequence(dims):
        dims = (dims,dims,dims)
    pdf = vtk.vtkPointDensityFilter()
    pdf.SetInputData(mesh.polydata())
    pdf.SetSampleDimensions(dims)
    pdf.SetDensityEstimateToFixedRadius()
    pdf.SetDensityFormToVolumeNormalized()
    pdf.SetDensityFormToNumberOfPoints ()
    if radius is None:
        radius = mesh.diagonalSize()/20
    pdf.SetRadius(radius)
    pdf.SetComputeGradient(computeGradient)
    if bounds is None:
        bounds = mesh.GetBounds()
    pdf.SetModelBounds(bounds)
    pdf.Update()
    img = pdf.GetOutput()
    vol = Volume(img)
    vol.name = "PointDensity"
    return vol


def visiblePoints(mesh, area=(), tol=None, invert=False):
    """Extract points based on whether they are visible or not.
    Visibility is determined by accessing the z-buffer of a rendering window.
    The position of each input point is converted into display coordinates,
    and then the z-value at that point is obtained.
    If within the user-specified tolerance, the point is considered visible.
    Associated data attributes are passed to the output as well.

    This filter also allows you to specify a rectangular window in display (pixel)
    coordinates in which the visible points must lie.

    :param list area: specify a rectangular region as (xmin,xmax,ymin,ymax)
    :param float tol: a tolerance in normalized display coordinate system
    :param bool invert: select invisible points instead.

    :Example:
        .. code-block:: python

            from vtkplotter import Ellipsoid, show, visiblePoints

            s = Ellipsoid().rotateY(30)

            #Camera options: pos, focalPoint, viewup, distance,
            # clippingRange, parallelScale, thickness, viewAngle
            camopts = dict(pos=(0,0,25), focalPoint=(0,0,0))
            show(s, camera=camopts, offscreen=True)

            m = visiblePoints(s)
            #print('visible pts:', m.points()) # numpy array
            show(m, newPlotter=True, axes=1)   # optionally draw result
    """
    # specify a rectangular region
    from vtkplotter import settings
    svp = vtk.vtkSelectVisiblePoints()
    svp.SetInputData(mesh.polydata())
    svp.SetRenderer(settings.plotter_instance.renderer)

    if len(area)==4:
        svp.SetSelection(area[0],area[1],area[2],area[3])
    if tol is not None:
        svp.SetTolerance(tol)
    if invert:
        svp.SelectInvisibleOn()
    svp.Update()

    m = Mesh(svp.GetOutput()).pointSize(5)
    m.name = "VisiblePoints"
    return m





