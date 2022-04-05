#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from deprecated import deprecated
import numpy as np
import vtk
import vedo
import vedo.colors as colors
import vedo.utils as utils
from vedo import settings
from vedo.base import BaseActor


__doc__ = """
Submodule to work with point clouds <br>
.. image:: https://vedo.embl.es/images/basic/pca.png
"""

__all__ = [
    "Points",
    "Point",
    "visiblePoints",
    "delaunay2D",
    "voronoi",
    "fitLine",
    "fitCircle",
    "fitPlane",
    "fitSphere",
    # "pcaEllipse",
    "pcaEllipsoid",
    "recoSurface",
]


def recoSurface(pts, dims=(100,100,100), radius=None,
                sampleSize=None, holeFilling=True, bounds=(), pad=0.1):
    """Please use `points.reconstructSurface()` instead."""
    colors.printc("Please use `points.reconstructSurface()` instead. Abort.", c='r')
    raise RuntimeError


def visiblePoints(mesh, area=(), tol=None, invert=False):
    """
    Extract points based on whether they are visible or not.
    Visibility is determined by accessing the z-buffer of a rendering window.
    The position of each input point is converted into display coordinates,
    and then the z-value at that point is obtained.
    If within the user-specified tolerance, the point is considered visible.
    Associated data attributes are passed to the output as well.

    This filter also allows you to specify a rectangular window in display (pixel)
    coordinates in which the visible points must lie.

    Parameters
    ----------
    area : list
        specify a rectangular region as (xmin,xmax,ymin,ymax)

    tol : float
        a tolerance in normalized display coordinate system

    invert : bool
        select invisible points instead.

    Example:
        .. code-block:: python

            from vedo import Ellipsoid, show, visiblePoints

            s = Ellipsoid().rotateY(30)

            #Camera options: pos, focalPoint, viewup, distance,
            # clippingRange, parallelScale, thickness, viewAngle
            camopts = dict(pos=(0,0,25), focalPoint=(0,0,0))
            show(s, camera=camopts, offscreen=True)

            m = visiblePoints(s)
            #print('visible pts:', m.points()) # numpy array
            show(m, new=True, axes=1) # optionally draw result on a new window
    """
    # specify a rectangular region
    svp = vtk.vtkSelectVisiblePoints()
    svp.SetInputData(mesh.polydata())
    svp.SetRenderer(vedo.plotter_instance.renderer)

    if len(area)==4:
        svp.SetSelection(area[0],area[1],area[2],area[3])
    if tol is not None:
        svp.SetTolerance(tol)
    if invert:
        svp.SelectInvisibleOn()
    svp.Update()

    m = Points(svp.GetOutput()).pointSize(5)
    m.name = "VisiblePoints"
    return m


def delaunay2D(plist, mode='scipy', boundaries=(), tol=None, alpha=0, offset=0, transform=None):
    """
    Create a mesh from points in the XY plane.
    If `mode='fit'` then the filter computes a best fitting
    plane and projects the points onto it.
    If `mode='fit'` then the xy plane is assumed.

    When mode=='fit' or 'xy'

    Parameters
    ----------
    tol : float
        specify a tolerance to control discarding of closely spaced points.
        This tolerance is specified as a fraction of the diagonal length of the bounding box of the points.

    alpha : float
        for a non-zero alpha value, only edges or triangles contained
        within a sphere centered at mesh vertices will be output.
        Otherwise, only triangles will be output.

    offset : float
        multiplier to control the size of the initial, bounding Delaunay triangulation.

    transform: vtkTransform
        a VTK transformation (eg. a thinplate spline)
        which is applied to points to generate a 2D problem.
        This maps a 3D dataset into a 2D dataset where triangulation can be done on the XY plane.
        The points are transformed and triangulated.
        The topology of triangulated points is used as the output topology.

    .. hint:: examples/basic/delaunay2d.py
        .. image:: https://vedo.embl.es/images/basic/delaunay2d.png
    """
    if isinstance(plist, Points):
        plist = plist.points()
    else:
        plist = np.ascontiguousarray(plist)
        if plist.shape[1] == 2: # make it 3d
            plist = np.c_[plist, np.zeros(len(plist))]

    #############################################
    if mode == 'scipy':
        from scipy.spatial import Delaunay as scipy_Delaunay
        tri = scipy_Delaunay(plist[:, 0:2])
        return vedo.mesh.Mesh([plist, tri.simplices])
    #############################################

    pd = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    vpts.SetData(utils.numpy2vtk(plist, dtype=float))
    pd.SetPoints(vpts)

    delny = vtk.vtkDelaunay2D()
    delny.SetInputData(pd)
    if tol:
        delny.SetTolerance(tol)
    delny.SetAlpha(alpha)
    delny.SetOffset(offset)
    if transform:
        if hasattr(transform, "transform"):
            transform = transform.transform
        delny.SetTransform(transform)

    if mode=='xy' and len(boundaries):
        boundary = vtk.vtkPolyData()
        boundary.SetPoints(vpts)
        aCellArray = vtk.vtkCellArray()
        for b in boundaries:
            cPolygon = vtk.vtkPolygon()
            for idd in b:
                cPolygon.GetPointIds().InsertNextId(idd)
            aCellArray.InsertNextCell(cPolygon)
        boundary.SetPolys(aCellArray)
        delny.SetSourceData(boundary)

    if mode=='fit':
        delny.SetProjectionPlaneMode(vtk.VTK_BEST_FITTING_PLANE)
    delny.Update()
    return vedo.mesh.Mesh(delny.GetOutput()).clean().lighting('off')

def voronoi(pts, padding=0, fit=False, method='vtk'):
    """
    Generate the 2D Voronoi convex tiling of the input points (z is ignored).
    The points are assumed to lie in a plane. The output is a Mesh. Each output cell is a convex polygon.

    The 2D Voronoi tessellation is a tiling of space, where each Voronoi tile represents the region nearest
    to one of the input points. Voronoi tessellations are important in computational geometry
    (and many other fields), and are the dual of Delaunay triangulations.

    Thus the triangulation is constructed in the x-y plane, and the z coordinate is ignored
    (although carried through to the output).
    If you desire to triangulate in a different plane, you can use fit=True.

    A brief summary is as follows. Each (generating) input point is associated with
    an initial Voronoi tile, which is simply the bounding box of the point set.
    A locator is then used to identify nearby points: each neighbor in turn generates a
    clipping line positioned halfway between the generating point and the neighboring point,
    and orthogonal to the line connecting them. Clips are readily performed by evaluationg the
    vertices of the convex Voronoi tile as being on either side (inside,outside) of the clip line.
    If two intersections of the Voronoi tile are found, the portion of the tile "outside" the clip
    line is discarded, resulting in a new convex, Voronoi tile. As each clip occurs,
    the Voronoi "Flower" error metric (the union of error spheres) is compared to the extent of the region
    containing the neighboring clip points. The clip region (along with the points contained in it) is grown
    by careful expansion (e.g., outward spiraling iterator over all candidate clip points).
    When the Voronoi Flower is contained within the clip region, the algorithm terminates and the Voronoi
    tile is output. Once complete, it is possible to construct the Delaunay triangulation from the Voronoi
    tessellation. Note that topological and geometric information is used to generate a valid triangulation
    (e.g., merging points and validating topology).

    Parameters
    ----------
    pts : list
        list of input points.

    padding : float
        padding distance. The default is 0.

    fit : bool
        detect automatically the best fitting plane. The default is False.

    .. hint:: examples/basic/voronoi1.py, voronoy2.py
        .. image:: https://vedo.embl.es/images/basic/voronoi1.png
    """
    if method=='scipy':
        from scipy.spatial import Voronoi as scipy_voronoi
        pts = np.asarray(pts)[:,(0,1)]
        vor = scipy_voronoi(pts)
        regs = [] # filter out invalid indices
        for r in vor.regions:
            flag=True
            for x in r:
                if x < 0:
                    flag=False
                    break
            if flag and len(r):
                regs.append(r)

        m = vedo.Mesh([vor.vertices, regs], c='orange5')
        m.celldata['VoronoiID'] = np.array(list(range(len(regs)))).astype(int)
        m.locator = None

    elif method=='vtk':
        vor = vtk.vtkVoronoi2D()
        if isinstance(pts, Points):
            vor.SetInputData(pts.polydata())
        else:
            pts = np.asarray(pts)
            if pts.shape[1] == 2:
                pts = np.c_[pts, np.zeros(len(pts))]
            pd = vtk.vtkPolyData()
            vpts = vtk.vtkPoints()
            vpts.SetData(utils.numpy2vtk(pts, dtype=float))
            pd.SetPoints(vpts)
            vor.SetInputData(pd)
        vor.SetPadding(padding)
        vor.SetGenerateScalarsToPointIds()
        if fit:
            vor.SetProjectionPlaneModeToBestFittingPlane()
        else:
            vor.SetProjectionPlaneModeToXYPlane()
        vor.Update()
        poly = vor.GetOutput()
        arr = poly.GetCellData().GetArray(0)
        if arr:
            arr.SetName("VoronoiID")
        m = vedo.Mesh(poly, c='orange5')
        m.locator = vor.GetLocator()

    else:
        vedo.logger.error(f"Unknown method {method} in voronoi()")
        raise RuntimeError

    m.lw(2).lighting('off').wireframe()
    m.name = "Voronoi"
    return m

def _rotatePoints(points, n0=None, n1=(0,0,1)):
    # Rotate a set of 3D points from direction n0 to direction n1.
    # Return the rotated points and the normal to the fitting plane (if n0 is None).
    # The pointing direction of the normal in this case is arbitrary.
    points = np.asarray(points)

    if points.ndim == 1:
        points = points[np.newaxis,:]

    if len(points[0])==2:
        return points, (0,0,1)

    if n0 is None: # fit plane
        datamean = points.mean(axis=0)
        vv = np.linalg.svd(points - datamean)[2]
        n0 = np.cross(vv[0], vv[1])

    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0, n1)
    l = np.linalg.norm(k)
    if not l:
        k = n0
    k /= np.linalg.norm(k)

    ct = np.dot(n0, n1)
    theta = np.arccos(ct)
    st = np.sin(theta)
    v = k * (1-ct)

    rpoints = []
    for p in points:
        a = p * ct
        b = np.cross(k,p) * st
        c = v * np.dot(k,p)
        rpoints.append(a + b + c)

    return np.array(rpoints), n0


def fitLine(points):
    """
    Fits a line through points.

    Extra info is stored in ``Line.slope``, ``Line.center``, ``Line.variances``.

    .. hint:: examples/advanced/fitline.py
        .. image:: https://vedo.embl.es/images/advanced/fitline.png
    """
    if isinstance(points, Points):
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
    l = vedo.shapes.Line(p1, p2, lw=1)
    l.slope = vv
    l.center = datamean
    l.variances = dd
    return l


def fitPlane(points, signed=False):
    """
    Fits a plane to a set of points.

    Extra info is stored in ``Plane.normal``, ``Plane.center``, ``Plane.variance``.

    Parameters
    ----------
    signed : bool
        if True flip sign of the normal based on the ordering of the points

    .. hint:: examples/basic/fitline.py
    """
    if isinstance(points, Points):
        points = points.points()
    data = np.array(points)
    datamean = data.mean(axis=0)
    pts = data - datamean
    res = np.linalg.svd(pts)
    dd, vv = res[1], res[2]
    n = np.cross(vv[0], vv[1])
    if signed:
        v = np.zeros_like(pts)
        for i in range(len(pts)-1):
            vi = np.cross(pts[i],  pts[i+1])
            v[i] = vi/np.linalg.norm(vi)
        ns = np.mean(v, axis=0) # normal to the points plane
        if np.dot(n,ns) < 0:
            n = -n
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    s = np.linalg.norm(xyz_max - xyz_min)
    pla = vedo.shapes.Plane(datamean, n, s=[s,s])
    pla.normal = n
    pla.center = datamean
    pla.variance = dd[2]
    pla.name = "fitPlane"
    return pla


def fitCircle(points):
    """
    Fits a circle through a set of 3D points, with a very fast non-iterative method.

    Returns the tuple `(center, radius, normal_to_circle)`.

    .. warning::
        trying to fit s-shaped points will inevitably lead to instabilities and
        circles of small radius.

    References: *J.F. Crawford, Nucl. Instr. Meth. 211, 1983, 223-225.*
    """
    data = np.asarray(points)

    offs = data.mean(axis=0)
    data, n0 = _rotatePoints(data-offs)

    xi = data[:,0]
    yi = data[:,1]

    x   = sum(xi)
    xi2 = xi*xi
    xx  = sum(xi2)
    xxx = sum(xi2*xi)

    y   = sum(yi)
    yi2 = yi*yi
    yy  = sum(yi2)
    yyy = sum(yi2*yi)

    xiyi = xi*yi
    xy  = sum(xiyi)
    xyy = sum(xiyi*yi)
    xxy = sum(xi*xiyi)

    N = len(xi)
    k = (xx+yy)/N

    a1 = xx-x*x/N
    b1 = xy-x*y/N
    c1 = 0.5*(xxx + xyy - x*k)

    a2 = xy-x*y/N
    b2 = yy-y*y/N
    c2 = 0.5*(xxy + yyy - y*k)

    d = a2*b1 - a1*b2
    if not d:
        return offs, 0, n0
    x0 = (b1*c2 - b2*c1)/d
    y0 = (c1 - a1*x0)/b1

    R = np.sqrt(x0*x0 + y0*y0 -1/N*(2*x0*x +2*y0*y -xx -yy))

    c, _ = _rotatePoints([x0,y0,0], (0,0,1), n0)

    return c[0]+offs, R, n0


def fitSphere(coords):
    """
    Fits a sphere to a set of points.

    Extra info is stored in ``Sphere.radius``, ``Sphere.center``, ``Sphere.residue``.

    .. hint:: examples/basic/fitspheres1.py, fitspheres2.py
        .. image:: https://vedo.embl.es/images/advanced/fitspheres1.jpg
    """
    if isinstance(coords, Points):
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
    s = vedo.shapes.Sphere(center, radius, c=(1,0,0)).wireframe(1)
    s.radius = radius # used by fitSphere
    s.center = center
    s.residue = residue
    s.name = "fitSphere"
    return s


# def pcaEllipse(points, pvalue=0.673):
#     """
#     Show the oriented PCA 2D ellipse that contains fraction `pvalue` of points.

#     Parameter `pvalue` sets the specified fraction of points inside the ellipse.

#     Eigenvalues can be accessed in ``ellips.eigenvalues``,
#     normalized directions are stored in ``ellips.axis1``, ``ellips.axis12``

#     .. hint:: examples/pyplot/histo_pca.py
#     """
#     from scipy.stats import f

#     if isinstance(points, Points):
#         coords = points.points()
#     else:
#         coords = points
#     if len(coords) < 4:
#         vedo.logger.warning("in pcaEllipse(), there are not enough points!")
#         return None

#     data = np.asarray(coords)[:,(0,1)]

#     center = np.mean(data, axis=0)
#     eigenvalues, eigenvectors = np.linalg.eig(np.cov(data.T))

#     ids = eigenvalues.argsort()[::-1]
#     # eigenvalues = eigenvalues[ids]
#     # eigenvectors = eigenvectors[:,ids]
#     eigenvalues_s = np.sqrt(np.abs(eigenvalues))
#     p = 2
#     n = data.shape[0]
#     fppf = f.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p)
#     eigenvalues_s *= np.sqrt(fppf)

#     elli = vedo.shapes.Circle().scale([eigenvalues_s[0], eigenvalues_s[1], 1])
#     angle = np.arctan2(-eigenvectors[0][1], eigenvectors[0][0])
#     elli.rotateZ(angle, rad=True).pos(center)

#     ## quick test
#     # ellibox = elli.z(-.2).extrude(1).triangulate()
#     # inp = ellibox.insidePoints(Points(coords))
#     # print("Points inside:",inp.N())
#     # vedo.show(Points(coords), elli, axes=1).close()

#     elli.center = center
#     elli.eigenvalues = eigenvalues
#     elli.eigenvectors = eigenvectors
#     elli.eigenangle = angle
#     elli.axis1 = eigenvectors[ids][0]
#     elli.axis2 = eigenvectors[ids][1]
#     elli.name = "pcaEllipse"
#     return elli


def pcaEllipsoid(points, pvalue=0.673):
    """
    Show the oriented PCA ellipsoid that contains fraction `pvalue` of points.

    Parameter `pvalue` sets the specified fraction of points inside the ellipsoid.

    Extra can be calculated with ``mesh.asphericity()``, ``mesh.asphericity_error()``
    (asphericity is equal to 0 for a perfect sphere).

    Axes can be accessed in ``ellips.va``, ``ellips.vb``, ``ellips.vc``,
    normalized directions are stored in ``ellips.axis1``, ``ellips.axis12``
    and ``ellips.axis3``.

    .. warning:: the meaning of ``ellips.axis1``, has changed wrt `vedo==2022.1.0`
        in that it's now the direction wrt the origin (e.i. the center is subtracted)

    .. hint:: examples/basic/pca.py, cell_colony.py
    """
    from scipy.stats import f

    if isinstance(points, Points):
        coords = points.points()
    else:
        coords = points
    if len(coords) < 4:
        vedo.logger.warning("in pcaEllipsoid(), there are not enough points!")
        return None

    P = np.array(coords, ndmin=2, dtype=float)
    cov = np.cov(P, rowvar=0)     # covariance matrix
    U, s, R = np.linalg.svd(cov)  # singular value decomposition
    p, n = s.size, P.shape[0]
    fppf = f.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p)  # f % point function
    cfac = 1 + 6/(n-1)            # correction factor for low statistics
    ua, ub, uc = np.sqrt(s*fppf)/cfac  # semi-axes (largest first)
    center = np.mean(P, axis=0)   # centroid of the hyperellipsoid

    elli = vedo.shapes.Ellipsoid((0,0,0), (1,0,0), (0,1,0), (0,0,1), alpha=0.2)
    elli.GetProperty().BackfaceCullingOn()

    matri = vtk.vtkMatrix4x4()
    matri.DeepCopy((
        R[0][0] * ua*2, R[1][0] * ub*2, R[2][0] * uc*2, center[0],
        R[0][1] * ua*2, R[1][1] * ub*2, R[2][1] * uc*2, center[1],
        R[0][2] * ua*2, R[1][2] * ub*2, R[2][2] * uc*2, center[2],
        0, 0, 0, 1)
    )
    vtra = vtk.vtkTransform()
    vtra.SetMatrix(matri)

    # assign the transformation
    elli.SetScale(vtra.GetScale())
    elli.SetOrientation(vtra.GetOrientation())
    elli.SetPosition(vtra.GetPosition())

    elli.center = np.array(vtra.GetPosition())
    elli.nr_of_points = n
    elli.va = ua
    elli.vb = ub
    elli.vc = uc
    elli.axis1 = np.array(vtra.TransformPoint([1,0,0])) - elli.center
    elli.axis2 = np.array(vtra.TransformPoint([0,1,0])) - elli.center
    elli.axis3 = np.array(vtra.TransformPoint([0,0,1])) - elli.center
    elli.axis1 /= np.linalg.norm(elli.axis1)
    elli.axis2 /= np.linalg.norm(elli.axis2)
    elli.axis3 /= np.linalg.norm(elli.axis3)
    elli.transformation = vtra
    elli.name = "pcaEllipsoid"
    return elli


###################################################
def Point(pos=(0, 0, 0), r=12, c="red", alpha=1):
    """
    Create a simple point in space.

    .. note:: if you are creating many points you should definitely use class `Points` instead.
    """
    if isinstance(pos, vtk.vtkActor):
        pos = pos.GetPosition()
    pd = utils.buildPolyData([[0,0,0]])
    if len(pos)==2:
        pos = (pos[0], pos[1], 0.)
    pt = Points(pd, c, alpha, r)
    pt.SetPosition(pos)
    pt.name = "Point"
    return pt

###################################################
class Points(vtk.vtkFollower, BaseActor):
    """
    Build a ``Mesh`` made of only vertex points for a list of 2D/3D points.
    Both shapes (N, 3) or (3, N) are accepted as input, if N>3.
    For very large point clouds a list of colors and alpha can be assigned to each
    point in the form c=[(R,G,B,A), ... ] where 0<=R<256, ... 0<=A<256.

    Parameters
    ----------
    inputobj : list, tuple
        The default is None.

    c : str, list
        Color. The default is (0.2,0.2,0.2).

    alpha : float
        Transparency in range [0,1]. The default is 1.

    r : int
        Point radius in units of pixels. The default is 4.

    Example:
        .. code-block:: python

            import numpy as np
            from vedo import *

            def fibonacci_sphere(n):
                s = np.linspace(0, n, num=n, endpoint=False)
                theta = s * 2.399963229728653
                y = 1 - s * (2/(n-1))
                r = np.sqrt(1 - y * y)
                x = np.cos(theta) * r
                z = np.sin(theta) * r
                return [x,y,z]

            Points(fibonacci_sphere(1000)).show(axes=1)

    More Examples: examples/pyplot/manypoints.py, lorenz.py
    """
    # .. note:: add some note here
    #   and on other line
    def __init__(self,
                 inputobj=None,
                 c=(0.2,0.2,0.2),
                 alpha=1,
                 r=4,
        ):
        vtk.vtkActor.__init__(self)
        BaseActor.__init__(self)

        self._data = None

        self._mapper = vtk.vtkPolyDataMapper()
        self.SetMapper(self._mapper)

        self._scals_idx = 0  # index of the active scalar changed from CLI
        self._ligthingnr = 0 # index of the lighting mode changed from CLI
        self._cmap_name = "" # remember the name for self._keypress
        #self.name = "Points" # better not to give it a name here

        self.property = self.GetProperty()
        try:
            self.property.RenderPointsAsSpheresOn()
        except:
            pass

        if inputobj is None:####################
            self._data = vtk.vtkPolyData()
            return
        ########################################

        self.property.SetRepresentationToPoints()
        self.property.SetPointSize(r)
        self.property.LightingOff()

        if isinstance(inputobj, vedo.BaseActor):
            inputobj = inputobj.points()  # numpy


        if isinstance(inputobj, vtk.vtkActor):
            polyCopy = vtk.vtkPolyData()
            pr = vtk.vtkProperty()
            pr.DeepCopy(inputobj.GetProperty())
            polyCopy.DeepCopy(inputobj.GetMapper().GetInput())
            pr.SetRepresentationToPoints()
            pr.SetPointSize(r)
            self._data = polyCopy
            self._mapper.SetInputData(polyCopy)
            self._mapper.SetScalarVisibility(inputobj.GetMapper().GetScalarVisibility())
            self.SetProperty(pr)
            self.property = pr

        elif isinstance(inputobj, vtk.vtkPolyData):
            if inputobj.GetNumberOfCells() == 0:
                carr = vtk.vtkCellArray()
                for i in range(inputobj.GetNumberOfPoints()):
                    carr.InsertNextCell(1)
                    carr.InsertCellPoint(i)
                inputobj.SetVerts(carr)
            self._data = inputobj  # cache vtkPolyData and mapper for speed

        elif utils.isSequence(inputobj): # passing point coords
            plist = inputobj
            n = len(plist)

            if n == 3:  # assume plist is in the format [all_x, all_y, all_z]
                if utils.isSequence(plist[0]) and len(plist[0]) > 3:
                    plist = np.stack((plist[0], plist[1], plist[2]), axis=1)
            elif n == 2:  # assume plist is in the format [all_x, all_y, 0]
                if utils.isSequence(plist[0]) and len(plist[0]) > 3:
                    plist = np.stack((plist[0], plist[1], np.zeros(len(plist[0]))), axis=1)

            if n and len(plist[0]) == 2: # make it 3d
                plist = np.c_[np.array(plist), np.zeros(len(plist))]

            if ((utils.isSequence(c)
                 and (len(c)>3
                      or (utils.isSequence(c[0]) and len(c[0])==4)
                     )
                )
                or utils.isSequence(alpha) ):

                cols = c

                n = len(plist)
                if n != len(cols):
                    vedo.logger.error(f"mismatch in Points() colors array lengths {n} and {len(cols)}")
                    raise RuntimeError()

                src = vtk.vtkPointSource()
                src.SetNumberOfPoints(n)
                src.Update()

                vgf = vtk.vtkVertexGlyphFilter()
                vgf.SetInputData(src.GetOutput())
                vgf.Update()
                pd = vgf.GetOutput()

                pd.GetPoints().SetData(utils.numpy2vtk(plist, dtype=float))

                ucols = vtk.vtkUnsignedCharArray()
                ucols.SetNumberOfComponents(4)
                ucols.SetName("Points_RGBA")
                if utils.isSequence(alpha):
                    if len(alpha) != n:
                        vedo.logger.error(f"mismatch in Points() alpha array lengths {n} and {len(cols)}")
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
                            rc,gc,bc = colors.getColor(cols[i])
                            ucols.InsertNextTuple4(rc*255, gc*255, bc*255, alphas[i]*255)
                else:
                    c = cols

                pd.GetPointData().AddArray(ucols)
                pd.GetPointData().SetActiveScalars("Points_RGBA")
                self._mapper.SetInputData(pd)
                self._mapper.ScalarVisibilityOn()
                self._data = pd

            else:

                pd = utils.buildPolyData(plist)
                self._mapper.SetInputData(pd)
                c = colors.getColor(c)
                self.property.SetColor(c)
                self.property.SetOpacity(alpha)
                self._data = pd

            ##########
            return
            ##########

        elif isinstance(inputobj, str):
            verts = vedo.io.load(inputobj)
            self.filename = inputobj
            self._data = verts.polydata()

        else:

            # try to extract the points from the VTK input data object
            try:
                vvpts = inputobj.GetPoints()
                pd = vtk.vtkPolyData()
                pd.SetPoints(vvpts)
                for i in range(inputobj.GetPointData().GetNumberOfArrays()):
                    arr = inputobj.GetPointData().GetArray(i)
                    pd.GetPointData().AddArray(arr)

                self._mapper.SetInputData(pd)
                c = colors.getColor(c)
                self.property.SetColor(c)
                self.property.SetOpacity(alpha)
                self._data = pd
            except:
                vedo.logger.error(f"cannot build Points from type {type(inputobj)}")
                raise RuntimeError()

        c = colors.getColor(c)
        self.property.SetColor(c)
        self.property.SetOpacity(alpha)

        self._mapper.SetInputData(self._data)
        return


    ##################################################################################
    def _update(self, polydata):
        # Overwrite the polygonal mesh with a new vtkPolyData
        self._data = polydata
        self._mapper.SetInputData(polydata)
        self._mapper.Modified()
        return self

    def __add__(self, meshs):
        if isinstance(meshs, list):
            alist = [self]
            for l in meshs:
                if isinstance(l, vtk.vtkAssembly):
                    alist += l.getMeshes()
                else:
                    alist += l
            return vedo.assembly.Assembly(alist)
        elif isinstance(meshs, vtk.vtkAssembly):
            meshs.AddPart(self)
            return meshs
        return vedo.assembly.Assembly([self, meshs])


    def polydata(self, transformed=True):
        """
        Returns the ``vtkPolyData`` object associated to a ``Mesh``.

        .. note::
            If ``transformed=True`` return a copy of polydata that corresponds
            to the current mesh position in space.
        """
        if not self._data:
            self._data = self._mapper.GetInput()
            return self._data


        if transformed:
            #if self.GetIsIdentity() or self._data.GetNumberOfPoints()==0: # commmentd out on 15th feb 2020
            if self._data.GetNumberOfPoints() == 0:
                # no need to do much
                return self._data
            else:
                # otherwise make a copy that corresponds to
                # the actual position in space of the mesh
                M = self.GetMatrix()
                transform = vtk.vtkTransform()
                transform.SetMatrix(M)
                tp = vtk.vtkTransformPolyDataFilter()
                tp.SetTransform(transform)
                tp.SetInputData(self._data)
                tp.Update()
                return tp.GetOutput()
        else:
            return self._data


    # def shader(stype="vertex", block="Normal", dcode="", icode="", before=True, repeat=False):
    #     """todo"""
    #     sp = self.GetShaderProperty()

    #     if "vertex" == stype and dcode:
    #         sp.AddShaderReplacement(
    #                                 vtk.vtkShader.Vertex,
    #                                 f"//VTK::{block}::Dec",  # replace the normal block
    #                                 before,                  # before the standard replacements
    #                                 "//VTK::Normal::Dec\n" + dcode+" \n", # we still want the default
    #                                 repeat,                 # only do it once
    #         )
    #         sp.AddVertexShaderReplacement(
    #                                 "//VTK::Normal::Impl",  # replace the normal block
    #                                 before,                  # before the standard replacements
    #                                 "//VTK::Normal::Impl\n" + icode+" \n"# we still want the default
    #                                 repeat,                 # only do it once
    #         )

    #     if "fragment" in stype and dcode:
    #         sp.AddFragmentShaderReplacement("//VTK::System::Dec",
    #                                False, # before the standard replacements
    #                                dcode,
    #                                False, # only do it once
    #         );

    #         sp.AddFragmentShaderReplacement(
    #           "//VTK::Normal::Dec",  # replace the normal block
    #           before,                  # before the standard replacements
    #           "//VTK::Normal::Dec\n" + icode, # we still want the default
    #           repeat,  # only do it once
    #         );
    #     return self


    def vertices(self, pts=None, transformed=True, copy=False):
        """Alias for ``points()``."""
        return self.points(pts, transformed, copy)


    def clone(self, deep=True, transformed=False):
        """
        Clone a ``PointCloud`` or ``Mesh`` object to make an exact copy of it.

        Parameters
        ----------
        deep : bool
            if False only build a shallow copy of the object (faster copy).

        transformed : bool
            if True reset the current transformation of the copy to unit.

        .. hint:: examples/basic/mirror.py
            .. image:: https://vedo.embl.es/images/basic/mirror.png
        """
        poly = self.polydata(transformed)
        polyCopy = vtk.vtkPolyData()
        if deep:
            polyCopy.DeepCopy(poly)
        else:
            polyCopy.ShallowCopy(poly)

        if isinstance(self, vedo.Mesh):
            cloned = vedo.Mesh(polyCopy)
        else:
            cloned = Points(polyCopy)

        pr = vtk.vtkProperty()
        pr.DeepCopy(self.GetProperty())
        cloned.SetProperty(pr)
        cloned.property = pr

        if self.GetBackfaceProperty():
            bfpr = vtk.vtkProperty()
            bfpr.DeepCopy(self.GetBackfaceProperty())
            cloned.SetBackfaceProperty(bfpr)

        if not transformed:
            # assign the same transformation to the copy
            cloned.SetOrigin(self.GetOrigin())
            cloned.SetScale(self.GetScale())
            cloned.SetOrientation(self.GetOrientation())
            cloned.SetPosition(self.GetPosition())

        cloned._mapper.SetScalarVisibility(self._mapper.GetScalarVisibility())
        cloned._mapper.SetScalarRange(self._mapper.GetScalarRange())
        cloned._mapper.SetColorMode(self._mapper.GetColorMode())
        lsr = self._mapper.GetUseLookupTableScalarRange()
        cloned._mapper.SetUseLookupTableScalarRange(lsr)
        cloned._mapper.SetScalarMode(self._mapper.GetScalarMode())
        lut = self._mapper.GetLookupTable()
        if lut:
            cloned._mapper.SetLookupTable(lut)

        cloned.SetPickable(self.GetPickable())

        cloned.base = np.array(self.base)
        cloned.top =  np.array(self.top)
        cloned.name = str(self.name)
        cloned.filename = str(self.filename)
        cloned.info = dict(self.info)
        if self.trail:
            n = len(self.trailPoints)
            cloned.addTrail(self.trailOffset, self.trailSegmentSize*n, n,
                            None, None, self.trail.GetProperty().GetLineWidth())
        if len(self.shadows) > 0:
            cloned.addShadows()

        cloned.point_locator = None # better not to share the same locators with original obj
        cloned.cell_locator = None

        return cloned


    def clone2D(self, pos=(0,0), coordsys=4, scale=None,
                c=None, alpha=None, ps=2, lw=1,
                sendback=False, layer=0):
        """
        Copy a 3D Mesh into a static 2D image. Returns a ``vtkActor2D``.

        Parameters
        ----------
        coordsys : int
            the coordinate system, options are

            - 0 = Displays
            - 1 = Normalized Display
            - 2 = Viewport (origin is the bottom-left corner of the window)
            - 3 = Normalized Viewport
            - 4 = View (origin is the center of the window)
            - 5 = World (anchor the 2d image to mesh)

        ps : int
            point size in pixel units

        lw : int
            line width in pixel units

        sendback : bool
            put it behind any other 3D object

        .. hint:: examples/other/clone2D.py
            .. image:: https://vedo.embl.es/images/other/clone2d.png
        """
        msiz = self.diagonalSize()
        if scale is None:
            if vedo.plotter_instance:
                sz = vedo.plotter_instance.window.GetSize()
                dsiz = utils.mag(sz)
                scale = dsiz/msiz/9
            else:
                scale = 350/msiz
            #colors.printc('clone2D(): scale set to', utils.precision(scale/300,3))
        else:
            scale *= 300

        cmsh = self.clone()
        poly = cmsh.pos(0,0,0).scale(scale).polydata()

        mapper2d = vtk.vtkPolyDataMapper2D()
        mapper2d.SetInputData(poly)
        act2d = vtk.vtkActor2D()
        act2d.SetMapper(mapper2d)
        act2d.SetLayerNumber(layer)
        csys = act2d.GetPositionCoordinate()
        csys.SetCoordinateSystem(coordsys)
        act2d.SetPosition(pos)
        if c is not None:
            c = colors.getColor(c)
            act2d.GetProperty().SetColor(c)
        else:
            act2d.GetProperty().SetColor(cmsh.color())
        if alpha is not None:
            act2d.GetProperty().SetOpacity(alpha)
        else:
            act2d.GetProperty().SetOpacity(cmsh.alpha())
        act2d.GetProperty().SetPointSize(ps)
        act2d.GetProperty().SetLineWidth(lw)
        act2d.GetProperty().SetDisplayLocationToForeground()
        if sendback:
            act2d.GetProperty().SetDisplayLocationToBackground()

        # print(csys.GetCoordinateSystemAsString())
        # print(act2d.GetHeight(), act2d.GetWidth(), act2d.GetLayerNumber())
        return act2d


    def addTrail(self, offset=None, maxlength=None, n=50, c=None, alpha=None, lw=2):
        """
        Add a trailing line to mesh.
        This new mesh is accessible through `mesh.trail`.

        Parameters
        ----------
        offset : float
            set an offset vector from the object center.

        maxlength : float
            length of trailing line in absolute units

        n : int
            number of segments to control precision

        lw : float
            line width of the trail

        .. hint:: examples/simulations/trail.py, airplane1.py, airplane2.py
            .. image:: https://vedo.embl.es/images/simulations/trail.gif
        """
        if maxlength is None:
            maxlength = self.diagonalSize() * 20
            if maxlength == 0:
                maxlength = 1

        if self.trail is None:
            pos = self.GetPosition()
            self.trailPoints = [None] * n
            self.trailSegmentSize = maxlength / n
            self.trailOffset = offset

            if c is None:
                if hasattr(self, "GetProperty"):
                    col = self.GetProperty().GetColor()
                else:
                    col = (0.1, 0.1, 0.1)
            else:
                col = colors.getColor(c)

            if alpha is None:
                alpha = 1
                if hasattr(self, "GetProperty"):
                    alpha = self.GetProperty().GetOpacity()

            tline = vedo.shapes.Line(pos, pos, res=n, c=col, alpha=alpha, lw=lw)
            self.trail = tline  # holds the Line
        return self

    def updateTrail(self):
        # internal use
        if isinstance(self, vedo.shapes.Arrow):
            currentpos= self.tipPoint() # the tip of Arrow
        else:
            currentpos = np.array(self.GetPosition())

        if self.trailOffset:
            currentpos += self.trailOffset

        lastpos = self.trailPoints[-1]
        if lastpos is None:  # reset list
            self.trailPoints = [currentpos] * len(self.trailPoints)
            return

        if np.linalg.norm(currentpos - lastpos) < self.trailSegmentSize:
            return

        self.trailPoints.append(currentpos)  # cycle
        self.trailPoints.pop(0)

        tpoly = self.trail.polydata(False)
        tpoly.GetPoints().SetData(
            utils.numpy2vtk(self.trailPoints-currentpos, dtype=float)
        )
        self.trail.SetPosition(currentpos)
        return self


    def deletePoints(self, indices, renamePoints=False):
        """
        Delete a list of vertices identified by their index.

        Parameters
        ----------
        renamePoints : bool
            if True, point indices and faces are renamed.
            If False, vertices are not really deleted and faces indices will
            stay unchanged (default, faster).

        .. hint:: examples/basic/deleteMeshPoints.py
            .. image:: https://vedo.embl.es/images/basic/deleteMeshPoints.png
        """
        cellIds = vtk.vtkIdList()
        self._data.BuildLinks()
        for i in indices:
            self._data.GetPointCells(i, cellIds)
            for j in range(cellIds.GetNumberOfIds()):
                self._data.DeleteCell(cellIds.GetId(j))  # flag cell

        self._data.RemoveDeletedCells()

        if renamePoints:
            coords = self.points(transformed=False)
            faces = self.faces()
            pts_inds = np.unique(faces) # flattened array

            newfaces = []
            for f in faces:
                newface=[]
                for i in f:
                    idx = np.where(pts_inds==i)[0][0]
                    newface.append(idx)
                newfaces.append(newface)

            newpoly = utils.buildPolyData(coords[pts_inds], newfaces)
            return self._update(newpoly)
        else:
            self._mapper.Modified()
            return self


    def delete(self, points=(), cells=()):
        """Delete points and/or cells from a point cloud or mesh."""
        rp = vtk.vtkRemovePolyData()

        if isinstance(points, Points):
            rp.SetInputData(self._data)
            poly = points._data
            rp.RemoveInputData(poly)
            rp.Update()
            out = rp.GetOutput()
            return self._update(out) ####

        if points:
            idarr = utils.numpy2vtk(points, dtype='id')
        elif cells:
            idarr = utils.numpy2vtk(cells, dtype='id')
        else:
            # utils.printc("delete(): nothing to delete, skip.", c='y')
            return self
        rp.SetPointIds(idarr)
        rp.Update()
        out = rp.GetOutput()
        return self._update(out)


    def computeNormalsWithPCA(self, n=20, orientationPoint=None, invert=False):
        """
        Generate point normals using PCA (principal component analysis).
        Basically this estimates a local tangent plane around each sample point p
        by considering a small neighborhood of points around p, and fitting a plane
        to the neighborhood (via PCA).

        Parameters
        ----------
        n : int
            neighborhood size to calculate the normal

        orientationPoint : list
            adjust the +/- sign of the normals so that
            the normals all point towards a specified point. If None, perform a traversal
            of the point cloud and flip neighboring normals so that they are mutually consistent.

        invert : bool
            flip all normals
        """
        poly = self.polydata()
        pcan = vtk.vtkPCANormalEstimation()
        pcan.SetInputData(poly)
        pcan.SetSampleSize(n)

        if orientationPoint is not None:
            pcan.SetNormalOrientationToPoint()
            pcan.SetOrientationPoint(orientationPoint)
        else:
            pcan.SetNormalOrientationToGraphTraversal()

        if invert:
            pcan.FlipNormalsOn()
        pcan.Update()

        varr = pcan.GetOutput().GetPointData().GetNormals()
        varr.SetName("Normals")
        self._data.GetPointData().SetNormals(varr)
        self._data.GetPointData().Modified()
        return self


    def distanceTo(self, pcloud, signed=False, invert=False, name="Distance"):
        """
        Computes the distance from one point cloud or mesh to another point cloud or mesh.
        This new `pointdata` array is saved with default name "Distance".

        Keywords ``signed`` and ``invert`` are used to compute signed distance,
        but the mesh in that case must have polygonal faces (not a simple point cloud),
        and normals must also be computed.

        Example:
            .. code-block:: python

                from vedo import *
                b1 = Sphere().pos(10,0,0)
                b2 = Sphere().pos(15,0,0)
                b1.distanceTo(b2, signed=True, invert=False).addScalarBar()
                print(b1.pointdata["Distance"])
                show(b1, b2, axes=1).close()

        .. hint:: examples/basic/distance2mesh.py
            .. image:: https://vedo.embl.es/images/basic/distance2mesh.png
        """
        if pcloud._data.GetNumberOfPolys():

            poly1 = self.polydata()
            poly2 = pcloud.polydata()
            df = vtk.vtkDistancePolyDataFilter()
            df.ComputeSecondDistanceOff()
            df.SetInputData(0, poly1)
            df.SetInputData(1, poly2)
            df.SetSignedDistance(signed)
            df.SetNegateDistance(invert)
            df.Update()
            scals = df.GetOutput().GetPointData().GetScalars()

        else: # has no polygons and vtkDistancePolyDataFilter wants them (dont know why)

            if signed:
                vedo.logger.warning("distanceTo() called with signed=True but input object has no polygons")

            if not pcloud.point_locator:
                pcloud.point_locator = vtk.vtkPointLocator()
                pcloud.point_locator.SetDataSet(pcloud.polydata())
                pcloud.point_locator.BuildLocator()

            ids = []
            ps1 = self.points()
            ps2 = pcloud.points()
            for p in ps1:
                pid = pcloud.point_locator.FindClosestPoint(p)
                ids.append(pid)

            deltas = ps2[ids] - ps1
            d = np.linalg.norm(deltas, axis=1).astype(np.float32)
            scals = utils.numpy2vtk(d)

        scals.SetName(name)
        self._data.GetPointData().AddArray(scals) # must be self._data !
        self._data.GetPointData().SetActiveScalars(scals.GetName())
        rng = scals.GetRange()
        self._mapper.SetScalarRange(rng[0], rng[1])
        self._mapper.ScalarVisibilityOn()
        return self

    @deprecated(reason=vedo.colors.red+"Please use distanceTo()"+vedo.colors.reset)
    def distanceToMesh(self, pcloud, signed=False, invert=False):
        """Deprecated. Please use distanceTo()"""
        return self.distanceTo(pcloud, signed, invert)


    def alpha(self, opacity=None):
        """Set/get mesh's transparency. Same as `mesh.opacity()`."""
        if opacity is None:
            return self.GetProperty().GetOpacity()

        self.GetProperty().SetOpacity(opacity)
        bfp = self.GetBackfaceProperty()
        if bfp:
            if opacity < 1:
                self._bfprop = bfp
                self.SetBackfaceProperty(None)
            else:
                self.SetBackfaceProperty(self._bfprop)
        return self

    def opacity(self, alpha=None):
        """Set/get mesh's transparency. Same as `mesh.alpha()`."""
        return self.alpha(alpha)

    def forceOpaque(self, value=True):
        """ Force the Mesh, Line or point cloud to be treated as opaque"""
        ## force the opaque pass, fixes picking in vtk9
        # but causes other bad troubles with lines..
        self.SetForceOpaque(value)
        return self

    def forceTranslucent(self, value=True):
        """ Force the Mesh, Line or point cloud to be treated as translucent"""
        self.SetForceTranslucent(value)
        return self

    def occlusion(self, value=None):
        """Occlusion strength in range [0,1]."""
        if value is None:
            return self.GetProperty().GetOcclusionStrength()
        else:
            self.GetProperty().SetOcclusionStrength(value)
            return self

    def pointSize(self, value):
        """Set/get mesh's point size of vertices. Same as `mesh.ps()`"""
        if not value:
            self.GetProperty().SetRepresentationToSurface()
        else:
            self.GetProperty().SetRepresentationToPoints()
            self.GetProperty().SetPointSize(value)
        return self

    def ps(self, pointSize=None):
        """Set/get mesh's point size of vertices. Same as `mesh.pointSize()`"""
        return self.pointSize(pointSize)

    def renderPointsAsSpheres(self, value=True):
        """Make points look spheric or make them look as squares."""
        self.GetProperty().SetRenderPointsAsSpheres(value)
        return self


    def color(self, c=False, alpha=None):
        """
        Set/get mesh's color.
        If None is passed as input, will use colors from active scalars.
        Same as `mesh.c()`.
        """
        # overrides base.color()
        if c is False:
            return np.array(self.GetProperty().GetColor())
        elif c is None:
            self._mapper.ScalarVisibilityOn()
            return self
        self._mapper.ScalarVisibilityOff()
        cc = colors.getColor(c)
        self.GetProperty().SetColor(cc)
        if self.trail:
            self.trail.GetProperty().SetColor(cc)
        if alpha is not None:
            self.alpha(alpha)
        return self


    def clean(self, tol=None):
        """
        Clean pointcloud or mesh by removing coincident points.
        """
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.PointMergingOn()
        cleanPolyData.ConvertLinesToPointsOn()
        cleanPolyData.ConvertPolysToLinesOn()
        cleanPolyData.ConvertStripsToPolysOn()
        cleanPolyData.SetInputData(self._data)
        if tol: #deprecation message
            vedo.logger.warning("clean(tol=...), please use subsample(fraction=...)")
            cleanPolyData.SetTolerance(tol)
        cleanPolyData.Update()
        return self._update(cleanPolyData.GetOutput())

    def subsample(self, fraction, absolute=False):
        """
        Subsample a point cloud by requiring that the points
        or vertices are far apart at least by the specified fraction of the object size.
        If a Mesh is passed the polygonal faces are not removed
        but holes can appear as vertices are removed.

        .. hint:: examples/advanced/moving_least_squares1D.py, recosurface.py
            .. image:: https://vedo.embl.es/images/advanced/moving_least_squares1D.png
        """
        if fraction > 1:
            vedo.logger.warning(f"subsample(fraction=...), fraction must be < 1, but is {fraction}")
        if fraction <= 0:
            return self
        cpd = vtk.vtkCleanPolyData()
        cpd.PointMergingOn()
        cpd.ConvertLinesToPointsOn()
        cpd.ConvertPolysToLinesOn()
        cpd.ConvertStripsToPolysOn()
        cpd.SetInputData(self._data)
        cpd.SetTolerance(fraction)
        cpd.SetToleranceIsAbsolute(absolute)
        cpd.Update()
        ps = 2
        if self.GetProperty().GetRepresentation() == 0:
            ps = self.GetProperty().GetPointSize()
        return self._update(cpd.GetOutput()).ps(ps)


    def threshold(self, scalars, above=None, below=None, on='points'):
        """
        Extracts cells where scalar value satisfies threshold criterion.

        scalars : str,list
            name of the scalars array.

        above : float
            minimum value of the scalar

        below : float
            maximum value of the scalar

        on : str
            if 'cells' assume array of scalars refers to cell data.

        .. hint:: examples/basic/mesh_threshold.py
        """
        if utils.isSequence(scalars):
            if on.startswith('c'):
                self.addCellArray(scalars, "threshold")
            else:
                self.addPointArray(scalars, "threshold")
            scalars = "threshold"
        else: # string is passed
            if on.startswith('c'):
                arr = self.celldata[scalars]
            else:
                arr = self.pointdata[scalars]
            if arr is None:
                vedo.logger.error(f"no scalars found with name/nr: {scalars}")
                raise RuntimeError()

        thres = vtk.vtkThreshold()
        thres.SetInputData(self._data)

        if on.startswith('c'):
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        thres.SetInputArrayToProcess(0, 0, 0, asso, scalars)
        #        if above is not None and below is not None:
        #            if above<below:
        #                thres.ThresholdBetween(above, below)
        #            elif above==below:
        #                return self
        #            else:
        #                thres.InvertOn()
        #                thres.ThresholdBetween(below, above)
        #        elif above is not None:
        #            thres.ThresholdByUpper(above)
        #        elif below is not None:
        #            thres.ThresholdByLower(below)

        if above is None and below is not None:
            thres.ThresholdByLower(below)
        elif below is None and above is not None:
            thres.ThresholdByUpper(above)
        else:
            thres.ThresholdBetween(above, below)
        thres.Update()

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(thres.GetOutput())
        gf.Update()
        return self._update(gf.GetOutput())


    def quantize(self, value):
        """
        The user should input a value and all {x,y,z} coordinates
        will be quantized to that absolute grain size.

        Example:
            .. code-block:: python

                from vedo import Paraboloid
                Paraboloid().lw(0.1).quantize(0.1).show()
        """
        poly = self._data
        qp = vtk.vtkQuantizePolyDataPoints()
        qp.SetInputData(poly)
        qp.SetQFactor(value)
        qp.Update()
        return self._update(qp.GetOutput()).flat()

    def averageSize(self):
        """
        Calculate the average size of a mesh.
        This is the mean of the vertex distances from the center of mass.
        """
        coords = self.points()
        cm = np.mean(coords, axis=0)
        if not len(coords):
            return 0.0
        cc = coords-cm
        return np.mean(np.linalg.norm(cc, axis=1))

    def centerOfMass(self):
        """Get the center of mass of mesh."""
        cmf = vtk.vtkCenterOfMass()
        cmf.SetInputData(self.polydata())
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)

    def normalAt(self, i):
        """Return the normal vector at vertex point `i`."""
        normals = self.polydata().GetPointData().GetNormals()
        return np.array(normals.GetTuple(i))

    def normals(self, cells=False, compute=True):
        """Retrieve vertex normals as a numpy array.

        cells : bool
            if `True` return cell normals.

        compute : bool
            if `True` normals are recalculated if not already present.
            Note that this might modify the number of mesh points.
        """
        if cells:
            vtknormals = self.polydata().GetCellData().GetNormals()
        else:
            vtknormals = self.polydata().GetPointData().GetNormals()
        if not vtknormals and compute:
            self.computeNormals(cells=cells)
            if cells:
                vtknormals = self.polydata().GetCellData().GetNormals()
            else:
                vtknormals = self.polydata().GetPointData().GetNormals()
        if not vtknormals:
            return np.array([])
        return utils.vtk2numpy(vtknormals)


    def labels(self, content=None, cells=False, scale=None,
               rotX=0, rotY=0, rotZ=0,
               ratio=1, precision=None,
               italic=False, font="", justify="bottom-left",
               c="black", alpha=1,
        ):
        """
        Generate value or ID labels for mesh cells or points.
        For large nr. of labels use ``font="VTK"`` which is much faster.

        See also: ``flag()``, ``vignette()``, ``caption()`` and ``legend()``.

        Parameters
        ----------
        content : list,int,str
             either 'id', 'cellid', array name or array number.
             A array can also be passed (must match the nr. of points or cells).

        cells : bool
            generate labels for cells instead of points [False]

        scale : float
            absolute size of labels, if left as None it is automatic

        rotZ : float
            local rotation angle of label in degrees

        ratio : int
            skipping ratio, to reduce nr of labels for large meshes

        precision : int
            numeric precision of labels

        Example:
            .. code-block:: python

                from vedo import *
                s = Sphere(res=10).lineWidth(1).c("orange").computeNormals()
                point_ids = s.labels('id', cells=False).c('green')
                cell_ids  = s.labels('id', cells=True ).c('black')
                show(s, point_ids, cell_ids)

        .. hint:: examples/basic/boundaries.py
            .. image:: https://vedo.embl.es/images/basic/boundaries.png
        """
        if isinstance(content, str):
            if "cellid" == content:
                cells=True
                content="id"

        if cells:
            elems = self.cellCenters()
            norms = self.normals(cells=True, compute=False)
            ns = np.sqrt(self.NCells())
        else:
            elems = self.points()
            norms = self.normals(cells=False, compute=False)
            ns = np.sqrt(self.NPoints())

        hasnorms=False
        if len(norms):
            hasnorms=True

        if scale is None:
            if not ns: ns = 100
            scale = self.diagonalSize()/ns/10

        arr = None
        mode = 0
        if content is None:
            mode=0
            if cells:
                if self._data.GetCellData().GetScalars():
                    name = self._data.GetCellData().GetScalars().GetName()
                    arr = self.celldata[name]
            else:
                if self._data.GetPointData().GetScalars():
                    name = self._data.GetPointData().GetScalars().GetName()
                    arr = self.pointdata[name]
        elif isinstance(content, (str, int)):
            if content=='id':
                mode = 1
            elif cells:
                mode = 0
                arr = self.celldata[content]
            else:
                mode = 0
                arr = self.pointdata[content]
        elif utils.isSequence(content):
            mode = 0
            arr = content
            # print('WEIRD labels() test', content)
            # exit()

        if arr is None and mode == 0:
            vedo.logger.error("in labels(), array not found for points or cells")
            return None

        tapp = vtk.vtkAppendPolyData()
        ninputs = 0

        for i,e in enumerate(elems):
            if i % ratio:
                continue

            if mode==1:
                txt_lab = str(i)
            else:
                if precision:
                    txt_lab = utils.precision(arr[i], precision)
                else:
                    txt_lab = str(arr[i])

            if not txt_lab:
                continue

            if font=="VTK":
                tx = vtk.vtkVectorText()
                tx.SetText(txt_lab)
                tx.Update()
                tx_poly = tx.GetOutput()
            else:
                tx_poly = vedo.shapes.Text3D(txt_lab, font=font, justify=justify)
                tx_poly = tx_poly._data

            if tx_poly.GetNumberOfPoints() == 0:
                continue #######################
            ninputs += 1

            T = vtk.vtkTransform()
            T.PostMultiply()
            if italic:
                T.Concatenate([1,0.2,0,0,
                               0,1,0,0,
                               0,0,1,0,
                               0,0,0,1])
            if hasnorms:
                ni = norms[i]
                if cells: # center-justify
                    bb = tx_poly.GetBounds()
                    dx, dy = (bb[1]-bb[0])/2, (bb[3]-bb[2])/2
                    T.Translate(-dx,-dy,0)
                if rotX: T.RotateX(rotX)
                if rotY: T.RotateY(rotY)
                if rotZ: T.RotateZ(rotZ)
                crossvec = np.cross([0,0,1], ni)
                angle = np.arccos(np.dot([0,0,1], ni))*57.3
                T.RotateWXYZ(angle, crossvec)
                if cells: # small offset along normal only for cells
                    T.Translate(ni*scale/2)
            else:
                if rotX: T.RotateX(rotX)
                if rotY: T.RotateY(rotY)
                if rotZ: T.RotateZ(rotZ)
            T.Scale(scale,scale,scale)
            T.Translate(e)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(tx_poly)
            tf.SetTransform(T)
            tf.Update()
            tapp.AddInputData(tf.GetOutput())

        if ninputs:
            tapp.Update()
            lpoly = tapp.GetOutput()
        else: #return an empty obj
            lpoly = vtk.vtkPolyData()

        ids = vedo.mesh.Mesh(lpoly, c=c, alpha=alpha)
        ids.GetProperty().LightingOff()
        ids.SetUseBounds(False)
        return ids

    def legend(self, txt):
        """Book a legend text."""
        self.info['legend'] = txt
        return self

    def vignette(
            self,
            txt=None,
            point=None,
            offset=None,
            s=None,
            font="",
            rounded=True,
            c=None,
            alpha=1,
            lw=2,
            italic=0,
        ):
        """
        Generate and return a vignette to describe an object.
        Returns a ``Mesh`` object.

        Parameters
        ----------
        txt : str
            Text to display. The default is the filename or the object name.

        point : list
            position of the vignette pointer. The default is None.

        offset : list
            text offset wrt the application point. The default is None.

        s : float
            size of the vignette. The default is None.

        font : str
            text font. The default is "".

        rounded : bool
            draw a rounded or squared box around the text. The default is True.

        c : list
            text and box color. The default is None.

        alpha : float
            transparency of text and box. The default is 1.

        lw : float
            line with of box frame. The default is 2.

        italic : float
            italicness of text. The default is 0.

        .. hint:: examples/pyplot/intersect2d.py, goniometer.py, flag_labels.py
            .. image:: https://vedo.embl.es/images/pyplot/intersect2d.png
        """
        acts = []

        if txt is None:
            if self.filename:
                txt = self.filename.split('/')[-1]
            elif self.name:
                txt = self.name
            else:
                return None

        sph = None
        x0, x1, y0, y1, z0, z1 = self.bounds()
        d = self.diagonalSize()
        if point is None:
            if d:
                point = self.closestPoint([(x0 + x1) / 2, (y0 + y1) / 2, z1])
            else:  # it's a Point
                point = self.GetPosition()

        if offset is None:
            offset = [(x1 - x0) / 3, (y1 - y0) / 6, 0]
        elif len(offset) == 2:
            offset = [offset[0], offset[1], 0] # make it 3d

        if s is None:
            s = d / 20

        sph = None
        if d and (z1 - z0) / d > 0.1:
            sph = vedo.shapes.Sphere(point, r=s*0.4, res=6)

        if c is None:
            c = np.array(self.color())/1.4

        if len(point) == 2:
            point = [point[0], point[1], 0.0]
        pt = np.asarray(point)

        lb = vedo.shapes.Text3D(txt, pos=pt+offset, s=s, font=font,
                                italic=italic, justify="center-left")
        acts.append(lb)

        if d and not sph:
            sph = vedo.shapes.Circle(pt, r=s/3, res=15)
        acts.append(sph)

        x0, x1, y0, y1, z0, z1 = lb.GetBounds()
        if rounded:
            box = vedo.shapes.KSpline(
                [(x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0)], closed=True
            )
        else:
            box = vedo.shapes.Line(
                [(x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0), (x0,y0,z0)]
            )
        box.origin([(x0+x1) / 2, (y0+y1) / 2, (z0+z1) / 2])
        box.scale([1.1,1.2,1])
        acts.append(box)

        x0, x1, y0, y1, z0, z1 = box.bounds()
        if x0 < pt[0] < x1:
            c0 = box.closestPoint(pt)
            c1 = [c0[0], c0[1] + (pt[1] - y0) / 4, pt[2]]
        elif (pt[0]-x0) < (x1-pt[0]):
            c0 = [x0, (y0 + y1) / 2, pt[2]]
            c1 = [x0 + (pt[0] - x0) / 4, (y0 + y1) / 2, pt[2]]
        else:
            c0 = [x1, (y0 + y1) / 2, pt[2]]
            c1 = [x1 + (pt[0] - x1) / 4, (y0 + y1) / 2, pt[2]]

        con = vedo.shapes.Line([c0, c1, pt])
        acts.append(con)

        macts = vedo.merge(acts).c(c).alpha(alpha)
        macts.SetOrigin(pt)
        macts.bc('t').pickable(False).GetProperty().LightingOff()
        macts.GetProperty().SetLineWidth(lw)
        macts.UseBoundsOff()
        return macts

    def caption(
            self,
            txt=None,
            point=None,
            size=(0.30, 0.15),
            padding=5,
            font="VictorMono",
            justify="center-right",
            vspacing=1,
            c=None,
            alpha=1,
            lw=1,
            ontop=True,
        ):
        """
        Add a 2D caption to an object which follows the camera movements.
        Latex is not supported. Returns the same input object for concatenation.

        See also ``vignette()``, ``flag()``, ``labels()`` and ``legend()``
        with similar functionality.

        Parameters
        ----------
        txt : str
            text to be rendered. The default is the file name.

        point : list
            anchoring point. The default is None.

        size : list
            (width, height) of the caption box. The default is (0.30, 0.15).

        padding : float
            padding space of the caption box in pixels. The default is 5.

        font : str
            font name. Font "LogoType" allows for Japanese and Chinese characters.
            Use a monospace font for better rendering. The default is "VictorMono".
            Type ``vedo -r fonts`` for a font demo.

        justify : str
            internal text justification. The default is "center-right".

        vspacing : float
            vertical spacing between lines. The default is 1.

        c : str
            text and box color. The default is 'lb'.

        alpha : float
            text and box transparency. The default is 1.

        lw : int,
            line width in pixels. The default is 1.

        ontop : bool
            keep the 2d caption always on top. The default is True.

        .. hint:: examples/pyplot/caption.py, flag_labels.py
            .. image:: https://vedo.embl.es/images/pyplot/caption.png
        """
        if txt is None:
            if self.filename:
                txt = self.filename.split('/')[-1]
            elif self.name:
                txt = self.name

        if not txt: # disable it
            self._caption = None
            return self

        for r in vedo.shapes._reps:
            txt = txt.replace(r[0], r[1])

        if c is None:
            c = np.array(self.GetProperty().GetColor())/2
        else:
            c = colors.getColor(c)

        if not font:
           font =  settings.defaultFont

        if point is None:
            x0,x1,y0,y1,z0,z1 = self.GetBounds()
            pt = [(x0+x1)/2, (y0+y1)/2, z1]
            point = self.closestPoint(pt)

        capt = vtk.vtkCaptionActor2D()
        capt.SetAttachmentPoint(point)
        capt.SetBorder(True)
        capt.SetLeader(True)
        sph = vtk.vtkSphereSource()
        sph.Update()
        capt.SetLeaderGlyphData(sph.GetOutput())
        capt.SetMaximumLeaderGlyphSize(5)
        capt.SetPadding(padding)
        capt.SetCaption(txt)
        capt.SetWidth(size[0])
        capt.SetHeight(size[1])
        capt.SetThreeDimensionalLeader(not ontop)

        pra = capt.GetProperty()
        pra.SetColor(c)
        pra.SetOpacity(alpha)
        pra.SetLineWidth(lw)

        pr = capt.GetCaptionTextProperty()
        pr.SetFontFamily(vtk.VTK_FONT_FILE)
        fl = utils.getFontPath(font)
        pr.SetFontFile(fl)
        pr.ShadowOff()
        pr.BoldOff()
        pr.FrameOff()
        pr.SetColor(c)
        pr.SetOpacity(alpha)
        pr.SetJustificationToLeft()
        if "top" in justify:
            pr.SetVerticalJustificationToTop()
        if "bottom" in justify:
            pr.SetVerticalJustificationToBottom()
        if "cent" in justify:
            pr.SetVerticalJustificationToCentered()
            pr.SetJustificationToCentered()
        if "left" in justify:
            pr.SetJustificationToLeft()
        if "right" in justify:
            pr.SetJustificationToRight()
        pr.SetLineSpacing(vspacing)
        self._caption = capt
        return self

    def flag(
            self,
            text=None,
            font="Normografo",
            size=18,
            angle=0,
            shadow=False,
            c='k',
            bg='w',
            justify=0,
            delay=150,
        ):
        """
        Add a flag label which becomes visible when hovering the object with mouse.
        Can be later disabled by setting `flag(False)`.

        See also: ``labels()``, ``vignette()``, ``caption()`` and ``legend()``.

        Parameters
        ----------
        text : str
            text string to be rendered. The default is the filename without extension.

        font : str
            name of font to use. The default is "Courier".

        size : int
            size of font. The default is 18. Fonts are: "Arial", "Courier", "Times".

        angle : float
            rotation angle. The default is 0.

        shadow : bool
            add a shadow to the font. The default is False.

        c : str
            color name or index. The default is 'k'.

        bg : str
            color name of the background. The default is 'w'.

        justify : TYPE
            justification code. The default is 0.

        delay : float
            pop up delay in milliseconds. The default is 150.

        .. hint:: examples/other/flag_labels.py
            .. image:: https://vedo.embl.es/images/other/flag_labels.png
        """
        if text is None:
            if self.filename:
                text = self.filename.split('/')[-1]
            elif self.name:
                text = self.name
            else:
                text = ""
        if "\\" in repr(text):
            for r in vedo.shapes._reps:
                text = text.replace(r[0], r[1])
        self.flagText = text
        settings.flagDelay    = delay
        settings.flagFont     = font
        settings.flagFontSize = size
        settings.flagAngle    = angle
        settings.flagShadow   = shadow
        settings.flagColor    = c
        settings.flagJustification = justify
        settings.flagBackgroundColor = bg
        return self

    def alignTo(
            self,
            target,
            iters=100,
            rigid=False,
            invert=False,
            useCentroids=False,
        ):
        """
        Aligned to target mesh through the `Iterative Closest Point` algorithm.

        The core of the algorithm is to match each vertex in one surface with
        the closest surface point on the other, then apply the transformation
        that modify one surface to best match the other (in the least-square sense).

        Parameters
        ----------
        rigid : bool
            if True do not allow scaling

        invert : bool
            if True start by aligning the target to the source but
             invert the transformation finally. Useful when the target is smaller
             than the source.

        useCentroids : bool
            start by matching the centroids of the two objects.

        .. hint:: examples/basic/align1.py, align2.py
            .. image:: https://vedo.embl.es/images/basic/align1.png
        """
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(self.polydata())
        icp.SetTarget(target.polydata())
        if invert:
            icp.Inverse()
        icp.SetMaximumNumberOfIterations(iters)
        if rigid:
            icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetStartByMatchingCentroids(useCentroids)
        icp.Update()

        M = icp.GetMatrix()
        if invert:
            M.Invert() # icp.GetInverse() doesnt work!
        # self.applyTransform(M)
        self.SetUserMatrix(M)

        self.transform = self.GetUserTransform()
        self.point_locator = None
        self.cell_locator = None
        return self


    def transformWithLandmarks(self, sourceLandmarks, targetLandmarks, rigid=False):
        """
        Trasform mesh orientation and position based on a set of landmarks points.
        The algorithm finds the best matching of source points to target points
        in the mean least square sense, in one single step.

        .. hint:: examples/basic/align5.py
            .. image:: https://vedo.embl.es/images/basic/align5.png
        """
        lmt = vtk.vtkLandmarkTransform()

        if utils.isSequence(sourceLandmarks):
            ss = vtk.vtkPoints()
            for p in sourceLandmarks:
                ss.InsertNextPoint(p)
        else:
            ss = sourceLandmarks.polydata().GetPoints()

        if utils.isSequence(targetLandmarks):
            st = vtk.vtkPoints()
            for p in targetLandmarks:
                st.InsertNextPoint(p)
        else:
            st = targetLandmarks.polydata().GetPoints()

        if ss.GetNumberOfPoints() != st.GetNumberOfPoints():
            vedo.logger.error("source and target have different nr of points")
            raise RuntimeError()

        lmt.SetSourceLandmarks(ss)
        lmt.SetTargetLandmarks(st)
        if rigid:
            lmt.SetModeToRigidBody()
        lmt.Update()
        # self.applyTransform(lmt)
        self.SetUserTransform(lmt)
        self.transform = lmt

        self.point_locator = None
        self.cell_locator = None
        return self


    def applyTransform(self, transformation, reset=False, concatenate=False):
        """
        Apply a linear or non-linear transformation to the mesh polygonal data.

        Parameters
        ----------
        transformation :
            a ``vtkTransform``, ``vtkMatrix4x4``
            or a 4x4 or 3x3 python or numpy matrix.

        reset : bool
            if True reset the current transformation matrix
            to identity after having moved the object, otherwise the internal
            matrix will stay the same (to only affect visualization).

            It the input transformation has no internal defined matrix (ie. non linear)
            then reset will be assumed as True.

        concatenate : bool
            concatenate the transformation with the current existing one

        Example:
            .. code-block:: python

                from vedo import Cube, show
                c1 = Cube().rotateZ(5).x(2).y(1)
                print("cube1 position", c1.pos())
                T = c1.getTransform()  # rotate by 5 degrees, sum 2 to x and 1 to y
                c2 = Cube().c('r4')
                c2.applyTransform(T)   # ignore previous movements
                c2.applyTransform(T, concatenate=True)
                c2.applyTransform(T, concatenate=True)
                print("cube2 position", c2.pos())
                show(c1, c2, axes=1).close()
        """
        self.point_locator = None
        self.cell_locator = None

        if isinstance(transformation, vtk.vtkMatrix4x4):
            tr = vtk.vtkTransform()
            tr.SetMatrix(transformation)
            transformation = tr
        elif utils.isSequence(transformation):
            M = vtk.vtkMatrix4x4()
            n = len(transformation[0])
            for i in range(n):
                for j in range(n):
                    M.SetElement(i, j, transformation[i][j])
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            transformation = tr

        if reset:# or not hasattr(transformation, 'GetMatrix'): # might be non-linear?

            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(transformation)
            tf.SetInputData(self.polydata())
            tf.Update()
            I = vtk.vtkMatrix4x4()
            ID = vtk.vtkTransform()
            ID.SetMatrix(I)
            self.transform = ID
            self.PokeMatrix(I)  # reset to identity
            self.SetUserTransform(None)
            return self._update(tf.GetOutput())

        else:

            if concatenate:

                M = vtk.vtkTransform()
                M.PostMultiply()
                M.SetMatrix(self.GetMatrix())

                M.Concatenate(transformation)

                self.SetScale(M.GetScale())
                self.SetOrientation(M.GetOrientation())
                self.SetPosition(M.GetPosition())
                self.transform = M
                self.SetUserTransform(None)

            else:

                try:
                    self.SetScale(transformation.GetScale())
                    self.SetOrientation(transformation.GetOrientation())
                    self.SetPosition(transformation.GetPosition())
                    self.SetUserTransform(None)
                except AttributeError: #GetScale might be missing for non linear an shear transf
                     self.SetUserTransform(transformation)

                self.transform = transformation

            return self


    def normalize(self):
        """Scale Mesh average size to unit."""
        coords = self.points()
        if not len(coords):
            return self
        cm = np.mean(coords, axis=0)
        pts = coords - cm
        xyz2 = np.sum(pts * pts, axis=0)
        scale = 1 / np.sqrt(np.sum(xyz2) / len(pts))
        t = vtk.vtkTransform()
        t.PostMultiply()
        # t.Translate(-cm)
        t.Scale(scale, scale, scale)
        # t.Translate(cm)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(self._data)
        tf.SetTransform(t)
        tf.Update()
        self.point_locator = None
        self.cell_locator = None
        return self._update(tf.GetOutput())


    def mirror(self, axis="x", origin=[0,0,0], reset=False):
        """
        Mirror the mesh  along one of the cartesian axes

        axis : str
            axis to use for mirroring, must be set to x, y, z or n.
            Or any combination of those. Adding 'n' reverses mesh faces (hence normals).

        origin : list
            use this point as the origin of the mirroring transformation.

        reset : bool
            if True keep into account the current position of the object,
            and then reset its internal transformation matrix to Identity.

        .. hint:: examples/basic/mirror.py
            .. image:: https://vedo.embl.es/images/basic/mirror.png
        """
        sx, sy, sz = 1, 1, 1
        if "x" in axis.lower(): sx = -1
        if "y" in axis.lower(): sy = -1
        if "z" in axis.lower(): sz = -1
        origin = np.array(origin)
        tr = vtk.vtkTransform()
        tr.PostMultiply()
        tr.Translate(-origin)
        tr.Scale(sx, sy, sz)
        tr.Translate(origin)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(self.polydata(reset))
        tf.SetTransform(tr)
        tf.Update()
        outpoly = tf.GetOutput()
        if reset:
            self.PokeMatrix(vtk.vtkMatrix4x4())  # reset to identity
        if sx*sy*sz<0 or 'n' in axis:
            rs = vtk.vtkReverseSense()
            rs.SetInputData(outpoly)
            rs.ReverseNormalsOff()
            rs.Update()
            outpoly = rs.GetOutput()

        self.point_locator = None
        self.cell_locator = None
        return self._update(outpoly)


    def shear(self, x=0, y=0, z=0):
        """Apply a shear deformation along one of the main axes"""
        t = vtk.vtkTransform()
        sx, sy, sz = self.GetScale()
        t.SetMatrix([sx, x, 0, 0,
                      y,sy, z, 0,
                      0, 0,sz, 0,
                      0, 0, 0, 1])
        self.applyTransform(t, reset=True)
        return self


    def flipNormals(self):
        """Flip all mesh normals. Same as `mesh.mirror('n')`."""
        rs = vtk.vtkReverseSense()
        rs.SetInputData(self._data)
        rs.ReverseCellsOff()
        rs.ReverseNormalsOn()
        rs.Update()
        return self._update(rs.GetOutput())


    #####################################################################################
    def cmap(
            self,
            cname,
            input_array=None,
            on="points",
            name="",
            vmin=None,
            vmax=None,
            alpha=1,
            n=256,
        ):
        """
        Set individual point/cell colors by providing a list of scalar values and a color map.
        `scalars` can be the string name of a ``vtkArray``.

        Parameters
        ----------

        cname
            :allowed types: str, list, vtkLookupTable, matplotlib.colors.LinearSegmentedColormap
            color map scheme to transform a real number into a color.

        on : str
            either 'points' or 'cells'.
            Apply the color map as defined on either point or cell data.

        name : str
            give a name to the numpy array

        vmin : float
            clip scalars to this minimum value

        vmax : float
            clip scalars to this maximum value

        alpha : float,list
            Mesh transparency. Can be a ``list`` of values one for each vertex.

        n : int
            number of distinct colors to be used.

        .. hint:: examples/basic/mesh_coloring.py, mesh_alphas.py, mesh_custom.py and many others
            .. image:: https://vedo.embl.es/images/basic/mesh_custom.png
        """
        self._cmap_name = cname

        if input_array is None:
            if len(self.pointdata.keys()) == 0 and len(self.celldata.keys()):
                on = 'cells'

        if on.startswith('p'):
            if not name: name="PointScalars"
            self._pointColors(input_array, cname, alpha, vmin, vmax, name, n)
        elif on.startswith('c'):
            if not name: name="CellScalars"
            self._cellColors(input_array, cname, alpha, vmin, vmax, name, n)
        else:
            vedo.logger.error("Must specify in cmap(on=...) either cells or points")
            raise RuntimeError()
        return self

    @deprecated(reason=vedo.colors.red+"Please use cmap(on='points')"+vedo.colors.reset)
    def pointColors(self, *args, **kwargs):
        "Deprecated, Please use cmap(on='points')"
        return self

    def _pointColors(self,
                     input_array,
                     cmap,
                     alpha,
                     vmin, vmax,
                     arrayName,
                     n=256,
        ):
        poly = self._data
        data = poly.GetPointData()

        if input_array is None:             # if None try to fetch the active scalars
            arr = data.GetScalars()
            if not arr:
                vedo.logger.error("cannot find any active Point array ...skip coloring.")
                return self

        elif isinstance(input_array, str):  # if a name string is passed
            arr = data.GetArray(input_array)
            if not arr:
                vedo.logger.error(f"cannot find point array {input_array} ...skip coloring.")
                return self

        elif isinstance(input_array, int):  # if an int is passed
            if input_array < data.GetNumberOfArrays():
                arr = data.GetArray(input_array)
            else:
                vedo.logger.error(f"cannot find point array at {input_array} ...skip coloring.")
                return self

        elif utils.isSequence(input_array): # if a numpy array is passed
            npts = len(input_array)
            if npts != poly.GetNumberOfPoints():
                n1 = poly.GetNumberOfPoints()
                vedo.logger.error(f"nr. of scalars {npts} != {n1} nr. of points ...skip coloring.")
                return self
            arr = utils.numpy2vtk(input_array, name=arrayName)
            data.AddArray(arr)

        elif isinstance(input_array, vtk.vtkArray): # if a vtkArray is passed
            arr = input_array
            data.AddArray(arr)

        else:
            vedo.logger.error(f"in cmap(), cannot understand input type {type(input_array)}")
            raise RuntimeError()

        ##########################
        if not arr.GetName(): # sometimes they dont have a name..
            arr.SetName("PointScalars")
        arrayName = arr.GetName()

        if arr.GetDataType() in [vtk.VTK_UNSIGNED_CHAR, vtk.VTK_UNSIGNED_SHORT,
                                 vtk.VTK_UNSIGNED_INT, vtk.VTK_UNSIGNED_LONG,
                                 vtk.VTK_UNSIGNED_LONG_LONG]:
            # dt = get_vtk_to_numpy_typemap()[arr.GetDataType()]
            # colors.printc(f"Warning in cmap(): your point array {arrayName}, "
            #               f"of data type {dt}, is not supported.", c='y')
            # make a copy as a float and add it...
            arr_float = vtk.vtkFloatArray() # fast type casting
            arr_float.ShallowCopy(arr)
            arr_float.SetName(arrayName+"_float")
            data.AddArray(arr_float)
            arr = arr_float
            arrayName = arrayName+"_float"

        if not utils.isSequence(alpha):
            alpha = [alpha]*n

        if vmin is None:
            vmin = arr.GetRange()[0]
        if vmax is None:
            vmax = arr.GetRange()[1]

        ########################### build the look-up table
        if isinstance(cmap, vtk.vtkLookupTable): # vtkLookupTable
            lut = cmap

        elif utils.isSequence(cmap):                 # manual sequence of colors
            lut = vtk.vtkLookupTable()
            lut.SetRange(vmin,vmax)
            ncols, nalpha = len(cmap), len(alpha)
            lut.SetNumberOfTableValues(ncols)
            for i, c in enumerate(cmap):
                r, g, b = colors.getColor(c)
                idx = int(i/ncols * nalpha)
                lut.SetTableValue(i, r, g, b, alpha[idx])
            lut.Build()

        else: # assume string cmap name OR matplotlib.colors.LinearSegmentedColormap
            lut = vtk.vtkLookupTable()
            lut.SetRange(vmin,vmax)
            ncols, nalpha = n, len(alpha)
            lut.SetNumberOfTableValues(ncols)
            mycols = colors.colorMap(range(ncols), cmap, 0,ncols)
            for i,c in enumerate(mycols):
                r, g, b = c
                idx = int(i/ncols * nalpha)
                lut.SetTableValue(i, r, g, b, alpha[idx])
            lut.Build()

        if self._data.GetPointData().GetScalars():
            self._data.GetPointData().GetScalars().SetLookupTable(lut)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetScalarModeToUsePointData()
        self._mapper.ScalarVisibilityOn()
        if hasattr(self._mapper, 'SetArrayName'):
            self._mapper.SetArrayName(arrayName)

        self._mapper.SetScalarRange(lut.GetRange())
        # data.SetScalars(arr)  # wrong! it deletes array in position 0, never use SetScalars
        # data.SetActiveAttribute(arrayName, 0) # boh!
        data.SetActiveScalars(arrayName)
        data.Modified()
        return self

    @deprecated(reason=vedo.colors.red+"Please use cmap(on='cells')"+vedo.colors.reset)
    def cellColors(self, *args, **kwargs): return self

    def _cellColors(self,
                   input_array,
                   cmap,
                   alpha,
                   vmin, vmax,
                   arrayName,
                   n,
        ):
        poly = self._data
        data = poly.GetCellData()

        if input_array is None:             # if None try to fetch the active scalars
            arr = data.GetScalars()
            if not arr:
                vedo.logger.error("cannot find any active cell array ...skip coloring.")
                return self

        elif isinstance(input_array, str):  # if a name string is passed
            arr = data.GetArray(input_array)
            if not arr:
                vedo.logger.error(f"cannot find cell array {input_array} ...skip coloring.")
                return self

        elif isinstance(input_array, int):  # if a int is passed
            if input_array < data.GetNumberOfArrays():
                arr = data.GetArray(input_array)
            else:
                vedo.logger.error(f"cannot find cell array at {input_array} ...skip coloring.")
                return self

        elif utils.isSequence(input_array): # if a numpy array is passed
            n = len(input_array)
            if n != poly.GetNumberOfCells():
                n1 = poly.GetNumberOfCells()
                vedo.logger.error(f"nr. of scalars {n} != {n1} nr. of cells ...skip coloring.")
                return self
            arr = utils.numpy2vtk(input_array, name=arrayName)
            data.AddArray(arr)

        elif isinstance(input_array, vtk.vtkArray): # if a vtkArray is passed
            arr = input_array
            data.AddArray(arr)

        else:
            vedo.logger.error(f"in cmap(): cannot understand input type {type(input_array)}")
            raise RuntimeError()

        ##########################
        if not arr.GetName(): # sometimes they dont have a name..
            arr.SetName("CellScalars")
        arrayName = arr.GetName()

        if arr.GetDataType() in [vtk.VTK_UNSIGNED_CHAR, vtk.VTK_UNSIGNED_SHORT,
                                 vtk.VTK_UNSIGNED_INT, vtk.VTK_UNSIGNED_LONG,
                                 vtk.VTK_UNSIGNED_LONG_LONG]:
            # dt = get_vtk_to_numpy_typemap()[arr.GetDataType()]
            # colors.printc(f"Warning in cmap(): your cell array {arrayName}, "
            #               f"of data type {dt}, is not supported.", c='y')
            # make a copy as a float and add it...
            arr_float = vtk.vtkFloatArray() # fast type casting
            arr_float.ShallowCopy(arr)
            arr_float.SetName(arrayName+"_float")
            data.AddArray(arr_float)
            arr = arr_float
            arrayName = arrayName+"_float"

        if not utils.isSequence(alpha):
            alpha = [alpha]*n

        if vmin is None:
            vmin = arr.GetRange()[0]
        if vmax is None:
            vmax = arr.GetRange()[1]

        ########################### build the look-up table
        if isinstance(cmap, vtk.vtkLookupTable):     # vtkLookupTable
            lut = cmap

        elif utils.isSequence(cmap):                 # manual sequence of colors
            lut = vtk.vtkLookupTable()
            lut.SetRange(vmin,vmax)
            ncols, nalpha = len(cmap), len(alpha)
            lut.SetNumberOfTableValues(ncols)
            for i, c in enumerate(cmap):
                r, g, b = colors.getColor(c)
                idx = int(i/ncols * nalpha)
                lut.SetTableValue(i, r, g, b, alpha[idx])
            lut.Build()

        else: # assume string cmap name OR matplotlib.colors.LinearSegmentedColormap
            lut = vtk.vtkLookupTable()
            lut.SetRange(vmin,vmax)
            ncols, nalpha = n, len(alpha)
            lut.SetNumberOfTableValues(ncols)
            mycols = colors.colorMap(range(ncols), cmap, 0,ncols)
            for i,c in enumerate(mycols):
                r, g, b = c
                idx = int(i/ncols * nalpha)
                lut.SetTableValue(i, r, g, b, alpha[idx])
            lut.Build()

        if self._data.GetCellData().GetScalars():
            self._data.GetCellData().GetScalars().SetLookupTable(lut)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetScalarModeToUseCellData()
        self._mapper.ScalarVisibilityOn()
        if hasattr(self._mapper, 'SetArrayName'):
            self._mapper.SetArrayName(arrayName)
        self._mapper.SetScalarRange(lut.GetRange())
        data.SetActiveScalars(arrayName)
        data.Modified()
        return self


    def cellIndividualColors(self, colorlist):
        """
        Colorize the faces of a mesh one by one
        passing a 1-to-1 list of colors in format [R,G,B] or [R,G,B,A].
        Colors levels and opacities must be in the range [0,255].

        A cell array named "CellIndividualColors" is automatically created.

        .. hint:: examples/basic/examples/basic/colorMeshCells.py
            .. image:: https://vedo.embl.es/images/basic/colorMeshCells.png
        """
        colorlist = np.asarray(colorlist).astype(np.uint8)
        self.celldata["CellIndividualColors"] = colorlist
        self.celldata.select("CellIndividualColors")
        return self


    def interpolateDataFrom(
            self,
            source,
            radius=None,
            N=None,
            kernel='shepard',
            exclude=('Normals',),
            on="points",
            nullStrategy=1,
            nullValue=0,
        ):
        """
        Interpolate over source to port its data onto the current object using various kernels.

        If N (number of closest points to use) is set then radius value is ignored.

        Parameters
        ----------
        kernel : str
            available kernels are [shepard, gaussian, linear]

        nullStrategy : int
            specify a strategy to use when encountering a "null" point
            during the interpolation process. Null points occur when the local neighborhood
            (of nearby points to interpolate from) is empty.

            Case 0: an output array is created that marks points
            as being valid (=1) or null (invalid =0), and the nullValue is set as well

            Case 1: the output data value(s) are set to the provided nullValue

            Case 2: simply use the closest point to perform the interpolation.

        nullValue : float
            see above.

        .. hint:: examples/advanced/interpolateMeshArray.py
            .. image:: https://vedo.embl.es/images/advanced/interpolateMeshArray.png
        """
        if radius is None and not N:
            vedo.logger.error("in interpolateDataFrom(): please set either radius or N")
            raise RuntimeError

        if on == "points":
            points = source.polydata()
        elif on == "cells":
            poly2 = vtk.vtkPolyData()
            poly2.ShallowCopy(source.polydata())
            c2p = vtk.vtkCellDataToPointData()
            c2p.SetInputData(poly2)
            c2p.Update()
            points = c2p.GetOutput()
        else:
            vedo.logger.error("in interpolateDataFrom(), on must be on points or cells")
            raise RuntimeError()

        locator = vtk.vtkPointLocator()
        locator.SetDataSet(points)
        locator.BuildLocator()

        if kernel.lower() == 'shepard':
            kern = vtk.vtkShepardKernel()
            kern.SetPowerParameter(2)
        elif kernel.lower() == 'gaussian':
            kern = vtk.vtkGaussianKernel()
            kern.SetSharpness(2)
        elif kernel.lower() == 'linear':
            kern = vtk.vtkLinearKernel()
        else:
            vedo.logger.error("available kernels are: [shepard, gaussian, linear]")
            raise RuntimeError()

        if N:
            kern.SetNumberOfPoints(N)
            kern.SetKernelFootprintToNClosest()
        else:
            kern.SetRadius(radius)

        interpolator = vtk.vtkPointInterpolator()
        interpolator.SetInputData(self.polydata())
        interpolator.SetSourceData(points)
        interpolator.SetKernel(kern)
        interpolator.SetLocator(locator)
        interpolator.PassFieldArraysOff()
        interpolator.SetNullPointsStrategy(nullStrategy)
        interpolator.SetNullValue(nullValue)
        interpolator.SetValidPointsMaskArrayName("ValidPointMask")
        for ex in exclude:
            interpolator.AddExcludedArray(ex)
        interpolator.Update()

        if on == "cells":
            p2c = vtk.vtkPointDataToCellData()
            p2c.SetInputData(interpolator.GetOutput())
            p2c.Update()
            cpoly = p2c.GetOutput()
        else:
            cpoly = interpolator.GetOutput()

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _data is
            M = vtk.vtkMatrix4x4()
            M.DeepCopy(self.GetMatrix())
            M.Invert()
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tr)
            tf.SetInputData(cpoly)
            tf.Update()
            self._update(tf.GetOutput())

        return self

    def addGaussNoise(self, sigma=1):
        """
        Add gaussian noise to point positions.
        An extra array is added named "GaussNoise" with the shifts.

        sigma : float
            nr. of standard deviations, expressed in percent of the diagonal size of mesh.
            Can also be a list [sigma_x, sigma_y, sigma_z].

        Example:
            .. code-block:: python

                from vedo import Sphere
                Sphere().pointGaussNoise(1.0).show()
        """
        sz = self.diagonalSize()
        pts = self.points()
        n = len(pts)
        ns = (np.random.randn(n, 3) * sigma) * (sz / 100)
        vpts = vtk.vtkPoints()
        vpts.SetNumberOfPoints(n)
        vpts.SetData(utils.numpy2vtk(pts + ns))
        self._data.SetPoints(vpts)
        self._data.GetPoints().Modified()
        self.addPointArray(-ns, 'GaussNoise')
        return self


    def closestPoint(self,
                     pt,
                     N=1,
                     radius=None,
                     returnPointId=False,
                     returnCellId=False,
        ):
        """
        Find the closest point(s) on a mesh given from the input point `pt`.

        Parameters
        ----------
        N : int
            if greater than 1, return a list of N ordered closest points

        radius : float
            if given, get all points within that radius. Then N is ignored.

        returnPointId : bool
            return point ID instead of coordinates

        returnCellId : bool
            return cell ID in which the closest point sits

        .. note::
            The appropriate tree search locator is built on the
            fly and cached for speed.
            If you want to reset it use ``mymesh.point_locator=None``

        .. hint:: examples/basic/align1.py, fitplanes.py, quadratic_morphing.py
        """
        #NB: every time the mesh moves or is warped the locateors are set to None
        if (N > 1 or radius) or (N==1 and returnPointId):

            poly = None
            if not self.point_locator:
                poly = self.polydata()
                self.point_locator = vtk.vtkPointLocator()
                self.point_locator.SetDataSet(poly)
                self.point_locator.BuildLocator()

            ##########
            if radius:
                vtklist = vtk.vtkIdList()
                self.point_locator.FindPointsWithinRadius(radius, pt, vtklist)
            elif N > 1:
                vtklist = vtk.vtkIdList()
                self.point_locator.FindClosestNPoints(N, pt, vtklist)
            else: # N==1 hence returnPointId==True
                ########
                return self.point_locator.FindClosestPoint(pt)
                ########

            if returnPointId:
                ########
                return utils.vtk2numpy(vtklist)
                ########
            else:
                if not poly:
                    poly = self.polydata()
                trgp = []
                for i in range(vtklist.GetNumberOfIds()):
                    trgp_ = [0, 0, 0]
                    vi = vtklist.GetId(i)
                    poly.GetPoints().GetPoint(vi, trgp_)
                    trgp.append(trgp_)
                ########
                return np.array(trgp)
                ########

        else:

            if not self.cell_locator:
                poly = self.polydata()

                # As per Miquel example with limbs the vtkStaticCellLocator doesnt work !!
                # https://discourse.vtk.org/t/vtkstaticcelllocator-problem-vtk9-0-3/7854/4
                self.cell_locator = vtk.vtkCellLocator()

#                try:
#                    self.cell_locator = vtk.vtkStaticCellLocator() # vtk7 doesn't have it
#                except:
#                    self.cell_locator = vtk.vtkCellLocator() # bugged if only 1 cell exists ? (#558)

                self.cell_locator.SetDataSet(poly)
                self.cell_locator.BuildLocator()

            trgp = [0, 0, 0]
            cid = vtk.mutable(0)
            dist2 = vtk.mutable(0)
            subid = vtk.mutable(0)
            self.cell_locator.FindClosestPoint(pt, trgp, cid, subid, dist2)

            if returnCellId:
                return int(cid)
            else:
                return np.array(trgp)


    def hausdorffDistance(self, points):
        """
        Compute the Hausdorff distance of two point sets.
        Returns a `float`.
        """
        hp = vtk.vtkHausdorffDistancePointSetFilter()
        hp.SetInputData(0, self.polydata())
        hp.SetInputData(1, points.polydata())
        hp.SetTargetDistanceMethodToPointToCell()
        hp.Update()
        return hp.GetHausdorffDistance()

    def chamferDistance(self, pcloud):
        """
        Compute the Chamfer distance of two point sets.
        Returns a `float`.
        """
        if not pcloud.point_locator:
            pcloud.point_locator = vtk.vtkPointLocator()
            pcloud.point_locator.SetDataSet(pcloud.polydata())
            pcloud.point_locator.BuildLocator()
        if not self.point_locator:
            self.point_locator = vtk.vtkPointLocator()
            self.point_locator.SetDataSet(self.polydata())
            self.point_locator.BuildLocator()

        ps1 = self.points()
        ps2 = pcloud.points()

        ids12 = []
        for p in ps1:
            pid12 = pcloud.point_locator.FindClosestPoint(p)
            ids12.append(pid12)
        deltav = ps2[ids12] - ps1
        da = np.mean(np.linalg.norm(deltav, axis=1))

        ids21 = []
        for p in ps2:
            pid21 = self.point_locator.FindClosestPoint(p)
            ids21.append(pid21)
        deltav = ps1[ids21] - ps2
        db = np.mean(np.linalg.norm(deltav, axis=1))
        return (da + db)/2



    def removeOutliers(self, radius, neighbors=5):
        """
        Remove outliers from a cloud of points within the specified `radius` search.

        Parameters
        ----------
        radius : float
            Specify the local search radius.

        neighbors : int
            Specify the number of neighbors that a point must have,
            within the specified radius, for the point to not be considered isolated.

        .. hint:: examples/basic/clustering.py
            .. image:: https://vedo.embl.es/images/basic/clustering.png
        """
        removal = vtk.vtkRadiusOutlierRemoval()
        removal.SetInputData(self.polydata())
        removal.SetRadius(radius)
        removal.SetNumberOfNeighbors(neighbors)
        removal.GenerateOutliersOff()
        removal.Update()
        return self._update(removal.GetOutput())

    def smoothMLS1D(self, f=0.2, radius=None):
        """
        Smooth mesh or points with a `Moving Least Squares` variant.
        The point data array "Variances" will contain the residue calculated for each point.
        Input mesh's polydata is modified.

        Parameters
        ----------
        f : float
            smoothing factor - typical range is [0,2].

        radius : float
            radius search in absolute units. If set then ``f`` is ignored.

        .. hint:: examples/advanced/moving_least_squares1D.py, skeletonize.py
        """
        coords = self.points()
        ncoords = len(coords)

        if radius:
            Ncp=0
        else:
            Ncp = int(ncoords * f / 10)
            if Ncp < 5:
                vedo.logger.warning(f"Please choose a fraction higher than {f}")
                Ncp = 5

        variances, newline = [], []
        for p in coords:
            points = self.closestPoint(p, N=Ncp, radius=radius)
            if len(points) < 4:
                continue

            points = np.array(points)
            pointsmean = points.mean(axis=0)  # plane center
            uu, dd, vv = np.linalg.svd(points - pointsmean)
            newp = np.dot(p - pointsmean, vv[0]) * vv[0] + pointsmean
            variances.append(dd[1] + dd[2])
            newline.append(newp)

        vdata = utils.numpy2vtk(np.array(variances))
        vdata.SetName("Variances")
        self._data.GetPointData().AddArray(vdata)
        self._data.GetPointData().Modified()
        return self.points(newline)

    def smoothMLS2D(self, f=0.2, radius=None):
        """
        Smooth mesh or points with a `Moving Least Squares` algorithm variant.
        The list ``mesh.info['variances']`` contains the residue calculated for each point.
        When a radius is specified points that are isolated will not be moved and will get
        a False entry in array ``mesh.info['isvalid']``.

        f : float
            smoothing factor - typical range is [0,2].

        radius : float
            radius search in absolute units. If set then ``f`` is ignored.

        .. hint:: examples/advanced/moving_least_squares2D.py, recosurface.py
            .. image:: https://vedo.embl.es/images/advanced/recosurface.png
        """
        coords = self.points()
        ncoords = len(coords)

        if radius:
            Ncp = 1
        else:
            Ncp = int(ncoords * f / 100)
            if Ncp < 4:
                vedo.logger.error(f"MLS2D: Please choose a fraction higher than {f}")
                Ncp = 4

        variances, newpts, valid = [], [], []
        pb = None
        if ncoords > 10000:
            pb = utils.ProgressBar(0, ncoords)
        for i, p in enumerate(coords):
            if pb:
                pb.print("smoothMLS2D working ...")
            pts = self.closestPoint(p, N=Ncp, radius=radius)
            if len(pts) > 3:
                ptsmean = pts.mean(axis=0)  # plane center
                _, dd, vv = np.linalg.svd(pts - ptsmean)
                cv = np.cross(vv[0], vv[1])
                t = (np.dot(cv, ptsmean) - np.dot(cv, p)) / np.dot(cv,cv)
                newp = p + cv*t
                newpts.append(newp)
                variances.append(dd[2])
                if radius:
                    valid.append(True)
            else:
                newpts.append(p)
                variances.append(0)
                if radius:
                    valid.append(False)

        self.info["variances"] = np.array(variances)
        self.info["isvalid"] = np.array(valid)
        return self.points(newpts)


    def smoothLloyd2D(self, interations=2, bounds=None, options='Qbb Qc Qx'):
        """Lloyd relaxation of a 2D pointcloud."""
        #Credits: https://hatarilabs.com/ih-en/
        # tutorial-to-create-a-geospatial-voronoi-sh-mesh-with-python-scipy-and-geopandas
        from scipy.spatial import Voronoi as scipy_voronoi

        def _constrain_points(points):
            #Update any points that have drifted beyond the boundaries of this space
            if bounds is not None:
                for point in points:
                    if point[0] < bounds[0]: point[0] = bounds[0]
                    if point[0] > bounds[1]: point[0] = bounds[1]
                    if point[1] < bounds[2]: point[1] = bounds[2]
                    if point[1] > bounds[3]: point[1] = bounds[3]
            return points

        def _find_centroid(vertices):
            #The equation for the method used here to find the centroid of a
            #2D polygon is given here: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
            area = 0
            centroid_x = 0
            centroid_y = 0
            for i in range(len(vertices)-1):
              step = (vertices[i  , 0] * vertices[i+1, 1]) - \
                     (vertices[i+1, 0] * vertices[i  , 1])
              centroid_x += (vertices[i, 0] + vertices[i+1, 0]) * step
              centroid_y += (vertices[i, 1] + vertices[i+1, 1]) * step
              area += step
            if area:
                centroid_x = (1.0/(3.0*area)) * centroid_x
                centroid_y = (1.0/(3.0*area)) * centroid_y
            # prevent centroids from escaping bounding box
            return _constrain_points([[centroid_x, centroid_y]])[0]

        def _relax(voronoi):
            #Moves each point to the centroid of its cell in the voronoi
            #map to "relax" the points (i.e. jitter the points so as
            #to spread them out within the space).
            centroids = []
            for idx in voronoi.point_region:
                # the region is a series of indices into voronoi.vertices
                # remove point at infinity, designated by index -1
                region = [i for i in voronoi.regions[idx] if i != -1]
                # enclose the polygon
                region = region + [region[0]]
                verts = voronoi.vertices[region]
                # find the centroid of those vertices
                centroids.append(_find_centroid(verts))
            return _constrain_points(centroids)

        if bounds is None:
            bounds = self.bounds()

        pts = self.points()[:,(0,1)]
        for i in range(interations):
            vor = scipy_voronoi(pts, qhull_options=options)
            _constrain_points(vor.vertices)
            pts = _relax(vor)
        # m = vedo.Mesh([pts, self.faces()]) # not yet working properly
        return Points(pts, c='k')


    def projectOnPlane(self, plane='z', point=None, direction=None):
        """
        Project the mesh on one of the Cartesian planes.

        plane : str,Plane
            if plane is `str`, plane can be one of ['x', 'y', 'z'],
            represents x-plane, y-plane and z-plane, respectively.
            Otherwise, plane should be an instance of `vedo.shapes.Plane`.

         point : float,array
             if plane is `str`, point should be a float represents the intercept.
            Otherwise, point is the camera point of perspective projection

        direction : array
            direction of oblique projection

        Note:
            Parameters `point` and `direction` are only used if the given plane
            is an instance of `vedo.shapes.Plane`. And one of these two params
            should be left as `None` to specify the projection type.

        Example:
            .. code-block:: python

                s.projectOnPlane(plane='z') # project to z-plane
                plane = Plane(pos=(4, 8, -4), normal=(-1, 0, 1), s=(5,5))
                s.projectOnPlane(plane=plane)                       # orthogonal projection
                s.projectOnPlane(plane=plane, point=(6, 6, 6))      # perspective projection
                s.projectOnPlane(plane=plane, direction=(1, 2, -1)) # oblique projection

        .. hint:: examples/basic/silhouette2.py
            .. image:: https://vedo.embl.es/images/basic/silhouette2.png
        """
        coords = self.points()

        if   'x' == plane:
            coords[:, 0] = self.GetOrigin()[0]
            intercept = self.xbounds()[0] if point is None else point
            self.x(intercept)
        elif 'y' == plane:
            coords[:, 1] = self.GetOrigin()[1]
            intercept = self.ybounds()[0] if point is None else point
            self.y(intercept)
        elif 'z' == plane:
            coords[:, 2] = self.GetOrigin()[2]
            intercept = self.zbounds()[0] if point is None else point
            self.z(intercept)

        elif isinstance(plane, vedo.shapes.Plane):
            normal = plane.normal / np.linalg.norm(plane.normal)
            pl = np.hstack((normal, -np.dot(plane.pos(), normal))).reshape(4, 1)
            if direction is None and point is None:
                # orthogonal projection
                pt = np.hstack((normal, [0])).reshape(4, 1)
                # proj_mat = pt.T @ pl * np.eye(4) - pt @ pl.T # python3 only
                proj_mat = np.matmul(pt.T, pl) * np.eye(4) - np.matmul(pt, pl.T)

            elif direction is None:
                # perspective projection
                pt = np.hstack((np.array(point), [1])).reshape(4, 1)
                # proj_mat = pt.T @ pl * np.eye(4) - pt @ pl.T
                proj_mat = np.matmul(pt.T, pl) * np.eye(4) - np.matmul(pt, pl.T)

            elif point is None:
                # oblique projection
                pt = np.hstack((np.array(direction), [0])).reshape(4, 1)
                # proj_mat = pt.T @ pl * np.eye(4) - pt @ pl.T
                proj_mat = np.matmul(pt.T, pl) * np.eye(4) - np.matmul(pt, pl.T)

            coords = np.concatenate([coords, np.ones((coords.shape[:-1] + (1,)))], axis=-1)
            # coords = coords @ proj_mat.T
            coords = np.matmul(coords, proj_mat.T)
            coords = coords[:, :3] / coords[:, 3:]

        else:
            vedo.logger.error(f"unknown plane {plane}")
            raise RuntimeError()

        self.alpha(0.1)
        self.points(coords)
        return self


    def warpToPoint(self, point, factor=0.1, absolute=True):
        """
        Modify the mesh coordinates by moving the vertices towards a specified point.

        factor : float
            value to scale displacement.
        point : array
            the position to warp towards.
        absolute : bool
            turning on causes scale factor of the new position to be one unit away from point.

        Example:
            .. code-block:: python

                from vedo import *
                s = Cylinder(height=3).wireframe()
                pt = [4,0,0]
                w = s.clone().warpToPoint(pt, factor=0.5).wireframe(False)
                show(w,s, Point(pt), axes=1)
        """
        warpTo = vtk.vtkWarpTo()
        warpTo.SetInputData(self._data)
        warpTo.SetPosition(point-self.pos())
        warpTo.SetScaleFactor(factor)
        warpTo.SetAbsolute(absolute)
        warpTo.Update()
        self.point_locator = None
        self.cell_locator = None
        return self._update(warpTo.GetOutput())

    @deprecated(reason=vedo.colors.red+"Please use mymesh.points(my_new_pts)"+vedo.colors.reset)
    def warpByVectors(self, vects, factor=1, useCells=False):
        """Deprecated. Please use mymesh.points(my_new_pts) """
        wf = vtk.vtkWarpVector()
        wf.SetInputDataObject(self.polydata())

        if useCells:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS

        vname = vects
        if utils.isSequence(vects):
            varr = utils.numpy2vtk(vects)
            vname = "WarpVectors"
            if useCells:
                self.addCellArray(varr, vname)
            else:
                self.addPointArray(varr, vname)
        wf.SetInputArrayToProcess(0, 0, 0, asso, vname)
        wf.SetScaleFactor(factor)
        wf.Update()
        self.point_locator = None
        self.cell_locator = None
        return self._update(wf.GetOutput())

    @deprecated(reason=vedo.colors.red+"Please use warp() with same syntax"+vedo.colors.reset)
    def thinPlateSpline(self, *args, **kwargs):
        """Deprecated. Please use warp() with same syntax"""
        return self.warp(*args, **kwargs)

    def warp(self, sourcePts, targetPts, sigma=1, mode="3d"):
        """
        `Thin Plate Spline` transformations describe a nonlinear warp transform defined by a set
        of source and target landmarks. Any point on the mesh close to a source landmark will
        be moved to a place close to the corresponding target landmark.
        The points in between are interpolated smoothly using
        Bookstein's Thin Plate Spline algorithm.

        Transformation object can be accessed with ``mesh.transform``.

        Parameters
        ----------
        sigma : float
            specify the 'stiffness' of the spline.

        mode : str
            set the basis function to either abs(R) (for 3d) or R2LogR (for 2d meshes)

        .. hint:: examples/advanced/warp1.py, warp3.py, interpolateField.py
            .. image:: https://vedo.embl.es/images/advanced/warp2.png
        """
        if isinstance(sourcePts, Points):
            sourcePts = sourcePts.points()
        if isinstance(targetPts, Points):
            targetPts = targetPts.points()

        ns = len(sourcePts)
        ptsou = vtk.vtkPoints()
        ptsou.SetNumberOfPoints(ns)
        for i in range(ns):
            ptsou.SetPoint(i, sourcePts[i])

        nt = len(targetPts)
        if ns != nt:
            vedo.logger.error(f"#source {ns} != {nt} #target points")
            raise RuntimeError()

        pttar = vtk.vtkPoints()
        pttar.SetNumberOfPoints(nt)
        for i in range(ns):
            pttar.SetPoint(i, targetPts[i])

        transform = vtk.vtkThinPlateSplineTransform()
        if mode.lower() == "3d":
            transform.SetBasisToR()
        elif mode.lower() == "2d":
            transform.SetBasisToR2LogR()
        else:
            vedo.logger.error(f"unknown mode {mode}")
            raise RuntimeError()

        transform.SetSigma(sigma)
        transform.SetSourceLandmarks(ptsou)
        transform.SetTargetLandmarks(pttar)
        self.transform = transform
        self.applyTransform(transform, reset=True)
        return self


    def cutWithPlane(self, origin=(0, 0, 0), normal=(1, 0, 0)):
        """
        Cut the mesh with the plane defined by a point and a normal.

        Parameters
        ----------
        origin : array
            the cutting plane goes through this point

        normal : array
            normal of the cutting plane

        Example:
            .. code-block:: python

                from vedo import Cube
                cube = Cube().cutWithPlane(normal=(1,1,1))
                cube.bc('pink').show()

        .. hint:: examples/simulations/trail.py
            .. image:: https://vedo.embl.es/images/simulations/trail.gif

        Check out also:
            ``crop()``, ``cutWithBox()``, ``cutWithCylinder()``, ``cutWithSphere()``
        """
        s = str(normal)
        if "x" in s:
            normal = (1, 0, 0)
            if '-' in s: normal = -np.array(normal)
        elif "y" in s:
            normal = (0, 1, 0)
            if '-' in s: normal = -np.array(normal)
        elif "z" in s:
            normal = (0, 0, 1)
            if '-' in s: normal = -np.array(normal)
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.polydata(True)) # must be True
        clipper.SetClipFunction(plane)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()

        cpoly = clipper.GetOutput()

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _data is
            M = vtk.vtkMatrix4x4()
            M.DeepCopy(self.GetMatrix())
            M.Invert()
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tr)
            tf.SetInputData(cpoly)
            tf.Update()
            self._update(tf.GetOutput())

        return self


    def cutWithBox(self, bounds, invert=False):
        """
        Cut the current mesh with a box. This is much faster than ``cutWithMesh()``.

        Input ``bounds`` can be either:
            - a Mesh or Points object
            - a list of 6 number representing a bounding box [xmin,xmax, ymin,ymax, zmin,zmax]
            - a list of bounding boxes like the above: [[xmin1,...], [xmin2,...], ...]

        Example:
            .. code-block:: python

                from vedo import Sphere, Cube, show
                mesh = Sphere(r=1, res=50)
                box  = Cube(side=1.5).wireframe()
                mesh.cutWithBox(box)
                show(mesh, box, axes=1)

        Check out also:
            ``crop()``, ``cutWithLine()``, ``cutWithPlane()``, ``cutWithCylinder()``
        """
        if isinstance(bounds, Points):
            bounds = bounds.GetBounds()

        box = vtk.vtkBox()
        if utils.isSequence(bounds[0]):
            for bs in bounds:
                box.AddBounds(bs)
        else:
            box.SetBounds(bounds)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.polydata(True)) # must be True
        clipper.SetClipFunction(box)
        clipper.SetInsideOut(not invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        cpoly = clipper.GetOutput()

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _data is
            M = vtk.vtkMatrix4x4()
            M.DeepCopy(self.GetMatrix())
            M.Invert()
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tr)
            tf.SetInputData(cpoly)
            tf.Update()
            self._update(tf.GetOutput())

        return self

    def cutWith2DLine(self, points, invert=False, closed=True):
        """
        Cut the current mesh with a line vertically in the z-axis direction like a cookie cutter.
        The polyline is defined by a set of points (z-coordinates are ignored).
        This is much faster than ``cutWithMesh()``.

        Check out also:
            ``crop()``, ``cutWithBox()``, ``cutWithPlane()``, ``cutWithSphere()``
        """
        pplane = vtk.vtkPolyPlane()
        if isinstance(points, Points):
            points = points.points().tolist()

        if closed:
            if isinstance(points, np.ndarray):
                points = points.tolist()
            points.append(points[0])

        vpoints = vtk.vtkPoints()
        for p in points:
            if len(p) == 2:
                p = [p[0], p[1], 0.0]
            vpoints.InsertNextPoint(p)

        n = len(points)
        polyLine = vtk.vtkPolyLine()
        polyLine.Initialize(n, vpoints)
        polyLine.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            polyLine.GetPointIds().SetId(i, i)
        pplane.SetPolyLine(polyLine)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.polydata(True)) # must be True
        clipper.SetClipFunction(pplane)
        clipper.SetInsideOut(invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        cpoly = clipper.GetOutput()

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _data is
            M = vtk.vtkMatrix4x4()
            M.DeepCopy(self.GetMatrix())
            M.Invert()
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tr)
            tf.SetInputData(cpoly)
            tf.Update()
            self._update(tf.GetOutput())

        return self

    def cutWithCylinder(self, center=(0,0,0), axis=(0,0,1), r=1, invert=False):
        """
        Cut the current mesh with an infinite cylinder.
        This is much faster than ``cutWithMesh()``.

        Parameters
        ----------
        center : array
            the center of the cylinder
        normal : array
            direction of the cylinder axis
        r : float
            radius of the cylinder

        Example:
            .. code-block:: python

                from vedo import Disc, show
                disc = Disc(r1=1, r2=1.2)
                mesh = disc.extrude(3, res=50).lineWidth(1)
                mesh.cutWithCylinder([0,0,2], r=0.4, axis='y', invert=True)
                show(mesh, axes=1)

        .. hint:: examples/simulations/optics_main1.py

        Check out also:
            ``crop()``, ``cutWithBox()``, ``cutWithPlane()``, ``cutWithSphere()``
        """
        s = str(axis)
        if "x" in s:
            axis = (1, 0, 0)
        elif "y" in s:
            axis = (0, 1, 0)
        elif "z" in s:
            axis = (0, 0, 1)
        cyl = vtk.vtkCylinder()
        cyl.SetCenter(center)
        cyl.SetAxis(axis[0], axis[1], axis[2])
        cyl.SetRadius(r)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.polydata(True)) # must be True
        clipper.SetClipFunction(cyl)
        clipper.SetInsideOut(not invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        cpoly = clipper.GetOutput()

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _data is
            M = vtk.vtkMatrix4x4()
            M.DeepCopy(self.GetMatrix())
            M.Invert()
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tr)
            tf.SetInputData(cpoly)
            tf.Update()
            self._update(tf.GetOutput())

        return self

    def cutWithSphere(self, center=(0,0,0), r=1, invert=False):
        """
        Cut the current mesh with an sphere.
        This is much faster than ``cutWithMesh()``.

        Parameters
        ----------
        center : array
            the center of the sphere

        r : float
            radius of the sphere

        Example:
            .. code-block:: python

                from vedo import Disc, show
                disc = Disc(r1=1, r2=1.2)
                mesh = disc.extrude(3, res=50).lineWidth(1)
                mesh.cutWithSphere([1,-0.7,2], r=0.5, invert=True)
                show(mesh, axes=1)

        Check out also:
            ``crop()``, ``cutWithBox()``, ``cutWithPlane()``, ``cutWithCylinder()``
        """
        sph = vtk.vtkSphere()
        sph.SetCenter(center)
        sph.SetRadius(r)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.polydata(True)) # must be True
        clipper.SetClipFunction(sph)
        clipper.SetInsideOut(not invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        cpoly = clipper.GetOutput()

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _data is
            M = vtk.vtkMatrix4x4()
            M.DeepCopy(self.GetMatrix())
            M.Invert()
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tr)
            tf.SetInputData(cpoly)
            tf.Update()
            self._update(tf.GetOutput())

        return self


    def cutWithMesh(self, mesh, invert=False, keep=False):
        """
        Cut an ``Mesh`` mesh with another ``Mesh``.

        Use ``invert`` to invert the selection.

        Use `keep` to keep the cutoff part, in this case an `Assembly` is returned:
        the "cut" object and the "discarded" part of the original object.
        You can access the via `assembly.unpack()` method.

        Example:
            .. code-block:: python

                from vedo import *
                import numpy as np
                x, y, z = np.mgrid[:30, :30, :30] / 15
                U = sin(6*x)*cos(6*y) + sin(6*y)*cos(6*z) + sin(6*z)*cos(6*x)
                iso = Volume(U).isosurface(0).smooth().c('silver').lw(1)
                cube = TessellatedBox(n=(29,29,29), spacing=(1,1,1))
                cube.cutWithMesh(iso).c('silver').alpha(1)
                show(iso, cube).close()

        Example:
            .. code-block:: python

                from vedo import *
                import numpy as np
                arr = np.random.randn(100000, 3)/2
                pts = Points(arr).c('red3').pos(5,0,0)
                cube = Cube().pos(4,0.5,0)
                assem = pts.cutWithMesh(cube, keep=True)
                show(assem.unpack(), axes=1).close()

       .. hint:: examples/advanced/cutWithMesh1.py, cutAndCap.py
           .. image:: https://vedo.embl.es/images/advanced/cutWithMesh1.jpg

       Check out also:
           ``crop()``, ``cutWithBox()``, ``cutWithPlane()``, ``cutWithCylinder()``
       """
        polymesh = mesh.polydata()
        poly = self.polydata()

        # Create an array to hold distance information
        signedDistances = vtk.vtkFloatArray()
        signedDistances.SetNumberOfComponents(1)
        signedDistances.SetName("SignedDistances")

        # implicit function that will be used to slice the mesh
        ippd = vtk.vtkImplicitPolyDataDistance()
        ippd.SetInput(polymesh)

        # Evaluate the signed distance function at all of the grid points
        for pointId in range(poly.GetNumberOfPoints()):
            p = poly.GetPoint(pointId)
            signedDistance = ippd.EvaluateFunction(p)
            signedDistances.InsertNextValue(signedDistance)

        currentscals = poly.GetPointData().GetScalars()
        if currentscals:
            currentscals = currentscals.GetName()

        poly.GetPointData().AddArray(signedDistances)
        poly.GetPointData().SetActiveScalars("SignedDistances")

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(poly)
        clipper.SetInsideOut(not invert)
        clipper.SetGenerateClippedOutput(keep)
        clipper.SetValue(0.0)
        clipper.Update()
        cpoly = clipper.GetOutput()
        if keep:
            kpoly = clipper.GetOutput(1)

        vis = False
        if currentscals:
            cpoly.GetPointData().SetActiveScalars(currentscals)
            vis = self._mapper.GetScalarVisibility()

        if self.GetIsIdentity() or cpoly.GetNumberOfPoints() == 0:
            self._update(cpoly)
        else:
            # bring the underlying polydata to where _data is
            M = vtk.vtkMatrix4x4()
            M.DeepCopy(self.GetMatrix())
            M.Invert()
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tr)
            tf.SetInputData(cpoly)
            tf.Update()
            self._update(tf.GetOutput())

        self.pointdata.remove("SignedDistances")
        self._mapper.SetScalarVisibility(vis)
        if keep:
            if isinstance(self, vedo.Mesh):
                cutoff = vedo.Mesh(kpoly)
            else:
                cutoff = vedo.Points(kpoly)
            cutoff.property = vtk.vtkProperty()
            cutoff.property.DeepCopy(self.property)
            cutoff.SetProperty(cutoff.property)
            cutoff.c('k5').alpha(0.2)
            return vedo.Assembly([self, cutoff])
        else:
            return self

    def implicitModeller(self, distance=0.05, res=(50,50,50), bounds=(), maxdist=None):
        """Find the surface which sits at the specified distance from the input one."""
        if not len(bounds):
            bounds = self.bounds()

        if not maxdist:
            maxdist = self.diagonalSize()/2

        imp = vtk.vtkImplicitModeller()
        imp.SetInputData(self.polydata())
        imp.SetSampleDimensions(res)
        imp.SetMaximumDistance(maxdist)
        imp.SetModelBounds(bounds)
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(imp.GetOutputPort())
        contour.SetValue(0, distance)
        contour.Update()
        poly = contour.GetOutput()
        return vedo.Mesh(poly, c='lb')

    def tomesh(
            self,
            resLine=None,
            resMesh=None,
            smooth=0,
            jitter=0.01,
            grid=None,
            quads=False,
            invert=False,
        ):
        """
        Generate a polygonal Mesh from a closed contour line.
        If line is not closed it will be closed with a straight segment.

        Parameters
        ----------
        resLine : int
            resolution of the contour line. The default is None, in this case
            the contour is not resampled.

        resMesh : int
            resolution of the intenal triangles not touching the boundary.
            The default is None.

        smooth : float
            smoothing of the contour before meshing. The default is 0.

        jitter : float
            add a small noise to the internal points. The default is 0.01.

        grid : Grid
            manually pass a Grid object. The default is True.

        quads : bool
            generate a mesh of quads instead of triangles.

        invert : bool
            flip the line orientation. The default is False.

        .. hint:: examples/advanced/cutWithMesh1.py, cutWithMesh2.py, line2mesh_quads.py
            .. image:: https://vedo.embl.es/images/advanced/cutWithMesh1.jpg
        """
        if resLine is None:
            contour = vedo.shapes.Line(self.points())
        else:
            contour = vedo.shapes.Spline(self.points(), smooth=smooth, res=resLine)
        contour.clean()

        length = contour.length()
        density= length/contour.N()
        vedo.logger.debug(f"tomesh():\n\tline length = {length}")
        vedo.logger.debug(f"\tdensity = {density} length/pt_separation")

        x0,x1 = contour.xbounds()
        y0,y1 = contour.ybounds()

        if grid is None:
            if resMesh is None:
                resx = int((x1-x0)/density+0.5)
                resy = int((y1-y0)/density+0.5)
                vedo.logger.debug(f"tresMesh = {[resx, resy]}")
            else:
                if utils.isSequence(resMesh):
                    resx, resy = resMesh
                else:
                    resx, resy = resMesh, resMesh
            grid = vedo.shapes.Grid([(x0+x1)/2, (y0+y1)/2, 0],
                                    s=((x1-x0)*1.025, (y1-y0)*1.025),
                                    res=(resx, resy),
            )
        else:
            grid = grid.clone()


        cpts = contour.points()

        # make sure it's closed
        p0,p1 = cpts[0], cpts[-1]
        nj = max(2, int(utils.mag(p1-p0)/density+0.5))
        joinline = vedo.shapes.Line(p1, p0, res=nj)
        contour = vedo.merge(contour, joinline).subsample(0.0001)

        ####################################### quads
        if quads:
            cmesh = grid.clone().cutWithPointLoop(contour, on='cells', invert=invert)
            return cmesh.wireframe(False).lw(0.5)
        #############################################

        grid_tmp = grid.points()

        if jitter:
            np.random.seed(0)
            sigma = 1.0/np.sqrt(grid.N())*grid.diagonalSize()*jitter
            vedo.logger.debug(f"\tsigma jittering = {sigma}")
            grid_tmp += np.random.rand(grid.N(),3) * sigma
            grid_tmp[:,2] = 0.0

        todel = []
        density /= np.sqrt(3)
        vgrid_tmp = Points(grid_tmp)

        for p in contour.points():
            out = vgrid_tmp.closestPoint(p, radius=density, returnPointId=True)
            todel += out.tolist()
        # cpoints = contour.points()
        # for i, p in enumerate(cpoints):
        #     if i:
        #         den = utils.mag(p-cpoints[i-1])/1.732
        #     else:
        #         den = density
        #     todel += vgrid_tmp.closestPoint(p, radius=den, returnPointId=True)

        grid_tmp = grid_tmp.tolist()
        for index in sorted(list(set(todel)), reverse=True):
            del grid_tmp[index]

        points = contour.points().tolist() + grid_tmp
        if invert:
            boundary = reversed(range(contour.N()))
        else:
            boundary = range(contour.N())

        dln = delaunay2D(points, mode='xy', boundaries=[boundary])
        dln.computeNormals(points=False)  # fixes reversd faces
        dln.lw(0.5)
        return dln


    def reconstructSurface(
            self,
            dims=(100,100,100),
            radius=None,
            sampleSize=None,
            holeFilling=True,
            bounds=(),
            padding=0.05,
        ):
        """
        Surface reconstruction from a scattered cloud of points.

        Parameters
        ----------
        dims : int
            number of voxels in x, y and z to control precision.

        radius : float, optiona
            radius of influence of each point.
            Smaller values generally improve performance markedly.
            Note that after the signed distance function is computed,
            any voxel taking on the value >= radius
            is presumed to be "unseen" or uninitialized.

        sampleSize : int
            if normals are not present
            they will be calculated using this sample size per point.

        holeFilling : bool
            enables hole filling, this generates
            separating surfaces between the empty and unseen portions of the volume.

        bounds : list
            region in space in which to perform the sampling
            in format (xmin,xmax, ymin,ymax, zim, zmax)

        padding : float
            increase by this fraction the bounding box

        .. hint:: examples/advanced/recosurface.py
            .. image:: https://vedo.embl.es/images/advanced/recosurface.png
        """
        if not utils.isSequence(dims):
            dims = (dims,dims,dims)

        polyData = self.polydata()

        sdf = vtk.vtkSignedDistance()

        if len(bounds)==6:
            sdf.SetBounds(bounds)
        else:
            x0, x1, y0, y1, z0, z1 = polyData.GetBounds()
            sdf.SetBounds(
                x0-(x1-x0)*padding, x1+(x1-x0)*padding,
                y0-(y1-y0)*padding, y1+(y1-y0)*padding,
                z0-(z1-z0)*padding, z1+(z1-z0)*padding
            )

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
            radius = self.diagonalSize() / (sum(dims)/3) * 5
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
        m = vedo.mesh.Mesh(surface.GetOutput(), c=self.color())
        return m


    def to_trimesh(self):
        """Return the ``trimesh`` equivalent object."""
        return utils.vedo2trimesh(self)

    def to_meshlab(self):
        """Return the ``pymeshlab.Mesh`` equivalent object."""
        return utils.vedo2meshlab(self)

    def addClustering(self, radius):
        """
        Clustering of points in space. The `radius` is the radius of local search.
        An array named "ClusterId" is added to the vertex points.

        .. hint:: examples/basic/clustering.py
            .. image:: https://vedo.embl.es/images/basic/clustering.png
        """
        cluster = vtk.vtkEuclideanClusterExtraction()
        cluster.SetInputData(self._data)
        cluster.SetExtractionModeToAllClusters()
        cluster.SetRadius(radius)
        cluster.ColorClustersOn()
        cluster.Update()
        idsarr = cluster.GetOutput().GetPointData().GetArray("ClusterId")
        self._data.GetPointData().AddArray(idsarr)
        return self

    def addConnection(self, radius, mode=0, regions=(), vrange=(0,1), seeds=(), angle=0):
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

        Parameters
        ----------
        radius : float
            variable specifying a local sphere used to define local point neighborhood

        mode : int
            - 0,  Extract all regions
            - 1,  Extract point seeded regions
            - 2,  Extract largest region
            - 3,  Test specified regions
            - 4,  Extract all regions with scalar connectivity
            - 5,  Extract point seeded regions

        regions : list
            a list of non-negative regions id to extract

        vrange : list
            scalar range to use to extract points based on scalar connectivity

        seeds : list
            a list of non-negative point seed ids

        angle : list
            points are connected if the angle between their normals is
            within this angle threshold (expressed in degrees).
        """
        # https://vtk.org/doc/nightly/html/classvtkConnectedPointsFilter.html
        cpf = vtk.vtkConnectedPointsFilter()
        cpf.SetInputData(self.polydata())
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
        return self._update(cpf.GetOutput())

    def density(self, dims=(40,40,40),
                bounds=None, radius=None,
                computeGradient=False, locator=None):
        """
        Generate a density field from a point cloud. Input can also be a set of 3D coordinates.
        Output is a ``Volume``.
        The local neighborhood is specified as the `radius` around each sample position (each voxel).
        The density is expressed as the number of counts in the radius search.

        Parameters
        ----------
        dims : int,list
            numer of voxels in x, y and z of the output Volume.

        computeGradient : bool
            Turn on/off the generation of the gradient vector,
            gradient magnitude scalar, and function classification scalar.
            By default this is off. Note that this will increase execution time
            and the size of the output. (The names of these point data arrays are:
            "Gradient", "Gradient Magnitude", and "Classification".)

        locator : vtkPointLocator
            can be assigned from a previous call for speed (access it via `object.point_locator`).

        .. hint:: examples/pyplot/plot_density3d.py
            .. image:: https://vedo.embl.es/images/pyplot/plot_density3d.png
        """
        pdf = vtk.vtkPointDensityFilter()

        poly = self.polydata()
        b = list(poly.GetBounds())
        diag = self.diagonalSize()

        if not utils.isSequence(dims):
            dims = [dims,dims,dims]

        if b[5]-b[4] == 0 or len(dims) == 2: # its 2D
            dims = list(dims)
            dims = [dims[0],dims[1], 2]
            b[5] = b[4] + diag/1000

        pdf.SetInputData(poly)
        pdf.SetSampleDimensions(dims)
        pdf.SetDensityEstimateToFixedRadius()
        pdf.SetDensityFormToNumberOfPoints()
        if locator:
            pdf.SetLocator(locator)
        if radius is None:
            radius = diag/15
        pdf.SetRadius(radius)
        if bounds is None:
            bounds = b
        pdf.SetModelBounds(bounds)
        pdf.SetComputeGradient(computeGradient)
        pdf.Update()
        img = pdf.GetOutput()
        vol = vedo.volume.Volume(img).mode(1)
        vol.name = "PointDensity"
        vol.info['radius'] = radius
        vol.locator = pdf.GetLocator()
        return vol


    def densify(self, targetDistance=0.1, nclosest=6, radius=None, niter=1, maxN=None):
        """
        Return a copy of the cloud with new added points.
        The new points are created in such a way that all points in any local neighborhood are
        within a target distance of one another.

        For each input point, the distance to all points in its neighborhood is computed.
        If any of its neighbors is further than the target distance,
        the edge connecting the point and its neighbor is bisected and
        a new point is inserted at the bisection point.
        A single pass is completed once all the input points are visited.
        Then the process repeats to the number of iterations.

        .. note::
            Points will be created in an iterative fashion until all points in their
            local neighborhood are the target distance apart or less.
            Note that the process may terminate early due to the
            number of iterations. By default the target distance is set to 0.5.
            Note that the targetDistance should be less than the radius
            or nothing will change on output.

        .. warning::
            This class can generate a lot of points very quickly.
            The maximum number of iterations is by default set to =1.0 for this reason.
            Increase the number of iterations very carefully.
            Also, `maxN` can be set to limit the explosion of points.
            It is also recommended that a N closest neighborhood is used.

        .. hint:: examples/volumetric/densifycloud.py
            .. image:: https://vedo.embl.es/images/volumetric/densifycloud.png
        """
        src = vtk.vtkProgrammableSource()
        opts = self.points()
        def _readPoints():
            output = src.GetPolyDataOutput()
            points = vtk.vtkPoints()
            for p in opts:
                points.InsertNextPoint(p)
            output.SetPoints(points)
        src.SetExecuteMethod(_readPoints)

        dens = vtk.vtkDensifyPointCloudFilter()
        dens.SetInputConnection(src.GetOutputPort())
        dens.InterpolateAttributeDataOn()
        dens.SetTargetDistance(targetDistance)
        dens.SetMaximumNumberOfIterations(niter)
        if maxN: dens.SetMaximumNumberOfPoints(maxN)

        if radius:
            dens.SetNeighborhoodTypeToRadius()
            dens.SetRadius(radius)
        elif nclosest:
            dens.SetNeighborhoodTypeToNClosest()
            dens.SetNumberOfClosestPoints(nclosest)
        else:
            vedo.logger.error("set either radius or nclosest")
            raise RuntimeError()
        dens.Update()
        pts = utils.vtk2numpy(dens.GetOutput().GetPoints().GetData())
        cld = Points(pts, c=None).pointSize(self.GetProperty().GetPointSize())
        cld.interpolateDataFrom(self, N=nclosest, radius=radius)
        cld.name = "densifiedCloud"
        return cld

    ###############################################################################
    ## stuff returning Volume
    def signedDistance(self, bounds=None, dims=(20,20,20), invert=False, maxradius=None):
        """
        Compute the ``Volume`` object whose voxels contains the signed distance from
        the point cloud. The point cloud must have Normals.

        Parameters
        ----------
        bounds : list, actor
            bounding box sizes

        dims : list
            dimensions (nr. of voxels) of the output volume.

        invert : bool
            flip the sign

        maxradius : float
            specify how far out to propagate distance calculation

        .. hint:: examples/basic/distance2mesh.py
            .. image:: https://vedo.embl.es/images/basic/distance2mesh.png
        """
        if bounds is None:
            bounds = self.GetBounds()
        if maxradius is None:
            maxradius = self.diagonalSize()/2
        dist = vtk.vtkSignedDistance()
        dist.SetInputData(self.polydata())
        dist.SetRadius(maxradius)
        dist.SetBounds(bounds)
        dist.SetDimensions(dims)
        dist.Update()
        img = dist.GetOutput()
        if invert:
            mat = vtk.vtkImageMathematics()
            mat.SetInput1Data(img)
            mat.SetOperationToMultiplyByK()
            mat.SetConstantK(-1)
            mat.Update()
            img = mat.GetOutput()

        vol = vedo.Volume(img)
        vol.name = "SignedDistanceVolume"
        return vol

    def tovolume(
            self,
            kernel='shepard',
            radius=None,
            N=None,
            bounds=None,
            nullValue=None,
            dims=(25,25,25),
        ):
        """
        Generate a ``Volume`` by interpolating a scalar
        or vector field which is only known on a scattered set of points or mesh.
        Available interpolation kernels are: shepard, gaussian, or linear.

        Parameters
        ----------
        kernel : str
            interpolation kernel type [shepard]

        radius : float
            radius of the local search

        bounds : list
            bounding box of the output Volume object

        dims : list
            dimensions of the output Volume object

        nullValue : float
            value to be assigned to invalid points

        .. hint:: examples/volumetric/interpolateVolume.py
            .. image:: https://vedo.embl.es/images/volumetric/59095175-1ec5a300-8918-11e9-8bc0-fd35c8981e2b.jpg
        """
        if radius is None and not N:
            vedo.logger.error("please set either radius or N")
            raise RuntimeError

        poly = self.polydata()

        # Create a probe volume
        probe = vtk.vtkImageData()
        probe.SetDimensions(dims)
        if bounds is None:
            bounds = poly.GetBounds()
        probe.SetOrigin(bounds[0],bounds[2],bounds[4])
        probe.SetSpacing((bounds[1]-bounds[0])/dims[0],
                         (bounds[3]-bounds[2])/dims[1],
                         (bounds[5]-bounds[4])/dims[2])

        if not self.point_locator:
            self.point_locator = vtk.vtkPointLocator()
            self.point_locator.SetDataSet(poly)
            self.point_locator.BuildLocator()

        if kernel == 'shepard':
            kern = vtk.vtkShepardKernel()
            kern.SetPowerParameter(2)
        elif kernel == 'gaussian':
            kern = vtk.vtkGaussianKernel()
        elif kernel == 'linear':
            kern = vtk.vtkLinearKernel()
        else:
            vedo.logger.error('Error in tovolume, available kernels are:')
            vedo.logger.error(' [shepard, gaussian, linear]')
            raise RuntimeError()

        if radius:
            kern.SetRadius(radius)

        interpolator = vtk.vtkPointInterpolator()
        interpolator.SetInputData(probe)
        interpolator.SetSourceData(poly)
        interpolator.SetKernel(kern)
        interpolator.SetLocator(self.point_locator)

        if N:
            kern.SetNumberOfPoints(N)
            kern.SetKernelFootprintToNClosest()
        else:
            kern.SetRadius(radius)

        if nullValue is not None:
            interpolator.SetNullValue(nullValue)
        else:
            interpolator.SetNullPointsStrategyToClosestPoint()
        interpolator.Update()
        return vedo.Volume(interpolator.GetOutput())
