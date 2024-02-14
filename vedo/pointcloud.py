#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from weakref import ref as weak_ref_to
import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import colors
from vedo import utils
from vedo.transformations import LinearTransform, NonLinearTransform
from vedo.core import PointAlgorithms
from vedo.visual import PointsVisual

__docformat__ = "google"

__doc__ = """
Submodule to work with point clouds <br>

![](https://vedo.embl.es/images/basic/pca.png)
"""

__all__ = [
    "Points",
    "Point",
    "CellCenters",
    "merge",
    "delaunay2d",  # deprecated, use .generate_delaunay2d()
    "fit_line",
    "fit_circle",
    "fit_plane",
    "fit_sphere",
    "pca_ellipse",
    "pca_ellipsoid",
]


####################################################
def merge(*meshs, flag=False):
    """
    Build a new Mesh (or Points) formed by the fusion of the inputs.

    Similar to Assembly, but in this case the input objects become a single entity.

    To keep track of the original identities of the inputs you can set `flag=True`.
    In this case a `pointdata` array of ids is added to the output with name "OriginalMeshID".

    Examples:
        - [warp1.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp1.py)

            ![](https://vedo.embl.es/images/advanced/warp1.png)

        - [value_iteration.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/value_iteration.py)

    """
    objs = [a for a in utils.flatten(meshs) if a]

    if not objs:
        return None

    idarr = []
    polyapp = vtki.new("AppendPolyData")
    for i, ob in enumerate(objs):
        polyapp.AddInputData(ob.dataset)
        if flag:
            idarr += [i] * ob.dataset.GetNumberOfPoints()
    polyapp.Update()
    mpoly = polyapp.GetOutput()

    if flag:
        varr = utils.numpy2vtk(idarr, dtype=np.uint16, name="OriginalMeshID")
        mpoly.GetPointData().AddArray(varr)

    has_mesh = False
    for ob in objs:
        if isinstance(ob, vedo.Mesh):
            has_mesh = True
            break

    if has_mesh:
        msh = vedo.Mesh(mpoly)
    else:
        msh = Points(mpoly)

    msh.copy_properties_from(objs[0])

    msh.pipeline = utils.OperationNode(
        "merge", parents=objs, comment=f"#pts {msh.dataset.GetNumberOfPoints()}"
    )
    return msh


def delaunay2d(plist, **kwargs):
    """delaunay2d() is deprecated, use Points().generate_delaunay2d() instead."""
    if isinstance(plist, Points):
        plist = plist.vertices
    else:
        plist = np.ascontiguousarray(plist)
        plist = utils.make3d(plist)
    pp = Points(plist).generate_delaunay2d(**kwargs)
    print("WARNING: delaunay2d() is deprecated, use Points().generate_delaunay2d() instead")
    return pp


def _rotate_points(points, n0=None, n1=(0, 0, 1)):
    # Rotate a set of 3D points from direction n0 to direction n1.
    # Return the rotated points and the normal to the fitting plane (if n0 is None).
    # The pointing direction of the normal in this case is arbitrary.
    points = np.asarray(points)

    if points.ndim == 1:
        points = points[np.newaxis, :]

    if len(points[0]) == 2:
        return points, (0, 0, 1)

    if n0 is None:  # fit plane
        datamean = points.mean(axis=0)
        vv = np.linalg.svd(points - datamean)[2]
        n0 = np.cross(vv[0], vv[1])

    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    l = np.linalg.norm(k)
    if not l:
        k = n0
    k /= np.linalg.norm(k)

    ct = np.dot(n0, n1)
    theta = np.arccos(ct)
    st = np.sin(theta)
    v = k * (1 - ct)

    rpoints = []
    for p in points:
        a = p * ct
        b = np.cross(k, p) * st
        c = v * np.dot(k, p)
        rpoints.append(a + b + c)

    return np.array(rpoints), n0


def fit_line(points):
    """
    Fits a line through points.

    Extra info is stored in `Line.slope`, `Line.center`, `Line.variances`.

    Examples:
        - [fitline.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/fitline.py)

            ![](https://vedo.embl.es/images/advanced/fitline.png)
    """
    if isinstance(points, Points):
        points = points.vertices
    data = np.array(points)
    datamean = data.mean(axis=0)
    _, dd, vv = np.linalg.svd(data - datamean)
    vv = vv[0] / np.linalg.norm(vv[0])
    # vv contains the first principal component, i.e. the direction
    # vector of the best fit line in the least squares sense.
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    a = np.linalg.norm(xyz_min - datamean)
    b = np.linalg.norm(xyz_max - datamean)
    p1 = datamean - a * vv
    p2 = datamean + b * vv
    line = vedo.shapes.Line(p1, p2, lw=1)
    line.slope = vv
    line.center = datamean
    line.variances = dd
    return line


def fit_circle(points):
    """
    Fits a circle through a set of 3D points, with a very fast non-iterative method.

    Returns the tuple `(center, radius, normal_to_circle)`.

    .. warning::
        trying to fit s-shaped points will inevitably lead to instabilities and
        circles of small radius.

    References:
        *J.F. Crawford, Nucl. Instr. Meth. 211, 1983, 223-225.*
    """
    data = np.asarray(points)

    offs = data.mean(axis=0)
    data, n0 = _rotate_points(data - offs)

    xi = data[:, 0]
    yi = data[:, 1]

    x = sum(xi)
    xi2 = xi * xi
    xx = sum(xi2)
    xxx = sum(xi2 * xi)

    y = sum(yi)
    yi2 = yi * yi
    yy = sum(yi2)
    yyy = sum(yi2 * yi)

    xiyi = xi * yi
    xy = sum(xiyi)
    xyy = sum(xiyi * yi)
    xxy = sum(xi * xiyi)

    N = len(xi)
    k = (xx + yy) / N

    a1 = xx - x * x / N
    b1 = xy - x * y / N
    c1 = 0.5 * (xxx + xyy - x * k)

    a2 = xy - x * y / N
    b2 = yy - y * y / N
    c2 = 0.5 * (xxy + yyy - y * k)

    d = a2 * b1 - a1 * b2
    if not d:
        return offs, 0, n0
    x0 = (b1 * c2 - b2 * c1) / d
    y0 = (c1 - a1 * x0) / b1

    R = np.sqrt(x0 * x0 + y0 * y0 - 1 / N * (2 * x0 * x + 2 * y0 * y - xx - yy))

    c, _ = _rotate_points([x0, y0, 0], (0, 0, 1), n0)

    return c[0] + offs, R, n0


def fit_plane(points, signed=False):
    """
    Fits a plane to a set of points.

    Extra info is stored in `Plane.normal`, `Plane.center`, `Plane.variance`.

    Arguments:
        signed : (bool)
            if True flip sign of the normal based on the ordering of the points

    Examples:
        - [fitline.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/fitline.py)

            ![](https://vedo.embl.es/images/advanced/fitline.png)
    """
    if isinstance(points, Points):
        points = points.vertices
    data = np.asarray(points)
    datamean = data.mean(axis=0)
    pts = data - datamean
    res = np.linalg.svd(pts)
    dd, vv = res[1], res[2]
    n = np.cross(vv[0], vv[1])
    if signed:
        v = np.zeros_like(pts)
        for i in range(len(pts) - 1):
            vi = np.cross(pts[i], pts[i + 1])
            v[i] = vi / np.linalg.norm(vi)
        ns = np.mean(v, axis=0)  # normal to the points plane
        if np.dot(n, ns) < 0:
            n = -n
    xyz_min = data.min(axis=0)
    xyz_max = data.max(axis=0)
    s = np.linalg.norm(xyz_max - xyz_min)
    pla = vedo.shapes.Plane(datamean, n, s=[s, s])
    pla.variance = dd[2]
    pla.name = "FitPlane"
    return pla


def fit_sphere(coords):
    """
    Fits a sphere to a set of points.

    Extra info is stored in `Sphere.radius`, `Sphere.center`, `Sphere.residue`.

    Examples:
        - [fitspheres1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/fitspheres1.py)

            ![](https://vedo.embl.es/images/advanced/fitspheres1.jpg)
    """
    if isinstance(coords, Points):
        coords = coords.vertices
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
    try:
        C, residue, rank, _ = np.linalg.lstsq(A, f, rcond=-1)  # solve AC=f
    except:
        C, residue, rank, _ = np.linalg.lstsq(A, f)  # solve AC=f
    if rank < 4:
        return None
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = np.sqrt(t)[0]
    center = np.array([C[0][0], C[1][0], C[2][0]])
    if len(residue) > 0:
        residue = np.sqrt(residue[0]) / n
    else:
        residue = 0
    sph = vedo.shapes.Sphere(center, radius, c=(1, 0, 0)).wireframe(1)
    sph.radius = radius
    sph.center = center
    sph.residue = residue
    sph.name = "FitSphere"
    return sph


def pca_ellipse(points, pvalue=0.673, res=60):
    """
    Create the oriented 2D ellipse that contains the fraction `pvalue` of points.
    PCA (Principal Component Analysis) is used to compute the ellipse orientation.

    Parameter `pvalue` sets the specified fraction of points inside the ellipse.
    Normalized directions are stored in `ellipse.axis1`, `ellipse.axis2`.
    Axes sizes are stored in `ellipse.va`, `ellipse.vb`

    Arguments:
        pvalue : (float)
            ellipse will include this fraction of points
        res : (int)
            resolution of the ellipse

    Examples:
        - [pca_ellipse.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/pca_ellipse.py)
        - [histo_pca.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_pca.py)

            ![](https://vedo.embl.es/images/pyplot/histo_pca.png)
    """
    from scipy.stats import f

    if isinstance(points, Points):
        coords = points.vertices
    else:
        coords = points
    if len(coords) < 4:
        vedo.logger.warning("in pca_ellipse(), there are not enough points!")
        return None

    P = np.array(coords, dtype=float)[:, (0, 1)]
    cov = np.cov(P, rowvar=0)      # covariance matrix
    _, s, R = np.linalg.svd(cov)   # singular value decomposition
    p, n = s.size, P.shape[0]
    fppf = f.ppf(pvalue, p, n - p) # f % point function
    u = np.sqrt(s * fppf / 2) * 2  # semi-axes (largest first)
    ua, ub = u
    center = utils.make3d(np.mean(P, axis=0)) # centroid of the ellipse

    t = LinearTransform(R.T * u).translate(center)
    elli = vedo.shapes.Circle(alpha=0.75, res=res)
    elli.apply_transform(t)
    elli.properties.LightingOff()

    elli.pvalue = pvalue
    elli.center = np.array([center[0], center[1], 0])
    elli.nr_of_points = n
    elli.va = ua
    elli.vb = ub
    
    # we subtract center because it's in t
    elli.axis1 = t.move([1, 0, 0]) - center
    elli.axis2 = t.move([0, 1, 0]) - center

    elli.axis1 /= np.linalg.norm(elli.axis1)
    elli.axis2 /= np.linalg.norm(elli.axis2)
    elli.name = "PCAEllipse"
    return elli


def pca_ellipsoid(points, pvalue=0.673, res=24):
    """
    Create the oriented ellipsoid that contains the fraction `pvalue` of points.
    PCA (Principal Component Analysis) is used to compute the ellipsoid orientation.

    Axes sizes can be accessed in `ellips.va`, `ellips.vb`, `ellips.vc`,
    normalized directions are stored in `ellips.axis1`, `ellips.axis2` and `ellips.axis3`.
    Center of mass is stored in `ellips.center`.

    Asphericity can be accessed in `ellips.asphericity()` and ellips.asphericity_error().
    A value of 0 means a perfect sphere.

    Arguments:
        pvalue : (float)
            ellipsoid will include this fraction of points
   
    Examples:
        [pca_ellipsoid.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/pca_ellipsoid.py)

            ![](https://vedo.embl.es/images/basic/pca.png)
    
    See also:
        `pca_ellipse()` for a 2D ellipse.
    """
    from scipy.stats import f

    if isinstance(points, Points):
        coords = points.vertices
    else:
        coords = points
    if len(coords) < 4:
        vedo.logger.warning("in pca_ellipsoid(), not enough input points!")
        return None

    P = np.array(coords, ndmin=2, dtype=float)
    cov = np.cov(P, rowvar=0)     # covariance matrix
    _, s, R = np.linalg.svd(cov)  # singular value decomposition
    p, n = s.size, P.shape[0]
    fppf = f.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p)  # f % point function
    u = np.sqrt(s*fppf)
    ua, ub, uc = u                # semi-axes (largest first)
    center = np.mean(P, axis=0)   # centroid of the hyperellipsoid

    t = LinearTransform(R.T * u).translate(center)
    elli = vedo.shapes.Ellipsoid((0,0,0), (1,0,0), (0,1,0), (0,0,1), res=res)
    elli.apply_transform(t)
    elli.alpha(0.25)
    elli.properties.LightingOff()

    elli.pvalue = pvalue
    elli.nr_of_points = n
    elli.center = center
    elli.va = ua
    elli.vb = ub
    elli.vc = uc
    # we subtract center because it's in t
    elli.axis1 = np.array(t.move([1, 0, 0])) - center
    elli.axis2 = np.array(t.move([0, 1, 0])) - center
    elli.axis3 = np.array(t.move([0, 0, 1])) - center
    elli.axis1 /= np.linalg.norm(elli.axis1)
    elli.axis2 /= np.linalg.norm(elli.axis2)
    elli.axis3 /= np.linalg.norm(elli.axis3)
    elli.name = "PCAEllipsoid"
    return elli


###################################################
def Point(pos=(0, 0, 0), r=12, c="red", alpha=1.0):
    """
    Create a simple point in space.

    .. note:: if you are creating many points you should use class `Points` instead!
    """
    try:
        pos = pos.pos()
    except AttributeError:
        pass
    pt = Points([[0, 0, 0]], r, c, alpha)
    pt.pos(pos)
    pt.name = "Point"
    return pt


###################################################
class Points(PointsVisual, PointAlgorithms):
    """Work with point clouds."""

    def __init__(self, inputobj=None, r=4, c=(0.2, 0.2, 0.2), alpha=1):
        """
        Build an object made of only vertex points for a list of 2D/3D points.
        Both shapes (N, 3) or (3, N) are accepted as input, if N>3.
        For very large point clouds a list of colors and alpha can be assigned to each
        point in the form c=[(R,G,B,A), ... ] where 0<=R<256, ... 0<=A<256.

        Arguments:
            inputobj : (list, tuple)
            r : (int)
                Point radius in units of pixels.
            c : (str, list)
                Color name or rgb tuple.
            alpha : (float)
                Transparency in range [0,1].

        Example:
            ```python
            from vedo import *

            def fibonacci_sphere(n):
                s = np.linspace(0, n, num=n, endpoint=False)
                theta = s * 2.399963229728653
                y = 1 - s * (2/(n-1))
                r = np.sqrt(1 - y * y)
                x = np.cos(theta) * r
                z = np.sin(theta) * r
                return np._c[x,y,z]

            Points(fibonacci_sphere(1000)).show(axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/fibonacci.png)
        """
        # print("INIT POINTS")
        super().__init__()

        self.name = ""
        self.filename = ""
        self.file_size = ""

        self.info = {}
        self.time = time.time()
        
        self.transform = LinearTransform()
        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None

        self.actor = vtki.vtkActor()
        self.properties = self.actor.GetProperty()
        self.properties_backface = self.actor.GetBackfaceProperty()
        self.mapper = vtki.new("PolyDataMapper")
        self.dataset = vtki.vtkPolyData()
        
        # Create weakref so actor can access this object (eg to pick/remove):
        self.actor.retrieve_object = weak_ref_to(self)

        try:
            self.properties.RenderPointsAsSpheresOn()
        except AttributeError:
            pass

        if inputobj is None:  ####################
            return
        ##########################################

        self.name = "Points"

        ######
        if isinstance(inputobj, vtki.vtkActor):
            self.dataset.DeepCopy(inputobj.GetMapper().GetInput())
            pr = vtki.vtkProperty()
            pr.DeepCopy(inputobj.GetProperty())
            self.actor.SetProperty(pr)
            self.properties = pr
            self.mapper.SetScalarVisibility(inputobj.GetMapper().GetScalarVisibility())

        elif isinstance(inputobj, vtki.vtkPolyData):
            self.dataset = inputobj
            if self.dataset.GetNumberOfCells() == 0:
                carr = vtki.vtkCellArray()
                for i in range(self.dataset.GetNumberOfPoints()):
                    carr.InsertNextCell(1)
                    carr.InsertCellPoint(i)
                self.dataset.SetVerts(carr)

        elif isinstance(inputobj, Points):
            self.dataset = inputobj.dataset
            self.copy_properties_from(inputobj)

        elif utils.is_sequence(inputobj):  # passing point coords
            self.dataset = utils.buildPolyData(utils.make3d(inputobj))

        elif isinstance(inputobj, str):
            verts = vedo.file_io.load(inputobj)
            self.filename = inputobj
            self.dataset = verts.dataset

        else:
            # try to extract the points from a generic VTK input data object
            if hasattr(inputobj, "dataset"):
                inputobj = inputobj.dataset
            try:
                vvpts = inputobj.GetPoints()
                self.dataset = vtki.vtkPolyData()
                self.dataset.SetPoints(vvpts)
                for i in range(inputobj.GetPointData().GetNumberOfArrays()):
                    arr = inputobj.GetPointData().GetArray(i)
                    self.dataset.GetPointData().AddArray(arr)
            except:
                vedo.logger.error(f"cannot build Points from type {type(inputobj)}")
                raise RuntimeError()

        self.actor.SetMapper(self.mapper)
        self.mapper.SetInputData(self.dataset)

        self.properties.SetColor(colors.get_color(c))
        self.properties.SetOpacity(alpha)
        self.properties.SetRepresentationToPoints()
        self.properties.SetPointSize(r)
        self.properties.LightingOff()

        self.pipeline = utils.OperationNode(
            self, parents=[], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )

    def _update(self, polydata, reset_locators=True):
        """Overwrite the polygonal dataset with a new vtkPolyData."""
        self.dataset = polydata
        self.mapper.SetInputData(self.dataset)
        self.mapper.Modified()
        if reset_locators:
            self.point_locator = None
            self.line_locator = None
            self.cell_locator = None
        return self

    def __str__(self):
        """Print a description of the Points/Mesh."""
        module = self.__class__.__module__
        name = self.__class__.__name__
        out = vedo.printc(
            f"{module}.{name} at ({hex(self.memory_address())})".ljust(75),
            c="g", bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m\x1b[32;1m"

        if self.name:
            out += "name".ljust(14) + ": " + self.name
            if "legend" in self.info.keys() and self.info["legend"]:
                out+= f", legend='{self.info['legend']}'"
            out += "\n"
 
        if self.filename:
            out+= "file name".ljust(14) + ": " + self.filename + "\n"

        if not self.mapper.GetScalarVisibility():
            col = utils.precision(self.properties.GetColor(), 3)
            cname = vedo.colors.get_color_name(self.properties.GetColor())
            out+= "color".ljust(14) + ": " + cname 
            out+= f", rgb={col}, alpha={self.properties.GetOpacity()}\n"
            if self.actor.GetBackfaceProperty():
                bcol = self.actor.GetBackfaceProperty().GetDiffuseColor()
                cname = vedo.colors.get_color_name(bcol)
                out+= "backface color".ljust(14) + ": " 
                out+= f"{cname}, rgb={utils.precision(bcol,3)}\n"

        npt = self.dataset.GetNumberOfPoints()
        npo, nln = self.dataset.GetNumberOfPolys(), self.dataset.GetNumberOfLines()
        out+= "elements".ljust(14) + f": vertices={npt:,} polygons={npo:,} lines={nln:,}"
        if self.dataset.GetNumberOfStrips():
            out+= f", strips={self.dataset.GetNumberOfStrips():,}"
        out+= "\n"
        if self.dataset.GetNumberOfPieces() > 1:
            out+= "pieces".ljust(14) + ": " + str(self.dataset.GetNumberOfPieces()) + "\n"

        out+= "position".ljust(14) + ": " + f"{utils.precision(self.pos(), 6)}\n"
        try:
            sc = self.transform.get_scale()
            out+= "scaling".ljust(14)  + ": "
            out+= utils.precision(sc, 6) + "\n"
        except AttributeError:
            pass

        if self.npoints:
            out+="size".ljust(14)+ ": average=" + utils.precision(self.average_size(),6)
            out+=", diagonal="+ utils.precision(self.diagonal_size(), 6)+ "\n"
            out+="center of mass".ljust(14) + ": " + utils.precision(self.center_of_mass(),6)+"\n"

        bnds = self.bounds()
        bx1, bx2 = utils.precision(bnds[0], 3), utils.precision(bnds[1], 3)
        by1, by2 = utils.precision(bnds[2], 3), utils.precision(bnds[3], 3)
        bz1, bz2 = utils.precision(bnds[4], 3), utils.precision(bnds[5], 3)
        out+= "bounds".ljust(14) + ":"
        out+= " x=(" + bx1 + ", " + bx2 + "),"
        out+= " y=(" + by1 + ", " + by2 + "),"
        out+= " z=(" + bz1 + ", " + bz2 + ")\n"

        for key in self.pointdata.keys():
            arr = self.pointdata[key]
            dim = arr.shape[1] if arr.ndim > 1 else 1
            mark_active = "pointdata"
            a_scalars = self.dataset.GetPointData().GetScalars()
            a_vectors = self.dataset.GetPointData().GetVectors()
            a_tensors = self.dataset.GetPointData().GetTensors()
            if   a_scalars and a_scalars.GetName() == key:
                mark_active += " *"
            elif a_vectors and a_vectors.GetName() == key:
                mark_active += " **"
            elif a_tensors and a_tensors.GetName() == key:
                mark_active += " ***"
            out += mark_active.ljust(14) + f': "{key}" ({arr.dtype}), dim={dim}'
            if dim == 1:
                rng = utils.precision(arr.min(), 3) + ", " + utils.precision(arr.max(), 3)
                out += f", range=({rng})\n"
            else:
                out += "\n"

        for key in self.celldata.keys():
            arr = self.celldata[key]
            dim = arr.shape[1] if arr.ndim > 1 else 1
            mark_active = "celldata"
            a_scalars = self.dataset.GetCellData().GetScalars()
            a_vectors = self.dataset.GetCellData().GetVectors()
            a_tensors = self.dataset.GetCellData().GetTensors()
            if   a_scalars and a_scalars.GetName() == key:
                mark_active += " *"
            elif a_vectors and a_vectors.GetName() == key:
                mark_active += " **"
            elif a_tensors and a_tensors.GetName() == key:
                mark_active += " ***"
            out += mark_active.ljust(14) + f': "{key}" ({arr.dtype}), dim={dim}'
            if dim == 1:
                rng = utils.precision(arr.min(), 3) + ", " + utils.precision(arr.max(), 3)
                out += f", range=({rng})\n"
            else:
                out += "\n"

        for key in self.metadata.keys():
            arr = self.metadata[key]
            if len(arr) > 3:
                out+= "metadata".ljust(14) + ": " + f'"{key}" ({len(arr)} values)\n'
            else:
                out+= "metadata".ljust(14) + ": " + f'"{key}" = {arr}\n'

        if self.picked3d is not None:
            idp = self.closest_point(self.picked3d, return_point_id=True)
            idc = self.closest_point(self.picked3d, return_cell_id=True)
            out+= "clicked point".ljust(14) + ": " + utils.precision(self.picked3d, 6)
            out+= f", pointID={idp}, cellID={idc}\n"

        return out.rstrip() + "\x1b[0m"

    def _repr_html_(self):
        """
        HTML representation of the Point cloud object for Jupyter Notebooks.

        Returns:
            HTML text with the image and some properties.
        """
        import io
        import base64
        from PIL import Image

        library_name = "vedo.pointcloud.Points"
        help_url = "https://vedo.embl.es/docs/vedo/pointcloud.html#Points"

        arr = self.thumbnail()
        im = Image.fromarray(arr)
        buffered = io.BytesIO()
        im.save(buffered, format="PNG", quality=100)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = "data:image/png;base64," + encoded
        image = f"<img src='{url}'></img>"

        bounds = "<br/>".join(
            [
                utils.precision(min_x, 4) + " ... " + utils.precision(max_x, 4)
                for min_x, max_x in zip(self.bounds()[::2], self.bounds()[1::2])
            ]
        )
        average_size = "{size:.3f}".format(size=self.average_size())

        help_text = ""
        if self.name:
            help_text += f"<b> {self.name}: &nbsp&nbsp</b>"
        help_text += '<b><a href="' + help_url + '" target="_blank">' + library_name + "</a></b>"
        if self.filename:
            dots = ""
            if len(self.filename) > 30:
                dots = "..."
            help_text += f"<br/><code><i>({dots}{self.filename[-30:]})</i></code>"

        pdata = ""
        if self.dataset.GetPointData().GetScalars():
            if self.dataset.GetPointData().GetScalars().GetName():
                name = self.dataset.GetPointData().GetScalars().GetName()
                pdata = "<tr><td><b> point data array </b></td><td>" + name + "</td></tr>"

        cdata = ""
        if self.dataset.GetCellData().GetScalars():
            if self.dataset.GetCellData().GetScalars().GetName():
                name = self.dataset.GetCellData().GetScalars().GetName()
                cdata = "<tr><td><b> cell data array </b></td><td>" + name + "</td></tr>"

        allt = [
            "<table>",
            "<tr>",
            "<td>",
            image,
            "</td>",
            "<td style='text-align: center; vertical-align: center;'><br/>",
            help_text,
            "<table>",
            "<tr><td><b> bounds </b> <br/> (x/y/z) </td><td>" + str(bounds) + "</td></tr>",
            "<tr><td><b> center of mass </b></td><td>"
            + utils.precision(self.center_of_mass(), 3)
            + "</td></tr>",
            "<tr><td><b> average size </b></td><td>" + str(average_size) + "</td></tr>",
            "<tr><td><b> nr. points </b></td><td>" + str(self.npoints) + "</td></tr>",
            pdata,
            cdata,
            "</table>",
            "</table>",
        ]
        return "\n".join(allt)

    ##################################################################################
    def __add__(self, meshs):
        """
        Add two meshes or a list of meshes together to form an `Assembly` object.
        """
        if isinstance(meshs, list):
            alist = [self]
            for l in meshs:
                if isinstance(l, vedo.Assembly):
                    alist += l.unpack()
                else:
                    alist += l
            return vedo.assembly.Assembly(alist)

        if isinstance(meshs, vedo.Assembly):
            return meshs + self  # use Assembly.__add__

        return vedo.assembly.Assembly([self, meshs])

    def polydata(self, **kwargs):
        """
        Obsolete. Use property `.dataset` instead.
        Returns the underlying `vtkPolyData` object.
        """
        colors.printc(
            "WARNING: call to .polydata() is obsolete, use property .dataset instead.",
            c="y")
        return self.dataset

    def __copy__(self):
        return self.clone(deep=False)

    def __deepcopy__(self, memo):
        return self.clone(deep=memo)
    
    def copy(self, deep=True):
        """Return a copy of the object. Alias of `clone()`."""
        return self.clone(deep=deep)

    def clone(self, deep=True):
        """
        Clone a `PointCloud` or `Mesh` object to make an exact copy of it.
        Alias of `copy()`.

        Arguments:
            deep : (bool)
                if False return a shallow copy of the mesh without copying the points array.

        Examples:
            - [mirror.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mirror.py)

               ![](https://vedo.embl.es/images/basic/mirror.png)
        """
        poly = vtki.vtkPolyData()
        if deep or isinstance(deep, dict): # if a memo object is passed this checks as True
            poly.DeepCopy(self.dataset)
        else:
            poly.ShallowCopy(self.dataset)

        if isinstance(self, vedo.Mesh):
            cloned = vedo.Mesh(poly)
        else:
            cloned = Points(poly)

        cloned.transform = self.transform.clone()

        cloned.copy_properties_from(self)

        cloned.name = str(self.name)
        cloned.filename = str(self.filename)
        cloned.info = dict(self.info)
        cloned.pipeline = utils.OperationNode("clone", parents=[self], shape="diamond", c="#edede9")

        if isinstance(deep, dict):
            deep[id(self)] = cloned

        return cloned

    def compute_normals_with_pca(self, n=20, orientation_point=None, invert=False):
        """
        Generate point normals using PCA (principal component analysis).
        This algorithm estimates a local tangent plane around each sample point p
        by considering a small neighborhood of points around p, and fitting a plane
        to the neighborhood (via PCA).

        Arguments:
            n : (int)
                neighborhood size to calculate the normal
            orientation_point : (list)
                adjust the +/- sign of the normals so that
                the normals all point towards a specified point. If None, perform a traversal
                of the point cloud and flip neighboring normals so that they are mutually consistent.
            invert : (bool)
                flip all normals
        """
        poly = self.dataset
        pcan = vtki.new("PCANormalEstimation")
        pcan.SetInputData(poly)
        pcan.SetSampleSize(n)

        if orientation_point is not None:
            pcan.SetNormalOrientationToPoint()
            pcan.SetOrientationPoint(orientation_point)
        else:
            pcan.SetNormalOrientationToGraphTraversal()

        if invert:
            pcan.FlipNormalsOn()
        pcan.Update()

        varr = pcan.GetOutput().GetPointData().GetNormals()
        varr.SetName("Normals")
        self.dataset.GetPointData().SetNormals(varr)
        self.dataset.GetPointData().Modified()
        return self

    def compute_acoplanarity(self, n=25, radius=None, on="points"):
        """
        Compute acoplanarity which is a measure of how much a local region of the mesh
        differs from a plane.
        
        The information is stored in a `pointdata` or `celldata` array with name 'Acoplanarity'.
        
        Either `n` (number of neighbour points) or `radius` (radius of local search) can be specified.
        If a radius value is given and not enough points fall inside it, then a -1 is stored.

        Example:
            ```python
            from vedo import *
            msh = ParametricShape('RandomHills')
            msh.compute_acoplanarity(radius=0.1, on='cells')
            msh.cmap("coolwarm", on='cells').add_scalarbar()
            msh.show(axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/acoplanarity.jpg)
        """
        acoplanarities = []
        if "point" in on:
            pts = self.vertices
        elif "cell" in on:
            pts = self.cell_centers
        else:
            raise ValueError(f"In compute_acoplanarity() set on to either 'cells' or 'points', not {on}")

        for p in utils.progressbar(pts, delay=5, width=15, title=f"{on} acoplanarity"):
            if n:
                data = self.closest_point(p, n=n)
                npts = n
            elif radius:
                data = self.closest_point(p, radius=radius)
                npts = len(data)

            try:
                center = data.mean(axis=0)
                res = np.linalg.svd(data - center)
                acoplanarities.append(res[1][2] / npts)
            except:
                acoplanarities.append(-1.0)

        if "point" in on:
            self.pointdata["Acoplanarity"] = np.array(acoplanarities, dtype=float)
        else:
            self.celldata["Acoplanarity"] = np.array(acoplanarities, dtype=float)
        return self

    def distance_to(self, pcloud, signed=False, invert=False, name="Distance"):
        """
        Computes the distance from one point cloud or mesh to another point cloud or mesh.
        This new `pointdata` array is saved with default name "Distance".

        Keywords `signed` and `invert` are used to compute signed distance,
        but the mesh in that case must have polygonal faces (not a simple point cloud),
        and normals must also be computed.

        Examples:
            - [distance2mesh.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/distance2mesh.py)

                ![](https://vedo.embl.es/images/basic/distance2mesh.png)
        """
        if pcloud.dataset.GetNumberOfPolys():

            poly1 = self.dataset
            poly2 = pcloud.dataset
            df = vtki.new("DistancePolyDataFilter")
            df.ComputeSecondDistanceOff()
            df.SetInputData(0, poly1)
            df.SetInputData(1, poly2)
            df.SetSignedDistance(signed)
            df.SetNegateDistance(invert)
            df.Update()
            scals = df.GetOutput().GetPointData().GetScalars()
            dists = utils.vtk2numpy(scals)

        else:  # has no polygons

            if signed:
                vedo.logger.warning("distance_to() called with signed=True but input object has no polygons")

            if not pcloud.point_locator:
                pcloud.point_locator = vtki.new("PointLocator")
                pcloud.point_locator.SetDataSet(pcloud.dataset)
                pcloud.point_locator.BuildLocator()

            ids = []
            ps1 = self.vertices
            ps2 = pcloud.vertices
            for p in ps1:
                pid = pcloud.point_locator.FindClosestPoint(p)
                ids.append(pid)

            deltas = ps2[ids] - ps1
            dists = np.linalg.norm(deltas, axis=1).astype(np.float32)
            scals = utils.numpy2vtk(dists)

        scals.SetName(name)
        self.dataset.GetPointData().AddArray(scals)
        self.dataset.GetPointData().SetActiveScalars(scals.GetName())
        rng = scals.GetRange()
        self.mapper.SetScalarRange(rng[0], rng[1])
        self.mapper.ScalarVisibilityOn()

        self.pipeline = utils.OperationNode(
            "distance_to",
            parents=[self, pcloud],
            shape="cylinder",
            comment=f"#pts {self.dataset.GetNumberOfPoints()}",
        )
        return dists

    def clean(self):
        """Clean pointcloud or mesh by removing coincident points."""
        cpd = vtki.new("CleanPolyData")
        cpd.PointMergingOn()
        cpd.ConvertLinesToPointsOff()
        cpd.ConvertPolysToLinesOff()
        cpd.ConvertStripsToPolysOff()
        cpd.SetInputData(self.dataset)
        cpd.Update()
        self._update(cpd.GetOutput())
        self.pipeline = utils.OperationNode(
            "clean", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def subsample(self, fraction, absolute=False):
        """
        Subsample a point cloud by requiring that the points
        or vertices are far apart at least by the specified fraction of the object size.
        If a Mesh is passed the polygonal faces are not removed
        but holes can appear as their vertices are removed.

        Examples:
            - [moving_least_squares1D.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares1D.py)

                ![](https://vedo.embl.es/images/advanced/moving_least_squares1D.png)

            - [recosurface.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/recosurface.py)

                ![](https://vedo.embl.es/images/advanced/recosurface.png)
        """
        if not absolute:
            if fraction > 1:
                vedo.logger.warning(
                    f"subsample(fraction=...), fraction must be < 1, but is {fraction}"
                )
            if fraction <= 0:
                return self

        cpd = vtki.new("CleanPolyData")
        cpd.PointMergingOn()
        cpd.ConvertLinesToPointsOn()
        cpd.ConvertPolysToLinesOn()
        cpd.ConvertStripsToPolysOn()
        cpd.SetInputData(self.dataset)
        if absolute:
            cpd.SetTolerance(fraction / self.diagonal_size())
            # cpd.SetToleranceIsAbsolute(absolute)
        else:
            cpd.SetTolerance(fraction)
        cpd.Update()

        ps = 2
        if self.properties.GetRepresentation() == 0:
            ps = self.properties.GetPointSize()

        self._update(cpd.GetOutput())
        self.ps(ps)

        self.pipeline = utils.OperationNode(
            "subsample", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def threshold(self, scalars, above=None, below=None, on="points"):
        """
        Extracts cells where scalar value satisfies threshold criterion.

        Arguments:
            scalars : (str)
                name of the scalars array.
            above : (float)
                minimum value of the scalar
            below : (float)
                maximum value of the scalar
            on : (str)
                if 'cells' assume array of scalars refers to cell data.

        Examples:
            - [mesh_threshold.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_threshold.py)
        """
        thres = vtki.new("Threshold")
        thres.SetInputData(self.dataset)

        if on.startswith("c"):
            asso = vtki.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            asso = vtki.vtkDataObject.FIELD_ASSOCIATION_POINTS

        thres.SetInputArrayToProcess(0, 0, 0, asso, scalars)

        if above is None and below is not None:
            try:  # vtk 9.2
                thres.ThresholdByLower(below)
            except AttributeError:  # vtk 9.3
                thres.SetUpperThreshold(below)

        elif below is None and above is not None:
            try:
                thres.ThresholdByUpper(above)
            except AttributeError:
                thres.SetLowerThreshold(above)
        else:
            try:
                thres.ThresholdBetween(above, below)
            except AttributeError:
                thres.SetUpperThreshold(below)
                thres.SetLowerThreshold(above)

        thres.Update()

        gf = vtki.new("GeometryFilter")
        gf.SetInputData(thres.GetOutput())
        gf.Update()
        self._update(gf.GetOutput())
        self.pipeline = utils.OperationNode("threshold", parents=[self])
        return self

    def quantize(self, value):
        """
        The user should input a value and all {x,y,z} coordinates
        will be quantized to that absolute grain size.
        """
        qp = vtki.new("QuantizePolyDataPoints")
        qp.SetInputData(self.dataset)
        qp.SetQFactor(value)
        qp.Update()
        self._update(qp.GetOutput())
        self.pipeline = utils.OperationNode("quantize", parents=[self])
        return self

    @property
    def vertex_normals(self):
        """
        Retrieve vertex normals as a numpy array. Same as `point_normals`.
        Check out also `compute_normals()` and `compute_normals_with_pca()`.
        """
        vtknormals = self.dataset.GetPointData().GetNormals()
        return utils.vtk2numpy(vtknormals)

    @property
    def point_normals(self):
        """
        Retrieve vertex normals as a numpy array. Same as `vertex_normals`.
        Check out also `compute_normals()` and `compute_normals_with_pca()`.
        """
        vtknormals = self.dataset.GetPointData().GetNormals()
        return utils.vtk2numpy(vtknormals)

    def align_to(self, target, iters=100, rigid=False, invert=False, use_centroids=False):
        """
        Aligned to target mesh through the `Iterative Closest Point` algorithm.

        The core of the algorithm is to match each vertex in one surface with
        the closest surface point on the other, then apply the transformation
        that modify one surface to best match the other (in the least-square sense).

        Arguments:
            rigid : (bool)
                if True do not allow scaling
            invert : (bool)
                if True start by aligning the target to the source but
                invert the transformation finally. Useful when the target is smaller
                than the source.
            use_centroids : (bool)
                start by matching the centroids of the two objects.

        Examples:
            - [align1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align1.py)

                ![](https://vedo.embl.es/images/basic/align1.png)

            - [align2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align2.py)

                ![](https://vedo.embl.es/images/basic/align2.png)
        """
        icp = vtki.new("IterativeClosestPointTransform")
        icp.SetSource(self.dataset)
        icp.SetTarget(target.dataset)
        if invert:
            icp.Inverse()
        icp.SetMaximumNumberOfIterations(iters)
        if rigid:
            icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetStartByMatchingCentroids(use_centroids)
        icp.Update()

        T = LinearTransform(icp.GetMatrix())
        self.apply_transform(T)

        self.pipeline = utils.OperationNode(
            "align_to", parents=[self, target], comment=f"rigid = {rigid}"
        )
        return self

    def align_to_bounding_box(self, msh, rigid=False):
        """
        Align the current object's bounding box to the bounding box
        of the input object.

        Use `rigid=True` to disable scaling.

        Example:
            [align6.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align6.py)
        """
        lmt = vtki.vtkLandmarkTransform()
        ss = vtki.vtkPoints()
        xss0, xss1, yss0, yss1, zss0, zss1 = self.bounds()
        for p in [
            [xss0, yss0, zss0],
            [xss1, yss0, zss0],
            [xss1, yss1, zss0],
            [xss0, yss1, zss0],
            [xss0, yss0, zss1],
            [xss1, yss0, zss1],
            [xss1, yss1, zss1],
            [xss0, yss1, zss1],
        ]:
            ss.InsertNextPoint(p)
        st = vtki.vtkPoints()
        xst0, xst1, yst0, yst1, zst0, zst1 = msh.bounds()
        for p in [
            [xst0, yst0, zst0],
            [xst1, yst0, zst0],
            [xst1, yst1, zst0],
            [xst0, yst1, zst0],
            [xst0, yst0, zst1],
            [xst1, yst0, zst1],
            [xst1, yst1, zst1],
            [xst0, yst1, zst1],
        ]:
            st.InsertNextPoint(p)

        lmt.SetSourceLandmarks(ss)
        lmt.SetTargetLandmarks(st)
        lmt.SetModeToAffine()
        if rigid:
            lmt.SetModeToRigidBody()
        lmt.Update()

        LT = LinearTransform(lmt)
        self.apply_transform(LT)
        return self

    def align_with_landmarks(
        self,
        source_landmarks,
        target_landmarks,
        rigid=False,
        affine=False,
        least_squares=False,
    ):
        """
        Transform mesh orientation and position based on a set of landmarks points.
        The algorithm finds the best matching of source points to target points
        in the mean least square sense, in one single step.

        If `affine` is True the x, y and z axes can scale independently but stay collinear.
        With least_squares they can vary orientation.

        Examples:
            - [align5.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align5.py)

                ![](https://vedo.embl.es/images/basic/align5.png)
        """

        if utils.is_sequence(source_landmarks):
            ss = vtki.vtkPoints()
            for p in source_landmarks:
                ss.InsertNextPoint(p)
        else:
            ss = source_landmarks.dataset.GetPoints()
            if least_squares:
                source_landmarks = source_landmarks.vertices

        if utils.is_sequence(target_landmarks):
            st = vtki.vtkPoints()
            for p in target_landmarks:
                st.InsertNextPoint(p)
        else:
            st = target_landmarks.GetPoints()
            if least_squares:
                target_landmarks = target_landmarks.vertices

        if ss.GetNumberOfPoints() != st.GetNumberOfPoints():
            n1 = ss.GetNumberOfPoints()
            n2 = st.GetNumberOfPoints()
            vedo.logger.error(f"source and target have different nr of points {n1} vs {n2}")
            raise RuntimeError()

        if int(rigid) + int(affine) + int(least_squares) > 1:
            vedo.logger.error(
                "only one of rigid, affine, least_squares can be True at a time"
            )
            raise RuntimeError()

        lmt = vtki.vtkLandmarkTransform()
        lmt.SetSourceLandmarks(ss)
        lmt.SetTargetLandmarks(st)
        lmt.SetModeToSimilarity()

        if rigid:
            lmt.SetModeToRigidBody()
            lmt.Update()

        elif affine:
            lmt.SetModeToAffine()
            lmt.Update()

        elif least_squares:
            cms = source_landmarks.mean(axis=0)
            cmt = target_landmarks.mean(axis=0)
            m = np.linalg.lstsq(source_landmarks - cms, target_landmarks - cmt, rcond=None)[0]
            M = vtki.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    M.SetElement(j, i, m[i][j])
            lmt = vtki.vtkTransform()
            lmt.Translate(cmt)
            lmt.Concatenate(M)
            lmt.Translate(-cms)

        else:
            lmt.Update()

        self.apply_transform(lmt)
        self.pipeline = utils.OperationNode("transform_with_landmarks", parents=[self])
        return self

    def normalize(self):
        """Scale average size to unit. The scaling is performed around the center of mass."""
        coords = self.vertices
        if not coords.shape[0]:
            return self
        cm = np.mean(coords, axis=0)
        pts = coords - cm
        xyz2 = np.sum(pts * pts, axis=0)
        scale = 1 / np.sqrt(np.sum(xyz2) / len(pts))
        self.scale(scale, origin=cm)
        self.pipeline = utils.OperationNode("normalize", parents=[self])
        return self

    def mirror(self, axis="x", origin=True):
        """
        Mirror reflect along one of the cartesian axes

        Arguments:
            axis : (str)
                axis to use for mirroring, must be set to `x, y, z`.
                Or any combination of those.
            origin : (list)
                use this point as the origin of the mirroring transformation.

        Examples:
            - [mirror.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mirror.py)

                ![](https://vedo.embl.es/images/basic/mirror.png)
        """
        sx, sy, sz = 1, 1, 1
        if "x" in axis.lower(): sx = -1
        if "y" in axis.lower(): sy = -1
        if "z" in axis.lower(): sz = -1

        self.scale([sx, sy, sz], origin=origin)

        self.pipeline = utils.OperationNode(
            "mirror", comment=f"axis = {axis}", parents=[self])

        if sx * sy * sz < 0:
            self.reverse()
        return self

    def flip_normals(self):
        """Flip all normals orientation."""
        rs = vtki.new("ReverseSense")
        rs.SetInputData(self.dataset)
        rs.ReverseCellsOff()
        rs.ReverseNormalsOn()
        rs.Update()
        self._update(rs.GetOutput())
        self.pipeline = utils.OperationNode("flip_normals", parents=[self])
        return self

    def add_gaussian_noise(self, sigma=1.0):
        """
        Add gaussian noise to point positions.
        An extra array is added named "GaussianNoise" with the displacements.

        Arguments:
            sigma : (float)
                nr. of standard deviations, expressed in percent of the diagonal size of mesh.
                Can also be a list `[sigma_x, sigma_y, sigma_z]`.

        Example:
            ```python
            from vedo import Sphere
            Sphere().add_gaussian_noise(1.0).point_size(8).show().close()
            ```
        """
        sz = self.diagonal_size()
        pts = self.vertices
        n = len(pts)
        ns = (np.random.randn(n, 3) * sigma) * (sz / 100)
        vpts = vtki.vtkPoints()
        vpts.SetNumberOfPoints(n)
        vpts.SetData(utils.numpy2vtk(pts + ns, dtype=np.float32))
        self.dataset.SetPoints(vpts)
        self.dataset.GetPoints().Modified()
        self.pointdata["GaussianNoise"] = -ns
        self.pipeline = utils.OperationNode(
            "gaussian_noise", parents=[self], shape="egg", comment=f"sigma = {sigma}"
        )
        return self

    def closest_point(
        self, pt, n=1, radius=None, return_point_id=False, return_cell_id=False
    ):
        """
        Find the closest point(s) on a mesh given from the input point `pt`.

        Arguments:
            n : (int)
                if greater than 1, return a list of n ordered closest points
            radius : (float)
                if given, get all points within that radius. Then n is ignored.
            return_point_id : (bool)
                return point ID instead of coordinates
            return_cell_id : (bool)
                return cell ID in which the closest point sits

        Examples:
            - [align1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align1.py)
            - [fitplanes.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/fitplanes.py)
            - [quadratic_morphing.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/quadratic_morphing.py)

        .. note::
            The appropriate tree search locator is built on the fly and cached for speed.

            If you want to reset it use `mymesh.point_locator=None`
            and / or `mymesh.cell_locator=None`.
        """
        if len(pt) != 3:
            pt = [pt[0], pt[1], 0]

        # NB: every time the mesh moves or is warped the locators are set to None
        if ((n > 1 or radius) or (n == 1 and return_point_id)) and not return_cell_id:
            poly = None
            if not self.point_locator:
                poly = self.dataset
                self.point_locator = vtki.new("StaticPointLocator")
                self.point_locator.SetDataSet(poly)
                self.point_locator.BuildLocator()

            ##########
            if radius:
                vtklist = vtki.vtkIdList()
                self.point_locator.FindPointsWithinRadius(radius, pt, vtklist)
            elif n > 1:
                vtklist = vtki.vtkIdList()
                self.point_locator.FindClosestNPoints(n, pt, vtklist)
            else:  # n==1 hence return_point_id==True
                ########
                return self.point_locator.FindClosestPoint(pt)
                ########

            if return_point_id:
                ########
                return utils.vtk2numpy(vtklist)
                ########

            if not poly:
                poly = self.dataset
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
                poly = self.dataset

                # As per Miquel example with limbs the vtkStaticCellLocator doesnt work !!
                # https://discourse.vtk.org/t/vtkstaticcelllocator-problem-vtk9-0-3/7854/4
                if vedo.vtk_version[0] >= 9 and vedo.vtk_version[1] > 0:
                    self.cell_locator = vtki.new("StaticCellLocator")
                else:
                    self.cell_locator = vtki.new("CellLocator")

                self.cell_locator.SetDataSet(poly)
                self.cell_locator.BuildLocator()

            if radius is not None:
                vedo.printc("Warning: closest_point() with radius is not implemented for cells.", c='r')   
 
            if n != 1:
                vedo.printc("Warning: closest_point() with n>1 is not implemented for cells.", c='r')   
 
            trgp = [0, 0, 0]
            cid = vtki.mutable(0)
            dist2 = vtki.mutable(0)
            subid = vtki.mutable(0)
            self.cell_locator.FindClosestPoint(pt, trgp, cid, subid, dist2)

            if return_cell_id:
                return int(cid)

            return np.array(trgp)

    def auto_distance(self):
        """
        Calculate the distance to the closest point in the same cloud of points.
        The output is stored in a new pointdata array called "AutoDistance",
        and it is also returned by the function.
        """
        points = self.vertices
        if not self.point_locator:
            self.point_locator = vtki.new("StaticPointLocator")
            self.point_locator.SetDataSet(self.dataset)
            self.point_locator.BuildLocator()
        qs = []
        vtklist = vtki.vtkIdList()
        vtkpoints = self.dataset.GetPoints()
        for p in points:
            self.point_locator.FindClosestNPoints(2, p, vtklist)
            q = [0, 0, 0]
            pid = vtklist.GetId(1)
            vtkpoints.GetPoint(pid, q)
            qs.append(q)
        dists = np.linalg.norm(points - np.array(qs), axis=1)
        self.pointdata["AutoDistance"] = dists
        return dists

    def hausdorff_distance(self, points):
        """
        Compute the Hausdorff distance to the input point set.
        Returns a single `float`.

        Example:
            ```python
            from vedo import *
            t = np.linspace(0, 2*np.pi, 100)
            x = 4/3 * sin(t)**3
            y = cos(t) - cos(2*t)/3 - cos(3*t)/6 - cos(4*t)/12
            pol1 = Line(np.c_[x,y], closed=True).triangulate()
            pol2 = Polygon(nsides=5).pos(2,2)
            d12 = pol1.distance_to(pol2)
            d21 = pol2.distance_to(pol1)
            pol1.lw(0).cmap("viridis")
            pol2.lw(0).cmap("viridis")
            print("distance d12, d21 :", min(d12), min(d21))
            print("hausdorff distance:", pol1.hausdorff_distance(pol2))
            print("chamfer distance  :", pol1.chamfer_distance(pol2))
            show(pol1, pol2, axes=1)
            ```
            ![](https://vedo.embl.es/images/feats/heart.png)
        """
        hp = vtki.new("HausdorffDistancePointSetFilter")
        hp.SetInputData(0, self.dataset)
        hp.SetInputData(1, points.dataset)
        hp.SetTargetDistanceMethodToPointToCell()
        hp.Update()
        return hp.GetHausdorffDistance()

    def chamfer_distance(self, pcloud):
        """
        Compute the Chamfer distance to the input point set.

        Example:
            ```python
            from vedo import *
            cloud1 = np.random.randn(1000, 3)
            cloud2 = np.random.randn(1000, 3) + [1, 2, 3]
            c1 = Points(cloud1, r=5, c="red")
            c2 = Points(cloud2, r=5, c="green")
            d = c1.chamfer_distance(c2)
            show(f"Chamfer distance = {d}", c1, c2, axes=1).close()
            ```
        """
        # Definition of Chamfer distance may vary, here we use the average
        if not pcloud.point_locator:
            pcloud.point_locator = vtki.new("PointLocator")
            pcloud.point_locator.SetDataSet(pcloud.dataset)
            pcloud.point_locator.BuildLocator()
        if not self.point_locator:
            self.point_locator = vtki.new("PointLocator")
            self.point_locator.SetDataSet(self.dataset)
            self.point_locator.BuildLocator()

        ps1 = self.vertices
        ps2 = pcloud.vertices

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
        return (da + db) / 2

    def remove_outliers(self, radius, neighbors=5):
        """
        Remove outliers from a cloud of points within the specified `radius` search.

        Arguments:
            radius : (float)
                Specify the local search radius.
            neighbors : (int)
                Specify the number of neighbors that a point must have,
                within the specified radius, for the point to not be considered isolated.

        Examples:
            - [clustering.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/clustering.py)

                ![](https://vedo.embl.es/images/basic/clustering.png)
        """
        removal = vtki.new("RadiusOutlierRemoval")
        removal.SetInputData(self.dataset)
        removal.SetRadius(radius)
        removal.SetNumberOfNeighbors(neighbors)
        removal.GenerateOutliersOff()
        removal.Update()
        inputobj = removal.GetOutput()
        if inputobj.GetNumberOfCells() == 0:
            carr = vtki.vtkCellArray()
            for i in range(inputobj.GetNumberOfPoints()):
                carr.InsertNextCell(1)
                carr.InsertCellPoint(i)
            inputobj.SetVerts(carr)
        self._update(removal.GetOutput())
        self.pipeline = utils.OperationNode("remove_outliers", parents=[self])
        return self

    def relax_point_positions(
            self, 
            n=10,
            iters=10,
            sub_iters=10,
            packing_factor=1,
            max_step=0,
            constraints=(),
        ):
        """
        Smooth mesh or points with a 
        [Laplacian algorithm](https://vtk.org/doc/nightly/html/classvtkPointSmoothingFilter.html)
        variant. This modifies the coordinates of the input points by adjusting their positions
        to create a smooth distribution (and thereby form a pleasing packing of the points).
        Smoothing is performed by considering the effects of neighboring points on one another
        it uses a cubic cutoff function to produce repulsive forces between close points
        and attractive forces that are a little further away.
        
        In general, the larger the neighborhood size, the greater the reduction in high frequency
        information. The memory and computational requirements of the algorithm may also
        significantly increase.

        The algorithm incrementally adjusts the point positions through an iterative process.
        Basically points are moved due to the influence of neighboring points. 
        
        As points move, both the local connectivity and data attributes associated with each point
        must be updated. Rather than performing these expensive operations after every iteration,
        a number of sub-iterations can be specified. If so, then the neighborhood and attribute
        value updates occur only every sub iteration, which can improve performance significantly.
        
        Arguments:
            n : (int)
                neighborhood size to calculate the Laplacian.
            iters : (int)
                number of iterations.
            sub_iters : (int)
                number of sub-iterations, i.e. the number of times the neighborhood and attribute
                value updates occur during each iteration.
            packing_factor : (float)
                adjust convergence speed.
            max_step : (float)
                Specify the maximum smoothing step size for each smoothing iteration.
                This limits the the distance over which a point can move in each iteration.
                As in all iterative methods, the stability of the process is sensitive to this parameter.
                In general, small step size and large numbers of iterations are more stable than a larger
                step size and a smaller numbers of iterations.
            constraints : (dict)
                dictionary of constraints.
                Point constraints are used to prevent points from moving,
                or to move only on a plane. This can prevent shrinking or growing point clouds.
                If enabled, a local topological analysis is performed to determine whether a point
                should be marked as fixed" i.e., never moves, or the point only moves on a plane,
                or the point can move freely.
                If all points in the neighborhood surrounding a point are in the cone defined by
                `fixed_angle`, then the point is classified as fixed.
                If all points in the neighborhood surrounding a point are in the cone defined by
                `boundary_angle`, then the point is classified as lying on a plane.
                Angles are expressed in degrees.
        
        Example:
            ```py
            import numpy as np
            from vedo import Points, show
            from vedo.pyplot import histogram

            vpts1 = Points(np.random.rand(10_000, 3))
            dists = vpts1.auto_distance()
            h1 = histogram(dists, xlim=(0,0.08)).clone2d()

            vpts2 = vpts1.clone().relax_point_positions(n=100, iters=20, sub_iters=10)
            dists = vpts2.auto_distance()
            h2 = histogram(dists, xlim=(0,0.08)).clone2d()

            show([[vpts1, h1], [vpts2, h2]], N=2).close()
            ```
        """
        smooth = vtki.new("PointSmoothingFilter")
        smooth.SetInputData(self.dataset)
        smooth.SetSmoothingModeToUniform()
        smooth.SetNumberOfIterations(iters)
        smooth.SetNumberOfSubIterations(sub_iters)
        smooth.SetPackingFactor(packing_factor)
        if self.point_locator:
            smooth.SetLocator(self.point_locator)
        if not max_step:
            max_step = self.diagonal_size() / 100
        smooth.SetMaximumStepSize(max_step)
        smooth.SetNeighborhoodSize(n)
        if constraints:
            fixed_angle = constraints.get("fixed_angle", 45)
            boundary_angle = constraints.get("boundary_angle", 110)
            smooth.EnableConstraintsOn()
            smooth.SetFixedAngle(fixed_angle)
            smooth.SetBoundaryAngle(boundary_angle)
            smooth.GenerateConstraintScalarsOn()
            smooth.GenerateConstraintNormalsOn()
        smooth.Update()
        self._update(smooth.GetOutput())
        self.metadata["PackingRadius"] = smooth.GetPackingRadius()
        self.pipeline = utils.OperationNode("relax_point_positions", parents=[self])
        return self

    def smooth_mls_1d(self, f=0.2, radius=None, n=0):
        """
        Smooth mesh or points with a `Moving Least Squares` variant.
        The point data array "Variances" will contain the residue calculated for each point.

        Arguments:
            f : (float)
                smoothing factor - typical range is [0,2].
            radius : (float)
                radius search in absolute units.
                If set then `f` is ignored.
            n : (int)
                number of neighbours to be used for the fit.
                If set then `f` and `radius` are ignored.

        Examples:
            - [moving_least_squares1D.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares1D.py)
            - [skeletonize.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/skeletonize.py)

            ![](https://vedo.embl.es/images/advanced/moving_least_squares1D.png)
        """
        coords = self.vertices
        ncoords = len(coords)

        if n:
            Ncp = n
        elif radius:
            Ncp = 1
        else:
            Ncp = int(ncoords * f / 10)
            if Ncp < 5:
                vedo.logger.warning(f"Please choose a fraction higher than {f}")
                Ncp = 5

        variances, newline = [], []
        for p in coords:
            points = self.closest_point(p, n=Ncp, radius=radius)
            if len(points) < 4:
                continue

            points = np.array(points)
            pointsmean = points.mean(axis=0)  # plane center
            _, dd, vv = np.linalg.svd(points - pointsmean)
            newp = np.dot(p - pointsmean, vv[0]) * vv[0] + pointsmean
            variances.append(dd[1] + dd[2])
            newline.append(newp)

        self.pointdata["Variances"] = np.array(variances).astype(np.float32)
        self.vertices = newline
        self.pipeline = utils.OperationNode("smooth_mls_1d", parents=[self])
        return self

    def smooth_mls_2d(self, f=0.2, radius=None, n=0):
        """
        Smooth mesh or points with a `Moving Least Squares` algorithm variant.

        The `mesh.pointdata['MLSVariance']` array will contain the residue calculated for each point.
        When a radius is specified, points that are isolated will not be moved and will get
        a 0 entry in array `mesh.pointdata['MLSValidPoint']`.

        Arguments:
            f : (float)
                smoothing factor - typical range is [0, 2].
            radius : (float | array)
                radius search in absolute units. Can be single value (float) or sequence
                for adaptive smoothing. If set then `f` is ignored.
            n : (int)
                number of neighbours to be used for the fit.
                If set then `f` and `radius` are ignored.

        Examples:
            - [moving_least_squares2D.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares2D.py)
            - [recosurface.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/recosurface.py)

                ![](https://vedo.embl.es/images/advanced/recosurface.png)
        """
        coords = self.vertices
        ncoords = len(coords)

        if n:
            Ncp = n
            radius = None
        elif radius is not None:
            Ncp = 1
        else:
            Ncp = int(ncoords * f / 100)
            if Ncp < 4:
                vedo.logger.error(f"please choose a f-value higher than {f}")
                Ncp = 4

        variances, newpts, valid = [], [], []
        radius_is_sequence = utils.is_sequence(radius)

        pb = None
        if ncoords > 10000:
            pb = utils.ProgressBar(0, ncoords, delay=3)

        for i, p in enumerate(coords):
            if pb:
                pb.print("smooth_mls_2d working ...")
            
            # if a radius was provided for each point
            if radius_is_sequence:
                pts = self.closest_point(p, n=Ncp, radius=radius[i])
            else:
                pts = self.closest_point(p, n=Ncp, radius=radius)

            if len(pts) > 3:
                ptsmean = pts.mean(axis=0)  # plane center
                _, dd, vv = np.linalg.svd(pts - ptsmean)
                cv = np.cross(vv[0], vv[1])
                t = (np.dot(cv, ptsmean) - np.dot(cv, p)) / np.dot(cv, cv)
                newpts.append(p + cv * t)
                variances.append(dd[2])
                if radius is not None:
                    valid.append(1)
            else:
                newpts.append(p)
                variances.append(0)
                if radius is not None:
                    valid.append(0)

        if radius is not None:
            self.pointdata["MLSValidPoint"] = np.array(valid).astype(np.uint8)
        self.pointdata["MLSVariance"] = np.array(variances).astype(np.float32)

        self.vertices = newpts

        self.pipeline = utils.OperationNode("smooth_mls_2d", parents=[self])
        return self

    def smooth_lloyd_2d(self, iterations=2, bounds=None, options="Qbb Qc Qx"):
        """
        Lloyd relaxation of a 2D pointcloud.
        
        Arguments:
            iterations : (int)
                number of iterations.
            bounds : (list)
                bounding box of the domain.
            options : (str)
                options for the Qhull algorithm.
        """
        # Credits: https://hatarilabs.com/ih-en/
        # tutorial-to-create-a-geospatial-voronoi-sh-mesh-with-python-scipy-and-geopandas
        from scipy.spatial import Voronoi as scipy_voronoi

        def _constrain_points(points):
            # Update any points that have drifted beyond the boundaries of this space
            if bounds is not None:
                for point in points:
                    if point[0] < bounds[0]: point[0] = bounds[0]
                    if point[0] > bounds[1]: point[0] = bounds[1]
                    if point[1] < bounds[2]: point[1] = bounds[2]
                    if point[1] > bounds[3]: point[1] = bounds[3]
            return points

        def _find_centroid(vertices):
            # The equation for the method used here to find the centroid of a
            # 2D polygon is given here: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
            area = 0
            centroid_x = 0
            centroid_y = 0
            for i in range(len(vertices) - 1):
                step = (vertices[i, 0] * vertices[i + 1, 1]) - (vertices[i + 1, 0] * vertices[i, 1])
                centroid_x += (vertices[i, 0] + vertices[i + 1, 0]) * step
                centroid_y += (vertices[i, 1] + vertices[i + 1, 1]) * step
                area += step
            if area:
                centroid_x = (1.0 / (3.0 * area)) * centroid_x
                centroid_y = (1.0 / (3.0 * area)) * centroid_y
            # prevent centroids from escaping bounding box
            return _constrain_points([[centroid_x, centroid_y]])[0]

        def _relax(voron):
            # Moves each point to the centroid of its cell in the voronoi
            # map to "relax" the points (i.e. jitter the points so as
            # to spread them out within the space).
            centroids = []
            for idx in voron.point_region:
                # the region is a series of indices into voronoi.vertices
                # remove point at infinity, designated by index -1
                region = [i for i in voron.regions[idx] if i != -1]
                # enclose the polygon
                region = region + [region[0]]
                verts = voron.vertices[region]
                # find the centroid of those vertices
                centroids.append(_find_centroid(verts))
            return _constrain_points(centroids)

        if bounds is None:
            bounds = self.bounds()

        pts = self.vertices[:, (0, 1)]
        for i in range(iterations):
            vor = scipy_voronoi(pts, qhull_options=options)
            _constrain_points(vor.vertices)
            pts = _relax(vor)
        out = Points(pts)
        out.name = "MeshSmoothLloyd2D"
        out.pipeline = utils.OperationNode("smooth_lloyd", parents=[self])
        return out

    def project_on_plane(self, plane="z", point=None, direction=None):
        """
        Project the mesh on one of the Cartesian planes.

        Arguments:
            plane : (str, Plane)
                if plane is `str`, plane can be one of ['x', 'y', 'z'],
                represents x-plane, y-plane and z-plane, respectively.
                Otherwise, plane should be an instance of `vedo.shapes.Plane`.
            point : (float, array)
                if plane is `str`, point should be a float represents the intercept.
                Otherwise, point is the camera point of perspective projection
            direction : (array)
                direction of oblique projection

        Note:
            Parameters `point` and `direction` are only used if the given plane
            is an instance of `vedo.shapes.Plane`. And one of these two params
            should be left as `None` to specify the projection type.

        Example:
            ```python
            s.project_on_plane(plane='z') # project to z-plane
            plane = Plane(pos=(4, 8, -4), normal=(-1, 0, 1), s=(5,5))
            s.project_on_plane(plane=plane)                       # orthogonal projection
            s.project_on_plane(plane=plane, point=(6, 6, 6))      # perspective projection
            s.project_on_plane(plane=plane, direction=(1, 2, -1)) # oblique projection
            ```

        Examples:
            - [silhouette2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/silhouette2.py)

                ![](https://vedo.embl.es/images/basic/silhouette2.png)
        """
        coords = self.vertices

        if plane == "x":
            coords[:, 0] = self.transform.position[0]
            intercept = self.xbounds()[0] if point is None else point
            self.x(intercept)
        elif plane == "y":
            coords[:, 1] = self.transform.position[1]
            intercept = self.ybounds()[0] if point is None else point
            self.y(intercept)
        elif plane == "z":
            coords[:, 2] = self.transform.position[2]
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
        self.vertices = coords
        return self

    def warp(self, source, target, sigma=1.0, mode="3d"):
        """
        "Thin Plate Spline" transformations describe a nonlinear warp transform defined by a set
        of source and target landmarks. Any point on the mesh close to a source landmark will
        be moved to a place close to the corresponding target landmark.
        The points in between are interpolated smoothly using
        Bookstein's Thin Plate Spline algorithm.

        Transformation object can be accessed with `mesh.transform`.

        Arguments:
            sigma : (float)
                specify the 'stiffness' of the spline.
            mode : (str)
                set the basis function to either abs(R) (for 3d) or R2LogR (for 2d meshes)

        Examples:
            - [interpolate_field.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/interpolate_field.py)
            - [warp1.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp1.py)
            - [warp2.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp2.py)
            - [warp3.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp3.py)
            - [warp4a.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp4a.py)
            - [warp4b.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp4b.py)
            - [warp6.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp6.py)

            ![](https://vedo.embl.es/images/advanced/warp2.png)
        """
        parents = [self]

        try:
            source = source.vertices
            parents.append(source)
        except AttributeError:
            source = utils.make3d(source)
        
        try:
            target = target.vertices
            parents.append(target)
        except AttributeError:
            target = utils.make3d(target)

        ns = len(source)
        nt = len(target)
        if ns != nt:
            vedo.logger.error(f"#source {ns} != {nt} #target points")
            raise RuntimeError()

        NLT = NonLinearTransform()
        NLT.source_points = source
        NLT.target_points = target
        self.apply_transform(NLT)

        self.pipeline = utils.OperationNode("warp", parents=parents)
        return self

    def cut_with_plane(self, origin=(0, 0, 0), normal=(1, 0, 0), invert=False):
        """
        Cut the mesh with the plane defined by a point and a normal.

        Arguments:
            origin : (array)
                the cutting plane goes through this point
            normal : (array)
                normal of the cutting plane

        Example:
            ```python
            from vedo import Cube
            cube = Cube().cut_with_plane(normal=(1,1,1))
            cube.back_color('pink').show().close()
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_plane_cube.png)

        Examples:
            - [trail.py](https://github.com/marcomusy/vedo/blob/master/examples/simulations/trail.py)

                ![](https://vedo.embl.es/images/simulations/trail.gif)

        Check out also:
            `cut_with_box()`, `cut_with_cylinder()`, `cut_with_sphere()`.
        """
        s = str(normal)
        if "x" in s:
            normal = (1, 0, 0)
            if "-" in s:
                normal = -np.array(normal)
        elif "y" in s:
            normal = (0, 1, 0)
            if "-" in s:
                normal = -np.array(normal)
        elif "z" in s:
            normal = (0, 0, 1)
            if "-" in s:
                normal = -np.array(normal)
        plane = vtki.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(plane)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetInsideOut(invert)
        clipper.SetValue(0)
        clipper.Update()

        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_plane", parents=[self])
        return self

    def cut_with_planes(self, origins, normals, invert=False):
        """
        Cut the mesh with a convex set of planes defined by points and normals.

        Arguments:
            origins : (array)
                each cutting plane goes through this point
            normals : (array)
                normal of each of the cutting planes
            invert : (bool)
                if True, cut outside instead of inside

        Check out also:
            `cut_with_box()`, `cut_with_cylinder()`, `cut_with_sphere()`
        """

        vpoints = vtki.vtkPoints()
        for p in utils.make3d(origins):
            vpoints.InsertNextPoint(p)
        normals = utils.make3d(normals)

        planes = vtki.vtkPlanes()
        planes.SetPoints(vpoints)
        planes.SetNormals(utils.numpy2vtk(normals, dtype=float))

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetInsideOut(invert)
        clipper.SetClipFunction(planes)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()

        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_planes", parents=[self])
        return self

    def cut_with_box(self, bounds, invert=False):
        """
        Cut the current mesh with a box or a set of boxes.
        This is much faster than `cut_with_mesh()`.

        Input `bounds` can be either:
        - a Mesh or Points object
        - a list of 6 number representing a bounding box `[xmin,xmax, ymin,ymax, zmin,zmax]`
        - a list of bounding boxes like the above: `[[xmin1,...], [xmin2,...], ...]`

        Example:
            ```python
            from vedo import Sphere, Cube, show
            mesh = Sphere(r=1, res=50)
            box  = Cube(side=1.5).wireframe()
            mesh.cut_with_box(box)
            show(mesh, box, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_box_cube.png)

        Check out also:
            `cut_with_line()`, `cut_with_plane()`, `cut_with_cylinder()`
        """
        if isinstance(bounds, Points):
            bounds = bounds.bounds()

        box = vtki.new("Box")
        if utils.is_sequence(bounds[0]):
            for bs in bounds:
                box.AddBounds(bs)
        else:
            box.SetBounds(bounds)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(box)
        clipper.SetInsideOut(not invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_box", parents=[self])
        return self

    def cut_with_line(self, points, invert=False, closed=True):
        """
        Cut the current mesh with a line vertically in the z-axis direction like a cookie cutter.
        The polyline is defined by a set of points (z-coordinates are ignored).
        This is much faster than `cut_with_mesh()`.

        Check out also:
            `cut_with_box()`, `cut_with_plane()`, `cut_with_sphere()`
        """
        pplane = vtki.new("PolyPlane")
        if isinstance(points, Points):
            points = points.vertices.tolist()

        if closed:
            if isinstance(points, np.ndarray):
                points = points.tolist()
            points.append(points[0])

        vpoints = vtki.vtkPoints()
        for p in points:
            if len(p) == 2:
                p = [p[0], p[1], 0.0]
            vpoints.InsertNextPoint(p)

        n = len(points)
        polyline = vtki.new("PolyLine")
        polyline.Initialize(n, vpoints)
        polyline.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            polyline.GetPointIds().SetId(i, i)
        pplane.SetPolyLine(polyline)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(pplane)
        clipper.SetInsideOut(invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_line", parents=[self])
        return self

    def cut_with_cookiecutter(self, lines):
        """
        Cut the current mesh with a single line or a set of lines.

        Input `lines` can be either:
        - a `Mesh` or `Points` object
        - a list of 3D points: `[(x1,y1,z1), (x2,y2,z2), ...]`
        - a list of 2D points: `[(x1,y1), (x2,y2), ...]`

        Example:
            ```python
            from vedo import *
            grid = Mesh(dataurl + "dolfin_fine.vtk")
            grid.compute_quality().cmap("Greens")
            pols = merge(
                Polygon(nsides=10, r=0.3).pos(0.7, 0.3),
                Polygon(nsides=10, r=0.2).pos(0.3, 0.7),
            )
            lines = pols.boundaries()
            cgrid = grid.clone().cut_with_cookiecutter(lines)
            grid.alpha(0.1).wireframe()
            show(grid, cgrid, lines, axes=8, bg='blackboard').close()
            ```
            ![](https://vedo.embl.es/images/feats/cookiecutter.png)

        Check out also:
            `cut_with_line()` and `cut_with_point_loop()`

        Note:
            In case of a warning message like:
                "Mesh and trim loop point data attributes are different"
            consider interpolating the mesh point data to the loop points,
            Eg. (in the above example):
            ```python
            lines = pols.boundaries().interpolate_data_from(grid, n=2)
            ```

        Note:
            trying to invert the selection by reversing the loop order
            will have no effect in this method, hence it does not have
            the `invert` option.
        """
        if utils.is_sequence(lines):
            lines = utils.make3d(lines)
            iline = list(range(len(lines))) + [0]
            poly = utils.buildPolyData(lines, lines=[iline])
        else:
            poly = lines.dataset

        # if invert: # not working
        #     rev = vtki.new("ReverseSense")
        #     rev.ReverseCellsOn()
        #     rev.SetInputData(poly)
        #     rev.Update()
        #     poly = rev.GetOutput()

        # Build loops from the polyline
        build_loops = vtki.new("ContourLoopExtraction")
        build_loops.SetGlobalWarningDisplay(0)
        build_loops.SetInputData(poly)
        build_loops.Update()
        boundary_poly = build_loops.GetOutput()

        ccut = vtki.new("CookieCutter")
        ccut.SetInputData(self.dataset)
        ccut.SetLoopsData(boundary_poly)
        ccut.SetPointInterpolationToMeshEdges()
        # ccut.SetPointInterpolationToLoopEdges()
        ccut.PassCellDataOn()
        ccut.PassPointDataOn()
        ccut.Update()
        self._update(ccut.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_cookiecutter", parents=[self])
        return self

    def cut_with_cylinder(self, center=(0, 0, 0), axis=(0, 0, 1), r=1, invert=False):
        """
        Cut the current mesh with an infinite cylinder.
        This is much faster than `cut_with_mesh()`.

        Arguments:
            center : (array)
                the center of the cylinder
            normal : (array)
                direction of the cylinder axis
            r : (float)
                radius of the cylinder

        Example:
            ```python
            from vedo import Disc, show
            disc = Disc(r1=1, r2=1.2)
            mesh = disc.extrude(3, res=50).linewidth(1)
            mesh.cut_with_cylinder([0,0,2], r=0.4, axis='y', invert=True)
            show(mesh, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_cylinder.png)

        Examples:
            - [optics_main1.py](https://github.com/marcomusy/vedo/blob/master/examples/simulations/optics_main1.py)

        Check out also:
            `cut_with_box()`, `cut_with_plane()`, `cut_with_sphere()`
        """
        s = str(axis)
        if "x" in s:
            axis = (1, 0, 0)
        elif "y" in s:
            axis = (0, 1, 0)
        elif "z" in s:
            axis = (0, 0, 1)
        cyl = vtki.new("Cylinder")
        cyl.SetCenter(center)
        cyl.SetAxis(axis[0], axis[1], axis[2])
        cyl.SetRadius(r)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(cyl)
        clipper.SetInsideOut(not invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_cylinder", parents=[self])
        return self

    def cut_with_sphere(self, center=(0, 0, 0), r=1.0, invert=False):
        """
        Cut the current mesh with an sphere.
        This is much faster than `cut_with_mesh()`.

        Arguments:
            center : (array)
                the center of the sphere
            r : (float)
                radius of the sphere

        Example:
            ```python
            from vedo import Disc, show
            disc = Disc(r1=1, r2=1.2)
            mesh = disc.extrude(3, res=50).linewidth(1)
            mesh.cut_with_sphere([1,-0.7,2], r=1.5, invert=True)
            show(mesh, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_sphere.png)

        Check out also:
            `cut_with_box()`, `cut_with_plane()`, `cut_with_cylinder()`
        """
        sph = vtki.new("Sphere")
        sph.SetCenter(center)
        sph.SetRadius(r)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(sph)
        clipper.SetInsideOut(not invert)
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())
        self.pipeline = utils.OperationNode("cut_with_sphere", parents=[self])
        return self

    def cut_with_mesh(self, mesh, invert=False, keep=False):
        """
        Cut an `Mesh` mesh with another `Mesh`.

        Use `invert` to invert the selection.

        Use `keep` to keep the cutoff part, in this case an `Assembly` is returned:
        the "cut" object and the "discarded" part of the original object.
        You can access both via `assembly.unpack()` method.

        Example:
        ```python
        from vedo import *
        arr = np.random.randn(100000, 3)/2
        pts = Points(arr).c('red3').pos(5,0,0)
        cube = Cube().pos(4,0.5,0)
        assem = pts.cut_with_mesh(cube, keep=True)
        show(assem.unpack(), axes=1).close()
        ```
        ![](https://vedo.embl.es/images/feats/cut_with_mesh.png)

       Check out also:
            `cut_with_box()`, `cut_with_plane()`, `cut_with_cylinder()`
       """
        polymesh = mesh.dataset
        poly = self.dataset

        # Create an array to hold distance information
        signed_distances = vtki.vtkFloatArray()
        signed_distances.SetNumberOfComponents(1)
        signed_distances.SetName("SignedDistances")

        # implicit function that will be used to slice the mesh
        ippd = vtki.new("ImplicitPolyDataDistance")
        ippd.SetInput(polymesh)

        # Evaluate the signed distance function at all of the grid points
        for pointId in range(poly.GetNumberOfPoints()):
            p = poly.GetPoint(pointId)
            signed_distance = ippd.EvaluateFunction(p)
            signed_distances.InsertNextValue(signed_distance)

        currentscals = poly.GetPointData().GetScalars()
        if currentscals:
            currentscals = currentscals.GetName()

        poly.GetPointData().AddArray(signed_distances)
        poly.GetPointData().SetActiveScalars("SignedDistances")

        clipper = vtki.new("ClipPolyData")
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
            vis = self.mapper.GetScalarVisibility()

        self._update(cpoly)

        self.pointdata.remove("SignedDistances")
        self.mapper.SetScalarVisibility(vis)
        if keep:
            if isinstance(self, vedo.Mesh):
                cutoff = vedo.Mesh(kpoly)
            else:
                cutoff = vedo.Points(kpoly)
            cutoff.properties = vtki.vtkProperty()
            cutoff.properties.DeepCopy(self.properties)
            cutoff.actor.SetProperty(cutoff.properties)
            cutoff.c("k5").alpha(0.2)
            return vedo.Assembly([self, cutoff])

        self.pipeline = utils.OperationNode("cut_with_mesh", parents=[self, mesh])
        return self

    def cut_with_point_loop(
        self, points, invert=False, on="points", include_boundary=False
    ):
        """
        Cut an `Mesh` object with a set of points forming a closed loop.

        Arguments:
            invert : (bool)
                invert selection (inside-out)
            on : (str)
                if 'cells' will extract the whole cells lying inside (or outside) the point loop
            include_boundary : (bool)
                include cells lying exactly on the boundary line. Only relevant on 'cells' mode

        Examples:
            - [cut_with_points1.py](https://github.com/marcomusy/vedo/blob/master/examples/advanced/cut_with_points1.py)

                ![](https://vedo.embl.es/images/advanced/cutWithPoints1.png)

            - [cut_with_points2.py](https://github.com/marcomusy/vedo/blob/master/examples/advanced/cut_with_points2.py)

                ![](https://vedo.embl.es/images/advanced/cutWithPoints2.png)
        """
        if isinstance(points, Points):
            parents = [points]
            vpts = points.dataset.GetPoints()
            points = points.vertices
        else:
            parents = [self]
            vpts = vtki.vtkPoints()
            points = utils.make3d(points)
            for p in points:
                vpts.InsertNextPoint(p)

        if "cell" in on:
            ippd = vtki.new("ImplicitSelectionLoop")
            ippd.SetLoop(vpts)
            ippd.AutomaticNormalGenerationOn()
            clipper = vtki.new("ExtractPolyDataGeometry")
            clipper.SetInputData(self.dataset)
            clipper.SetImplicitFunction(ippd)
            clipper.SetExtractInside(not invert)
            clipper.SetExtractBoundaryCells(include_boundary)
        else:
            spol = vtki.new("SelectPolyData")
            spol.SetLoop(vpts)
            spol.GenerateSelectionScalarsOn()
            spol.GenerateUnselectedOutputOff()
            spol.SetInputData(self.dataset)
            spol.Update()
            clipper = vtki.new("ClipPolyData")
            clipper.SetInputData(spol.GetOutput())
            clipper.SetInsideOut(not invert)
            clipper.SetValue(0.0)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_pointloop", parents=parents)
        return self

    def cut_with_scalar(self, value, name="", invert=False):
        """
        Cut a mesh or point cloud with some input scalar point-data.

        Arguments:
            value : (float)
                cutting value
            name : (str)
                array name of the scalars to be used
            invert : (bool)
                flip selection

        Example:
            ```python
            from vedo import *
            s = Sphere().lw(1)
            pts = s.vertices
            scalars = np.sin(3*pts[:,2]) + pts[:,0]
            s.pointdata["somevalues"] = scalars
            s.cut_with_scalar(0.3)
            s.cmap("Spectral", "somevalues").add_scalarbar()
            s.show(axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_scalars.png)
        """
        if name:
            self.pointdata.select(name)
        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetValue(value)
        clipper.GenerateClippedOutputOff()
        clipper.SetInsideOut(not invert)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_scalar", parents=[self])
        return self

    def crop(self, top=None, bottom=None, right=None, left=None, front=None, back=None):
        """
        Crop an `Mesh` object.
        Use this method at creation (before moving the object).

        Arguments:
            top : (float)
                fraction to crop from the top plane (positive z)
            bottom : (float)
                fraction to crop from the bottom plane (negative z)
            front : (float)
                fraction to crop from the front plane (positive y)
            back : (float)
                fraction to crop from the back plane (negative y)
            right : (float)
                fraction to crop from the right plane (positive x)
            left : (float)
                fraction to crop from the left plane (negative x)

        Example:
            ```python
            from vedo import Sphere
            Sphere().crop(right=0.3, left=0.1).show()
            ```
            ![](https://user-images.githubusercontent.com/32848391/57081955-0ef1e800-6cf6-11e9-99de-b45220939bc9.png)
        """
        cu = vtki.new("Box")
        pos = np.array(self.pos())
        x0, x1, y0, y1, z0, z1 = self.bounds()
        x0, y0, z0 = [x0, y0, z0] - pos
        x1, y1, z1 = [x1, y1, z1] - pos

        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        if top:
            z1 = z1 - top * dz
        if bottom:
            z0 = z0 + bottom * dz
        if front:
            y1 = y1 - front * dy
        if back:
            y0 = y0 + back * dy
        if right:
            x1 = x1 - right * dx
        if left:
            x0 = x0 + left * dx
        bounds = (x0, x1, y0, y1, z0, z1)

        cu.SetBounds(bounds)

        clipper = vtki.new("ClipPolyData")
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(cu)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode(
            "crop", parents=[self], comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def generate_surface_halo(
            self, 
            distance=0.05,
            res=(50, 50, 50),
            bounds=(),
            maxdist=None,
    ):
        """
        Generate the surface halo which sits at the specified distance from the input one.
        Uses the `vtkImplicitModeller` class.

        Arguments:
            distance : (float)
                distance from the input surface
            res : (int)
                resolution of the surface
            bounds : (list)
                bounding box of the surface
            maxdist : (float)
                maximum distance to generate the surface
        """
        if not bounds:
            bounds = self.bounds()

        if not maxdist:
            maxdist = self.diagonal_size() / 2

        imp = vtki.new("ImplicitModeller")
        imp.SetInputData(self.dataset)
        imp.SetSampleDimensions(res)
        if maxdist:
            imp.SetMaximumDistance(maxdist)
        if len(bounds) == 6:
            imp.SetModelBounds(bounds)
        contour = vtki.new("ContourFilter")
        contour.SetInputConnection(imp.GetOutputPort())
        contour.SetValue(0, distance)
        contour.Update()
        out = vedo.Mesh(contour.GetOutput())
        out.c("lightblue").alpha(0.25).lighting("off")
        out.pipeline = utils.OperationNode("generate_surface_halo", parents=[self])
        return out

    def generate_mesh(
        self,
        line_resolution=None,
        mesh_resolution=None,
        smooth=0.0,
        jitter=0.001,
        grid=None,
        quads=False,
        invert=False,
    ):
        """
        Generate a polygonal Mesh from a closed contour line.
        If line is not closed it will be closed with a straight segment.

        Check also `generate_delaunay2d()`.

        Arguments:
            line_resolution : (int)
                resolution of the contour line. The default is None, in this case
                the contour is not resampled.
            mesh_resolution : (int)
                resolution of the internal triangles not touching the boundary.
            smooth : (float)
                smoothing of the contour before meshing.
            jitter : (float)
                add a small noise to the internal points.
            grid : (Grid)
                manually pass a Grid object. The default is True.
            quads : (bool)
                generate a mesh of quads instead of triangles.
            invert : (bool)
                flip the line orientation. The default is False.

        Examples:
            - [line2mesh_tri.py](https://github.com/marcomusy/vedo/blob/master/examples/advanced/line2mesh_tri.py)

                ![](https://vedo.embl.es/images/advanced/line2mesh_tri.jpg)

            - [line2mesh_quads.py](https://github.com/marcomusy/vedo/blob/master/examples/advanced/line2mesh_quads.py)

                ![](https://vedo.embl.es/images/advanced/line2mesh_quads.png)
        """
        if line_resolution is None:
            contour = vedo.shapes.Line(self.vertices)
        else:
            contour = vedo.shapes.Spline(self.vertices, smooth=smooth, res=line_resolution)
        contour.clean()

        length = contour.length()
        density = length / contour.npoints
        # print(f"tomesh():\n\tline length = {length}")
        # print(f"\tdensity = {density} length/pt_separation")

        x0, x1 = contour.xbounds()
        y0, y1 = contour.ybounds()

        if grid is None:
            if mesh_resolution is None:
                resx = int((x1 - x0) / density + 0.5)
                resy = int((y1 - y0) / density + 0.5)
                # print(f"tmesh_resolution = {[resx, resy]}")
            else:
                if utils.is_sequence(mesh_resolution):
                    resx, resy = mesh_resolution
                else:
                    resx, resy = mesh_resolution, mesh_resolution
            grid = vedo.shapes.Grid(
                [(x0 + x1) / 2, (y0 + y1) / 2, 0],
                s=((x1 - x0) * 1.025, (y1 - y0) * 1.025),
                res=(resx, resy),
            )
        else:
            grid = grid.clone()

        cpts = contour.vertices

        # make sure it's closed
        p0, p1 = cpts[0], cpts[-1]
        nj = max(2, int(utils.mag(p1 - p0) / density + 0.5))
        joinline = vedo.shapes.Line(p1, p0, res=nj)
        contour = vedo.merge(contour, joinline).subsample(0.0001)

        ####################################### quads
        if quads:
            cmesh = grid.clone().cut_with_point_loop(contour, on="cells", invert=invert)
            cmesh.wireframe(False).lw(0.5)
            cmesh.pipeline = utils.OperationNode(
                "generate_mesh",
                parents=[self, contour],
                comment=f"#quads {cmesh.dataset.GetNumberOfCells()}",
            )
            return cmesh
        #############################################

        grid_tmp = grid.vertices.copy()

        if jitter:
            np.random.seed(0)
            sigma = 1.0 / np.sqrt(grid.npoints) * grid.diagonal_size() * jitter
            # print(f"\tsigma jittering = {sigma}")
            grid_tmp += np.random.rand(grid.npoints, 3) * sigma
            grid_tmp[:, 2] = 0.0

        todel = []
        density /= np.sqrt(3)
        vgrid_tmp = Points(grid_tmp)

        for p in contour.vertices:
            out = vgrid_tmp.closest_point(p, radius=density, return_point_id=True)
            todel += out.tolist()

        grid_tmp = grid_tmp.tolist()
        for index in sorted(list(set(todel)), reverse=True):
            del grid_tmp[index]

        points = contour.vertices.tolist() + grid_tmp
        if invert:
            boundary = list(reversed(range(contour.npoints)))
        else:
            boundary = list(range(contour.npoints))

        dln = Points(points).generate_delaunay2d(mode="xy", boundaries=[boundary])
        dln.compute_normals(points=False)  # fixes reversd faces
        dln.lw(1)

        dln.pipeline = utils.OperationNode(
            "generate_mesh",
            parents=[self, contour],
            comment=f"#cells {dln.dataset.GetNumberOfCells()}",
        )
        return dln

    def reconstruct_surface(
        self,
        dims=(100, 100, 100),
        radius=None,
        sample_size=None,
        hole_filling=True,
        bounds=(),
        padding=0.05,
    ):
        """
        Surface reconstruction from a scattered cloud of points.

        Arguments:
            dims : (int)
                number of voxels in x, y and z to control precision.
            radius : (float)
                radius of influence of each point.
                Smaller values generally improve performance markedly.
                Note that after the signed distance function is computed,
                any voxel taking on the value >= radius
                is presumed to be "unseen" or uninitialized.
            sample_size : (int)
                if normals are not present
                they will be calculated using this sample size per point.
            hole_filling : (bool)
                enables hole filling, this generates
                separating surfaces between the empty and unseen portions of the volume.
            bounds : (list)
                region in space in which to perform the sampling
                in format (xmin,xmax, ymin,ymax, zim, zmax)
            padding : (float)
                increase by this fraction the bounding box

        Examples:
            - [recosurface.py](https://github.com/marcomusy/vedo/blob/master/examples/advanced/recosurface.py)

                ![](https://vedo.embl.es/images/advanced/recosurface.png)
        """
        if not utils.is_sequence(dims):
            dims = (dims, dims, dims)

        sdf = vtki.new("SignedDistance")

        if len(bounds) == 6:
            sdf.SetBounds(bounds)
        else:
            x0, x1, y0, y1, z0, z1 = self.bounds()
            sdf.SetBounds(
                x0 - (x1 - x0) * padding,
                x1 + (x1 - x0) * padding,
                y0 - (y1 - y0) * padding,
                y1 + (y1 - y0) * padding,
                z0 - (z1 - z0) * padding,
                z1 + (z1 - z0) * padding,
            )
        
        bb = sdf.GetBounds()
        if bb[0]==bb[1]:
            vedo.logger.warning("reconstruct_surface(): zero x-range")
        if bb[2]==bb[3]:
            vedo.logger.warning("reconstruct_surface(): zero y-range")
        if bb[4]==bb[5]:
            vedo.logger.warning("reconstruct_surface(): zero z-range")

        pd = self.dataset

        if pd.GetPointData().GetNormals():
            sdf.SetInputData(pd)
        else:
            normals = vtki.new("PCANormalEstimation")
            normals.SetInputData(pd)
            if not sample_size:
                sample_size = int(pd.GetNumberOfPoints() / 50)
            normals.SetSampleSize(sample_size)
            normals.SetNormalOrientationToGraphTraversal()
            sdf.SetInputConnection(normals.GetOutputPort())
            # print("Recalculating normals with sample size =", sample_size)

        if radius is None:
            radius = self.diagonal_size() / (sum(dims) / 3) * 5
            # print("Calculating mesh from points with radius =", radius)

        sdf.SetRadius(radius)
        sdf.SetDimensions(dims)
        sdf.Update()

        surface = vtki.new("ExtractSurface")
        surface.SetRadius(radius * 0.99)
        surface.SetHoleFilling(hole_filling)
        surface.ComputeNormalsOff()
        surface.ComputeGradientsOff()
        surface.SetInputConnection(sdf.GetOutputPort())
        surface.Update()
        m = vedo.mesh.Mesh(surface.GetOutput(), c=self.color())

        m.pipeline = utils.OperationNode(
            "reconstruct_surface",
            parents=[self],
            comment=f"#pts {m.dataset.GetNumberOfPoints()}",
        )
        return m

    def compute_clustering(self, radius):
        """
        Cluster points in space. The `radius` is the radius of local search.
        
        An array named "ClusterId" is added to `pointdata`.

        Examples:
            - [clustering.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/clustering.py)

                ![](https://vedo.embl.es/images/basic/clustering.png)
        """
        cluster = vtki.new("EuclideanClusterExtraction")
        cluster.SetInputData(self.dataset)
        cluster.SetExtractionModeToAllClusters()
        cluster.SetRadius(radius)
        cluster.ColorClustersOn()
        cluster.Update()
        idsarr = cluster.GetOutput().GetPointData().GetArray("ClusterId")
        self.dataset.GetPointData().AddArray(idsarr)

        self.pipeline = utils.OperationNode(
            "compute_clustering", parents=[self], comment=f"radius = {radius}"
        )
        return self

    def compute_connections(self, radius, mode=0, regions=(), vrange=(0, 1), seeds=(), angle=0):
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

        Arguments:
            radius : (float)
                variable specifying a local sphere used to define local point neighborhood
            mode : (int)
                - 0,  Extract all regions
                - 1,  Extract point seeded regions
                - 2,  Extract largest region
                - 3,  Test specified regions
                - 4,  Extract all regions with scalar connectivity
                - 5,  Extract point seeded regions
            regions : (list)
                a list of non-negative regions id to extract
            vrange : (list)
                scalar range to use to extract points based on scalar connectivity
            seeds : (list)
                a list of non-negative point seed ids
            angle : (list)
                points are connected if the angle between their normals is
                within this angle threshold (expressed in degrees).
        """
        # https://vtk.org/doc/nightly/html/classvtkConnectedPointsFilter.html
        cpf = vtki.new("ConnectedPointsFilter")
        cpf.SetInputData(self.dataset)
        cpf.SetRadius(radius)
        if mode == 0:  # Extract all regions
            pass

        elif mode == 1:  # Extract point seeded regions
            cpf.SetExtractionModeToPointSeededRegions()
            for s in seeds:
                cpf.AddSeed(s)

        elif mode == 2:  # Test largest region
            cpf.SetExtractionModeToLargestRegion()

        elif mode == 3:  # Test specified regions
            cpf.SetExtractionModeToSpecifiedRegions()
            for r in regions:
                cpf.AddSpecifiedRegion(r)

        elif mode == 4:  # Extract all regions with scalar connectivity
            cpf.SetExtractionModeToLargestRegion()
            cpf.ScalarConnectivityOn()
            cpf.SetScalarRange(vrange[0], vrange[1])

        elif mode == 5:  # Extract point seeded regions
            cpf.SetExtractionModeToLargestRegion()
            cpf.ScalarConnectivityOn()
            cpf.SetScalarRange(vrange[0], vrange[1])
            cpf.AlignedNormalsOn()
            cpf.SetNormalAngle(angle)

        cpf.Update()
        self._update(cpf.GetOutput(), reset_locators=False)
        return self

    def compute_camera_distance(self):
        """
        Calculate the distance from points to the camera.
        
        A pointdata array is created with name 'DistanceToCamera' and returned.
        """
        if vedo.plotter_instance.renderer:
            poly = self.dataset
            dc = vtki.new("DistanceToCamera")
            dc.SetInputData(poly)
            dc.SetRenderer(vedo.plotter_instance.renderer)
            dc.Update()
            self._update(dc.GetOutput(), reset_locators=False)
        return self.pointdata["DistanceToCamera"]

    def density(
        self, dims=(40, 40, 40), bounds=None, radius=None, compute_gradient=False, locator=None
    ):
        """
        Generate a density field from a point cloud. Input can also be a set of 3D coordinates.
        Output is a `Volume`.

        The local neighborhood is specified as the `radius` around each sample position (each voxel).
        If left to None, the radius is automatically computed as the diagonal of the bounding box
        and can be accessed via `vol.metadata["radius"]`.
        The density is expressed as the number of counts in the radius search.

        Arguments:
            dims : (int, list)
                number of voxels in x, y and z of the output Volume.
            compute_gradient : (bool)
                Turn on/off the generation of the gradient vector,
                gradient magnitude scalar, and function classification scalar.
                By default this is off. Note that this will increase execution time
                and the size of the output. (The names of these point data arrays are:
                "Gradient", "Gradient Magnitude", and "Classification")
            locator : (vtkPointLocator)
                can be assigned from a previous call for speed (access it via `object.point_locator`).

        Examples:
            - [plot_density3d.py](https://github.com/marcomusy/vedo/blob/master/examples/pyplot/plot_density3d.py)

                ![](https://vedo.embl.es/images/pyplot/plot_density3d.png)
        """
        pdf = vtki.new("PointDensityFilter")
        pdf.SetInputData(self.dataset)

        if not utils.is_sequence(dims):
            dims = [dims, dims, dims]

        if bounds is None:
            bounds = list(self.bounds())
        elif len(bounds) == 4:
            bounds = [*bounds, 0, 0]

        if bounds[5] - bounds[4] == 0 or len(dims) == 2:  # its 2D
            dims = list(dims)
            dims = [dims[0], dims[1], 2]
            diag = self.diagonal_size()
            bounds[5] = bounds[4] + diag / 1000
        pdf.SetModelBounds(bounds)

        pdf.SetSampleDimensions(dims)

        if locator:
            pdf.SetLocator(locator)

        pdf.SetDensityEstimateToFixedRadius()
        if radius is None:
            radius = self.diagonal_size() / 20
        pdf.SetRadius(radius)

        pdf.SetComputeGradient(compute_gradient)
        pdf.Update()
        img = pdf.GetOutput()
        vol = vedo.volume.Volume(img).mode(1)
        vol.name = "PointDensity"
        vol.metadata["radius"] = radius
        vol.locator = pdf.GetLocator()
        vol.pipeline = utils.OperationNode(
            "density", parents=[self], comment=f"dims={tuple(vol.dimensions())}"
        )
        return vol

    def densify(self, target_distance=0.1, nclosest=6, radius=None, niter=1, nmax=None):
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

        Examples:
            - [densifycloud.py](https://github.com/marcomusy/vedo/blob/master/examples/volumetric/densifycloud.py)

                ![](https://vedo.embl.es/images/volumetric/densifycloud.png)

        .. note::
            Points will be created in an iterative fashion until all points in their
            local neighborhood are the target distance apart or less.
            Note that the process may terminate early due to the
            number of iterations. By default the target distance is set to 0.5.
            Note that the target_distance should be less than the radius
            or nothing will change on output.

        .. warning::
            This class can generate a lot of points very quickly.
            The maximum number of iterations is by default set to =1.0 for this reason.
            Increase the number of iterations very carefully.
            Also, `nmax` can be set to limit the explosion of points.
            It is also recommended that a N closest neighborhood is used.

        """
        src = vtki.new("ProgrammableSource")
        opts = self.vertices

        def _read_points():
            output = src.GetPolyDataOutput()
            points = vtki.vtkPoints()
            for p in opts:
                points.InsertNextPoint(p)
            output.SetPoints(points)

        src.SetExecuteMethod(_read_points)

        dens = vtki.new("DensifyPointCloudFilter")
        dens.SetInputConnection(src.GetOutputPort())
        dens.InterpolateAttributeDataOn()
        dens.SetTargetDistance(target_distance)
        dens.SetMaximumNumberOfIterations(niter)
        if nmax:
            dens.SetMaximumNumberOfPoints(nmax)

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
        cld = Points(pts, c=None).point_size(self.properties.GetPointSize())
        cld.interpolate_data_from(self, n=nclosest, radius=radius)
        cld.name = "DensifiedCloud"

        cld.pipeline = utils.OperationNode(
            "densify",
            parents=[self],
            c="#e9c46a:",
            comment=f"#pts {cld.dataset.GetNumberOfPoints()}",
        )
        return cld

    ###############################################################################
    ## stuff returning Volume

    def signed_distance(self, bounds=None, dims=(20, 20, 20), invert=False, maxradius=None):
        """
        Compute the `Volume` object whose voxels contains the signed distance from
        the point cloud. The point cloud must have Normals.

        Arguments:
            bounds : (list, actor)
                bounding box sizes
            dims : (list)
                dimensions (nr. of voxels) of the output volume.
            invert : (bool)
                flip the sign
            maxradius : (float)
                specify how far out to propagate distance calculation

        Examples:
            - [distance2mesh.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/distance2mesh.py)

                ![](https://vedo.embl.es/images/basic/distance2mesh.png)
        """
        if bounds is None:
            bounds = self.bounds()
        if maxradius is None:
            maxradius = self.diagonal_size() / 2
        dist = vtki.new("SignedDistance")
        dist.SetInputData(self.dataset)
        dist.SetRadius(maxradius)
        dist.SetBounds(bounds)
        dist.SetDimensions(dims)
        dist.Update()
        img = dist.GetOutput()
        if invert:
            mat = vtki.new("ImageMathematics")
            mat.SetInput1Data(img)
            mat.SetOperationToMultiplyByK()
            mat.SetConstantK(-1)
            mat.Update()
            img = mat.GetOutput()

        vol = vedo.Volume(img)
        vol.name = "SignedDistanceVolume"

        vol.pipeline = utils.OperationNode(
            "signed_distance",
            parents=[self],
            comment=f"dims={tuple(vol.dimensions())}",
            c="#e9c46a:#0096c7",
        )
        return vol

    def tovolume(
        self,
        kernel="shepard",
        radius=None,
        n=None,
        bounds=None,
        null_value=None,
        dims=(25, 25, 25),
    ):
        """
        Generate a `Volume` by interpolating a scalar
        or vector field which is only known on a scattered set of points or mesh.
        Available interpolation kernels are: shepard, gaussian, or linear.

        Arguments:
            kernel : (str)
                interpolation kernel type [shepard]
            radius : (float)
                radius of the local search
            n : (int)
                number of point to use for interpolation
            bounds : (list)
                bounding box of the output Volume object
            dims : (list)
                dimensions of the output Volume object
            null_value : (float)
                value to be assigned to invalid points

        Examples:
            - [interpolate_volume.py](https://github.com/marcomusy/vedo/blob/master/examples/volumetric/interpolate_volume.py)

                ![](https://vedo.embl.es/images/volumetric/59095175-1ec5a300-8918-11e9-8bc0-fd35c8981e2b.jpg)
        """
        if radius is None and not n:
            vedo.logger.error("please set either radius or n")
            raise RuntimeError

        poly = self.dataset

        # Create a probe volume
        probe = vtki.vtkImageData()
        probe.SetDimensions(dims)
        if bounds is None:
            bounds = self.bounds()
        probe.SetOrigin(bounds[0], bounds[2], bounds[4])
        probe.SetSpacing(
            (bounds[1] - bounds[0]) / dims[0],
            (bounds[3] - bounds[2]) / dims[1],
            (bounds[5] - bounds[4]) / dims[2],
        )

        if not self.point_locator:
            self.point_locator = vtki.new("PointLocator")
            self.point_locator.SetDataSet(poly)
            self.point_locator.BuildLocator()

        if kernel == "shepard":
            kern = vtki.new("ShepardKernel")
            kern.SetPowerParameter(2)
        elif kernel == "gaussian":
            kern = vtki.new("GaussianKernel")
        elif kernel == "linear":
            kern = vtki.new("LinearKernel")
        else:
            vedo.logger.error("Error in tovolume(), available kernels are:")
            vedo.logger.error(" [shepard, gaussian, linear]")
            raise RuntimeError()

        if radius:
            kern.SetRadius(radius)

        interpolator = vtki.new("PointInterpolator")
        interpolator.SetInputData(probe)
        interpolator.SetSourceData(poly)
        interpolator.SetKernel(kern)
        interpolator.SetLocator(self.point_locator)

        if n:
            kern.SetNumberOfPoints(n)
            kern.SetKernelFootprintToNClosest()
        else:
            kern.SetRadius(radius)

        if null_value is not None:
            interpolator.SetNullValue(null_value)
        else:
            interpolator.SetNullPointsStrategyToClosestPoint()
        interpolator.Update()

        vol = vedo.Volume(interpolator.GetOutput())

        vol.pipeline = utils.OperationNode(
            "signed_distance",
            parents=[self],
            comment=f"dims={tuple(vol.dimensions())}",
            c="#e9c46a:#0096c7",
        )
        return vol

    #################################################################################
    def generate_random_data(self):
        """Fill a dataset with random attributes"""
        gen = vtki.new("RandomAttributeGenerator")
        gen.SetInputData(self.dataset)
        gen.GenerateAllDataOn()
        gen.SetDataTypeToFloat()
        gen.GeneratePointNormalsOff()
        gen.GeneratePointTensorsOn()
        gen.GenerateCellScalarsOn()
        gen.Update()

        self._update(gen.GetOutput(), reset_locators=False)

        self.pipeline = utils.OperationNode("generate_random_data", parents=[self])
        return self
    
    def generate_segments(self, istart=0, rmax=1e30, niter=3):
        """
        Generate a line segments from a set of points.
        The algorithm is based on the closest point search.

        Returns a `Line` object.
        This object contains the a metadata array of used vertex counts in "UsedVertexCount"
        and the sum of the length of the segments in "SegmentsLengthSum".

        Arguments:
            istart : (int)
                index of the starting point
            rmax : (float)
                maximum length of a segment
            niter : (int)
                number of iterations or passes through the points

        Examples:
            - [moving_least_squares1D.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares1D.py)
        """
        points = self.vertices
        segments = []
        dists = []
        n = len(points)
        used = np.zeros(n, dtype=int)
        for _ in range(niter):
            i = istart
            for _ in range(n):
                p = points[i]
                ids = self.closest_point(p, n=4, return_point_id=True)
                j = ids[1]
                if used[j] > 1 or [j, i] in segments:
                    j = ids[2]
                if used[j] > 1:
                    j = ids[3]
                d = np.linalg.norm(p - points[j])
                if used[j] > 1 or used[i] > 1 or d > rmax:
                    i += 1
                    if i >= n:
                        i = 0
                    continue
                used[i] += 1
                used[j] += 1
                segments.append([i, j])
                dists.append(d)
                i = j
        segments = np.array(segments, dtype=int)

        line = vedo.shapes.Lines(points[segments], c="k", lw=3)
        line.metadata["UsedVertexCount"] = used
        line.metadata["SegmentsLengthSum"] = np.sum(dists)
        line.pipeline = utils.OperationNode("generate_segments", parents=[self])
        line.name = "Segments"
        return line

    def generate_delaunay2d(
        self,
        mode="scipy",
        boundaries=(),
        tol=None,
        alpha=0.0,
        offset=0.0,
        transform=None,
    ):
        """
        Create a mesh from points in the XY plane.
        If `mode='fit'` then the filter computes a best fitting
        plane and projects the points onto it.

        Check also `generate_mesh()`.

        Arguments:
            tol : (float)
                specify a tolerance to control discarding of closely spaced points.
                This tolerance is specified as a fraction of the diagonal length of the bounding box of the points.
            alpha : (float)
                for a non-zero alpha value, only edges or triangles contained
                within a sphere centered at mesh vertices will be output.
                Otherwise, only triangles will be output.
            offset : (float)
                multiplier to control the size of the initial, bounding Delaunay triangulation.
            transform: (LinearTransform, NonLinearTransform)
                a transformation which is applied to points to generate a 2D problem.
                This maps a 3D dataset into a 2D dataset where triangulation can be done on the XY plane.
                The points are transformed and triangulated.
                The topology of triangulated points is used as the output topology.

        Examples:
            - [delaunay2d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/delaunay2d.py)

                ![](https://vedo.embl.es/images/basic/delaunay2d.png)
        """
        plist = self.vertices.copy()

        #########################################################
        if mode == "scipy":
            from scipy.spatial import Delaunay as scipy_delaunay

            tri = scipy_delaunay(plist[:, 0:2])
            return vedo.mesh.Mesh([plist, tri.simplices])
        ##########################################################

        pd = vtki.vtkPolyData()
        vpts = vtki.vtkPoints()
        vpts.SetData(utils.numpy2vtk(plist, dtype=np.float32))
        pd.SetPoints(vpts)

        delny = vtki.new("Delaunay2D")
        delny.SetInputData(pd)
        if tol:
            delny.SetTolerance(tol)
        delny.SetAlpha(alpha)
        delny.SetOffset(offset)

        if transform:
            delny.SetTransform(transform.T)
        elif mode == "fit":
            delny.SetProjectionPlaneMode(vtki.get_class("VTK_BEST_FITTING_PLANE"))
        elif mode == "xy" and boundaries:
            boundary = vtki.vtkPolyData()
            boundary.SetPoints(vpts)
            cell_array = vtki.vtkCellArray()
            for b in boundaries:
                cpolygon = vtki.vtkPolygon()
                for idd in b:
                    cpolygon.GetPointIds().InsertNextId(idd)
                cell_array.InsertNextCell(cpolygon)
            boundary.SetPolys(cell_array)
            delny.SetSourceData(boundary)

        delny.Update()

        msh = vedo.mesh.Mesh(delny.GetOutput())
        msh.name = "Delaunay2D"
        msh.clean().lighting("off")
        msh.pipeline = utils.OperationNode(
            "delaunay2d",
            parents=[self],
            comment=f"#cells {msh.dataset.GetNumberOfCells()}",
        )
        return msh

    def generate_voronoi(self, padding=0.0, fit=False, method="vtk"):
        """
        Generate the 2D Voronoi convex tiling of the input points (z is ignored).
        The points are assumed to lie in a plane. The output is a Mesh. Each output cell is a convex polygon.

        A cell array named "VoronoiID" is added to the output Mesh.

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

        Arguments:
            pts : (list)
                list of input points.
            padding : (float)
                padding distance. The default is 0.
            fit : (bool)
                detect automatically the best fitting plane. The default is False.

        Examples:
            - [voronoi1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/voronoi1.py)

                ![](https://vedo.embl.es/images/basic/voronoi1.png)

            - [voronoi2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/voronoi2.py)

                ![](https://vedo.embl.es/images/advanced/voronoi2.png)
        """
        pts = self.vertices

        if method == "scipy":
            from scipy.spatial import Voronoi as scipy_voronoi

            pts = np.asarray(pts)[:, (0, 1)]
            vor = scipy_voronoi(pts)
            regs = []  # filter out invalid indices
            for r in vor.regions:
                flag = True
                for x in r:
                    if x < 0:
                        flag = False
                        break
                if flag and len(r) > 0:
                    regs.append(r)

            m = vedo.Mesh([vor.vertices, regs])
            m.celldata["VoronoiID"] = np.array(list(range(len(regs)))).astype(int)
            m.locator = None

        elif method == "vtk":
            vor = vtki.new("Voronoi2D")
            if isinstance(pts, Points):
                vor.SetInputData(pts)
            else:
                pts = np.asarray(pts)
                if pts.shape[1] == 2:
                    pts = np.c_[pts, np.zeros(len(pts))]
                pd = vtki.vtkPolyData()
                vpts = vtki.vtkPoints()
                vpts.SetData(utils.numpy2vtk(pts, dtype=np.float32))
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
            m = vedo.Mesh(poly, c="orange5")
            m.locator = vor.GetLocator()

        else:
            vedo.logger.error(f"Unknown method {method} in voronoi()")
            raise RuntimeError

        m.lw(2).lighting("off").wireframe()
        m.name = "Voronoi"
        return m

    ##########################################################################
    def generate_delaunay3d(self, radius=0, tol=None):
        """
        Create 3D Delaunay triangulation of input points.

        Arguments:
            radius : (float)
                specify distance (or "alpha") value to control output.
                For a non-zero values, only tetra contained within the circumsphere
                will be output.
            tol : (float)
                Specify a tolerance to control discarding of closely spaced points.
                This tolerance is specified as a fraction of the diagonal length of
                the bounding box of the points.
        """
        deln = vtki.new("Delaunay3D")
        deln.SetInputData(self.dataset)
        deln.SetAlpha(radius)
        deln.AlphaTetsOn()
        deln.AlphaTrisOff()
        deln.AlphaLinesOff()
        deln.AlphaVertsOff()
        deln.BoundingTriangulationOff()
        if tol:
            deln.SetTolerance(tol)
        deln.Update()
        m = vedo.TetMesh(deln.GetOutput())
        m.pipeline = utils.OperationNode(
            "generate_delaunay3d", c="#e9c46a:#edabab", parents=[self],
        )
        m.name = "Delaunay3D"
        return m

    ####################################################
    def visible_points(self, area=(), tol=None, invert=False):
        """
        Extract points based on whether they are visible or not.
        Visibility is determined by accessing the z-buffer of a rendering window.
        The position of each input point is converted into display coordinates,
        and then the z-value at that point is obtained.
        If within the user-specified tolerance, the point is considered visible.
        Associated data attributes are passed to the output as well.

        This filter also allows you to specify a rectangular window in display (pixel)
        coordinates in which the visible points must lie.

        Arguments:
            area : (list)
                specify a rectangular region as (xmin,xmax,ymin,ymax)
            tol : (float)
                a tolerance in normalized display coordinate system
            invert : (bool)
                select invisible points instead.

        Example:
            ```python
            from vedo import Ellipsoid, show
            s = Ellipsoid().rotate_y(30)

            # Camera options: pos, focal_point, viewup, distance
            camopts = dict(pos=(0,0,25), focal_point=(0,0,0))
            show(s, camera=camopts, offscreen=True)

            m = s.visible_points()
            # print('visible pts:', m.vertices)  # numpy array
            show(m, new=True, axes=1).close() # optionally draw result in a new window
            ```
            ![](https://vedo.embl.es/images/feats/visible_points.png)
        """
        svp = vtki.new("SelectVisiblePoints")
        svp.SetInputData(self.dataset)

        ren = None
        if vedo.plotter_instance:
            if vedo.plotter_instance.renderer:
                ren = vedo.plotter_instance.renderer
                svp.SetRenderer(ren)
        if not ren:
            vedo.logger.warning(
                "visible_points() can only be used after a rendering step"
            )
            return None

        if len(area) == 2:
            area = utils.flatten(area)
        if len(area) == 4:
            # specify a rectangular region
            svp.SetSelection(area[0], area[1], area[2], area[3])
        if tol is not None:
            svp.SetTolerance(tol)
        if invert:
            svp.SelectInvisibleOn()
        svp.Update()

        m = Points(svp.GetOutput())
        m.name = "VisiblePoints"
        return m

####################################################
class CellCenters(Points):
    def __init__(self, pcloud):
        """
        Generate `Points` at the center of the cells of any type of object.

        Check out also `cell_centers()`.
        """
        vcen = vtki.new("CellCenters")
        vcen.CopyArraysOn()
        vcen.VertexCellsOn()
        # vcen.ConvertGhostCellsToGhostPointsOn()
        try:
            vcen.SetInputData(pcloud.dataset)
        except AttributeError:
            vcen.SetInputData(pcloud)
        vcen.Update()
        super().__init__(vcen.GetOutput())
        self.name = "CellCenters"
