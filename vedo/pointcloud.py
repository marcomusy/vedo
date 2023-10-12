#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

import vedo
from vedo import colors
from vedo import utils
from vedo.base import BaseActor
from vedo.transformations import LinearTransform

__docformat__ = "google"

__doc__ = """
Submodule to work with point clouds <br>

![](https://vedo.embl.es/images/basic/pca.png)
"""

__all__ = [
    "Points",
    "Point",
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

    To keep track of the original identities of the inputs you can use `flag`.
    In this case a point array of IDs is added to the output.

    Examples:
        - [warp1.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp1.py)

            ![](https://vedo.embl.es/images/advanced/warp1.png)

        - [value_iteration.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/value_iteration.py)

    """
    objs = [a for a in utils.flatten(meshs) if a]

    if not objs:
        return None

    idarr = []
    polyapp = vtk.vtkAppendPolyData()
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
        "merge",
        parents=objs,
        comment=f"#pts {msh.dataset.GetNumberOfPoints()}",
    )
    return msh


def delaunay2d(plist, **kwargs):
    """delaunay2d() is deprecated, use Points().generate_delaunay2d() instead"""
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
    pla.top = n
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
    Show the oriented PCA 2D ellipse that contains the fraction `pvalue` of points.

    Parameter `pvalue` sets the specified fraction of points inside the ellipse.
    Normalized directions are stored in `ellipse.axis1`, `ellipse.axis12`
    axes sizes are stored in `ellipse.va`, `ellipse.vb`

    Arguments:
        pvalue : (float)
            ellipse will include this fraction of points
        res : (int)
            resolution of the ellipse

    Examples:
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

    P = np.array(coords, dtype=float)[:,(0,1)]
    cov = np.cov(P, rowvar=0)     # covariance matrix
    _, s, R = np.linalg.svd(cov)  # singular value decomposition
    p, n = s.size, P.shape[0]
    fppf = f.ppf(pvalue, p, n-p)  # f % point function
    ua, ub = np.sqrt(s*fppf/2)*2  # semi-axes (largest first)
    center = np.mean(P, axis=0)   # centroid of the ellipse

    matri = vtk.vtkMatrix4x4()
    matri.DeepCopy((
        R[0][0] * ua, R[1][0] * ub, 0, center[0],
        R[0][1] * ua, R[1][1] * ub, 0, center[1],
                   0,            0, 1,         0,
        0, 0, 0, 1)
    )
    vtra = vtk.vtkTransform()
    vtra.SetMatrix(matri)

    elli = vedo.shapes.Circle(alpha=0.75, res=res)
    elli.apply_transform(vtra)

    elli.pvalue = pvalue
    elli.center = np.array([center[0], center[1], 0])
    elli.nr_of_points = n
    elli.va = ua
    elli.vb = ub
    # we subtract center because it's in M
    elli.axis1 = np.array(vtra.TransformPoint([1, 0, 0])) - elli.center
    elli.axis2 = np.array(vtra.TransformPoint([0, 1, 0])) - elli.center
    elli.axis1 /= np.linalg.norm(elli.axis1)
    elli.axis2 /= np.linalg.norm(elli.axis2)
    elli.name = "PCAEllipse"
    return elli


def pca_ellipsoid(points, pvalue=0.673):
    """
    Show the oriented PCA ellipsoid that contains fraction `pvalue` of points.

    Parameter `pvalue` sets the specified fraction of points inside the ellipsoid.

    Extra can be calculated with `mesh.asphericity()`, `mesh.asphericity_error()`
    (asphericity is equal to 0 for a perfect sphere).

    Axes sizes can be accessed in `ellips.va`, `ellips.vb`, `ellips.vc`,
    normalized directions are stored in `ellips.axis1`, `ellips.axis12`
    and `ellips.axis3`.

    .. warning:: the meaning of `ellips.axis1`, has changed wrt `vedo==2022.1.0`
        in that it's now the direction wrt the origin (e.i. the center is subtracted)

    Examples:
        - [pca_ellipsoid.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/pca_ellipsoid.py)

            ![](https://vedo.embl.es/images/basic/pca.png)
    """
    from scipy.stats import f

    if isinstance(points, Points):
        coords = points.vertices
    else:
        coords = points
    if len(coords) < 4:
        vedo.logger.warning("in pca_ellipsoid(), there are not enough points!")
        return None

    P = np.array(coords, ndmin=2, dtype=float)
    cov = np.cov(P, rowvar=0)     # covariance matrix
    _, s, R = np.linalg.svd(cov)  # singular value decomposition
    p, n = s.size, P.shape[0]
    fppf = f.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p)  # f % point function
    cfac = 1 + 6/(n-1)            # correction factor for low statistics
    ua, ub, uc = np.sqrt(s*fppf)/cfac  # semi-axes (largest first)
    center = np.mean(P, axis=0)   # centroid of the hyperellipsoid

    M = vtk.vtkMatrix4x4()
    M.DeepCopy((
        R[0][0] * ua*2, R[1][0] * ub*2, R[2][0] * uc*2, center[0],
        R[0][1] * ua*2, R[1][1] * ub*2, R[2][1] * uc*2, center[1],
        R[0][2] * ua*2, R[1][2] * ub*2, R[2][2] * uc*2, center[2],
        0, 0, 0, 1)
    )
    vtra = vtk.vtkTransform()
    vtra.SetMatrix(M)

    elli = vedo.shapes.Ellipsoid(
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), alpha=0.25)
    elli.property.LightingOff()
    elli.apply_transform(vtra)

    elli.pvalue = pvalue
    elli.nr_of_points = n
    elli.center = center
    elli.va = ua
    elli.vb = ub
    elli.vc = uc
    # we subtract center because it's in M
    elli.axis1 = np.array(vtra.TransformPoint([1, 0, 0])) - center
    elli.axis2 = np.array(vtra.TransformPoint([0, 1, 0])) - center
    elli.axis3 = np.array(vtra.TransformPoint([0, 0, 1])) - center
    elli.axis1 /= np.linalg.norm(elli.axis1)
    elli.axis2 /= np.linalg.norm(elli.axis2)
    elli.axis3 /= np.linalg.norm(elli.axis3)
    elli.name = "PCAEllipsoid"
    return elli


###################################################
def Point(pos=(0, 0, 0), r=12, c="red", alpha=1.0):
    """
    Create a simple point in space.

    .. note:: if you are creating many points you should definitely use class `Points` instead!
    """
    if isinstance(pos, vedo.Base3DProp):
        pos = pos.pos()
    if len(pos) == 2:
        pos = (pos[0], pos[1], 0.0)
    pt = Points([[0,0,0]], r, c, alpha)
    pt.pos(pos)
    pt.name = "Point"
    return pt


###################################################
class PointsVisual:
    """Class to manage the visual aspects of a ``Points`` object."""

    def __init__(self):

        self.actor = vtk.vtkActor()
        self.property = self.actor.GetProperty()
        self.mapper = vtk.vtkPolyDataMapper()
        
        self.property_backface = self.actor.GetBackfaceProperty()     

        self._scals_idx = 0   # index of the active scalar changed from CLI
        self._ligthingnr = 0  # index of the lighting mode changed from CLI
        self._cmap_name = ""  # remember the cmap name for self._keypress

        try:
            self.property.RenderPointsAsSpheresOn()
        except AttributeError:
            pass

    ##################################################
    def copy_properties_from(self, source, deep=True, actor_related=True):
        """
        Copy properties from another ``Points`` object.
        """
        pr = vtk.vtkProperty()
        if deep:
            pr.DeepCopy(source.property)
        else:
            pr.ShallowCopy(source.property)
        self.actor.SetProperty(pr)
        self.property = pr

        if self.actor.GetBackfaceProperty():
            bfpr = vtk.vtkProperty()
            bfpr.DeepCopy(source.actor.GetBackfaceProperty())
            self.actor.SetBackfaceProperty(bfpr)
            self.property_backface = bfpr
    
        if not actor_related:
            return self
        
        # mapper related:
        self.mapper.SetScalarVisibility(source.mapper.GetScalarVisibility())
        self.mapper.SetScalarMode(source.mapper.GetScalarMode())
        self.mapper.SetScalarRange(source.mapper.GetScalarRange())
        self.mapper.SetLookupTable(source.mapper.GetLookupTable())
        self.mapper.SetColorMode(source.mapper.GetColorMode())
        self.mapper.SetInterpolateScalarsBeforeMapping(source.mapper.GetInterpolateScalarsBeforeMapping())
        self.mapper.SetUseLookupTableScalarRange(source.mapper.GetUseLookupTableScalarRange())

        self.actor.SetPickable(source.actor.GetPickable())
        self.actor.SetDragable(source.actor.GetDragable())
        self.actor.SetTexture(source.actor.GetTexture())
        self.actor.SetVisibility(source.actor.GetVisibility())
        return self
    
    def color(self, c=False, alpha=None):
        """
        Set/get mesh's color.
        If None is passed as input, will use colors from active scalars.
        Same as `mesh.c()`.
        """
        # overrides base.color()
        if c is False:
            return np.array(self.property.GetColor())
        if c is None:
            self.mapper.ScalarVisibilityOn()
            return self
        self.mapper.ScalarVisibilityOff()
        cc = colors.get_color(c)
        self.property.SetColor(cc)
        if self.trail:
            self.trail.GetProperty().SetColor(cc)
        if alpha is not None:
            self.alpha(alpha)
        return self

    def c(self, color=False, alpha=None):
        """
        Shortcut for `color()`.
        If None is passed as input, will use colors from current active scalars.
        """
        return self.color(color, alpha)

    def alpha(self, opacity=None):
        """Set/get mesh's transparency. Same as `mesh.opacity()`."""
        if opacity is None:
            return self.property.GetOpacity()

        self.property.SetOpacity(opacity)
        bfp = self.actor.GetBackfaceProperty()
        if bfp:
            if opacity < 1:
                self.property_backface = bfp
                self.actor.SetBackfaceProperty(None)
            else:
                self.actor.SetBackfaceProperty(self.property_backface)
        return self


    def opacity(self, alpha=None):
        """Set/get mesh's transparency. Same as `mesh.alpha()`."""
        return self.alpha(alpha)

    def force_opaque(self, value=True):
        """ Force the Mesh, Line or point cloud to be treated as opaque"""
        ## force the opaque pass, fixes picking in vtk9
        # but causes other bad troubles with lines..
        self.actor.SetForceOpaque(value)
        return self

    def force_translucent(self, value=True):
        """ Force the Mesh, Line or point cloud to be treated as translucent"""
        self.actor.SetForceTranslucent(value)
        return self

    def point_size(self, value=None):
        """Set/get mesh's point size of vertices. Same as `mesh.ps()`"""
        if value is None:
            return self.property.GetPointSize()
            #self.property.SetRepresentationToSurface()
        else:
            self.property.SetRepresentationToPoints()
            self.property.SetPointSize(value)
        return self

    def ps(self, pointsize=None):
        """Set/get mesh's point size of vertices. Same as `mesh.point_size()`"""
        return self.point_size(pointsize)

    def render_points_as_spheres(self, value=True):
        """Make points look spheric or else make them look as squares."""
        self.property.SetRenderPointsAsSpheres(value)
        return self

    def lighting(
        self,
        style="",
        ambient=None,
        diffuse=None,
        specular=None,
        specular_power=None,
        specular_color=None,
        metallicity=None,
        roughness=None,
    ):
        """
        Set the ambient, diffuse, specular and specular_power lighting constants.

        Arguments:
            style : (str)
                preset style, options are `[metallic, plastic, shiny, glossy, ambient, off]`
            ambient : (float)
                ambient fraction of emission [0-1]
            diffuse : (float)
                emission of diffused light in fraction [0-1]
            specular : (float)
                fraction of reflected light [0-1]
            specular_power : (float)
                precision of reflection [1-100]
            specular_color : (color)
                color that is being reflected by the surface

        <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Phong_components_version_4.png" alt="", width=700px>

        Examples:
            - [specular.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/specular.py)
        """
        pr = self.property

        if style:

            if isinstance(pr, vtk.vtkVolumeProperty):
                self.shade(True)
                if style == "off":
                    self.shade(False)
                elif style == "ambient":
                    style = "default"
                    self.shade(False)
            else:
                if style != "off":
                    pr.LightingOn()

            if style == "off":
                pr.SetInterpolationToFlat()
                pr.LightingOff()
                return self  ##############

            if hasattr(pr, "GetColor"):  # could be Volume
                c = pr.GetColor()
            else:
                c = (1, 1, 0.99)
            mpr = self.mapper
            if hasattr(mpr, 'GetScalarVisibility') and mpr.GetScalarVisibility():
                c = (1,1,0.99)
            if   style=='metallic': pars = [0.1, 0.3, 1.0, 10, c]
            elif style=='plastic' : pars = [0.3, 0.4, 0.3,  5, c]
            elif style=='shiny'   : pars = [0.2, 0.6, 0.8, 50, c]
            elif style=='glossy'  : pars = [0.1, 0.7, 0.9, 90, (1,1,0.99)]
            elif style=='ambient' : pars = [0.8, 0.1, 0.0,  1, (1,1,1)]
            elif style=='default' : pars = [0.1, 1.0, 0.05, 5, c]
            else:
                vedo.logger.error("in lighting(): Available styles are")
                vedo.logger.error("[default, metallic, plastic, shiny, glossy, ambient, off]")
                raise RuntimeError()
            pr.SetAmbient(pars[0])
            pr.SetDiffuse(pars[1])
            pr.SetSpecular(pars[2])
            pr.SetSpecularPower(pars[3])
            if hasattr(pr, "GetColor"):
                pr.SetSpecularColor(pars[4])

        if ambient is not None: pr.SetAmbient(ambient)
        if diffuse is not None: pr.SetDiffuse(diffuse)
        if specular is not None: pr.SetSpecular(specular)
        if specular_power is not None: pr.SetSpecularPower(specular_power)
        if specular_color is not None: pr.SetSpecularColor(colors.get_color(specular_color))
        if utils.vtk_version_at_least(9):
            if metallicity is not None:
                pr.SetInterpolationToPBR()
                pr.SetMetallic(metallicity)
            if roughness is not None:
                pr.SetInterpolationToPBR()
                pr.SetRoughness(roughness)

        return self

    def point_blurring(self, r=1, emissive=False):
        """Set point blurring.
        Apply a gaussian convolution filter to the points.
        In this case the radius `r` is in absolute units of the mesh coordinates.
        With emissive set, the halo of point becomes light-emissive.
        """
        self.property.SetRepresentationToPoints()
        if emissive:
            self.mapper.SetEmissive(bool(emissive))
        self.mapper.SetScaleFactor(r * 1.4142)

        # https://kitware.github.io/vtk-examples/site/Python/Meshes/PointInterpolator/
        if alpha < 1:
            self.mapper.SetSplatShaderCode(
                "//VTK::Color::Impl\n"
                "float dist = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);\n"
                "if (dist > 1.0) {\n"
                "   discard;\n"
                "} else {\n"
                f"  float scale = ({alpha} - dist);\n"
                "   ambientColor *= scale;\n"
                "   diffuseColor *= scale;\n"
                "}\n"
            )
            alpha = 1

        self.mapper.Modified()
        self.actor.Modified()
        self.property.SetOpacity(alpha)
        self.actor.SetMapper(self.mapper)
        return self


    @property
    def cellcolors(self):
        """
        Colorize each cell (face) of a mesh by passing
        a 1-to-1 list of colors in format [R,G,B] or [R,G,B,A].
        Colors levels and opacities must be in the range [0,255].

        A single constant color can also be passed as string or RGBA.

        A cell array named "CellsRGBA" is automatically created.

        Examples:
            - [color_mesh_cells1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/color_mesh_cells1.py)
            - [color_mesh_cells2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/color_mesh_cells2.py)

            ![](https://vedo.embl.es/images/basic/colorMeshCells.png)
        """
        if "CellsRGBA" not in self.celldata.keys():
            lut = self.mapper.GetLookupTable()
            vscalars = self.dataset.GetCellData().GetScalars()
            if vscalars is None or lut is None:
                arr = np.zeros([self.ncells, 4], dtype=np.uint8)
                col = np.array(self.property.GetColor())
                col = np.round(col * 255).astype(np.uint8)
                alf = self.property.GetOpacity()
                alf = np.round(alf * 255).astype(np.uint8)
                arr[:, (0, 1, 2)] = col
                arr[:, 3] = alf
            else:
                cols = lut.MapScalars(vscalars, 0, 0)
                arr = utils.vtk2numpy(cols)
            self.celldata["CellsRGBA"] = arr
        self.celldata.select("CellsRGBA")
        return self.celldata["CellsRGBA"]

    @cellcolors.setter
    def cellcolors(self, value):
        if isinstance(value, str):
            c = colors.get_color(value)
            value = np.array([*c, 1]) * 255
            value = np.round(value)

        value = np.asarray(value)
        n = self.ncells

        if value.ndim == 1:
            value = np.repeat([value], n, axis=0)

        if value.shape[1] == 3:
            z = np.zeros((n, 1), dtype=np.uint8)
            value = np.append(value, z + 255, axis=1)

        assert n == value.shape[0]

        self.celldata["CellsRGBA"] = value.astype(np.uint8)
        self.celldata.select("CellsRGBA")


    @property
    def pointcolors(self):
        """
        Colorize each point (or vertex of a mesh) by passing
        a 1-to-1 list of colors in format [R,G,B] or [R,G,B,A].
        Colors levels and opacities must be in the range [0,255].

        A single constant color can also be passed as string or RGBA.

        A point array named "PointsRGBA" is automatically created.
        """
        if "PointsRGBA" not in self.pointdata.keys():
            lut = self.mapper.GetLookupTable()
            vscalars = self.dataset.GetPointData().GetScalars()
            if vscalars is None or lut is None:
                arr = np.zeros([self.npoints, 4], dtype=np.uint8)
                col = np.array(self.property.GetColor())
                col = np.round(col * 255).astype(np.uint8)
                alf = self.property.GetOpacity()
                alf = np.round(alf * 255).astype(np.uint8)
                arr[:, (0, 1, 2)] = col
                arr[:, 3] = alf
            else:
                cols = lut.MapScalars(vscalars, 0, 0)
                arr = utils.vtk2numpy(cols)
            self.pointdata["PointsRGBA"] = arr
        self.pointdata.select("PointsRGBA")
        return self.pointdata["PointsRGBA"]

    @pointcolors.setter
    def pointcolors(self, value):
        if isinstance(value, str):
            c = colors.get_color(value)
            value = np.array([*c, 1]) * 255
            value = np.round(value)

        value = np.asarray(value)
        n = self.npoints

        if value.ndim == 1:
            value = np.repeat([value], n, axis=0)

        if value.shape[1] == 3:
            z = np.zeros((n, 1), dtype=np.uint8)
            value = np.append(value, z + 255, axis=1)

        assert n == value.shape[0]

        self.pointdata["PointsRGBA"] = value.astype(np.uint8)
        self.pointdata.select("PointsRGBA")

    #####################################################################################
    def cmap(
        self,
        input_cmap,
        input_array=None,
        on="points",
        name="Scalars",
        vmin=None,
        vmax=None,
        n_colors=256,
        alpha=1.0,
        logscale=False,
    ):
        """
        Set individual point/cell colors by providing a list of scalar values and a color map.

        Arguments:
            input_cmap : (str, list, vtkLookupTable, matplotlib.colors.LinearSegmentedColormap)
                color map scheme to transform a real number into a color.
            input_array : (str, list, vtkArray)
                can be the string name of an existing array, a numpy array or a `vtkArray`.
            on : (str)
                either 'points' or 'cells'.
                Apply the color map to data which is defined on either points or cells.
            name : (str)
                give a name to the provided numpy array (if input_array is a numpy array)
            vmin : (float)
                clip scalars to this minimum value
            vmax : (float)
                clip scalars to this maximum value
            n_colors : (int)
                number of distinct colors to be used in colormap table.
            alpha : (float, list)
                Mesh transparency. Can be a `list` of values one for each vertex.
            logscale : (bool)
                Use logscale

        Examples:
            - [mesh_coloring.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_coloring.py)
            - [mesh_alphas.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_alphas.py)
            - [mesh_custom.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_custom.py)
            (and many others)

                ![](https://vedo.embl.es/images/basic/mesh_custom.png)
        """
        self._cmap_name = input_cmap

        if input_array is None:
            if not self.pointdata.keys() and self.celldata.keys():
                on = "cells"
                if not self.dataset.GetCellData().GetScalars():
                    input_array = 0  # pick the first at hand

        if on.startswith("point"):
            data = self.dataset.GetPointData()
            n = self.dataset.GetNumberOfPoints()
        elif on.startswith("cell"):
            data = self.dataset.GetCellData()
            n = self.dataset.GetNumberOfCells()
        else:
            vedo.logger.error("Must specify in cmap(on=...) to either 'cells' or 'points'")
            raise RuntimeError()

        if input_array is None:  # if None try to fetch the active scalars
            arr = data.GetScalars()
            if not arr:
                vedo.logger.error(f"in cmap(), cannot find any {on} active array ...skip coloring.")
                return self

            if not arr.GetName():  # sometimes arrays dont have a name..
                arr.SetName(name)

        elif isinstance(input_array, str):  # if a string is passed
            arr = data.GetArray(input_array)
            if not arr:
                vedo.logger.error(f"in cmap(), cannot find {on} array {input_array} ...skip coloring.")
                return self

        elif isinstance(input_array, int):  # if an int is passed
            if input_array < data.GetNumberOfArrays():
                arr = data.GetArray(input_array)
            else:
                vedo.logger.error(f"in cmap(), cannot find {on} array at {input_array} ...skip coloring.")
                return self

        elif utils.is_sequence(input_array):  # if a numpy array is passed
            npts = len(input_array)
            if npts != n:
                vedo.logger.error(f"in cmap(), nr. of input {on} scalars {npts} != {n} ...skip coloring.")
                return self
            arr = utils.numpy2vtk(input_array, name=name, dtype=float)
            data.AddArray(arr)
            data.Modified()

        elif isinstance(input_array, vtk.vtkArray):  # if a vtkArray is passed
            arr = input_array
            data.AddArray(arr)
            data.Modified()

        else:
            vedo.logger.error(f"in cmap(), cannot understand input type {type(input_array)}")
            raise RuntimeError()

        # Now we have array "arr"
        array_name = arr.GetName()

        if arr.GetNumberOfComponents() == 1:
            if vmin is None:
                vmin = arr.GetRange()[0]
            if vmax is None:
                vmax = arr.GetRange()[1]
        else:
            if vmin is None or vmax is None:
                vn = utils.mag(utils.vtk2numpy(arr))
            if vmin is None:
                vmin = vn.min()
            if vmax is None:
                vmax = vn.max()

        # interpolate alphas if they are not constant
        if not utils.is_sequence(alpha):
            alpha = [alpha] * n_colors
        else:
            v = np.linspace(0, 1, n_colors, endpoint=True)
            xp = np.linspace(0, 1, len(alpha), endpoint=True)
            alpha = np.interp(v, xp, alpha)

        ########################### build the look-up table
        if isinstance(input_cmap, vtk.vtkLookupTable):  # vtkLookupTable
            lut = input_cmap

        elif utils.is_sequence(input_cmap):  # manual sequence of colors
            lut = vtk.vtkLookupTable()
            if logscale:
                lut.SetScaleToLog10()
            lut.SetRange(vmin, vmax)
            ncols = len(input_cmap)
            lut.SetNumberOfTableValues(ncols)

            for i, c in enumerate(input_cmap):
                r, g, b = colors.get_color(c)
                lut.SetTableValue(i, r, g, b, alpha[i])
            lut.Build()

        else:  # assume string cmap name OR matplotlib.colors.LinearSegmentedColormap
            lut = vtk.vtkLookupTable()
            if logscale:
                lut.SetScaleToLog10()
            lut.SetVectorModeToMagnitude()
            lut.SetRange(vmin, vmax)
            lut.SetNumberOfTableValues(n_colors)
            mycols = colors.color_map(range(n_colors), input_cmap, 0, n_colors)
            for i, c in enumerate(mycols):
                r, g, b = c
                lut.SetTableValue(i, r, g, b, alpha[i])
            lut.Build()

        arr.SetLookupTable(lut)

        data.SetActiveScalars(array_name)
        # data.SetScalars(arr)  # wrong! it deletes array in position 0, never use SetScalars
        # data.SetActiveAttribute(array_name, 0) # boh!

        if data.GetScalars():
            data.GetScalars().SetLookupTable(lut)
            data.GetScalars().Modified()

        self.mapper.SetLookupTable(lut)
        self.mapper.SetColorModeToMapScalars()  # so we dont need to convert uint8 scalars

        self.mapper.ScalarVisibilityOn()
        self.mapper.SetScalarRange(lut.GetRange())
        if on.startswith("point"):
            self.mapper.SetScalarModeToUsePointData()
        else:
            self.mapper.SetScalarModeToUseCellData()
        if hasattr(self.mapper, "SetArrayName"):
            self.mapper.SetArrayName(array_name)

        return self

    def add_trail(self, offset=(0, 0, 0), n=50, c=None, alpha=1.0, lw=2):
        """
        Add a trailing line to mesh.
        This new mesh is accessible through `mesh.trail`.

        Arguments:
            offset : (float)
                set an offset vector from the object center.
            n : (int)
                number of segments
            lw : (float)
                line width of the trail

        Examples:
            - [trail.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/trail.py)

                ![](https://vedo.embl.es/images/simulations/trail.gif)

            - [airplane1.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/airplane1.py)
            - [airplane2.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/airplane2.py)
        """
        if self.trail is None:
            pos = self.pos()
            self.trail_offset = np.asarray(offset)
            self.trail_points = [pos] * n

            if c is None:
                col = self.property.GetColor()
            else:
                col = colors.get_color(c)

            tline = vedo.shapes.Line(pos, pos, res=n, c=col, alpha=alpha, lw=lw)
            self.trail = tline  # holds the Line
        return self

    def update_trail(self):
        """
        Update the trailing line of a moving object.
        """
        currentpos = self.pos()

        self.trail_points.append(currentpos)  # cycle
        self.trail_points.pop(0)

        data = np.array(self.trail_points) - currentpos + self.trail_offset
        tpoly = self.trail.dataset
        tpoly.GetPoints().SetData(utils.numpy2vtk(data, dtype=np.float32))
        self.trail.pos(currentpos)
        return self


    def _compute_shadow(self, plane, point, direction):
        shad = self.clone()
        shad.dataset.GetPointData().SetTCoords(None) # remove any texture coords
        shad.name = "Shadow"

        pts = shad.vertices
        if plane == 'x':
            # shad = shad.project_on_plane('x')
            # instead do it manually so in case of alpha<1 
            # we dont see glitches due to coplanar points
            # we leave a small tolerance of 0.1% in thickness
            x0, x1 = self.xbounds()
            pts[:, 0] = (pts[:, 0] - (x0 + x1) / 2) / 1000 + self.actor.GetOrigin()[0]
            shad.vertices = pts
            shad.x(point)
        elif plane == 'y':
            x0, x1 = self.ybounds()
            pts[:, 1] = (pts[:, 1] - (x0 + x1) / 2) / 1000 + self.actor.GetOrigin()[1]
            shad.vertices = pts
            shad.y(point)
        elif plane == "z":
            x0, x1 = self.zbounds()
            pts[:, 2] = (pts[:, 2] - (x0 + x1) / 2) / 1000 + self.actor.GetOrigin()[2]
            shad.vertices = pts
            shad.z(point)
        else:
            shad = shad.project_on_plane(plane, point, direction)
        return shad

    def add_shadow(self, plane, point, direction=None, c=(0.6, 0.6, 0.6), alpha=1, culling=0):
        """
        Generate a shadow out of an `Mesh` on one of the three Cartesian planes.
        The output is a new `Mesh` representing the shadow.
        This new mesh is accessible through `mesh.shadow`.
        By default the shadow mesh is placed on the bottom wall of the bounding box.

        See also `pointcloud.project_on_plane()`.

        Arguments:
            plane : (str, Plane)
                if plane is `str`, plane can be one of `['x', 'y', 'z']`,
                represents x-plane, y-plane and z-plane, respectively.
                Otherwise, plane should be an instance of `vedo.shapes.Plane`
            point : (float, array)
                if plane is `str`, point should be a float represents the intercept.
                Otherwise, point is the camera point of perspective projection
            direction : (list)
                direction of oblique projection
            culling : (int)
                choose between front [1] or backface [-1] culling or None.

        Examples:
            - [shadow1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/shadow1.py)
            - [airplane1.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/airplane1.py)
            - [airplane2.py](https://github.com/marcomusy/vedo/tree/master/examples/simulations/airplane2.py)

            ![](https://vedo.embl.es/images/simulations/57341963-b8910900-713c-11e9-898a-84b6d3712bce.gif)
        """
        shad = self._compute_shadow(plane, point, direction)
        shad.c(c).alpha(alpha)

        try:
            # Points dont have these methods
            shad.flat()
            if culling in (1, True):
                shad.frontface_culling()
            elif culling == -1:
                shad.backface_culling()
        except AttributeError:
            pass

        shad.property.LightingOff()
        shad.actor.SetPickable(False)
        shad.actor.SetUseBounds(True)

        if shad not in self.shadows:
            self.shadows.append(shad)
            shad.info = dict(plane=plane, point=point, direction=direction)
        return self

    def update_shadows(self):
        """
        Update the shadows of a moving object.
        """
        for sha in self.shadows:
            plane = sha.info['plane']
            point = sha.info['point']
            direction = sha.info['direction']
            new_sha = self._compute_shadow(plane, point, direction)
            # sha.DeepCopy(new_sha)
            sha._update(new_sha.dataset)
        return self


    def labels(
        self,
        content=None,
        on="points",
        scale=None,
        xrot=0.0,
        yrot=0.0,
        zrot=0.0,
        ratio=1,
        precision=None,
        italic=False,
        font="",
        justify="bottom-left",
        c="black",
        alpha=1.0,
        cells=None,
    ):
        """
        Generate value or ID labels for mesh cells or points.
        For large nr. of labels use `font="VTK"` which is much faster.

        See also:
            `labels2d()`, `flagpole()`, `caption()` and `legend()`.

        Arguments:
            content : (list,int,str)
                either 'id', 'cellid', array name or array number.
                A array can also be passed (must match the nr. of points or cells).
            on : (str)
                generate labels for "cells" instead of "points"
            scale : (float)
                absolute size of labels, if left as None it is automatic
            zrot : (float)
                local rotation angle of label in degrees
            ratio : (int)
                skipping ratio, to reduce nr of labels for large meshes
            precision : (int)
                numeric precision of labels

        ```python
        from vedo import *
        s = Sphere(res=10).linewidth(1).c("orange").compute_normals()
        point_ids = s.labels('id', on="points").c('green')
        cell_ids  = s.labels('id', on="cells" ).c('black')
        show(s, point_ids, cell_ids)
        ```
        ![](https://vedo.embl.es/images/feats/labels.png)

        Examples:
            - [boundaries.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/boundaries.py)

                ![](https://vedo.embl.es/images/basic/boundaries.png)
        """
        if cells is not None:  # deprecation message
            vedo.logger.warning("In labels(cells=...) please use labels(on='cells') instead")

        if "cell" in on or "face" in on:
            cells = True

        if isinstance(content, str):
            if content in ("cellid", "cellsid"):
                cells = True
                content = "id"

        if cells:
            elems = self.cell_centers
            norms = self.normals(cells=True, recompute=False)
            ns = np.sqrt(self.ncells)
        else:
            elems = self.vertices
            norms = self.normals(cells=False, recompute=False)
            ns = np.sqrt(self.npoints)

        hasnorms = False
        if len(norms) > 0:
            hasnorms = True

        if scale is None:
            if not ns:
                ns = 100
            scale = self.diagonal_size() / ns / 10

        arr = None
        mode = 0
        if content is None:
            mode = 0
            if cells:
                if self.dataset.GetCellData().GetScalars():
                    name = self.dataset.GetCellData().GetScalars().GetName()
                    arr = self.celldata[name]
            else:
                if self.dataset.GetPointData().GetScalars():
                    name = self.dataset.GetPointData().GetScalars().GetName()
                    arr = self.pointdata[name]
        elif isinstance(content, (str, int)):
            if content == "id":
                mode = 1
            elif cells:
                mode = 0
                arr = self.celldata[content]
            else:
                mode = 0
                arr = self.pointdata[content]
        elif utils.is_sequence(content):
            mode = 0
            arr = content
            # print('WEIRD labels() test', content)
            # exit()

        if arr is None and mode == 0:
            vedo.logger.error("in labels(), array not found for points or cells")
            return None

        tapp = vtk.vtkAppendPolyData()
        ninputs = 0

        for i, e in enumerate(elems):
            if i % ratio:
                continue

            if mode == 1:
                txt_lab = str(i)
            else:
                if precision:
                    txt_lab = utils.precision(arr[i], precision)
                else:
                    txt_lab = str(arr[i])

            if not txt_lab:
                continue

            if font == "VTK":
                tx = vtk.vtkVectorText()
                tx.SetText(txt_lab)
                tx.Update()
                tx_poly = tx.GetOutput()
            else:
                tx_poly = vedo.shapes.Text3D(txt_lab, font=font, justify=justify).dataset

            if tx_poly.GetPointData() == 0:
                continue  #######################
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
                if cells:  # center-justify
                    bb = tx_poly.GetBounds()
                    dx, dy = (bb[1] - bb[0]) / 2, (bb[3] - bb[2]) / 2
                    T.Translate(-dx, -dy, 0)
                if xrot:
                    T.RotateX(xrot)
                if yrot:
                    T.RotateY(yrot)
                if zrot:
                    T.RotateZ(zrot)
                crossvec = np.cross([0, 0, 1], ni)
                angle = np.arccos(np.dot([0, 0, 1], ni)) * 57.3
                T.RotateWXYZ(angle, crossvec)
                if cells:  # small offset along normal only for cells
                    T.Translate(ni * scale / 2)
            else:
                if xrot:
                    T.RotateX(xrot)
                if yrot:
                    T.RotateY(yrot)
                if zrot:
                    T.RotateZ(zrot)
            T.Scale(scale, scale, scale)
            T.Translate(e)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(tx_poly)
            tf.SetTransform(T)
            tf.Update()
            tapp.AddInputData(tf.GetOutput())

        if ninputs:
            tapp.Update()
            lpoly = tapp.GetOutput()
        else:  # return an empty obj
            lpoly = vtk.vtkPolyData()

        ids = vedo.mesh.Mesh(lpoly, c=c, alpha=alpha)
        ids.property.LightingOff()
        ids.actor.PickableOff()
        ids.actor.SetUseBounds(False)
        return ids

    def labels2d(
        self,
        content="id",
        on="points",
        scale=1.0,
        precision=4,
        font="Calco",
        justify="bottom-left",
        angle=0.0,
        frame=False,
        c="black",
        bc=None,
        alpha=1.0,
    ):
        """
        Generate value or ID bi-dimensional labels for mesh cells or points.

        See also: `labels()`, `flagpole()`, `caption()` and `legend()`.

        Arguments:
            content : (str)
                either 'id', 'cellid', or array name
            on : (str)
                generate labels for "cells" instead of "points" (the default)
            scale : (float)
                size scaling of labels
            precision : (int)
                precision of numeric labels
            angle : (float)
                local rotation angle of label in degrees
            frame : (bool)
                draw a frame around the label
            bc : (str)
                background color of the label

        ```python
        from vedo import Sphere, show
        sph = Sphere(quads=True, res=4).compute_normals().wireframe()
        sph.celldata["zvals"] = sph.cell_centers[:,2]
        l2d = sph.labels("zvals", on="cells", precision=2).backcolor('orange9')
        show(sph, l2d, axes=1).close()
        ```
        ![](https://vedo.embl.es/images/feats/labels2d.png)
        """
        cells = False
        if isinstance(content, str):
            if content in ("cellid", "cellsid"):
                cells = True
                content = "id"

        if "cell" in on:
            cells = True
        elif "point" in on:
            cells = False

        if cells:
            if content != "id" and content not in self.celldata.keys():
                vedo.logger.error(f"In labels2d: cell array {content} does not exist.")
                return None
            cellcloud = Points(self.cell_centers)
            arr = self.dataset.GetCellData().GetScalars()
            poly = cellcloud
            poly.GetPointData().SetScalars(arr)
        else:
            poly = self
            if content != "id" and content not in self.pointdata.keys():
                vedo.logger.error(f"In labels2d: point array {content} does not exist.")
                return None
            self.pointdata.select(content)

        mp = vtk.vtkLabeledDataMapper()

        if content == "id":
            mp.SetLabelModeToLabelIds()
        else:
            mp.SetLabelModeToLabelScalars()
            if precision is not None:
                mp.SetLabelFormat(f"%-#.{precision}g")

        pr = mp.GetLabelTextProperty()
        c = colors.get_color(c)
        pr.SetColor(c)
        pr.SetOpacity(alpha)
        pr.SetFrame(frame)
        pr.SetFrameColor(c)
        pr.SetItalic(False)
        pr.BoldOff()
        pr.ShadowOff()
        pr.UseTightBoundingBoxOn()
        pr.SetOrientation(angle)
        pr.SetFontFamily(vtk.VTK_FONT_FILE)
        fl = utils.get_font_path(font)
        pr.SetFontFile(fl)
        pr.SetFontSize(int(20 * scale))

        if "cent" in justify or "mid" in justify:
            pr.SetJustificationToCentered()
        elif "rig" in justify:
            pr.SetJustificationToRight()
        elif "left" in justify:
            pr.SetJustificationToLeft()
        # ------
        if "top" in justify:
            pr.SetVerticalJustificationToTop()
        else:
            pr.SetVerticalJustificationToBottom()

        if bc is not None:
            bc = colors.get_color(bc)
            pr.SetBackgroundColor(bc)
            pr.SetBackgroundOpacity(alpha)

        mp.SetInputData(poly)
        a2d = vtk.vtkActor2D()
        a2d.PickableOff()
        a2d.SetMapper(mp)
        return a2d

    def legend(self, txt):
        """Book a legend text."""
        self.info["legend"] = txt
        return self

    def flagpole(
        self,
        txt=None,
        point=None,
        offset=None,
        s=None,
        font="",
        rounded=True,
        c=None,
        alpha=1.0,
        lw=2,
        italic=0.0,
        padding=0.1,
    ):
        """
        Generate a flag pole style element to describe an object.
        Returns a `Mesh` object.

        Use flagpole.follow_camera() to make it face the camera in the scene.

        Consider using `settings.use_parallel_projection = True` 
        to avoid perspective distortions.

        See also `flagpost()`.

        Arguments:
            txt : (str)
                Text to display. The default is the filename or the object name.
            point : (list)
                position of the flagpole pointer. 
            offset : (list)
                text offset wrt the application point. 
            s : (float)
                size of the flagpole.
            font : (str)
                font face. Check [available fonts here](https://vedo.embl.es/fonts).
            rounded : (bool)
                draw a rounded or squared box around the text.
            c : (list)
                text and box color.
            alpha : (float)
                opacity of text and box.
            lw : (float)
                line with of box frame.
            italic : (float)
                italicness of text.

        Examples:
            - [intersect2d.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/intersect2d.py)

                ![](https://vedo.embl.es/images/pyplot/intersect2d.png)

            - [goniometer.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/goniometer.py)
            - [flag_labels1.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels1.py)
            - [flag_labels2.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels2.py)
        """
        acts = []

        if txt is None:
            if self.filename:
                txt = self.filename.split("/")[-1]
            elif self.name:
                txt = self.name
            else:
                return None

        x0, x1, y0, y1, z0, z1 = self.bounds()
        d = self.diagonal_size()
        if point is None:
            if d:
                point = self.closest_point([(x0 + x1) / 2, (y0 + y1) / 2, z1])
                # point = self.closest_point([x1, y0, z1])
            else:  # it's a Point
                point = self.transform.position

        pt = utils.make3d(point)

        if offset is None:
            offset = [(x1 - x0) / 1.75, (y1 - y0) / 5, 0]
        offset = utils.make3d(offset)

        if s is None:
            s = d / 20

        sph = None
        if d and (z1 - z0) / d > 0.1:
            sph = vedo.shapes.Sphere(pt, r=s * 0.4, res=6)

        if c is None:
            c = np.array(self.color()) / 1.4

        lab = vedo.shapes.Text3D(
            txt, pos=pt+offset, s=s, 
            font=font, italic=italic, justify="center"
        )
        acts.append(lab)

        if d and not sph:
            sph = vedo.shapes.Circle(pt, r=s / 3, res=15)
        acts.append(sph)

        x0, x1, y0, y1, z0, z1 = lab.bounds()
        aline = [(x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0)]
        if rounded:
            box = vedo.shapes.KSpline(aline, closed=True)
        else:
            box = vedo.shapes.Line(aline, closed=True)

        cnt = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]

        box.actor.SetOrigin(cnt)
        box.scale([1 + padding, 1 + 2 * padding, 1], origin=cnt)
        acts.append(box)

        x0, x1, y0, y1, z0, z1 = box.bounds()
        if x0 < pt[0] < x1:
            c0 = box.closest_point(pt)
            c1 = [c0[0], c0[1] + (pt[1] - y0) / 4, pt[2]]
        elif (pt[0] - x0) < (x1 - pt[0]):
            c0 = [x0, (y0 + y1) / 2, pt[2]]
            c1 = [x0 + (pt[0] - x0) / 4, (y0 + y1) / 2, pt[2]]
        else:
            c0 = [x1, (y0 + y1) / 2, pt[2]]
            c1 = [x1 + (pt[0] - x1) / 4, (y0 + y1) / 2, pt[2]]

        con = vedo.shapes.Line([c0, c1, pt])
        acts.append(con)

        macts = vedo.merge(acts).c(c).alpha(alpha)
        macts.actor.SetOrigin(pt)
        macts.bc("tomato").pickable(False)
        macts.property.LightingOff()
        macts.property.SetLineWidth(lw)
        macts.actor.UseBoundsOff()
        macts.name = "FlagPole"
        return macts

    def flagpost(
        self,
        txt=None,
        point=None,
        offset=None,
        s=1.0,
        c="k9",
        bc="k1",
        alpha=1,
        lw=0,
        font="Calco",
        justify="center-left",
        vspacing=1.0,
    ):
        """
        Generate a flag post style element to describe an object.

        Arguments:
            txt : (str)
                Text to display. The default is the filename or the object name.
            point : (list)
                position of the flag anchor point. The default is None.
            offset : (list)
                a 3D displacement or offset. The default is None.
            s : (float)
                size of the text to be shown
            c : (list)
                color of text and line
            bc : (list)
                color of the flag background
            alpha : (float)
                opacity of text and box.
            lw : (int)
                line with of box frame. The default is 0.
            font : (str)
                font name. Use a monospace font for better rendering. The default is "Calco".
                Type `vedo -r fonts` for a font demo.
                Check [available fonts here](https://vedo.embl.es/fonts).
            justify : (str)
                internal text justification. The default is "center-left".
            vspacing : (float)
                vertical spacing between lines.

        Examples:
            - [flag_labels2.py](https://github.com/marcomusy/vedo/tree/master/examples/examples/other/flag_labels2.py)

            ![](https://vedo.embl.es/images/other/flag_labels2.png)
        """
        if txt is None:
            if self.filename:
                txt = self.filename.split("/")[-1]
            elif self.name:
                txt = self.name
            else:
                return None

        x0, x1, y0, y1, z0, z1 = self.bounds()
        d = self.diagonal_size()
        if point is None:
            if d:
                point = self.closest_point([(x0 + x1) / 2, (y0 + y1) / 2, z1])
            else:  # it's a Point
                point = self.transform.position

        point = utils.make3d(point)

        if offset is None:
            offset = [0, 0, (z1 - z0) / 2]
        offset = utils.make3d(offset)

        fpost = vedo.addons.Flagpost(
            txt, point, point + offset, s, c, bc, alpha, lw, font, justify, vspacing
        )
        self._caption = fpost
        return fpost

    def caption(
        self,
        txt=None,
        point=None,
        size=(0.30, 0.15),
        padding=5,
        font="Calco",
        justify="center-right",
        vspacing=1.0,
        c=None,
        alpha=1.0,
        lw=1,
        ontop=True,
    ):
        """
        Add a 2D caption to an object which follows the camera movements.
        Latex is not supported. Returns the same input object for concatenation.

        See also `flagpole()`, `flagpost()`, `labels()` and `legend()`
        with similar functionality.

        Arguments:
            txt : (str)
                text to be rendered. The default is the file name.
            point : (list)
                anchoring point. The default is None.
            size : (list)
                (width, height) of the caption box. The default is (0.30, 0.15).
            padding : (float)
                padding space of the caption box in pixels. The default is 5.
            font : (str)
                font name. Use a monospace font for better rendering. The default is "VictorMono".
                Type `vedo -r fonts` for a font demo.
                Check [available fonts here](https://vedo.embl.es/fonts).
            justify : (str)
                internal text justification. The default is "center-right".
            vspacing : (float)
                vertical spacing between lines. The default is 1.
            c : (str)
                text and box color. The default is 'lb'.
            alpha : (float)
                text and box transparency. The default is 1.
            lw : (int)
                line width in pixels. The default is 1.
            ontop : (bool)
                keep the 2d caption always on top. The default is True.

        Examples:
            - [caption.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/caption.py)

                ![](https://vedo.embl.es/images/pyplot/caption.png)

            - [flag_labels1.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels1.py)
            - [flag_labels2.py](https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels2.py)
        """
        if txt is None:
            if self.filename:
                txt = self.filename.split("/")[-1]
            elif self.name:
                txt = self.name

        if not txt:  # disable it
            self._caption = None
            return self

        for r in vedo.shapes._reps:
            txt = txt.replace(r[0], r[1])

        if c is None:
            c = np.array(self.property.GetColor()) / 2
        else:
            c = colors.get_color(c)

        if point is None:
            x0, x1, y0, y1, _, z1 = self.dataset.GetBounds()
            pt = [(x0 + x1) / 2, (y0 + y1) / 2, z1]
            point = self.closest_point(pt)

        capt = vtk.vtkCaptionActor2D()
        capt.SetAttachmentPoint(point)
        capt.SetBorder(True)
        capt.SetLeader(True)
        sph = vtk.vtkSphereSource()
        sph.Update()
        capt.SetLeaderGlyphData(sph.GetOutput())
        capt.SetMaximumLeaderGlyphSize(5)
        capt.SetPadding(int(padding))
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
        fl = utils.get_font_path(font)
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


###################################################
class Points(PointsVisual, BaseActor):
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
                return [x,y,z]

            Points(fibonacci_sphere(1000)).show(axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/fibonacci.png)
        """
        # super().__init__()  ## super is not working here
        BaseActor.__init__(self)
        PointsVisual.__init__(self)

        self.dataset = vtk.vtkPolyData()

        self.transform = LinearTransform()
        self.actor.data = self

        if inputobj is None:  ####################
            return
        ########################################

        self.name = "Points" # better not to give it a name here?

        if isinstance(inputobj, vedo.BaseActor):
            inputobj = inputobj.vertices  # numpy

        ######
        if isinstance(inputobj, vtk.vtkActor):
            self.dataset.DeepCopy(inputobj.GetMapper().GetInput())
            pr = vtk.vtkProperty()
            pr.DeepCopy(inputobj.GetProperty())
            self.actor.SetProperty(pr)
            self.property = pr
            self.mapper.SetScalarVisibility(inputobj.GetMapper().GetScalarVisibility())

        elif isinstance(inputobj, vtk.vtkPolyData):
            self.dataset.DeepCopy(inputobj)
            if self.dataset.GetNumberOfCells() == 0:
                carr = vtk.vtkCellArray()
                for i in range(self.dataset.GetNumberOfPoints()):
                    carr.InsertNextCell(1)
                    carr.InsertCellPoint(i)
                self.dataset.SetVerts(carr)
 
        elif utils.is_sequence(inputobj):  # passing point coords
            self.dataset = utils.buildPolyData(utils.make3d(inputobj))
        
        elif isinstance(inputobj, str):
            verts = vedo.file_io.load(inputobj)
            self.filename = inputobj
            self.dataset = verts.dataset

        else:
            # try to extract the points from a generic VTK input data object
            try:
                vvpts = inputobj.GetPoints()
                self.dataset.SetPoints(vvpts)
                for i in range(inputobj.GetPointData().GetNumberOfArrays()):
                    arr = inputobj.GetPointData().GetArray(i)
                    self.dataset.GetPointData().AddArray(arr)
            except:
                vedo.logger.error(f"cannot build Points from type {type(inputobj)}")
                raise RuntimeError()

        self.actor.SetMapper(self.mapper)
        self.mapper.SetInputData(self.dataset)

        self.property.SetColor(colors.get_color(c))
        self.property.SetOpacity(alpha)
        self.property.SetRepresentationToPoints()
        self.property.SetPointSize(r)
        self.property.LightingOff()

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
    
    @property
    def vertices(self):
        """Return the vertices (points) coordinates."""
        varr = self.dataset.GetPoints().GetData()
        narr = utils.vtk2numpy(varr)
        return narr

    #setter
    @vertices.setter
    def vertices(self, pts):
        """Set vertices (points) coordinates."""
        arr = utils.numpy2vtk(pts, dtype=np.float32)
        vpts = self.dataset.GetPoints()
        vpts.SetData(arr)
        vpts.Modified()
        # reset mesh to identity matrix position/rotation:
        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None
        self.actor.PokeMatrix(vtk.vtkMatrix4x4())
        self.transform = LinearTransform()
        # BUG
        # from vedo import *
        # plt = Plotter(interactive=False, axes=7)
        # s = Disc(res=(8,120)).linewidth(0.1)
        # print([s.dataset.GetPoints().GetData()])
        # # plt.show(s)  # depends if I show it or not..
        # # plt.renderer.AddActor(s.actor)
        # print([s.dataset.GetPoints().GetData()])
        # for i in progressbar(100):
        #     s.vertices[:,2] = sin(i/10.*s.vertices[:,0])/5 # move vertices in z
        # show(s, interactive=1)
        # exit()
        return self

    def polydata(self):
        """Return the underlying ``vtkPolyData`` object."""
        print("WARNING: call to .polydata() is obsolete, you can use property `.dataset`.")
        return self.dataset

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
        help_url = "https://vedo.embl.es/docs/vedo/pointcloud.html"

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
        """Obsolete.
        You can remove it anywhere from your code.
        """
        print("WARNING: call to .polydata() is obsolete, you can remove it from your code.")
        return self

    def clone(self, deep=True):
        """
        Clone a `PointCloud` or `Mesh` object to make an exact copy of it.

        Arguments:
            deep : (bool)
                if False only build a shallow copy of the object (faster copy).

        Examples:
            - [mirror.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mirror.py)

               ![](https://vedo.embl.es/images/basic/mirror.png)
        """
        if isinstance(self, vedo.Mesh):
            cloned = vedo.Mesh(self.dataset)
        else:
            cloned = Points(self.dataset)
    
        cloned.transform = self.transform.clone()

        cloned.copy_properties_from(self)

        cloned.base = np.array(self.base)
        cloned.top = np.array(self.top)
        cloned.name = str(self.name)
        cloned.filename = str(self.filename)
        cloned.info = dict(self.info)

        # dont share the same locators with original obj
        cloned.point_locator = None
        cloned.cell_locator = None
        cloned.line_locator = None

        cloned.pipeline = utils.OperationNode("clone", parents=[self], shape="diamond", c="#edede9")
        return cloned

    def clone2d(
        self,
        pos=(0, 0),
        coordsys=4,
        scale=None,
        c=None,
        alpha=None,
        ps=2,
        lw=1,
        sendback=False,
        layer=0,
    ):
        """
        Copy a 3D Mesh into a static 2D image. Returns a `vtkActor2D`.

        Arguments:
            coordsys : (int)
                the coordinate system, options are
                - 0 = Displays
                - 1 = Normalized Display
                - 2 = Viewport (origin is the bottom-left corner of the window)
                - 3 = Normalized Viewport
                - 4 = View (origin is the center of the window)
                - 5 = World (anchor the 2d image to mesh)

            ps : (int)
                point size in pixel units

            lw : (int)
                line width in pixel units

            sendback : (bool)
                put it behind any other 3D object

        Examples:
            - [clone2d.py](https://github.com/marcomusy/vedo/tree/master/examples/other/clone2d.py)

                ![](https://vedo.embl.es/images/other/clone2d.png)
        """
        if scale is None:
            msiz = self.diagonal_size()
            if vedo.plotter_instance and vedo.plotter_instance.window:
                sz = vedo.plotter_instance.window.GetSize()
                dsiz = utils.mag(sz)
                scale = dsiz / msiz / 10
            else:
                scale = 350 / msiz

        cmsh = self.clone()
        poly = cmsh.pos(0, 0, 0).scale(scale)

        mapper3d = self.mapper
        cm = mapper3d.GetColorMode()
        lut = mapper3d.GetLookupTable()
        sv = mapper3d.GetScalarVisibility()
        use_lut = mapper3d.GetUseLookupTableScalarRange()
        vrange = mapper3d.GetScalarRange()
        sm = mapper3d.GetScalarMode()

        mapper2d = vtk.vtkPolyDataMapper2D()
        mapper2d.ShallowCopy(mapper3d)
        mapper2d.SetInputData(poly)
        mapper2d.SetColorMode(cm)
        mapper2d.SetLookupTable(lut)
        mapper2d.SetScalarVisibility(sv)
        mapper2d.SetUseLookupTableScalarRange(use_lut)
        mapper2d.SetScalarRange(vrange)
        mapper2d.SetScalarMode(sm)

        act2d = vtk.vtkActor2D()
        act2d.SetMapper(mapper2d)
        act2d.SetLayerNumber(layer)
        csys = act2d.GetPositionCoordinate()
        csys.SetCoordinateSystem(coordsys)
        act2d.SetPosition(pos)
        if c is not None:
            c = colors.get_color(c)
            act2d.GetProperty().SetColor(c)
            mapper2d.SetScalarVisibility(False)
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

    def delete_cells_by_point_index(self, indices):
        """
        Delete a list of vertices identified by any of their vertex index.

        See also `delete_cells()`.

        Examples:
            - [elete_mesh_pts.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/elete_mesh_pts.py)

                ![](https://vedo.embl.es/images/basic/deleteMeshPoints.png)
        """
        cell_ids = vtk.vtkIdList()
        self.dataset.BuildLinks()
        n = 0
        for i in np.unique(indices):
            self.dataset.GetPointCells(i, cell_ids)
            for j in range(cell_ids.GetNumberOfIds()):
                self.dataset.DeleteCell(cell_ids.GetId(j))  # flag cell
                n += 1

        self.dataset.RemoveDeletedCells()
        self.mapper.Modified()
        self.pipeline = utils.OperationNode(f"delete {n} cells\nby point index", parents=[self])
        return self

    def compute_normals_with_pca(self, n=20, orientation_point=None, invert=False):
        """
        Generate point normals using PCA (principal component analysis).
        Basically this estimates a local tangent plane around each sample point p
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
        pcan = vtk.vtkPCANormalEstimation()
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
            df = vtk.vtkDistancePolyDataFilter()
            df.ComputeSecondDistanceOff()
            df.SetInputData(0, poly1)
            df.SetInputData(1, poly2)
            df.SetSignedDistance(signed)
            df.SetNegateDistance(invert)
            df.Update()
            scals = df.GetOutput().GetPointData().GetScalars()
            dists = utils.vtk2numpy(scals)

        else:  # has no polygons and vtkDistancePolyDataFilter wants them (dont know why)

            if signed:
                vedo.logger.warning("distance_to() called with signed=True but input object has no polygons")

            if not pcloud.point_locator:
                pcloud.point_locator = vtk.vtkPointLocator()
                pcloud.point_locator.SetDataSet(pcloud)
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
            comment=f"#pts {self.dataset.GetPointData()}",
        )
        return dists

    def clean(self):
        """
        Clean pointcloud or mesh by removing coincident points.
        """
        cpd = vtk.vtkCleanPolyData()
        cpd.PointMergingOn()
        cpd.ConvertLinesToPointsOn()
        cpd.ConvertPolysToLinesOn()
        cpd.ConvertStripsToPolysOn()
        cpd.SetInputData(self.dataset)
        cpd.Update()
        self.dataset.DeepCopy(cpd.GetOutput())
        self.pipeline = utils.OperationNode(
            "clean", parents=[self],
            comment=f"#pts {self.dataset.GetNumberOfPoints()}"
        )
        return self

    def subsample(self, fraction, absolute=False):
        """
        Subsample a point cloud by requiring that the points
        or vertices are far apart at least by the specified fraction of the object size.
        If a Mesh is passed the polygonal faces are not removed
        but holes can appear as vertices are removed.

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

        cpd = vtk.vtkCleanPolyData()
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
        if self.property.GetRepresentation() == 0:
            ps = self.property.GetPointSize()

        self.dataset.DeepCopy(cpd.GetOutput())
        self.ps(ps)

        self.pipeline = utils.OperationNode(
            "subsample", parents=[self], comment=f"#pts {self.dataset.GetPointData()}"
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
        thres = vtk.vtkThreshold()
        thres.SetInputData(self.dataset)

        if on.startswith("c"):
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS

        thres.SetInputArrayToProcess(0, 0, 0, asso, scalars)

        if above is None and below is not None:
            try: # vtk 9.2
                thres.ThresholdByLower(below)
            except AttributeError: # vtk 9.3
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

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(thres.GetOutput())
        gf.Update()
        self.dataset.DeepCopy(gf.GetOutput())
        self.pipeline = utils.OperationNode("threshold", parents=[self])
        return self

    def quantize(self, value):
        """
        The user should input a value and all {x,y,z} coordinates
        will be quantized to that absolute grain size.
        """
        qp = vtk.vtkQuantizePolyDataPoints()
        qp.SetInputData(self.dataset)
        qp.SetQFactor(value)
        qp.Update()
        self.dataset.DeepCopy(qp.GetOutput())
        self.flat()
        self.pipeline = utils.OperationNode("quantize", parents=[self])
        return self

    def average_size(self):
        """
        Calculate the average size of a mesh.
        This is the mean of the vertex distances from the center of mass.
        """
        coords = self.vertices
        cm = np.mean(coords, axis=0)
        if coords.shape[0] == 0:
            return 0.0
        cc = coords - cm
        return np.mean(np.linalg.norm(cc, axis=1))

    def center_of_mass(self):
        """Get the center of mass of mesh."""
        cmf = vtk.vtkCenterOfMass()
        cmf.SetInputData(self.dataset)
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)

    def normal_at(self, i):
        """Return the normal vector at vertex point `i`."""
        normals = self.dataset.GetPointData().GetNormals()
        return np.array(normals.GetTuple(i))

    def normals(self, cells=False, recompute=True):
        """Retrieve vertex normals as a numpy array.

        Arguments:
            cells : (bool)
                if `True` return cell normals.

            recompute : (bool)
                if `True` normals are recalculated if not already present.
                Note that this might modify the number of mesh points.
        """
        if cells:
            vtknormals = self.dataset.GetCellData().GetNormals()
        else:
            vtknormals = self.dataset.GetPointData().GetNormals()
        if not vtknormals and recompute:
            try:
                self.compute_normals(cells=cells)
                if cells:
                    vtknormals = self.dataset.GetCellData().GetNormals()
                else:
                    vtknormals = self.dataset.GetPointData().GetNormals()
            except AttributeError:
                # can be that 'Points' object has no attribute 'compute_normals'
                pass

        if not vtknormals:
            return np.array([])
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
        icp = vtk.vtkIterativeClosestPointTransform()
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

    def transform_with_landmarks(
        self, source_landmarks, target_landmarks, rigid=False, affine=False, least_squares=False
    ):
        """
        Transform mesh orientation and position based on a set of landmarks points.
        The algorithm finds the best matching of source points to target points
        in the mean least square sense, in one single step.

        If affine is True the x, y and z axes can scale independently but stay collinear.
        With least_squares they can vary orientation.

        Examples:
            - [align5.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align5.py)

                ![](https://vedo.embl.es/images/basic/align5.png)
        """

        if utils.is_sequence(source_landmarks):
            ss = vtk.vtkPoints()
            for p in source_landmarks:
                ss.InsertNextPoint(p)
        else:
            ss = source_landmarks.dataset.GetPoints()
            if least_squares:
                source_landmarks = source_landmarks.vertices

        if utils.is_sequence(target_landmarks):
            st = vtk.vtkPoints()
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

        lmt = vtk.vtkLandmarkTransform()
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
            M = vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    M.SetElement(j, i, m[i][j])
            lmt = vtk.vtkTransform()
            lmt.Translate(cmt)
            lmt.Concatenate(M)
            lmt.Translate(-cms)

        else:
            lmt.Update()

        self.apply_transform(lmt)
        self.pipeline = utils.OperationNode("transform_with_landmarks", parents=[self])
        return self

    def normalize(self):
        """Scale Mesh average size to unit."""
        coords = self.vertices
        if not coords.shape[0]:
            return self
        cm = np.mean(coords, axis=0)
        pts = coords - cm
        xyz2 = np.sum(pts * pts, axis=0)
        scale = 1 / np.sqrt(np.sum(xyz2) / len(pts))
        self.scale(scale).pos(cm)
        return self

    def mirror(self, axis="x", origin=True):
        """
        Mirror the mesh  along one of the cartesian axes

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
        
        self.pipeline = utils.OperationNode(f"mirror\naxis = {axis}", parents=[self])

        if sx * sy * sz < 0:
            self.reverse()
        return self

    def flip_normals(self):
        """Flip all mesh normals. Same as `mesh.mirror('n')`."""
        rs = vtk.vtkReverseSense()
        rs.SetInputData(self.dataset)
        rs.ReverseCellsOff()
        rs.ReverseNormalsOn()
        rs.Update()
        self.dataset.DeepCopy(rs.GetOutput())
        self.pipeline = utils.OperationNode("flip_normals", parents=[self])
        return self

    def interpolate_data_from(
        self,
        source,
        radius=None,
        n=None,
        kernel="shepard",
        exclude=("Normals",),
        on="points",
        null_strategy=1,
        null_value=0,
    ):
        """
        Interpolate over source to port its data onto the current object using various kernels.

        If n (number of closest points to use) is set then radius value is ignored.

        Arguments:
            kernel : (str)
                available kernels are [shepard, gaussian, linear, voronoi]
            null_strategy : (int)
                specify a strategy to use when encountering a "null" point
                during the interpolation process. Null points occur when the local neighborhood
                (of nearby points to interpolate from) is empty.

                - Case 0: an output array is created that marks points
                  as being valid (=1) or null (invalid =0), and the null_value is set as well
                - Case 1: the output data value(s) are set to the provided null_value
                - Case 2: simply use the closest point to perform the interpolation.
            null_value : (float)
                see above.

        Examples:
            - [interpolate_scalar3.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/interpolate_scalar3.py)

                ![](https://vedo.embl.es/images/advanced/interpolateMeshArray.png)
        """
        if radius is None and not n:
            vedo.logger.error("in interpolate_data_from(): please set either radius or n")
            raise RuntimeError

        if on == "points":
            points = source.dataset
        elif on == "cells":
            poly2 = vtk.vtkPolyData()
            poly2.ShallowCopy(source.dataset)
            c2p = vtk.vtkCellDataToPointData()
            c2p.SetInputData(poly2)
            c2p.Update()
            points = c2p.GetOutput()
        else:
            vedo.logger.error("in interpolate_data_from(), on must be on points or cells")
            raise RuntimeError()

        locator = vtk.vtkPointLocator()
        locator.SetDataSet(points)
        locator.BuildLocator()

        if kernel.lower() == "shepard":
            kern = vtk.vtkShepardKernel()
            kern.SetPowerParameter(2)
        elif kernel.lower() == "gaussian":
            kern = vtk.vtkGaussianKernel()
            kern.SetSharpness(2)
        elif kernel.lower() == "linear":
            kern = vtk.vtkLinearKernel()
        elif kernel.lower() == "voronoi":
            kern = vtk.vtkProbabilisticVoronoiKernel()
        else:
            vedo.logger.error("available kernels are: [shepard, gaussian, linear, voronoi]")
            raise RuntimeError()

        if n:
            kern.SetNumberOfPoints(n)
            kern.SetKernelFootprintToNClosest()
        else:
            kern.SetRadius(radius)
            kern.SetKernelFootprintToRadius()

        interpolator = vtk.vtkPointInterpolator()
        interpolator.SetInputData(self.dataset)
        interpolator.SetSourceData(points)
        interpolator.SetKernel(kern)
        interpolator.SetLocator(locator)
        interpolator.PassFieldArraysOff()
        interpolator.SetNullPointsStrategy(null_strategy)
        interpolator.SetNullValue(null_value)
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

        self.dataset.DeepCopy(cpoly)

        self.pipeline = utils.OperationNode("interpolate_data_from", parents=[self, source])
        return self


    def add_gaussian_noise(self, sigma=1.0):
        """
        Add gaussian noise to point positions.
        An extra array is added named "GaussianNoise" with the shifts.

        Arguments:
            sigma : (float)
                nr. of standard deviations, expressed in percent of the diagonal size of mesh.
                Can also be a list [sigma_x, sigma_y, sigma_z].

        Examples:
            ```python
            from vedo import Sphere
            Sphere().add_gaussian_noise(1.0).point_size(8).show().close()
            ```
        """
        sz = self.diagonal_size()
        pts = self.vertices
        n = len(pts)
        ns = (np.random.randn(n, 3) * sigma) * (sz / 100)
        vpts = vtk.vtkPoints()
        vpts.SetNumberOfPoints(n)
        vpts.SetData(utils.numpy2vtk(pts + ns, dtype=np.float32))
        self.dataset.SetPoints(vpts)
        self.dataset.GetPoints().Modified()
        self.pointdata["GaussianNoise"] = -ns
        self.pipeline = utils.OperationNode(
            "gaussian_noise", parents=[self], shape="egg", comment=f"sigma = {sigma}"
        )
        return self


    def closest_point(self, pt, n=1, radius=None, return_point_id=False, return_cell_id=False):
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
        """
        # NB: every time the mesh moves or is warped the locators are set to None
        if ((n > 1 or radius) or (n == 1 and return_point_id)) and not return_cell_id:
            poly = None
            if not self.point_locator:
                poly = self.dataset
                self.point_locator = vtk.vtkStaticPointLocator()
                self.point_locator.SetDataSet(poly)
                self.point_locator.BuildLocator()

            ##########
            if radius:
                vtklist = vtk.vtkIdList()
                self.point_locator.FindPointsWithinRadius(radius, pt, vtklist)
            elif n > 1:
                vtklist = vtk.vtkIdList()
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
                    self.cell_locator = vtk.vtkStaticCellLocator()
                else:
                    self.cell_locator = vtk.vtkCellLocator()

                self.cell_locator.SetDataSet(poly)
                self.cell_locator.BuildLocator()

            if radius is not None:
                vedo.printc("Warning: closest_point() with radius is not implemented for cells.", c='r')   
 
            if n != 1:
                vedo.printc("Warning: closest_point() with n>1 is not implemented for cells.", c='r')   
 
            trgp = [0, 0, 0]
            cid = vtk.mutable(0)
            dist2 = vtk.mutable(0)
            subid = vtk.mutable(0)
            self.cell_locator.FindClosestPoint(pt, trgp, cid, subid, dist2)

            if return_cell_id:
                return int(cid)

            return np.array(trgp)


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
        hp = vtk.vtkHausdorffDistancePointSetFilter()
        hp.SetInputData(0, self)
        hp.SetInputData(1, points)
        hp.SetTargetDistanceMethodToPointToCell()
        hp.Update()
        return hp.GetHausdorffDistance()

    def chamfer_distance(self, pcloud):
        """
        Compute the Chamfer distance to the input point set.
        Returns a single `float`.
        """
        if not pcloud.point_locator:
            pcloud.point_locator = vtk.vtkPointLocator()
            pcloud.point_locator.SetDataSet(pcloud)
            pcloud.point_locator.BuildLocator()
        if not self.point_locator:
            self.point_locator = vtk.vtkPointLocator()
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
        removal = vtk.vtkRadiusOutlierRemoval()
        removal.SetInputData(self.dataset)
        removal.SetRadius(radius)
        removal.SetNumberOfNeighbors(neighbors)
        removal.GenerateOutliersOff()
        removal.Update()
        inputobj = removal.GetOutput()
        if inputobj.GetNumberOfCells() == 0:
            carr = vtk.vtkCellArray()
            for i in range(inputobj.GetNumberOfPoints()):
                carr.InsertNextCell(1)
                carr.InsertCellPoint(i)
            inputobj.SetVerts(carr)
        self.dataset.DeepCopy(inputobj)
        self.mapper.ScalarVisibilityOff()
        self.pipeline = utils.OperationNode("remove_outliers", parents=[self])
        return self

    def smooth_mls_1d(self, f=0.2, radius=None):
        """
        Smooth mesh or points with a `Moving Least Squares` variant.
        The point data array "Variances" will contain the residue calculated for each point.
        Input mesh's polydata is modified.

        Arguments:
            f : (float)
                smoothing factor - typical range is [0,2].
            radius : (float)
                radius search in absolute units. If set then `f` is ignored.

        Examples:
            - [moving_least_squares1D.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares1D.py)
            - [skeletonize.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/skeletonize.py)

            ![](https://vedo.embl.es/images/advanced/moving_least_squares1D.png)
        """
        coords = self.vertices
        ncoords = len(coords)

        if radius:
            Ncp = 0
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

        vdata = utils.numpy2vtk(np.array(variances))
        vdata.SetName("Variances")
        self.dataset.GetPointData().AddArray(vdata)
        self.dataset.GetPointData().Modified()
        self.vertices = newline
        self.pipeline = utils.OperationNode("smooth_mls_1d", parents=[self])
        return self

    def smooth_mls_2d(self, f=0.2, radius=None):
        """
        Smooth mesh or points with a `Moving Least Squares` algorithm variant.
        The list `mesh.info['variances']` contains the residue calculated for each point.
        When a radius is specified points that are isolated will not be moved and will get
        a False entry in array `mesh.info['is_valid']`.

        Arguments:
            f : (float)
                smoothing factor - typical range is [0,2].
            radius : (float)
                radius search in absolute units. If set then `f` is ignored.

        Examples:
            - [moving_least_squares2D.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares2D.py)
            - [recosurface.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/recosurface.py)

                ![](https://vedo.embl.es/images/advanced/recosurface.png)
        """
        coords = self.vertices
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
        for p in coords:
            if pb:
                pb.print("smoothMLS2D working ...")
            pts = self.closest_point(p, n=Ncp, radius=radius)
            if len(pts) > 3:
                ptsmean = pts.mean(axis=0)  # plane center
                _, dd, vv = np.linalg.svd(pts - ptsmean)
                cv = np.cross(vv[0], vv[1])
                t = (np.dot(cv, ptsmean) - np.dot(cv, p)) / np.dot(cv, cv)
                newp = p + cv * t
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
        self.info["is_valid"] = np.array(valid)
        self.vertices = newpts

        self.pipeline = utils.OperationNode("smooth_mls_2d", parents=[self])
        return self

    def smooth_lloyd_2d(self, iterations=2, bounds=None, options="Qbb Qc Qx"):
        """Lloyd relaxation of a 2D pointcloud."""
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
        # m = vedo.Mesh([pts, self.cells]) # not yet working properly
        out = Points(pts, c="k")
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
        `Thin Plate Spline` transformations describe a nonlinear warp transform defined by a set
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

                ![](https://vedo.embl.es/images/advanced/warp2.png)
        """
        parents = [self]
        if isinstance(source, Points):
            parents.append(source)
            source = source.vertices
        else:
            source = utils.make3d(source)

        if isinstance(target, Points):
            parents.append(target)
            target = target.vertices
        else:
            target = utils.make3d(target)

        ns = len(source)
        ptsou = vtk.vtkPoints()
        ptsou.SetNumberOfPoints(ns)
        for i in range(ns):
            ptsou.SetPoint(i, source[i])

        nt = len(target)
        if ns != nt:
            vedo.logger.error(f"#source {ns} != {nt} #target points")
            raise RuntimeError()

        pttar = vtk.vtkPoints()
        pttar.SetNumberOfPoints(nt)
        for i in range(ns):
            pttar.SetPoint(i, target[i])

        T = vtk.vtkThinPlateSplineTransform()
        if mode.lower() == "3d":
            T.SetBasisToR()
        elif mode.lower() == "2d":
            T.SetBasisToR2LogR()
        else:
            vedo.logger.error(f"unknown mode {mode}")
            raise RuntimeError()

        T.SetSigma(sigma)
        T.SetSourceLandmarks(ptsou)
        T.SetTargetLandmarks(pttar)
        # self.transform = T
        self.apply_transform(T)

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
            cube.back_color('pink').show()
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
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        clipper = vtk.vtkClipPolyData()
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

        Check out also:
            `cut_with_box()`, `cut_with_cylinder()`, `cut_with_sphere()`
        """

        vpoints = vtk.vtkPoints()
        for p in utils.make3d(origins):
            vpoints.InsertNextPoint(p)
        normals = utils.make3d(normals)

        planes = vtk.vtkPlanes()
        planes.SetPoints(vpoints)
        planes.SetNormals(utils.numpy2vtk(normals, dtype=float))

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.dataset)  # must be True
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
            show(mesh, box, axes=1)
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_box_cube.png)

        Check out also:
            `cut_with_line()`, `cut_with_plane()`, `cut_with_cylinder()`
        """
        if isinstance(bounds, Points):
            bounds = bounds.bounds()

        box = vtk.vtkBox()
        if utils.is_sequence(bounds[0]):
            for bs in bounds:
                box.AddBounds(bs)
        else:
            box.SetBounds(bounds)

        clipper = vtk.vtkClipPolyData()
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
        pplane = vtk.vtkPolyPlane()
        if isinstance(points, Points):
            points = points.vertices.tolist()

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
        polyline = vtk.vtkPolyLine()
        polyline.Initialize(n, vpoints)
        polyline.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            polyline.GetPointIds().SetId(i, i)
        pplane.SetPolyLine(polyline)

        clipper = vtk.vtkClipPolyData()
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
            grid.cut_with_cookiecutter(lines)
            show(grid, lines, axes=8, bg='blackboard').close()
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
            poly = lines

        # if invert: # not working
        #     rev = vtk.vtkReverseSense()
        #     rev.ReverseCellsOn()
        #     rev.SetInputData(poly)
        #     rev.Update()
        #     poly = rev.GetOutput()

        # Build loops from the polyline
        build_loops = vtk.vtkContourLoopExtraction()
        build_loops.SetInputData(poly)
        build_loops.Update()
        boundaryPoly = build_loops.GetOutput()

        ccut = vtk.vtkCookieCutter()
        ccut.SetInputData(self.dataset)
        ccut.SetLoopsData(boundaryPoly)
        ccut.SetPointInterpolationToMeshEdges()
        # ccut.SetPointInterpolationToLoopEdges()
        ccut.PassCellDataOn()
        # ccut.PassPointDataOn()
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
            show(mesh, axes=1)
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
        cyl = vtk.vtkCylinder()
        cyl.SetCenter(center)
        cyl.SetAxis(axis[0], axis[1], axis[2])
        cyl.SetRadius(r)

        clipper = vtk.vtkClipPolyData()
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
            show(mesh, axes=1)
            ```
            ![](https://vedo.embl.es/images/feats/cut_with_sphere.png)

        Check out also:
            `cut_with_box()`, `cut_with_plane()`, `cut_with_cylinder()`
        """
        sph = vtk.vtkSphere()
        sph.SetCenter(center)
        sph.SetRadius(r)

        clipper = vtk.vtkClipPolyData()
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
        signed_distances = vtk.vtkFloatArray()
        signed_distances.SetNumberOfComponents(1)
        signed_distances.SetName("SignedDistances")

        # implicit function that will be used to slice the mesh
        ippd = vtk.vtkImplicitPolyDataDistance()
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
            vis = self.mapper.GetScalarVisibility()
        
        self._update(cpoly)

        self.pointdata.remove("SignedDistances")
        self.mapper.SetScalarVisibility(vis)
        if keep:
            if isinstance(self, vedo.Mesh):
                cutoff = vedo.Mesh(kpoly)
            else:
                cutoff = vedo.Points(kpoly)
            cutoff.property = vtk.vtkProperty()
            cutoff.property.DeepCopy(self.property)
            cutoff.actor.SetProperty(cutoff.property)
            cutoff.c("k5").alpha(0.2)
            return vedo.Assembly([self, cutoff])

        self.pipeline = utils.OperationNode("cut_with_mesh", parents=[self, mesh])
        return self

    def cut_with_point_loop(self, points, invert=False, on="points", include_boundary=False):
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
            vpts = vtk.vtkPoints()
            points = utils.make3d(points)
            for p in points:
                vpts.InsertNextPoint(p)

        if "cell" in on:
            ippd = vtk.vtkImplicitSelectionLoop()
            ippd.SetLoop(vpts)
            ippd.AutomaticNormalGenerationOn()
            clipper = vtk.vtkExtractPolyDataGeometry()
            clipper.SetInputData(self.dataset)
            clipper.SetImplicitFunction(ippd)
            clipper.SetExtractInside(not invert)
            clipper.SetExtractBoundaryCells(include_boundary)
        else:
            spol = vtk.vtkSelectPolyData()
            spol.SetLoop(vpts)
            spol.GenerateSelectionScalarsOn()
            spol.GenerateUnselectedOutputOff()
            spol.SetInputData(self.dataset)
            spol.Update()
            clipper = vtk.vtkClipPolyData()
            clipper.SetInputData(spol.GetOutput())
            clipper.SetInsideOut(not invert)
            clipper.SetValue(0.0)
        clipper.Update()
        cpoly = clipper.GetOutput()
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
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.dataset)
        clipper.SetValue(value)
        clipper.GenerateClippedOutputOff()
        clipper.SetInsideOut(not invert)
        clipper.Update()
        cpoly = clipper.GetOutput()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode("cut_with_scalars", parents=[self])
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
        cu = vtk.vtkBox()
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

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self.dataset)
        clipper.SetClipFunction(cu)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()
        cpoly = clipper.GetOutput()
        self._update(clipper.GetOutput())

        self.pipeline = utils.OperationNode(
            "crop", parents=[self], comment=f"#pts {self.dataset.GetPointData()}"
        )
        return self

    def implicit_modeller(self, distance=0.05, res=(50, 50, 50), bounds=(), maxdist=None):
        """Find the surface which sits at the specified distance from the input one."""
        if not bounds:
            bounds = self.bounds()

        if not maxdist:
            maxdist = self.diagonal_size() / 2

        imp = vtk.vtkImplicitModeller()
        imp.SetInputData(self.dataset)
        imp.SetSampleDimensions(res)
        imp.SetMaximumDistance(maxdist)
        imp.SetModelBounds(bounds)
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(imp.GetOutputPort())
        contour.SetValue(0, distance)
        contour.Update()
        poly = contour.GetOutput()
        out = vedo.Mesh(poly, c="lb")

        out.pipeline = utils.OperationNode("implicit_modeller", parents=[self])
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
        vedo.logger.debug(f"tomesh():\n\tline length = {length}")
        vedo.logger.debug(f"\tdensity = {density} length/pt_separation")

        x0, x1 = contour.xbounds()
        y0, y1 = contour.ybounds()

        if grid is None:
            if mesh_resolution is None:
                resx = int((x1 - x0) / density + 0.5)
                resy = int((y1 - y0) / density + 0.5)
                vedo.logger.debug(f"tmesh_resolution = {[resx, resy]}")
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

        grid_tmp = grid.vertices

        if jitter:
            np.random.seed(0)
            sigma = 1.0 / np.sqrt(grid.npoints) * grid.diagonal_size() * jitter
            vedo.logger.debug(f"\tsigma jittering = {sigma}")
            grid_tmp += np.random.rand(grid.npoints, 3) * sigma
            grid_tmp[:, 2] = 0.0

        todel = []
        density /= np.sqrt(3)
        vgrid_tmp = Points(grid_tmp)

        for p in contour.vertices:
            out = vgrid_tmp.closest_point(p, radius=density, return_point_id=True)
            todel += out.tolist()
        # cpoints = contour.vertices
        # for i, p in enumerate(cpoints):
        #     if i:
        #         den = utils.mag(p-cpoints[i-1])/1.732
        #     else:
        #         den = density
        #     todel += vgrid_tmp.closest_point(p, radius=den, return_point_id=True)

        grid_tmp = grid_tmp.tolist()
        for index in sorted(list(set(todel)), reverse=True):
            del grid_tmp[index]

        points = contour.vertices.tolist() + grid_tmp
        if invert:
            boundary = reversed(range(contour.npoints))
        else:
            boundary = range(contour.npoints)

        dln = Points(points).generate_delaunay2d(mode="xy", boundaries=[boundary])
        dln.compute_normals(points=False)  # fixes reversd faces
        dln.lw(0.5)

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

        sdf = vtk.vtkSignedDistance()

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

        pd = self.dataset

        if pd.GetPointData().GetNormals():
            sdf.SetInputData(pd)
        else:
            normals = vtk.vtkPCANormalEstimation()
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

        surface = vtk.vtkExtractSurface()
        surface.SetRadius(radius * 0.99)
        surface.SetHoleFilling(hole_filling)
        surface.ComputeNormalsOff()
        surface.ComputeGradientsOff()
        surface.SetInputConnection(sdf.GetOutputPort())
        surface.Update()
        m = vedo.mesh.Mesh(surface.GetOutput(), c=self.color())

        m.pipeline = utils.OperationNode(
            "reconstruct_surface", parents=[self],
            comment=f"#pts {m.dataset.GetPointData()}"
        )
        return m

    def compute_clustering(self, radius):
        """
        Cluster points in space. The `radius` is the radius of local search.
        An array named "ClusterId" is added to the vertex points.

        Examples:
            - [clustering.py](https://github.com/marcomusy/vedo/blob/master/examples/basic/clustering.py)

                ![](https://vedo.embl.es/images/basic/clustering.png)
        """
        cluster = vtk.vtkEuclideanClusterExtraction()
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
        cpf = vtk.vtkConnectedPointsFilter()
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
        A pointdata array is created with name 'DistanceToCamera'.
        """
        if vedo.plotter_instance.renderer:
            poly = self.dataset
            dc = vtk.vtkDistanceToCamera()
            dc.SetInputData(poly)
            dc.SetRenderer(vedo.plotter_instance.renderer)
            dc.Update()
            self._update(dc.GetOutput(), reset_locators=False)
        return self

    def density(
        self, dims=(40, 40, 40), bounds=None, radius=None, compute_gradient=False, locator=None
    ):
        """
        Generate a density field from a point cloud. Input can also be a set of 3D coordinates.
        Output is a `Volume`.

        The local neighborhood is specified as the `radius` around each sample position (each voxel).
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
        pdf = vtk.vtkPointDensityFilter()
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
        vol.info["radius"] = radius
        vol.locator = pdf.GetLocator()

        vol.pipeline = utils.OperationNode(
            "density", parents=[self], comment=f"dims = {tuple(vol.dimensions())}"
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
        src = vtk.vtkProgrammableSource()
        opts = self.vertices

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
        cld = Points(pts, c=None).point_size(self.property.GetPointSize())
        cld.interpolate_data_from(self, n=nclosest, radius=radius)
        cld.name = "densifiedCloud"

        cld.pipeline = utils.OperationNode(
            "densify", parents=[self], c="#e9c46a:",
            comment=f"#pts {cld.dataset.GetPointData()}"
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
        dist = vtk.vtkSignedDistance()
        dist.SetInputData(self.dataset)
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

        vol.pipeline = utils.OperationNode(
            "signed_distance",
            parents=[self],
            comment=f"dim = {tuple(vol.dimensions())}",
            c="#e9c46a:#0096c7",
        )
        return vol

    def tovolume(
        self, kernel="shepard", radius=None, n=None, bounds=None, null_value=None, dims=(25, 25, 25)
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
        probe = vtk.vtkImageData()
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
            self.point_locator = vtk.vtkPointLocator()
            self.point_locator.SetDataSet(poly)
            self.point_locator.BuildLocator()

        if kernel == "shepard":
            kern = vtk.vtkShepardKernel()
            kern.SetPowerParameter(2)
        elif kernel == "gaussian":
            kern = vtk.vtkGaussianKernel()
        elif kernel == "linear":
            kern = vtk.vtkLinearKernel()
        else:
            vedo.logger.error("Error in tovolume(), available kernels are:")
            vedo.logger.error(" [shepard, gaussian, linear]")
            raise RuntimeError()

        if radius:
            kern.SetRadius(radius)

        interpolator = vtk.vtkPointInterpolator()
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
            comment=f"dim = {tuple(vol.dimensions())}",
            c="#e9c46a:#0096c7",
        )
        return vol

    def generate_random_data(self):
        """Fill a dataset with random attributes"""
        gen = vtk.vtkRandomAttributeGenerator()
        gen.SetInputData(self.dataset)
        gen.GenerateAllDataOn()
        gen.SetDataTypeToFloat()
        gen.GeneratePointNormalsOff()
        gen.GeneratePointTensorsOn()
        gen.GenerateCellScalarsOn()
        gen.Update()

        self._update(gen.GetOutput(), reset_locators=False)

        self.pipeline = utils.OperationNode("generate\nrandom data", parents=[self])
        return self


    def generate_delaunay2d(self, mode="scipy", boundaries=(), tol=None, alpha=0.0, offset=0.0, transform=None):
        """
        Create a mesh from points in the XY plane.
        If `mode='fit'` then the filter computes a best fitting
        plane and projects the points onto it.

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
            transform: vtkTransform
                a VTK transformation (eg. a thinplate spline)
                which is applied to points to generate a 2D problem.
                This maps a 3D dataset into a 2D dataset where triangulation can be done on the XY plane.
                The points are transformed and triangulated.
                The topology of triangulated points is used as the output topology.

        Examples:
            - [delaunay2d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/delaunay2d.py)

                ![](https://vedo.embl.es/images/basic/delaunay2d.png)
        """
        plist = self.vertices

        #########################################################
        if mode == "scipy":
            from scipy.spatial import Delaunay as scipy_delaunay
            tri = scipy_delaunay(plist[:, 0:2])
            return vedo.mesh.Mesh([plist, tri.simplices])
        ##########################################################

        pd = vtk.vtkPolyData()
        vpts = vtk.vtkPoints()
        vpts.SetData(utils.numpy2vtk(plist, dtype=np.float32))
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

        if mode == "xy" and boundaries:
            boundary = vtk.vtkPolyData()
            boundary.SetPoints(vpts)
            cell_array = vtk.vtkCellArray()
            for b in boundaries:
                cpolygon = vtk.vtkPolygon()
                for idd in b:
                    cpolygon.GetPointIds().InsertNextId(idd)
                cell_array.InsertNextCell(cpolygon)
            boundary.SetPolys(cell_array)
            delny.SetSourceData(boundary)

        if mode == "fit":
            delny.SetProjectionPlaneMode(vtk.VTK_BEST_FITTING_PLANE)
        delny.Update()
        msh = vedo.mesh.Mesh(delny.GetOutput()).clean().lighting("off")

        msh.pipeline = utils.OperationNode(
            "delaunay2d", parents=[self],
            comment=f"#cells {msh.dataset.GetNumberOfCells()}"
        )
        return msh


    def generate_voronoi(self, padding=0.0, fit=False, method="vtk"):
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

            m = vedo.Mesh([vor.vertices, regs], c="orange5")
            m.celldata["VoronoiID"] = np.array(list(range(len(regs)))).astype(int)
            m.locator = None

        elif method == "vtk":
            vor = vtk.vtkVoronoi2D()
            if isinstance(pts, Points):
                vor.SetInputData(pts)
            else:
                pts = np.asarray(pts)
                if pts.shape[1] == 2:
                    pts = np.c_[pts, np.zeros(len(pts))]
                pd = vtk.vtkPolyData()
                vpts = vtk.vtkPoints()
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
            from vedo import Ellipsoid, show, visible_points
            s = Ellipsoid().rotate_y(30)

            #Camera options: pos, focal_point, viewup, distance,
            camopts = dict(pos=(0,0,25), focal_point=(0,0,0))
            show(s, camera=camopts, offscreen=True)

            m = s.visible_points()
            #print('visible pts:', m.vertices) # numpy array
            show(m, new=True, axes=1) # optionally draw result on a new window
            ```
            ![](https://vedo.embl.es/images/feats/visible_points.png)
        """
        svp = vtk.vtkSelectVisiblePoints()
        svp.SetInputData(self.dataset)
        svp.SetRenderer(vedo.plotter_instance.renderer)

        if len(area) == 4:
            # specify a rectangular region
            svp.SetSelection(area[0], area[1], area[2], area[3])
        if tol is not None:
            svp.SetTolerance(tol)
        if invert:
            svp.SelectInvisibleOn()
        svp.Update()

        m = Points(svp.GetOutput()).point_size(5)
        m.name = "VisiblePoints"
        return m
