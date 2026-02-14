#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Standalone fitting and pointcloud helper functions."""

from typing_extensions import Self

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import colors
from vedo import utils
from vedo.transformations import LinearTransform
from .core import Points

__all__ = [
    "Point",
    "merge",
    "fit_line",
    "fit_circle",
    "fit_plane",
    "fit_sphere",
    "pca_ellipse",
    "pca_ellipsoid",
    "project_point_on_variety",
]

def merge(*meshs, flag=False) -> vedo.Mesh | vedo.Points | None:
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
        try:
            polyapp.AddInputData(ob.dataset)
        except AttributeError:
            polyapp.AddInputData(ob)
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
        msh = Points(mpoly) # type: ignore

    msh.copy_properties_from(objs[0])

    msh.pipeline = utils.OperationNode(
        "merge", parents=objs, comment=f"#pts {msh.dataset.GetNumberOfPoints()}"
    )
    return msh


def _rotate_points(points, n0=None, n1=(0, 0, 1)) -> np.ndarray | tuple:
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


def fit_line(points: np.ndarray | vedo.Points) -> vedo.shapes.Line:
    """
    Fits a line through points.

    Extra info is stored in `Line.slope`, `Line.center`, `Line.variances`.

    Examples:
        - [fitline.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/fitline.py)

            ![](https://vedo.embl.es/images/advanced/fitline.png)
    """
    if isinstance(points, Points):
        points = points.coordinates
    data = np.asarray(points)
    datamean = data.mean(axis=0)
    _, dd, vv = np.linalg.svd(data - datamean)
    vv = vv[0] / np.linalg.norm(vv[0])
    # vv contains the first principal component, i.e. the direction
    # vector of the best fit line in the least squares sense.
    xyz_min = data.min(axis=0)
    xyz_max = data.max(axis=0)
    a = np.linalg.norm(xyz_min - datamean)
    b = np.linalg.norm(xyz_max - datamean)
    p1 = datamean - a * vv
    p2 = datamean + b * vv
    line = vedo.shapes.Line(p1, p2, lw=1)
    line.slope = vv
    line.center = datamean
    line.variances = dd
    return line


def fit_circle(points: np.ndarray | vedo.Points) -> tuple:
    """
    Fits a circle through a set of 3D points, with a very fast non-iterative method.

    Returns the tuple `(center, radius, normal_to_circle)`.

    .. warning::
        trying to fit s-shaped points will inevitably lead to instabilities and
        circles of small radius.

    References:
        *J.F. Crawford, Nucl. Instr. Meth. 211, 1983, 223-225.*
    """
    if isinstance(points, Points):
        points = points.coordinates
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


def fit_plane(points: np.ndarray | vedo.Points, signed=False) -> vedo.shapes.Plane:
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
        points = points.coordinates
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


def project_point_on_variety(
        pt, points, degree=3, compute_surface=False, compute_curvature=False
    ) -> tuple:
    """
    Project a point in 3D space onto a polynomial surface defined by a set of points
    around it. The polynomial degree can be adjusted.

    Arguments:
        pt : (list or np.ndarray)
            The point to smooth (3D coordinates).
        points: (np.ndarray)
            A set of points (Nx3 array) to fit the polynomial surface.
        degree: (int)
            The degree of the polynomial to fit.
        compute_surface: (bool)
            If True, returns a surface mesh of the fitted polynomial.

    Returns:
        transformed_pt : (np.ndarray)
            The projected point on the polynomial surface.
        surface_data : (tuple)
            If compute_surface is True, the first element is a vedo.Grid object representing the surface.
            If compute_curvature is True, the second element contains curvature information.
            Contains the fitted polynomial coefficients, rotation matrix, and centroid.

    Example:
        ```python
        import vedo
        from vedo.pointcloud import project_point_on_variety

        mesh = vedo.Mesh(vedo.dataurl+"bunny.obj").subdivide().scale(100)
        mesh.wireframe().alpha(0.1)

        pt = mesh.coordinates[30]
        points = mesh.closest_point(pt, n=200)

        pt_trans, res = project_point_on_variety(pt, points, degree=3, compute_surface=True)
        vpoints = vedo.Points(points, r=6, c="yellow2")

        plotter = vedo.Plotter(size=(1200, 800))
        plotter += mesh, vedo.Point(pt), vpoints, res[0], f"Residue: {pt - pt_trans}"
        plotter.show(axes=1).close()
        ```
    
    Check out also the `fit_plane()` function for a simpler case of plane fitting.
    """

    def _fit_polynomial_3d(points, degree):
        x, y, z = points.T
        # Create a Vandermonde matrix for polynomial fitting
        terms = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                terms.append(x**i * y**j)
        V = np.vstack(terms).T
        coeffs = np.linalg.lstsq(V, z, rcond=None)[0]
        return coeffs

    def _predict_polynomial_3d(x, y, coeffs, degree):
        idx = 0
        z_pred = 0
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                z_pred += coeffs[idx] * x**i * y**j
                idx += 1
        return z_pred

    def _compute_curvature(coeffs, terms, degree):
        if compute_curvature==False or degree < 2:
            return 0, 0 
        terms = [f"x^{i}y^{j}" for i in range(degree + 1) for j in range(degree + 1 - i)]
        # Print coefficients
        # for term, coeff in zip(terms, coeffs):
        #     print(f"Coefficient for {term}: {coeff}")
        # Find indices of curvature-related terms
        a_20 = a_02 = a_11 = 0
        for idx, term in enumerate(terms):
            if term == "x^2y^0":
                a_20 = coeffs[idx]
            elif term == "x^0y^2":
                a_02 = coeffs[idx]
            elif term == "x^1y^1":
                a_11 = coeffs[idx]
        # Second derivatives
        f_xx = 2 * a_20
        f_yy = 2 * a_02
        f_xy = a_11
        # Gaussian and mean curvature
        gaussian = f_xx * f_yy - f_xy**2
        mean = (f_xx**2 + f_yy**2) / 2
        return float(gaussian), float(mean)

    # Fit the plane: compute centroid and normal
    points = np.asarray(points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD to find the normal vector
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[-1, :]
    normal /= np.linalg.norm(normal)

    # Find an orthogonal basis: v1 perpendicular to normal
    axes = np.eye(3)
    crosses = np.cross(normal, axes)
    norms = np.linalg.norm(crosses, axis=1)
    max_idx = np.argmax(norms)
    v1 = crosses[max_idx]
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 /= np.linalg.norm(v2)

    # Old basis matrix with columns v1, v2, normal
    old_basis = np.column_stack((v1, v2, normal))

    # Ensure positive determinant for proper rotation
    if np.linalg.det(old_basis) < 0:
        v2 = -v2
        old_basis = np.column_stack((v1, v2, normal))

    # Rotation matrix R such that new_coords = centered @ R.T
    R = old_basis.T

    # Transform points to new coordinate system (plane aligns with XY)
    transformed = np.dot(centered, R.T)

    tpt = (pt - centroid) @ R.T  # Transform point to new coordinate system

    # Fit polynomial of arbitrary degree
    coeffs = _fit_polynomial_3d(transformed, degree)
    gauss_curv, mean_curv = _compute_curvature(coeffs, transformed, degree)

    # Predict z for a new point
    x_new, y_new = tpt[0], tpt[1]
    z_pred = _predict_polynomial_3d(x_new, y_new, coeffs, degree)

    # Transform back to original
    transformed_pt = np.array([x_new, y_new, z_pred])
    back_transformed = np.dot(transformed_pt, R) + centroid
    grid = None
    if compute_surface:
        # Create a surface mesh from the polynomial fit
        x_min, x_max = transformed[:, 0].min(), transformed[:, 0].max()
        y_min, y_max = transformed[:, 1].min(), transformed[:, 1].max()
        grid = vedo.Grid([x_min, x_max, y_min, y_max], res=(20, 20))
        grid.flat().use_bounds(False)
        gpts = grid.points
        for g in gpts:
            zg = _predict_polynomial_3d(g[0], g[1], coeffs, degree)
            tzg = np.array([g[0], g[1], zg])
            g[:] = np.dot(tzg, R)
        grid.shift(centroid).compute_normals()
        grid.lw(0).c("lightblue").alpha(0.5).lighting("glossy")
    return back_transformed, (grid, coeffs, R, centroid, gauss_curv, mean_curv)


def fit_sphere(coords: np.ndarray | vedo.Points) -> vedo.shapes.Sphere:
    """
    Fits a sphere to a set of points.

    Extra info is stored in `Sphere.radius`, `Sphere.center`, `Sphere.residue`.

    Examples:
        - [fitspheres1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/fitspheres1.py)

            ![](https://vedo.embl.es/images/advanced/fitspheres1.jpg)
    """
    if isinstance(coords, Points):
        coords = coords.coordinates
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
    except TypeError:
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


def pca_ellipse(points: np.ndarray | vedo.Points, pvalue=0.673, res=60) -> vedo.shapes.Circle | None:
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
        coords = points.coordinates
    else:
        coords = points
    if len(coords) < 4:
        vedo.logger.warning("in pca_ellipse(), there are not enough points!")
        return None

    P = np.array(coords, dtype=float)[:, (0, 1)]
    cov = np.cov(P, rowvar=0)      # type: ignore
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


def pca_ellipsoid(points: np.ndarray | vedo.Points, pvalue=0.673, res=24) -> vedo.shapes.Ellipsoid | None:
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
        coords = points.coordinates
    else:
        coords = points
    if len(coords) < 4:
        vedo.logger.warning("in pca_ellipsoid(), not enough input points!")
        return None

    P = np.array(coords, ndmin=2, dtype=float)
    cov = np.cov(P, rowvar=0)     # type: ignore
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
