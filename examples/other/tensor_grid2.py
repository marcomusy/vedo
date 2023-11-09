"""Cauchy-Green and Green-Lagrange strain tensors on a 2D grid."""
import numpy as np
import vedo

# Define a simple deformation function
def deform(x, y):
    xd = np.array([x + 0.25 * x * x, 
                   y + 0.25 * x * y])
    # Add rotation to the deformation.
    # note that the rotation is applied to the deformed configuration
    # and it has no effect on the deformation gradient tensor C and E
    rotation_angle_degrees = 10
    rotation_angle_radians = np.radians(rotation_angle_degrees)
    cos_angle = np.cos(rotation_angle_radians)
    sin_angle = np.sin(rotation_angle_radians)
    x_def, y_def = xd
    x_rot = x_def * cos_angle - y_def * sin_angle
    y_rot = x_def * sin_angle + y_def * cos_angle
    return np.array([x_rot, y_rot])

# Compute the deformation gradient tensor F
def deformation_gradient(x, y, ds=0.001):
    # Compute the deformation gradient tensor F
    # F = (df/dx, df/dy)
    fxy = deform(x, y)
    fxy_x = deform(x + ds, y)
    fxy_y = deform(x, y + ds)
    F = np.zeros((2, 2))
    F[0, 0] = (fxy_x[0] - fxy[0]) / ds
    F[0, 1] = (fxy_y[0] - fxy[0]) / ds
    F[1, 0] = (fxy_x[1] - fxy[1]) / ds
    F[1, 1] = (fxy_y[1] - fxy[1]) / ds
    return F

# Compute the right Cauchy-Green deformation tensor C
def cauchy_green(F):
    return F.T @ F

# Right Cauchy-Green tensor (C) is used to define the Green-Lagrange
# strain tensor (E), which is a measure of deformation in the reference
# (undeformed) configuration:
def green_lagrange(C):
    return 0.5 * (C - np.eye(2))

# Left Cauchy-Green tensor (B) is used to define the Almansi strain tensor (e),
# which is a measure of deformation in the current (deformed) configuration:
# e = 0.5 * (I - B^-1) (this is less used in practice)
def almansi(F):
    B = F @ F.T
    return 0.5 * (np.eye(2) - np.linalg.inv(B))

# Compute the principal stretches and directions
def principal_stretches_directions(T):
    # T is a symmetric tensor
    # eigenvalues are sorted in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    principal_stretches = np.sqrt(np.abs(eigenvalues))*np.sign(eigenvalues)
    return principal_stretches, eigenvectors

######################################################################
# Define the original grid (undeformed configuration)
x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))

grid = vedo.Grid(s=(x[0], y.T[0]))
grid_pts = grid.vertices
grid_pts_defo = deform(grid_pts[:, 0], grid_pts[:, 1])
grid_defo = grid.clone()
grid_defo.vertices = grid_pts_defo.T

# Initialize the vedo plotter
plotter = vedo.Plotter()

for i in range(x.shape[0]):
    for j in range(y.shape[1]):
        pt = x[i, j], y[i, j]
        displaced_pt = deform(*pt)
        F = deformation_gradient(*pt)

        C = cauchy_green(F)
        stretches, directions = principal_stretches_directions(C)
        ellipsoid_axes = np.diag(stretches) @ directions.T / 8
        ellipsoid_C = vedo.Ellipsoid(
            axis1=ellipsoid_axes[0],
            axis2=ellipsoid_axes[1],
            axis3=[0, 0, 0.01],
            pos=(*pt, 0),
        )
        ellipsoid_C.lighting("off").color("blue5")

        E = green_lagrange(C)
        # E = almansi(F)
        stretches, directions = principal_stretches_directions(E)
        ellipsoid_axes = np.diag(stretches) @ directions.T / 8
        ellipsoid_E = vedo.Ellipsoid(
            axis1=ellipsoid_axes[0],
            axis2=ellipsoid_axes[1],
            axis3=[0, 0, 0.01],
            pos=(*pt, 0),
        ).z(0.01)
        ellipsoid_E.lighting("off").color("purple5")
        if stretches[0] < 0 or stretches[1] < 0:
            ellipsoid_E.c("red4")

        # Plot the deformation gradient tensor, we cannot compute the
        # principal stretches and directions of the deformation gradient
        # tensor because it is not a symmetric tensor.
        # F = deformation_gradient(*pt)
        # circle = vedo.Circle(r=0.05).pos(*pt).color("black")
        # cpts = circle.vertices
        # cpts_defo = F @ cpts.T[:2]
        # circle.vertices = cpts_defo.T
        # Same as:
        circle = vedo.Circle(r=0.06).pos(*pt).color("black")
        cpts = circle.vertices
        cpts_defo = deform(cpts[:,0], cpts[:,1])
        circle.vertices = cpts_defo.T

        plotter += [ellipsoid_C, ellipsoid_E, circle]

pts  = np.array([x, y]).T.reshape(-1, 2)
defo_pts = deform(x, y).T.reshape(-1, 2)

plotter += vedo.Arrows2D(pts, defo_pts, s=0.2).color("blue5")
plotter += grid_defo
plotter += __doc__
plotter.show(axes=8, zoom=1.2)

#####################################################################
# Resources:
# https://en.wikipedia.org/wiki/Deformation_gradient
# https://en.wikipedia.org/wiki/Almansi_strain_tensor
# https://en.wikipedia.org/wiki/Principal_stretch
# https://www.continuummechanics.org/deformationstrainintro.html