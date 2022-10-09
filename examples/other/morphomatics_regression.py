"""Geodesic regression for data in SO(3)
with the Morphometrics library"""
try:
    from morphomatics.manifold import SO3
    from morphomatics.stats import RiemannianRegression
except ModuleNotFoundError:
    print("Install with:")
    print("pip install git+https://github.com/morphomatics/morphomatics.git#egg=morphomatics")
import numpy as np
import vedo

# z-axis is axis of rotation
I = np.eye(3)
R = np.array(
    [[np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
    [np.sin( np.pi / 6),  np.cos(np.pi / 6), 0],
    [0, 0, 1]],
)

# 6 points in SO(3). The extra dimension is not needed here but comes
# into play when the data consists of tuples of matrices.
M = SO3()
Y = np.zeros((6,) + tuple(M.point_shape))  # -> (6,1,3,3)
eval_perturbed = lambda t, vec: M.connec.exp(M.connec.geopoint(I,R,t), vec)
Y[0, 0] = eval_perturbed(-2 / 3, np.array([[0, 0, 0.1], [0, 0, 0.0], [-0.1, 0.0, 0]]))
Y[1, 0] = eval_perturbed(-1 / 3, np.array([[0, 0, 0.0], [0, 0, 0.2], [ 0.0,-0.2, 0]]))
Y[2, 0] = I
Y[3, 0] = eval_perturbed( 1 / 3, np.array([[0, 0, 0.0], [0, 0, 0.2], [ 0.0,-0.2, 0]]))
Y[4, 0] = eval_perturbed( 2 / 3, np.array([[0, 0, 0.1], [0, 0, 0.0], [-0.1, 0.0, 0]]))
Y[5, 0] = R

# corresponding time points
t = np.array([0, 1/5, 2/5, 3/5, 4/5, 1])

# geodesic has degree 1
degrees = np.array([1])
# cubic regression
# degrees = np.array([3])

# solve
regression = RiemannianRegression(M, Y, t, degrees)

# geodesic least-squares estimator
gam = regression.trend

# evaluate geodesic at 100 equidistant points
X = gam.eval()

# rotate [1,0,0] by rotations in X, i.e. take first column of X
x = X[...,0].squeeze()
time = np.linspace(0,1, x.shape[0])
pts = [y[..., 0][0] for y in Y]

# visualize
pts = vedo.Points(pts, r=15)
line = vedo.Line(x).lw(10).cmap("jet", time)
line.aadd_scalarbar("time")
sphere = vedo.Sphere(c='white').flat()
vedo.show(sphere, line, pts, __doc__, axes=1, viewup='z').close()

