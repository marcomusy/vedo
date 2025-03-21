"""This script shows how to fit a symbolic regression model to a 2D dataset.
The dataset is generated from a simple function and some noise is added.
The script then shows the true function and learned function on a 2D grid."""
# Check out the documentation for more information:
# https://astroautomata.com/PySR/
# https://github.com/MilesCranmer/PySR
# install with: 
#   pip install pysr
#
import numpy as np
from pysr import PySRRegressor
import vedo


model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (use julia syntax)
)

def compute_z(X):
    return 0.42345 * np.cos(X[:,1]) + 0.1 * X[:,0]**2 - 0.5

X = 2 * np.random.randn(100, 2)
z = compute_z(X)
# add noise to z values to make it more realistic
z += 0.15 * np.random.randn(100)

model.fit(X, z)
print(model)

grid = vedo.Grid(pos=(0,0,0), res=(100,100)).scale(10)
X_grid = grid.points[:, :2]  # 2D points on the grid
z_pred = model.predict(X_grid)
grid.points[:, 2] = z_pred

coords = np.c_[X[:,0], X[:,1], z]

# the truth
grid_truth = grid.clone().alpha(0.1).c("black")
z_truth = compute_z(X_grid)
grid_truth.points[:, 2] = z_truth

grid.compute_normals().cmap("ocean", z_pred)
grid.wireframe(False).lw(0).lighting("glossy")
levels = grid.isolines(n=10).color('white')

loss = model.equations_["loss"]
complexity = model.equations_["complexity"]

pts = vedo.Points(coords, r=8, c="k6")

pl = vedo.pyplot.plot(complexity, loss, xtitle="Complexity", ytitle="Loss", c="blue4")
pl = pl.clone2d("bottom-left", size=0.5)

vedo.show(pts, grid, grid_truth, levels, pl, __doc__, axes=1, viewup="z")
