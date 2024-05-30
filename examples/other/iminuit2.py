"""Fit a 3D polynomial surface to a set of noisy data using iminuit.
You can rotate the scene by dragging with the left mouse button."""
from iminuit import Minuit
from vedo import Points, Arrows2D, show, dataurl, printc, settings


def func(x, y, *pars):
    # the actual surface model to fit
    a, b, c, u, v = pars
    z = c - ((x + u) * a)**2 - (y + v) * b
    return z

def cost_fcn(pars):
    cost = 0.0
    for p in pts:
        x, y, z = p
        f = func(x, y, *pars)
        cost += (f - z)**2  # compute the chi-square
    return cost / pts.size

# Load a set of points from a file and fit a surface to them
points = Points(dataurl + "data_points.vtk").ps(6).color("k3")
pts = points.coordinates

# Run the fit (minimize the cost_fcn) and print the result
m = Minuit(cost_fcn, [0.01, 0.05, 200, 550, 400]) # init values
m.errordef = m.LEAST_SQUARES
m.migrad()  # migrad is a sophisticated gradient descent algorithm
printc(m, c="green7", italic=True)

# Create a set of points that lie on the fitted surface
fit_pts = []
for x, y, _ in pts:
    fit_pts.append([x, y, func(x, y, *m.values)])
fit_pts = Points(fit_pts).ps(9).color("r5")

lines = Arrows2D(pts, fit_pts, rotation=90).color("k5", 0.25)

# Show the result of the fit with the original points in grey
settings.use_parallel_projection = True
show(points, fit_pts, lines, __doc__,
     axes=1, size=(1250, 750), bg2="lavender", zoom=1.4).close()

