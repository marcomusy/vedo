"""Nelder-Mead minimization algorithm for the 4D function:
F = (x/4-1)**2 + (y+2)**2 + z**2 + (w-6)**2 + 3"""
from vedo import Minimizer, Line, show
from vedo.pyplot import plot

def func(pars):
    x, y, z, w = pars  # unpack parameters for convenience
    F = (x/4-1)**2 + (y+2)**2 + z**2 + (w-6)**2 + 3
    return F

mini = Minimizer(func)
mini.set_parameter("x", 4.0) # set initial values
mini.set_parameter("y", -3.0)
mini.set_parameter("z", 1.0)
mini.set_parameter("w", 1.0)
res = mini.minimize()  # run the minimization
mini.compute_hessian() # compute the Hessian to estimate the errors
print(mini)

# Draw the path of the minimization
path = res["parameters_path"]
vals = res["function_path"]
line = Line(path[:,:3], lw=5).cmap("jet", path[:,3]).add_scalarbar()
plo = plot(vals, xtitle="iteration", ytitle="function eval", lw=3)
show(line, plo.clone2d(), __doc__, axes=1)

