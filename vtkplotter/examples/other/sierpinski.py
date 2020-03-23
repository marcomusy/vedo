"""
Sierpinski3d fractal
"""
# Credits: K3D authors at
# https://github.com/K3D-tools/K3D-jupyter/tree/master/examples
#https://simple.wikipedia.org/wiki/Sierpinski_triangle
import numpy as np

N = 30000
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

x1 = np.empty_like(x)
y1 = np.empty_like(y)
z1 = np.empty_like(z)

# Sierpinski triangle iterative functions
def f1(x, y, z, x1, y1, z1, c):
    x1[c] = 0.5*x[c]
    y1[c] = 0.5*y[c]
    z1[c] = 0.5*z[c]

def f2(x, y, z, x1, y1, z1, c):
    x1[c] = 0.5*x[c] + 1 / 2.0
    y1[c] = 0.5*y[c]
    z1[c] = 0.5*z[c]

def f3(x, y, z, x1, y1, z1, c):
    x1[c] = 0.5*x[c] + 1 / 4.0
    y1[c] = 0.5*y[c] + np.sqrt(3) / 4
    z1[c] = 0.5*z[c]

def f4(x, y, z, x1, y1, z1, c):
    x1[c] = 0.5*x[c] + 1 / 4.0
    y1[c] = 0.5*y[c] + 1.0 / 4
    z1[c] = 0.5*z[c] + np.sqrt(3) / 4


functions = [f1, f2, f3, f4]
probabilities = [1 / 4.0] * 4
assert len(functions) == len(probabilities)

X, Y, Z = x, y, z
for i in range(20):
    # pick indices for each function to be applied
    r = np.random.choice(len(probabilities), size=N, p=probabilities)
    for i, f in enumerate(functions):
        f(x, y, z, x1, y1, z1, r == i)
    x, x1 = x1, x
    y, y1 = y1, y
    z, z1 = z1, z
    if i > 0:
        X, Y, Z = np.hstack([X,x]), np.hstack([Y,y]), np.hstack([Z,z])

# how much memory are we using, how many points there are
print("used mem, Npts=", 3 * X.nbytes // 1024 ** 2, "MB", X.shape[0])

from vtkplotter import Points

Points([X, Y, Z], c="tomato").show(axes=1)
