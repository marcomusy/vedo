from dolfin import *
import sympy as sp
# Credits:
# https://github.com/pf4d/fenics_scripts/calc_surface_area.py

x, y = sp.symbols('x, y')

# surface :
def s(x,y):        return sp.exp(x)

# x-derivative of surface
def dsdx(x,y):     return s(x,y).diff(x, 1)

# y-derivative of surface
def dsdy(x,y):     return s(x,y).diff(y, 1)

# outward-pointing-normal-vector magnitude at surface :
def n_mag_s(x,y):  return sp.sqrt(1 + dsdx(x,y)**2 + dsdy(x,y)**2)

# surface area of surface :
def area(x,y):     return sp.integrate(n_mag_s(x,y), (x,0,1), (y,0,1))

A_exact = area(x,y)

for n in [5,10,100,500]:
	mesh  = UnitSquareMesh(n,n)
	Q     = FunctionSpace(mesh, "CG", 1)
	e     = Expression('exp(x[0])', degree=2)
	f     = interpolate(e, Q)
	A_num = assemble( sqrt(f.dx(0)**2 + f.dx(1)**2 + 1) *  dx)
	print('for n = %i -- error = %.2e' % (n, abs(A_exact.evalf()-A_num)))

n        = 10
mesh     = UnitSquareMesh(n,n)
Q        = FunctionSpace(mesh, "CG", 1)
e        = Expression('exp(x[0])', degree=2)
f        = interpolate(e, Q)
A_vector = project( sqrt(f.dx(0)**2 + f.dx(1)**2 + 1), Q)


from vedo.dolfin import plot
plot(A_vector)

