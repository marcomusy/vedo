from dolfin import *
from mshr   import Circle, generate_mesh
from vedo.dolfin import plot, printc
from vedo import Latex
# Credits:
# https://github.com/pf4d/fenics_scripts/blob/master/pi_estimate.py

domain = Circle(Point(0.0,0.0), 1.0)


for res in [2**k for k in range(7)]:
	mesh = generate_mesh(domain, res)
	A    = assemble(Constant(1) * dx(domain=mesh))
	printc("resolution = %i, \t A-pi = %.5e" % (res, A-pi))

printc('~pi is about', A, c='yellow')

l = Latex(r'\mathrm{Area}(r)=\pi=\int\int_D 1 \cdot d(x,y)', s=0.3)
l.crop(0.3,0.3).z(0.1) # crop invisible top and bottom and set at z=0.1

plot(mesh, l, alpha=0.4, ztitle='', style=1, axes=3)
