from dolfin import *
from mshr   import Circle, generate_mesh
from vtkplotter.dolfin import plot, printc
# Credits:
# https://github.com/pf4d/fenics_scripts/pi_estimate.py

domain = Circle(Point(0.0,0.0), 1.0)

for res in [2**k for k in range(8)]:
	mesh = generate_mesh(domain, res)
	A    = assemble(Constant(1) * dx(domain=mesh))
	printc("resolution = %i, \t |A - pi| = %.5e" % (res, abs(A-pi)))
printc('~pi is about', A, c='yellow')

plot(mesh, style=1, axes=3)
