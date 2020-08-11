"""
Extract submesh boundaries.
"""
# https://fenicsproject.discourse.group/t/output-parts-of-boundary/537
from fenics import *
from mshr import *
from numpy import array
from numpy.linalg import norm

domain = Box(Point(0,0,0), Point(10,10,10)) - Sphere(Point(5,5,5), 3)
mesh = generate_mesh(domain, 32)
exterior = BoundaryMesh(mesh, "exterior")


def inSphere(x):
    v = x - array([5, 5, 5])
    return norm(v) < 3 + 1e2 * DOLFIN_EPS

class SphereDomain(SubDomain):
    def inside(self, x, on_boundary):
        return inSphere(x)

class BoxDomain(SubDomain):
    def inside(self, x, on_boundary):
        return not inSphere(x)

sph = SubMesh(exterior, SphereDomain())
box = SubMesh(exterior, BoxDomain())


from vedo.dolfin import plot

plot(sph, at=0, N=2, c='red', text=__doc__)
plot(box, at=1, wireframe=True)
