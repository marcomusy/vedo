"""Time integration of the wave equation
with the Crank-Nicolson method.
"""
#Credits:
#https://fenicsproject.discourse.group/t/
# aritymismatch-for-crank-nicolson-method-on-mixed-function-space
#https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method
from fenics import *
set_log_level(30)

T = 2
c = 1
nt = 500
nx = 200
dt = T / nt

mesh = UnitIntervalMesh(nx)
V1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V2 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement([V1, V2])
V = FunctionSpace(mesh, element)

def boundary(x, on_boundary): return on_boundary
bc1 = DirichletBC(V.sub(0), Constant(0), boundary)
bc2 = DirichletBC(V.sub(1), Constant(0), boundary)

class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        super(InitialConditions, self).__init__(**kwargs)
    def eval(self, values, val):
        x = val[0]
        values[0] = x*(1-x) * exp(-(x-0.4)**2/0.01)
        values[1] = 0.0
    def value_shape(self):
        return (2,)

u0 = InitialConditions(degree=2)
un = Function(V)
un.interpolate(u0)
un1, un2 = split(un)

u1, u2 = TrialFunctions(V)
v1, v2 = TestFunctions(V)
a1 =  u1 * v1 - Constant(0.5*dt) * u2 * v1
L1 = un1 * v1 + Constant(0.5*dt) * un2 * v1
a2 =  u2 * v2 + Constant(0.5*c**2*dt) * inner(grad(u1),  grad(v2))
L2 = un2 * v2 - Constant(0.5*c**2*dt) * inner(grad(un1), grad(v2))
a = (a1 + a2) * dx
L = (L1 + L2) * dx
uh = Function(V)

############################################################
from vedo.dolfin import plot
from vedo import Grid

#build a thin gray frame to avoid camera jumping around
frame = Grid(pos=[0.5, 0, 0]).c('gray').alpha(0.1)

for i in range(nt):

    solve(a==L, uh, [bc1, bc2])
    uk1, uk2 = uh.split()
    un.assign(uh)
#    
#    plot(uk1, warpYfactor=.0)
#    exit()

    if not i%4:
        plot(uk1, frame,
             at=0, shape=(2,1), # plot on first of 2 renderers
             warpYfactor=2.0,   # warp y-axis with solution uk1
             lw=3,              # line width
             lc='white',        # line color
             ytitle="diplacement at  T=%g" % (i*dt),
             scalarbar=False,
             bg='bb',
             size=(500,1000),
             interactive=False,
             )
        plot(uk2, frame,
             at=1,              # plot on second of 2 renderers
             warpYfactor=0.2,
             lw=3,
             lc='tomato',
             ytitle="velocity [a.u.]",
             scalarbar=False,
             bg='bb',
             interactive=False,
             )

plot() # enter interactive mode
