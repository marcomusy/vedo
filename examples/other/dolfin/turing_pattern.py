from dolfin import *
from numpy.random import random
set_log_level(30)


class TuringPattern(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x): assemble(self.L, tensor=b)
    def J(self, A, x): assemble(self.a, tensor=A)

mesh = UnitSquareMesh(48, 48)

U = FiniteElement("CG", mesh.ufl_cell(), 2)

W = FunctionSpace(mesh, U * U)

du   = TrialFunction(W)
q, p = TestFunctions(W)

w = Function(W)
w0 =  Function(W)

# Split mixed functions
dact, dhib = split(du)
act, hib = split(w)
act0, hib0 = split(w0)

dt = 0.04
T = 20.0

class IC(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 1.0*random() +0.25
        values[1] = 1.0*random() +0.25
    def value_shape(self): return (2,)

w_init = IC(element=W.ufl_element(), degree=2)
w.interpolate(w_init)
w0.interpolate(w_init)

L0 = act*q - act0*q \
    + dt*0.0005*inner(grad(act), grad(q)) \
    - dt*inner(act*act*hib,q) \
    + 1.0*dt*inner(act,q)
L1 = hib*p -hib0*p \
    + dt*0.1*inner(grad(hib), grad(p)) \
    + dt*inner(act*act*hib, p) \
    - dt*inner(Constant(1.0),p)
L  = (L0 + L1) *dx

# Compute directional derivative about u in the direction of du
a = derivative(L, w, du)

problem = TuringPattern(a, L)
solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-2


########################################### time steps
from vedo.dolfin import plot, printc

t = 0
printc('\bomb Press F1 to abort.', c='y', invert=True)
while t < T:
    t += dt
    w0.vector()[:] = w.vector()
    solver.solve(problem, w.vector())

    plot(w.split()[0],
         style=4,
         lw=0,
         scalarbar='h',
         interactive=False,
    )
    printc(f'time: {t}')

plot()
