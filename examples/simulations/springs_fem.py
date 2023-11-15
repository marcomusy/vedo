"""Solving a system of springs using the finite element method."""
# https://www.youtube.com/watch?v=YqpIEDWJCwc
import numpy as np
from vedo import *
# np.random.seed(0)

num_springs = 7
k = 1.0  # Stiffness of the springs

# Define applied forces at each node
num_nodes = num_springs + 1  # One more node than springs
F = np.random.randn(num_nodes) /5

# Discretize the system
nodes = np.arange(num_nodes)
elements = list(zip(nodes[:-1], nodes[1:]))

# Assemble global stiffness matrix and force vector
K = np.zeros((num_nodes, num_nodes))
for element in elements:
    i, j = element
    K[i, i] += k
    K[j, j] += k
    K[i, j] -= k
    K[j, i] -= k

# Apply boundary conditions (fixed nodes at both ends)
fixed_nodes = [0, num_nodes - 1]
for node in fixed_nodes:
    K[node, :] = 0
    K[:, node] = 0
    K[node, node] = 1
    F[node] = 0

# Solve for displacements
u = np.linalg.solve(K, F)

yvals = np.zeros(num_nodes)
nodes = np.c_[nodes, yvals]
u = np.c_[u, yvals]
F = np.c_[F, yvals]

nodes_displaced = nodes + u

# Visualize the solution
vnodes1 = Points(nodes).color("k", 0.25).ps(20)
vline1  = Line(nodes).color("k", 0.25)

arr_disp = Arrows2D(nodes, nodes_displaced).y(0.4)
arr_force= Arrows2D(nodes, nodes + F).y(-0.25)
arr_disp.c("red4",0.8).legend('Displacements')
arr_force.c("blue4",0.8).legend('Forces')

vnodes2 = Points(nodes_displaced).color("k").ps(20).y(0.1)
vline2  = Lines(vnodes1, vnodes2).color("k", 0.25)

springs = []
for i in range(num_springs):
    s = Spring(nodes_displaced[i], nodes_displaced[i+1], r1=0.04).y(0.1)
    s.lighting("metallic")
    springs.append(s)

lbox = LegendBox([arr_disp, arr_force], width=0.2, height=0.25, markers='s')
lbox.font("Calco")

show(
    __doc__, lbox, 
    vnodes1, vnodes2, vline1, vline2, arr_disp, arr_force, springs,
    axes=8, size=(1900, 490), zoom=3.6,
).close()


