"""The Lotka-Volterra model where:
x is the number of preys
y is the number of predators"""
#Credits:
#http://visual.icse.us.edu.pl/NPB/notebooks/Lotka_Volterra_with_SAGE.html
#as implemented in K3D_Animations/Lotka-Volterra.ipynb
#https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
import numpy as np
from scipy.integrate import odeint

def rhs(y0, t, a):
    x, y = y0[0], y0[1]
    return [x-x*y, a*(x*y-y)]

a_1 = 1.2
x0_1, x0_2, x0_3 = 2.0, 1.2, 1.0
y0_1, y0_2, y0_3 = 4.2, 3.7, 2.4

T = np.arange(0, 8, 0.02)
sol1 = odeint(rhs, [x0_1, y0_1], T, args=(a_1,))
sol2 = odeint(rhs, [x0_2, y0_2], T, args=(a_1,))
sol3 = odeint(rhs, [x0_3, y0_3], T, args=(a_1,))

limx = np.linspace(np.min(sol1[:,0]), np.max(sol1[:,0]), 20)
limy = np.linspace(np.min(sol1[:,1]), np.max(sol1[:,1]), 20)
vx, vy = np.meshgrid(limx, limy)
vx, vy = np.ravel(vx), np.ravel(vy)
vec = rhs([vx, vy], t=0.01, a=a_1)

origins = np.stack([np.zeros(np.shape(vx)), vx, vy]).T
vectors = np.stack([np.zeros(np.shape(vec[0])), vec[0], vec[1]]).T
vectors /= np.stack([np.linalg.norm(vectors, axis=1)]).T * 5

curve_points1 = np.vstack([np.zeros(sol1[:,0].shape), sol1[:,0], sol1[:,1]]).T
curve_points2 = np.vstack([np.zeros(sol2[:,0].shape), sol2[:,0], sol2[:,1]]).T
curve_points3 = np.vstack([np.zeros(sol3[:,0].shape), sol3[:,0], sol3[:,1]]).T

########################################################################
from vedo import Plotter, Arrows, Points, Line

plt = Plotter(bg="blackboard")
plt += Arrows(origins, origins+vectors, c='lr')

plt += Points(curve_points1, c='y')
plt += Line(curve_points1, c='y')
plt += Line(np.vstack([T, sol1[:,0], sol1[:,1]]).T, c='y')

plt += Points(curve_points2, c='g')
plt += Line(curve_points2, c='g')
plt += Line(np.vstack([T, sol2[:,0], sol2[:,1]]).T, c='g')

plt += Points(curve_points3, c='lb')
plt += Line(curve_points3, c='lb')
plt += Line(np.vstack([T, sol3[:,0], sol3[:,1]]).T, c='lb')

plt += __doc__

plt.show(axes={'xtitle':'time',
               'ytitle':'x',
               'ztitle':'y',
               'zxGrid':True,
               'yzGrid':False},
         viewup='x',
)
plt.close()
