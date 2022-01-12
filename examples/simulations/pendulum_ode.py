"""Double pendulum with ODE integration"""
# Copyright (c) 2018, N. Rougier, https://github.com/rougier/pendulum
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c
# Adapted for vedo by M. Musy, 2021
import numpy as np
import scipy.integrate as integrate
from vedo import Axes, Line, Points, show, sin, cos, ProgressBar, settings

G  = 9.81   # acceleration due to gravity, in m/s^2
L1 = 1.0    # length of pendulum 1 in m
L2 = 1.0    # length of pendulum 2 in m
M1 = 1.0    # mass of pendulum 1 in kg
M2 = 1.0    # mass of pendulum 2 in kg
th1= 120    # initial angles (degrees)
th2= -20
w1 = 0      # initial angular velocities (degrees per second)
w2 = 0
dt = 0.015

settings.allowInteraction = True

def derivs(state, t):
    dydx = np.zeros_like(state)
    dydx[0] = state[1]
    a = state[2] - state[0]
    sina, cosa = sin(a), cos(a)
    den1 = (M1 + M2)*L1 - M2*L1*cosa*cosa
    dydx[1] = (M2*L1*state[1]*state[1]*sina*cosa +
               M2*G*sin(state[2])*cosa +
               M2*L2*state[3]*state[3]*sina -
               (M1+M2)*G*sin(state[0]) )/den1
    dydx[2] = state[3]
    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sina*cosa +
               (M1+M2)*G*sin(state[0])*cosa -
               (M1+M2)*L1*state[1]*state[1]*sina -
               (M1+M2)*G*sin(state[2]) )/den2
    return dydx

t = np.arange(0.0, 10.0, dt)
state = np.radians([th1, w1, th2, w2])
y = integrate.odeint(derivs, state, t)

P1 = np.dstack([L1*sin(y[:,0]), -L1*cos(y[:,0])]).squeeze()
P2 = P1 + np.dstack([L2*sin(y[:,2]), -L2*cos(y[:,2])]).squeeze()

ax = Axes(xrange=(-2,2), yrange=(-2,1), htitle=__doc__)
pb = ProgressBar(0, len(t), c="b")
for i in pb.range():
    j = max(i- 5,0)
    k = max(i-10,0)
    l1 = Line([[0,0], P1[i], P2[i]]).lw(7).c("blue2")
    l2 = Line([[0,0], P1[j], P2[j]]).lw(6).c("blue2", 0.3)
    l3 = Line([[0,0], P1[k], P2[k]]).lw(5).c("blue2", 0.1)
    pt = Points([P1[i], P2[i], P1[j], P2[j], P1[k], P2[k]], r=8).c("blue2", 0.2)
    show(l1, l2, l3, pt, ax, interactive=False, size=(900,700), zoom=1.4)
    pb.print()
