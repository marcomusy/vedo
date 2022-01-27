"""Simulation of a gyroscope hanging from a spring"""
# (adapted by M. Musy from Bruce Sherwood, 2009)
from vedo import *

# ############################################################ parameters
dt = 0.005  # time step
ks = 15  # spring stiffness
Lrest = 1  # unstretched length of spring
Ls = 1  # length of gyroscope shaft
M = 1  # mass of gyroscope (massless shaft)
R = 0.4  # radius of gyroscope rotor
omega = 50  # angular velocity of rotor (rad/s, not shown)
gpos = vector(0, 0, 0)  # initial position of spring free end

# ############################################################ inits
top = vector(0, 2, 0)  # where top of spring is held
precess = vector(0, 0, 0)  # initial momentum of center of mass
Fgrav = vector(0, -M * 9.81, 0)
gaxis = vector(0, 0, 1)  # initial orientation of gyroscope
gaxis = versor(gaxis)
I = 1 / 2 * M * R ** 2  # moment of inertia of gyroscope
Lrot = I * omega * gaxis  # angular momentum
cm = gpos + 0.5 * Ls * gaxis  # center of mass of shaft

# ############################################################ the scene
settings.allowInteraction = True
plt = Plotter(axes=0, interactive=0)
plt += __doc__

shaft = Cylinder([[0, 0, 0], Ls * gaxis], r=0.03, c="dg")
rotor = Cylinder([(Ls - 0.55) * gaxis, (Ls - 0.45) * gaxis], r=R, c="t")
bar   = Cylinder([Ls*gaxis/2-R*vector(0,1,0), Ls*gaxis/2+R*vector(0,1,0)], r=R/6, c="r")
gyro = shaft + rotor + bar  # group meshes into a single one of type Assembly

spring = Spring(top, gpos, r=0.06, thickness=0.01, c="gray")
plt += [gyro, spring] # add it to Plotter.
plt += Box(top, length=0.2, width=0.02, height=0.2, c="gray")
plt += Box(pos=(0, 0.5, 0), length=2.6, width=3, height=2.6, c="gray", alpha=0.2).wireframe()

# ############################################################ the physics
pb = ProgressBar(0, 5, dt, c="b")
for t in pb.range():
    Fspring = -ks * versor(gpos - top) * (mag(gpos - top) - Lrest)
    torque = cross(-1 / 2 * Ls * versor(Lrot), Fspring)  # torque about center of mass
    Lrot += torque * dt
    precess += (Fgrav + Fspring) * dt  # momentum of center of mass
    cm += (precess / M) * dt
    gpos = cm - 1 / 2 * Ls * versor(Lrot)

    # set orientation along gaxis and rotate it around its axis by omega*t degrees
    gyro.orientation(Lrot, rotation=omega * t, rad=True).pos(gpos)
    spring.stretch(top, gpos)
    plt.show()
    if plt.escaped: break # if ESC is hit during the loop
    pb.print()

plt.interactive().close()
