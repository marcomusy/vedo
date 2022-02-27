"""A gyroscope sitting on a pedestal.
The analysis is in terms of Lagrangian mechanics.
The Lagrangian variables are polar angle theta,
azimuthal angle phi, and spin angle psi"""
# (adapted from http://www.glowscript.org)
from vedo import *

# ############################################################ parameters
dt = 5e-05  # time step
Lshaft = 1  # length of gyroscope shaft
M = 1  # mass of gyroscope (massless shaft)
R = 0.4  # radius of gyroscope rotor
theta = 1.3  # initial polar angle of shaft (from vertical)
psidot = -40  # spinning angular velocity (rad/s)
phidot = 0  # (try -1 and +1 to get first and second pattern)

# ############################################################
g, r = 9.81, Lshaft / 2
I3 = 1 / 2 * M * R ** 2  # moment of inertia, I, of gyroscope about its own axis
I1 = M * r ** 2 + 1 / 2 * I3  # I about a line through the support, perpendicular to axis
phi = psi = thetadot = 0
x = vector(theta, phi, psi)  # Lagrangian coordinates
v = vector(thetadot, phidot, psidot)

# ############################################################ the scene
settings.allowInteraction = True
plt = Plotter(interactive=False)
plt += __doc__

shaft = Cylinder([[0, 0, 0], [Lshaft, 0, 0]], r=0.03, c="dg")
rotor = Cylinder([[Lshaft / 2.2, 0, 0], [Lshaft / 1.8, 0, 0]], r=R).texture(dataurl+'textures/white.jpg')
base  = Sphere([0, 0, 0], c="dg", r=0.03)
tip   = Sphere([Lshaft, 0, 0], c="dg", r=0.03)
gyro = shaft + rotor + base + tip  # group relevant meshes into single one of type Assembly
plt += gyro  # add it to Plotter list

pedestal = Box([0, -0.63, 0], height=0.1, length=0.1, width=1).texture(dataurl+'textures/wood1.jpg')
pedbase = Box([0, -1.13, 0], height=0.5, length=0.5, width=0.05).texture(dataurl+'textures/wood1.jpg')
pedpin = Pyramid([0, -0.08, 0], axis=[0, 1, 0], s=0.05, height=0.12).texture(dataurl+'textures/wood1.jpg')
formulas = Picture(dataurl+"images/gyro_formulas.png").alpha(0.9)
formulas.scale(0.0035).pos(-1.4, -1.1, -1.1)
plt += [pedestal + pedbase + pedpin + formulas]

# ############################################################ the physics
pb = ProgressBar(0, 4, dt, c="b")
for i, t in enumerate(pb.range()):
    st, ct, sp, cp = sin(x[0]), cos(x[0]), sin(x[1]), cos(x[1])

    thetadot, phidot, psidot = v  # unpack
    atheta = (
        st * ct * phidot ** 2 + (M * g * r * st - I3 * (psidot + phidot * ct) * phidot * st) / I1
    )
    aphi = (I3 / I1) * (psidot + phidot * ct) * thetadot / st - 2 * ct * thetadot * phidot / st
    apsi = phidot * thetadot * st - aphi * ct
    a = vector(atheta, aphi, apsi)

    v += a * dt  # update velocities
    x += v * dt  # update Lagrangian coordinates

    gaxis = (Lshaft + 0.03) * vector(st * sp, ct, st * cp)
    # set orientation along gaxis and rotate it around its axis by psidot*t degrees
    gyro.orientation(gaxis, rotation=psidot * t, rad=True)
    if not i % 200:  # add trace and render all, every 200 iterations
        plt += Point(gaxis, r=3, c="r")
        plt.show()
        if plt.escaped: break # if ESC is hit during the loop
    pb.print()

plt.interactive().close()
