# Gyroscope hanging from a spring
# (adapetd from Bruce Sherwood)
from __future__ import division, print_function
from plotter import vtkPlotter, ProgressBar, vector, mag, norm
from numpy import cross

# ############################################################
dt = 0.005      # time step
ks = 15         # spring stiffness
Lspring = 1     # unstretched length of spring
Lshaft = 1      # length of gyroscope shaft
M = 1           # mass of gyroscope (massless shaft)
R = 0.4         # radius of gyroscope rotor
omega = 50      # angular velocity of rotor (rad/s, not shown)
gpos  = vector(-1, 0, 0) # initial position of spring free end
gaxis = vector(-1, 0, 0) # initial orientation of gyroscope

# ############################################################
top = vector(0, 0, 0)     # where top of spring is held
precess = vector(0, 0, 0) # initial momentum of center of mass
I = 1/2*M*R**2            # moment of inertia of gyroscope
Fgrav = vector(0, -M*9.81, 0)
gaxis = norm(gaxis)
cm = gpos + 0.5*Lshaft*gaxis # center of mass of shaft
Lrot = I*omega*gaxis      # angular momentum

# ############################################################
vp = vtkPlotter(verbose=0, axes=3, interactive=0, size=[300,300])
shaft = vp.cylinder([[0, 0, 0], Lshaft*gaxis], radius=.03, c='dg')
rotor = vp.cylinder([Lshaft*gaxis/1.8, Lshaft*gaxis/2.2], radius=R, c='t', edges=1)
spring= vp.helix(top, gpos, coils=20, radius=.06, thickness=.01, c='gray')
box   = vp.box(top, length=.2, width=.02, height=.2, c='gray')
gyro  = vp.makeAssembly([shaft, rotor]).pos(gpos) # group relevant actors
vp.actors = [] # clean up the list of actors to avoid duplications
vp.show([gyro, spring, box])

pb = ProgressBar(0, 5, dt, c='b')
for t in pb.range():
    Fspring = -ks*norm(gpos-top)*(mag(gpos-top)-Lspring)
    torque  = cross(-0.5*Lshaft*norm(Lrot), Fspring) # torque about center of mass
    Lrot    = Lrot + torque*dt
    precess = precess + (Fgrav+Fspring)*dt  # momentum of center of mass
    cm      = cm + (precess/M)*dt
    gpos    = cm - 0.5*Lshaft*norm(Lrot)
    gyro.orientation(gaxis, Lrot).pos(gpos)
    spring.stretch(top, gpos)
    trace = vp.point(gpos + Lshaft*norm(Lrot), r=1, c='g')
    vp.render(trace, resetcam=1) # add trace and render all
    pb.print()

vp.show(interactive=1)


