# Gyroscope hanging from a spring
# (adapted by M. Musy from Bruce Sherwood, 2009)
from __future__ import division, print_function
from vtkplotter import Plotter, ProgressBar, vector, mag, norm, cross

# ############################################################ parameters
dt = 0.005      # time step
ks = 15         # spring stiffness
Lrest  = 1      # unstretched length of spring
Ls = 1          # length of gyroscope shaft
M = 1           # mass of gyroscope (massless shaft)
R = 0.4         # radius of gyroscope rotor
omega = 50      # angular velocity of rotor (rad/s, not shown)
gpos = vector(0, 0, 0) # initial position of spring free end

# ############################################################ inits
top   = vector(0, 2, 0)   # where top of spring is held
precess = vector(0, 0, 0) # initial momentum of center of mass
Fgrav = vector(0, -M*9.81, 0)
gaxis = vector(0, 0, 1)   # initial orientation of gyroscope
gaxis = norm(gaxis)
I = 1/2*M*R**2            # moment of inertia of gyroscope
Lrot = I*omega*gaxis      # angular momentum
cm = gpos + 0.5*Ls*gaxis  # center of mass of shaft

# ############################################################ the scene
vp = Plotter(verbose=0, axes=3, interactive=0)

shaft = vp.cylinder([[0,0,0], Ls*gaxis], r=0.03, c='dg')
rotor = vp.cylinder([(Ls-0.55)*gaxis, (Ls-0.45)*gaxis], r=R, c='t')
bar   = vp.cylinder([Ls*gaxis/2-R*vector(0,1,0), Ls*gaxis/2+R*vector(0,1,0)], r=R/6, c='r')
gyro  = vp.Assembly([shaft, rotor, bar]) # group actors into a single one

spring= vp.helix(top, gpos, r=0.06, thickness=0.01, c='gray')
box   = vp.box(top, length=0.2, width=0.02, height=0.2, c='gray')

# ############################################################ the physics
pb = ProgressBar(0, 5, dt, c='b')
for t in pb.range():
    Fspring = -ks*norm(gpos-top)*(mag(gpos-top)-Lrest)
    torque  = cross(-1/2*Ls*norm(Lrot), Fspring) # torque about center of mass
    Lrot    += torque*dt
    precess += (Fgrav+Fspring)*dt  # momentum of center of mass
    cm      += (precess/M)*dt
    gpos    = cm - 1/2*Ls*norm(Lrot)

    # set orientation along gaxis and rotate it around its axis by omega*t degrees
    gyro.orientation(Lrot, rotation=omega*t*57.3).pos(gpos)
    spring.stretch(top, gpos)
    vp.point(gpos + Ls*norm(Lrot), r=1, c='g') # add trace point to show in the end
    vp.render() 
    pb.print()

vp.show(interactive=1)


