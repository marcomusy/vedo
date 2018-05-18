from __future__ import division, print_function
from plotter import vtkPlotter, ProgressBar

vp = vtkPlotter(title='Spring in viscous medium', verbose=0, axes=3)

l_rest = 0.1 # spring x position at rest
x0 = 0.85 # initial x-coordinate of the block
k = 25    # spring constant
m = 20    # block mass
b = 0.5   # viscosity friction (proportional to velocity)
dt= 0.1   # time step

#initial conditions
v  = vp.vector(0, 0, 0.2)
x  = vp.vector(x0, 0, 0)
xr = vp.vector(l_rest, 0, 0)
sx0 = vp.vector(-0.8, 0, 0)
offx= vp.vector(0, 0.3, 0)

vp.box(pos=(0, -0.1, 0), length=2.0, width=0.02, height=0.5)  #surface
vp.box(pos=(-.82,.15,0), length=.04, width=0.50, height=0.3)  #wall
block = vp.cube(pos=x, length=0.2, c='t')
spring= vp.helix(sx0, x, r=.06, thickness=.01, texture='metal1')

pb = ProgressBar(0,500, c='r')
for i in pb.range(): 
    F = -k*(x-xr) - b*v             # Force and friction
    a = F/m                         # acceleration
    v = v + a * dt                  # velocity
    x = x + v*dt + 1/2 * a * dt**2  # position
    
    block.pos(x)                    # update block position
    spring.stretch(sx0, x)          # stretch helix accordingly
    trace = vp.point(x+offx, c='r/0.5', r=3) # leave a red trace
    
    vp.camera.Azimuth(.1)
    vp.camera.Elevation(.1)
    vp.render(trace) # add trace to the list of actors and render
    pb.print('Fx='+str(F[0]))

vp.show(interactive=1)
 
    
