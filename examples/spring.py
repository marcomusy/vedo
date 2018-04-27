from __future__ import division, print_function
import plotter

vp = plotter.vtkPlotter(title='Spring in viscous medium', verbose=0)

l_rest = 0.1 # spring x position at rest
x0 = 0.85 # initial x-coordinate of the block
k = 25    # spring constant
m = 20    # block mass
b = 0.1   # viscosity friction (prop. to velocity)
dt= 0.1   # time step

#initial conditions
v  = vp.vector(0, 0, 0.2)
x  = vp.vector(x0, 0, 0)
xr = vp.vector(l_rest, 0, 0)

vp.box(pos=(0,-0.1,0),   length=2.0, width=0.02, height=0.5)  #surface
vp.box(pos=(-.82,.15,0), length=.04, width=0.50, height=0.3)  #wall
block  = vp.cube(pos=x, length=.2, c='t')
spring = vp.helix([-0.8,0,0], x, radius=0.06, thickness=.01, 
                  coils=25, texture='metal1')

pb = vp.ProgressBar(0,500, c='r')
for i in pb.range(): 
    F = -k*(x-xr) - b*v              # Force
    a = F/m                          # acceleration
    v = v + a * dt                   # velocity
    x = x + v*dt + 1/2 * a * dt**2   # position
    
    block.pos(x)                     # update block position
    spring.stretch([-0.8,0,0], x)    # stretch helix accordingly
    trace = vp.point(x+[0,.3,0], c='r', r=3, alpha=.5) # leave a trace
    
    vp.camera.Azimuth(.1)
    vp.camera.Elevation(.1)
    vp.render(addActor=trace)
    pb.print('Fx='+str(F[0]))

vp.show(interactive=1)
 
    