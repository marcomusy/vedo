from __future__ import division, print_function
import plotter

vp = plotter.vtkPlotter(verbose=0, interactive=0)

l_rest = 0.1 # spring length at rest
x0 = 0.85 # initial x-coordinate of the block
k = 25    # spring constant
m = 20    # block mass
b = 0.1   # viscosity friction
dt= .1    # time step

#initial conditions
v  = vp.vector(0,0,0)
x  = vp.vector(x0,0,0)
xr = vp.vector(l_rest,0,0)

vp.box(pos=(0,-.1,0),   length=2,   width=.02, height=.5)  #surface
vp.box(pos=(-.82,.15,0), length=.04, width=.5,  height=.3) #wall
block  = vp.cube(pos=x, length=.2, c='r')
spring = vp.helix(pos=(-.8,0,0), axis=x, coils=20, radius=.08, lw=1, c='grey')

pb = vp.ProgressBar(0,500, c='r')
for i in pb.range(): 
    F = -k*(x-xr) - b*v              # Force
    a = F/m                          # acceleration
    v = v + a * dt                   # velocity
    x = x + v*dt + 1/2 * a * dt**2   # position
    
    block.pos(x)
    f = (x[0]-spring.pos()[0])/(x0-spring.pos()[0])
    spring.SetScale(f, 1, 1)
    
    vp.camera.Azimuth(.1)
    vp.camera.Elevation(.1)
    vp.render()
    pb.print('F='+str(F[0]))

vp.show(interactive=1)
 
    