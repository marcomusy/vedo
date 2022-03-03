"""Manually define the number, shape and position
of the renderers inside the rendering window"""
from vedo import *

settings.immediateRendering = False  # faster for multi-renderers

# (0,0) is the bottom-left corner of the window, (1,1) the top-right
# the order in the list defines the priority when overlapping
custom_shape = [
        dict(bottomleft=(0.0,0.0), topright=(1.00,1.00), bg='wheat', bg2='w' ),# ren0
        dict(bottomleft=(0.0,0.0), topright=(0.40,0.30), bg='blue3', bg2='lb'),# ren1
        dict(bottomleft=(0.5,0.4), topright=(0.95,0.95), bg='green', bg2='lg'),# ren2
        dict(bottomleft=(0.7,0.2), topright=(0.90,0.50), bg='red', bg2='pink'),# ren3
        dict(bottomleft=(0.1,0.6), topright=(0.30,0.80), bg='violet', bg2='w'),# ren4
]

plt = Plotter(shape=custom_shape, size=(1200,900))

for i in range(len(custom_shape)):
    s = ParametricShape(i).color(i).lighting('glossy')
    msg = 'Renderer nr.'+str(i)+'\n'+str(custom_shape[i])+'\nShape = '+s.name
    plt.at(i).show(s, msg)

plt.at(0).add(Text2D(__doc__, pos='bottom-right', font="Quikhand", s=1.5))
plt.interactive().close()
