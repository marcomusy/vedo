"""A textured gyroid shape cut by a sphere"""
from vedo import *
import numpy as np

# Equation of a "gyroid" (https://en.wikipedia.org/wiki/Gyroid)
x, y, z = np.mgrid[:30,:30,:30] * 0.4
U = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)

# Create a Volume, take the isosurface at 0, smooth and subdivide it
gyr = Volume(U).isosurface(0).smooth().subdivide()

# Intersect it with a sphere made of quads
sph = Sphere(pos=(15,15,15), r=14, quads=True, res=30).triangulate()
printc("Please wait a few secs while I'm cutting your gyroid", c='y')
gxs = gyr.boolean('intersect', sph).clean().flat()
gxs.texture('https://vedo.embl.es/examples/data/images/marblings.jpg')

plt = show(gxs, __doc__, bg='wheat', bg2='lightblue', axes=5, zoom=1.4)
# Video('gyroid.mp4').action().close().interactive() # shoot video
plt.close()

