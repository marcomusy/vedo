from vedo import *
import numpy as np

x, y, z = np.mgrid[:30,:30,:30] * 0.4
U = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)

# Create a Volume, take the isosurface at 0, smooth and subdivide it
gyr = Volume(U).isosurface(0).smoothLaplacian().subdivide()

# Intersect it with a sphere made of quads
sph = Sphere(pos=(15,15,15), r=14, quads=True, res=30).triangulate()
printc("Please wait a few secs while I'm cutting your gyroid", c='y')
gxs = gyr.boolean('intersect', sph).clean().computeNormals()
gxs.texture('https://www.dropbox.com/s/d99h7sh7rx7drah/marblings.jpg')

show(gxs, bg='wheat', bg2='lightblue', zoom=1.45, viewup='z')

vd=Video('gyroid.mp4')
for i in range(360):
    ele=-.0
    if i>180: ele *=-1
    show(gxs, bg='wheat', bg2='lightblue', interactive=0, resetcam=0, elevation=ele, azimuth=1)
    vd.addFrame()
vd.close()


# Video('gyroid.mp4').action().close().interactive() # shoot video


