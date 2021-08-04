"""Cut a cube with a surface
to create a 'capping' mesh"""
from vedo import *
import numpy as np


# Equation of the "gyroid" (https://en.wikipedia.org/wiki/Gyroid)
x, y, z = np.mgrid[:30,:30,:30] * 0.4
U = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)

# Create a Volume, take the isosurface at 0, smooth it and set mesh edges
s = Volume(U).isosurface(0).smooth().lineWidth(1)

# Create a gridded cube
c = TessellatedBox(n=(29,29,29), spacing=(1,1,1)).alpha(1)

s.cutWithMesh(c).color('silver')  # take what's inside of cube
c.cutWithMesh(s).color('grey')    # take what's inside of isosurface

# Show all the created objects
show(s, c, __doc__, bg='darkseagreen', bg2='lightblue', axes=5).close()
