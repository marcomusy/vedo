"""Create an empty Plot to be filled in a loop.
Any 3D Mesh object can be added to the plot as well."""
from vedo import *
from vedo.pyplot import plot, Plot
import numpy as np


settings.defaultFont = "Cartoons123"

x = np.linspace(0, 4*np.pi, 200)
y = np.sin(x) * np.sin(x/12)

# dictionary of options for the axes
ax_opts = dict(xtitle="distance", xyPlaneColor='blue7', xyGridColor='red4')

# Create an empty Plot and fill it
pl = Plot(xlim=(0, 12), ylim=(-1.2, 1.2), aspect=16/9, axes=ax_opts)
for i in range(10):
    pl += plot(x, y * i/5, lc=-i)  # lc= line color

# Add any number of polygonal Meshes.
# Use add() to keep the object aspect ratio inside the Plot coord system:
mesh = Mesh(dataurl+'cessna.vtk').c('green4').scale(0.4)
circle = Circle([5,0.5, 1], c='orange5')
pl.add(mesh.pos(4, 0.5, 2), circle)

show(pl, __doc__, size=(1000,800), zoom='tight').close()


