"""Create an empty Figure to be filled in a loop
Any 3D Mesh object can be added to the figure as well"""
from vedo import settings, dataurl, Mesh, Circle, show
from vedo.pyplot import plot, Figure
import numpy as np


settings.defaultFont = "Cartoons123"
settings.palette = 2

x = np.linspace(0, 4*np.pi, 200)
y = np.sin(x) * np.sin(x/12)

# dictionary of options for the axes
ax_opts = dict(xtitle="distance", xyGrid=1, axesLineWidth=3, xyFrameLine=3)

# Create an empty Plot and fill it
fig = Figure(xlim=(0,12), ylim=(-1.1, 1.1), aspect=16/9, axes=ax_opts)
for i in range(10):
    fig += plot(x, y * i/5, lc=i)  # lc= line color (10-colors palette index)

# Add any number of polygonal Meshes.
# Use add() to keep the object aspect ratio inside the Plot coord system:
mesh = Mesh(dataurl+'cessna.vtk').c('blue5').scale(0.4).pos(4, 0.5)
circle = Circle([5,0.5], c='orange5')
fig.add(mesh, circle, as3d=True)

show(fig, __doc__, size=(1000,700), zoom=1.5).close()


