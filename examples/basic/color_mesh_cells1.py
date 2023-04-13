"""Colorize faces of a Mesh by passing
a 1-to-1 list of colors and transparencies"""
from vedo import *

# Enable depth peeling for better rendering of transparent objects
settings.use_depth_peeling = True

# Generate a torus and assign a linewidth of 1
tor = Torus(res=9).linewidth(1)

# Generate an array of random RGBA color values for each cell of the mesh
rgba = np.random.rand(tor.ncells, 4)*255
tor.cellcolors = rgba

# Print information about the cell arrays of the mesh and their shape
printc(
    'Mesh cell arrays:', tor.celldata.keys(),
    'shape:', tor.celldata['CellsRGBA'].shape,
)

# Display the mesh with the assigned colors and the docstring
show(tor, __doc__).close()
