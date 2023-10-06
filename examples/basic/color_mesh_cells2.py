"""Colorize a mesh cell by clicking on it"""
from vedo import Mesh, Plotter, dataurl

# Define the callback function to change the color of the clicked cell to red
def func(evt):
    msh = evt.object
    if not msh:
        return
    pt = evt.picked3d
    idcell = msh.closest_point(pt, return_cell_id=True)
    m.cellcolors[idcell] = [255,0,0,200] #RGBA 

# Load a 3D mesh of a panther from a file and set its color to blue
m = Mesh(dataurl + "panther.stl").c("blue7")

# Make the mesh opaque and set its line width to 1
m.force_opaque().linewidth(1)

# Create a Plotter object and add the callback function to it
plt = Plotter()
plt.add_callback("mouse click", func)

# Display the mesh with the Plotter object and the docstring
plt.show(m, __doc__, axes=1).close()
