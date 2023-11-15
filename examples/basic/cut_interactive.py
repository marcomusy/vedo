"""Manipulate a box to cut a mesh interactively.
Use mouse buttons to zoom and pan.
Press r to reset the cutting box
Press i to toggle the cutting box on/off
Press u to invert the selection"""
from vedo import *

# settings.enable_default_keyboard_callbacks = False
# settings.enable_default_mouse_callbacks = False

msh = Mesh(dataurl+'mouse_brain.stl').subdivide()
msh.backcolor("purple8").print()

# Create the plotter with the mesh, do not block the execution
plt = Plotter(bg='blackboard', interactive=False)
plt.show(msh, __doc__, viewup='z')

# Create the cutter object
cutter = PlaneCutter(msh)
# cutter = BoxCutter(msh)
# cutter = SphereCutter(msh)

# Add the cutter to the renderer and show
plt.add(cutter).interactive()

# Remove the cutter from the renderer and show
plt.remove(cutter).interactive()

# close the plotter
plt.close()
