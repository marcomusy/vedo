"""Manipulate a box to cut a mesh interactively.
Use mouse buttons to zoom and pan.
Press r to reset the cutting box
Press spacebar to toggle the cutting box on/off
Press i to invert the selection"""
from vedo import *

# settings.enable_default_keyboard_callbacks = False

cow = Mesh(dataurl+'cow.vtk').backcolor("purple8")
# cow.cmap('jet', cow.points()[:,2])

plt = Plotter(bg='blackboard', interactive=False)
plt.show(cow, __doc__, viewup='z')

cutter = BoxCutter(cow)
plt.add(cutter)
plt.interactive()

plt.remove(cutter)
plt.interactive()

plt.close()
