"""Manipulate a box to cut a mesh interactively.
Use mouse buttons to zoom and pan.
Press r to reset the cutting box
Press spacebar to toggle the cutting box on/off
Press i to invert the selection"""
from vedo import *

# settings.enable_default_keyboard_callbacks = False

cow = Mesh(dataurl+'cow.vtk')

plt = Plotter(bg='blackboard', interactive=False)
plt.show(cow, __doc__, viewup='z')

cutter = BoxCutter(cow)

cutter.on()
plt.interactive()

cutter.off()
plt.interactive()

plt.close()
