"""Manipulate a box to cut a mesh interactively.
Use mouse buttons to zoom and pan.
Press r to reset the cutting box
Press i to toggle the cutting box on/off
Press u to invert the selection"""
from vedo import *

# settings.enable_default_keyboard_callbacks = False
# settings.enable_default_mouse_callbacks = False

msh = Mesh(dataurl+'mouse_brain.stl').backcolor("purple8")

plt = Plotter(bg='blackboard', interactive=False)
plt.show(msh, __doc__, viewup='z')

cutter = PlaneCutter(msh)
# cutter = BoxCutter(msh)
# cutter = SphereCutter(msh)

plt.add(cutter)
plt.interactive()

plt.remove(cutter)
plt.interactive()

plt.close()
