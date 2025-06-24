"""Manipulate a box to cut a mesh interactively.
Use mouse buttons to zoom and pan. Press:
o to toggle the cutter on/off
i to invert the selection
t to print the cutter's transform
x to get the cut mesh
q to break interaction"""
from vedo import Mesh, settings, dataurl, Plotter
from vedo import BoxCutter, PlaneCutter, SphereCutter


def func(evt):
    """Callback function to handle key presses."""
    k = evt.keypress
    if k == "o":
        print("toggle the cutter on/off")
        cutter.toggle()
    if k == "i":
        print("invert the selection")
        cutter.invert().render()
    if k == "t":
        print(cutter.transform)
    if k == "q":
        print("break interaction")
        plt.break_interaction()
    if k == "x":
        print("get the cut mesh")
        cut_mesh = cutter.get_cut_mesh(invert=False)
        print(cut_mesh.clean())
    if k == "Escape":
        print("exit program")
        plt.close()

##################################################
settings.enable_default_keyboard_callbacks = False
settings.enable_default_mouse_callbacks = False
settings.default_font = "Calco"

msh = Mesh(dataurl + "mouse_brain.stl")
msh.subdivide().backcolor("purple8")

# Create the plotter with the mesh, do not block the execution
plt = Plotter(bg="blackboard")
plt.add_callback("on key press", func)

# ######## Create the cutter object
# cutter = PlaneCutter(msh)
cutter = BoxCutter(msh)
# cutter = SphereCutter(msh)

cutter.enable_rotation(True)
cutter.enable_translation(True)
cutter.enable_scaling(True)

plt.add(cutter, __doc__)
cutter.on()  # enable the cutter after adding it to the plotter

plt.show(viewup="z")
plt.close()
