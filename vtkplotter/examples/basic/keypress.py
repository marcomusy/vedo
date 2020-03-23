"""
This example shows how to implement a custom function that is triggered by
 pressing a keyboard button when the rendering window is in interactive mode.
 Every time a key is pressed the picked point of the mesh is used
 to add a sphere and some info is printed.
"""
from vtkplotter import *

##############################################################################
def myfnc(key):
    mesh = vp.clickedActor
    if not mesh or key != "c":
        printc("click a mesh and press c.", c="r")
        return
    printc("clicked mesh    :", mesh.filename[-40:], c=4)
    printc("clicked 3D point:", mesh.picked3d, c=4)
    printc("clicked renderer:", [vp.renderer], c=2)

    vp.add(Sphere(pos=mesh.picked3d, r=0.004, c="v"))


##############################################################################

vp = Plotter(verbose=0)

vp.keyPressFunction = myfnc  # make it known to Plotter class

vp.load(datadir+"bunny.obj")

vp += Text2D(__doc__)

printc("\nPress c to execute myfnc()", c=1)
vp.show()
