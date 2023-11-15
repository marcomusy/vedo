"""Implement a custom function that is triggered by
pressing a keyboard button when the rendering window
is in interactive mode.
Place the pointer anywhere on the mesh and press c"""
from vedo import dataurl, printc, Plotter, Point, Mesh

#############################################################
def myfnc(evt):
    mesh = evt.object
    # printc('dump event info', evt)
    if not mesh or evt.keypress != "c":
        printc("click mesh and press c", c="r", invert=True)
        return
    printc("point:", mesh.picked3d, c="v")
    cpt = Point(mesh.picked3d)
    cpt.color("violet").ps(20).pickable(False)
    plt.add(cpt).render()

##############################################################
plt = Plotter(axes=1)
plt+= Mesh(dataurl+"bunny.obj").color("gold")
plt+= __doc__
plt.add_callback('on key press', myfnc)
plt.show().close()
