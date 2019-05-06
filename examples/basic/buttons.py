"""
Add a square button with N possible internal states
 to a rendering window that calls an external function.
 Available fonts: arial, courier, times
"""
print(__doc__)

from vtkplotter import Plotter, printc, datadir


vp = Plotter(shape=(2, 1), axes=4)

act = vp.load(datadir+"magnolia.vtk", c="v")

vp.show(act, at=0)
vp.show(act, at=1)

# add a button to the current renderer (e.i. nr1)
def buttonfunc():
    act.alpha(1 - act.alpha())  # toggle mesh transparency
    bu.switch()                 # change to next status
    printc(bu.status(), box="_", dim=True)


bu = vp.addButton(
    buttonfunc,
    pos=(350, 20),    # x,y pixels from bottom left corner
    states=["press to hide", "press to show"],
    c=["w", "w"],
    bc=["dg", "dv"],  # colors of states
    font="courier",
    size=18,
    bold=True,
    italic=False,
)

vp.show(interactive=1)
