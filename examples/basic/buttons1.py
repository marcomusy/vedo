"""Add a square button with N possible internal states
to a rendering window that calls an external function"""
from vedo import Plotter, Mesh, dataurl, printc

# Toggle mesh visibility and cycle button label.
def buttonfunc(_obj, _ename):
    mesh.alpha(1 - mesh.alpha())  # toggle mesh transparency
    bu.switch()                   # change to next status
    printc(bu.status(), box="_", dim=True)

mesh = Mesh(dataurl+"magnolia.vtk").c("violet").flat()

plt = Plotter(axes=11)

# Button coordinates are normalized to window size.
bu = plt.add_button(
    buttonfunc,
    pos=(0.7, 0.1),   # x,y fraction from bottom left corner
    states=["click to hide", "click to show"],  # text for each state
    c=["w", "w"],     # font color for each state
    bc=["dg", "dv"],  # background color for each state
    font="courier",   # font type
    size=30,          # font size
    bold=True,        # bold font
    italic=False,     # non-italic font style
)

plt.show(mesh, __doc__).close()
