"""Use two sliders to change
color and transparency of a mesh"""
from vedo import Plotter, Mesh, dataurl


def slider1(widget, event):
    mesh.color(widget.value)

def slider2(widget, event):
    mesh.alpha(widget.value)


mesh = Mesh(dataurl+"magnolia.vtk").flat().lw(1)

plt = Plotter()
plt += [mesh, __doc__]

plt.add_slider(
    slider1,
    xmin=-9,
    xmax=9,
    value=0,
    pos="bottom-right",
    title="color number",
)

plt.add_slider(
    slider2,
    xmin=0.01,
    xmax=0.99,
    value=0.5,
    c="blue",
    pos="bottom-right-vertical",
    title="alpha value (opacity)",
)

plt.show().close()
