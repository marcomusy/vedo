"""Use two sliders to change
color and transparency of a mesh"""
from vedo import Plotter, dataurl, load


def slider1(widget, event):
    mesh.color(widget.value())

def slider2(widget, event):
    mesh.alpha(widget.value())


mesh = load(dataurl+"magnolia.vtk").flat().lw(0.1)

plt = Plotter()
plt += [mesh, __doc__]

plt.addSlider2D(
    slider1,
    xmin=-9,
    xmax=9,
    value=0,
    pos="bottom-right",
    title="color number",
)

plt.addSlider2D(
    slider2,
    xmin=0.01,
    xmax=0.99,
    value=0.5,
    c="blue",
    pos="bottom-right-vertical",
    title="alpha value (opacity)",
)

plt.show().close()
