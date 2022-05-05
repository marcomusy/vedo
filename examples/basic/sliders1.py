"""Use two sliders to change
color and transparency of a mesh"""
from vedo import Plotter, dataurl, load


def slider1(widget, event):
    value = widget.GetRepresentation().GetValue()
    mesh.color(value)

def slider2(widget, event):
    value = widget.GetRepresentation().GetValue()
    mesh.alpha(value)


plt = Plotter(axes=0)
mesh = load(dataurl+"magnolia.vtk").flat().lw(0.1)
plt += mesh
plt += __doc__

plt.addSlider2D(
    slider1, -9, 9,
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
