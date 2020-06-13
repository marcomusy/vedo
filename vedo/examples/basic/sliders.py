"""Use two sliders to change
color and transparency of a mesh.
"""
from vedo import Plotter, datadir


def slider1(widget, event):
    value = widget.GetRepresentation().GetValue()
    mesh.color(value)

def slider2(widget, event):
    value = widget.GetRepresentation().GetValue()
    mesh.alpha(value)


vp = Plotter(axes=0)
mesh = vp.load(datadir+"magnolia.vtk").flat().lw(0.1)

# pos = position corner number: horizontal [1-4] or vertical [11-14]
vp.addSlider2D(slider1, -9, 9, value=0, pos=4, title="color number")

vp.addSlider2D(slider2, xmin=0.01, xmax=0.99, value=0.5,
               pos=14, c="blue", title="alpha value (opacity)")

vp += __doc__
vp.show()
