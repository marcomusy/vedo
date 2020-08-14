"""
3D slider to move a mesh interactively
"""
from vedo import Plotter, datadir

vp = Plotter()

mesh = vp.load(datadir+"spider.ply")
mesh.normalize().rotateZ(190)


def slider_y(widget, event):
    value = widget.GetRepresentation().GetValue()
    mesh.y(value)  # set y coordinate position

vp.addSlider3D(
    slider_y,
    pos1=[.5, -3.5, .35],
    pos2=[.5, -1.0, .35],
    xmin=-1,
    xmax=1,
    value=0,
    s=0.04,
    c="r",
    rotation=45,
    title="y position",
)


vp.show(viewup="z", axes=11, bg='bb', bg2='navy')
