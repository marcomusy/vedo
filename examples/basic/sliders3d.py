"""3D slider to move a mesh interactively"""
from vedo import Plotter, Mesh, dataurl

plt = Plotter()

mesh = Mesh(dataurl+"spider.ply")
# mesh.normalize().rotate_z(190)


def slider_y(widget, event):
    mesh.x(widget.value)  # set y coordinate position

plt.add_slider3d(
    slider_y,
    pos1=[1, 0, 0.35],
    pos2=[6, 0, 0.35],
    xmin=-2,
    xmax=2,
    value=0,
    s=0.04,
    c="r",
    rotation=45,
    title="position",
)

plt.show(mesh, __doc__, axes=11, bg='bb', bg2='navy', elevation=-30)
plt.close()
