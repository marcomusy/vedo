"""3D slider to move a mesh interactively"""
from vedo import Plotter, Mesh, dataurl

plt = Plotter()

mesh = Mesh(dataurl+"spider.ply")
mesh.normalize().rotate_z(190)


def slider_y(widget, event):
    mesh.y(widget.value)  # set y coordinate position

plt.add_slider3d(
    slider_y,
    pos1=[0.5, -3.5, 0.35],
    pos2=[0.5, -1.0, 0.35],
    xmin=-1,
    xmax=1,
    value=0,
    s=0.04,
    c="r",
    rotation=45,
    title="y position",
)

plt.show(mesh, __doc__, axes=11, bg='bb', bg2='navy')
plt.close()
