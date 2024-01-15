"""Insert 2D and 3D scalarbars in the rendering scene"""
from vedo import Mesh, dataurl, show

shape = Mesh(dataurl + "lamp.vtk")

ms = []
cmaps = ("jet", "PuOr", "viridis")
for i in range(3):
    s = shape.clone(deep=False).pos([0, i * 2.2, 0])
    # colorize mesh
    scalars = s.vertices[:, 2]
    s.cmap(cmaps[i], scalars)
    ms.append(s)

# add 2D scalar bar to first mesh
ms[0].add_scalarbar(title="my scalarbar\nnumber #0")  # 2D

# add 3D scalar bars
ms[1].add_scalarbar3d(c="k", title="scalarbar #1", size=[0, 3])

sc = ms[2].add_scalarbar3d(
    c="k",
    size=[None, 2.8],  # change y-size only
    title="A viridis colormap\nscalarbar to play with",
    title_font="Quikhand",
    title_xoffset=-2,  # offset of labels
    title_size=1.5,
)
sc.scalarbar.rotate_x(90).scale(1.2).shift(0,2,0)  # make it vertical

# create a 2D copy scalarbar to the 3D one
sc2d = sc.scalarbar.clone2d(size=0.3, ontop=True)

show(ms, sc2d, __doc__, axes=1, viewup="z").close()
