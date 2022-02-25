"""Insert 2D and 3D scalarbars
in the rendering scene"""
from vedo import Mesh, dataurl, show

shape = Mesh(dataurl+"lamp.vtk")

ms = []
cmaps = ("jet", "PuOr", "viridis")
for i in range(3):
    s = shape.clone(deep=False).pos(0, i*2.2, 0)
    # colorize mesh
    scals = s.points()[:,2]
    s.cmap(cmaps[i], scals)
    ms.append(s)

# add 2D scalar bar to first mesh
ms[0].addScalarBar(title="my scalarbar\nnumber #0") #2D

# add 3D scalar bars
ms[1].addScalarBar3D(c="k", title="scalarbar #1", s=[None,3])

sc = ms[2].addScalarBar3D(pos=(1,0,-5),
                          c="k",
                          s=[None, 2.8],             # change y-size only
                          title="A viridis 3D\nscalarbar to play with",
                          titleFont='Quikhand',
                          titleXOffset=-2,           # offset of labels
                          titleSize=1.5)
sc.scalarbar.rotateX(90) # make it vertical

show(ms, __doc__, axes=1, viewup='z').close()

