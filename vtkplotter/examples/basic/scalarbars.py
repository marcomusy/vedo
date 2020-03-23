"""Insert 2D and 3D scalarbars
in the rendering scene"""
from vtkplotter import load, datadir, show, Text2D

shape = load(datadir + "lamp.vtk").normalize()

ms = []
cmaps = ("jet", "PuOr", "viridis")
for i in range(3):
    s = shape.clone().pos(0, i*2.2, 0)

    # colorize cells
    scals = range(s.NCells())
    s.cellColors(scals, cmap=cmaps[i])

    # Or
    # colorize vertices:
    #scals = s.points()[:,i]  # define some dummy point scalar
    #s.pointColors(scals, cmap=cmaps[i])

    ms.append(s)

# use flat shading and add a 2D scalar bar to first mesh
ms[0].flat().addScalarBar(title="my scalarbar\nnumber #0", c="k")

# add 3D scalar bars
ms[1].addScalarBar3D(pos=(1.0, 2.2, -2.), c="k")
ms[2].addScalarBar3D(pos=(1,0,-5), c="k",
    sy=2.8,                    # change y-size
    title="A viridis 3D\nscalarbar to play with",
    titleXOffset=-2,           # offset of labels
    titleSize=1.5).rotateX(90) # make it vertical

show(ms, Text2D(__doc__), axes=1, viewup='z')

# can save colors to vtk or ply format:
#ms[1].write('lamp.ply', binary=False)
