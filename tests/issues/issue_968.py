from vedo import *

letter = Text3D("A") #can be any Mesh
letter.pointdata["mydata"] = range(letter.npoints)
letter.cmap("Set1")
eletter = letter.clone().extrude(0.1)
eletter.cmap("Set1")
eletter.print()

show(letter, eletter, N=2, axes=1, viewup='z').close()


