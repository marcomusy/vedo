# Files can be grouped into a single data entity
# which can be loaded as a single object and unpacked.
#
from vedo import *

cube  = Cube(side=30)
scals = cube.points()[:,1]
poly  = cube.addPointArray(scals, 'scalsname').polydata()

img = load(datadir+'vase.vti').imagedata()

filename = "multiblock.vtm"

mblock = write([poly, img], filename) #returns a vtkMultiBlockData
printc("~save wrote file", filename,
	   "and corresponding directory", c='g')

# load back from file into a list of meshes/volumes
mbacts = load(filename) # loads and unpacks a MultiBlockData obj

show(mbacts, Text2D('load("file.vtm") #MultiBlockData', c='k'))

show(mblock, Text2D('show(multiblock)', c='w'), new=True, pos=(800,0))
