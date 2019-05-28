# Files can be grouped into a single data entity
# which can be loaded as a single object and unpacked.
#
from vtkplotter import *

cube  = Cube(side=30)
scals = cube.coordinates()[:,1]
poly  = cube.addPointScalars(scals, 'scalsname').polydata()

img = load(datadir+'vase.vti').imagedata()

filename = "multiblock.vtm"

mblock = write([poly, img], filename) #returns a vtkMultiBlockData
printc("~save wrote file", filename,
	   "and corresponding directory", c='g')

# load back from file into a list of actors/volumes
mbacts = load(filename) # loads and unpacks a MultiBlockData obj

show(mbacts, Text('load("file.vtm") #MultiBlockData', c='k'), bg='w')

show(mblock, Text('show(multiblock)', c='w'), newPlotter=True, pos=(800,0))
