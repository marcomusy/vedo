'''
Warp the tip of a mesh using Thin Plate Splines.
Red points stay fixed while a single point close to the tip moves up.
'''
from vtkplotter import *


mesh = load('data/270_flank.vtk').normalize()

meshd = mesh.clone().decimate(N=500) # a heavily decimated copy

sources = [[1.2, -0.3, -0.6]] # this point moves 
targets = [[1.4, -0.2,  0.1]] # to this.
for pt in meshd.coordinates(): 
    if pt[0]<0:               # these pts don't move
        sources.append(pt)    # source = target
        targets.append(pt)

# calculate the warping T on the reduced mesh
T = thinPlateSpline(meshd, sources, targets).info['transform']

warped = transformFilter(mesh, T).color('blue').alpha(0.4)

apts = points(sources).color('red')

arro = arrow(sources[0], targets[0])

show([mesh, arro, warped, apts, 
     text(__doc__, c='white')], viewup='z', verbose=0)

