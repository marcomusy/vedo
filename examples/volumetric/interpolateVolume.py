'''
Generate a voxel dataset (vtkImageData) by interpolating a scalar 
which is only known on a scattered set of points.
This is obtained by using RBF (radial basis function).
'''
# @Author: Giovanni Dalmasso
from __future__ import print_function
import vtk
from vtkplotter import Plotter, Volume, ProgressBar, vector, mag2, text
import numpy as np


bins = 30                        # nr. of voxels per axis
npts = 60                        # nr. of points of known scalar value

img = vtk.vtkImageData()
img.SetDimensions(bins, bins, bins) # range is [0, bins-1]
img.AllocateScalars(vtk.VTK_FLOAT, 1)

coords = np.random.rand(npts, 3) # range is [0, 1]
scals  = np.abs(coords[:, 2])    # let the scalar be the z of point itself
fact   = 1./(bins-1)             # conversion factor btw the 2 ranges

vp = Plotter(verbose=0)
vp.ztitle = 'z == scalar value'
cloud = vp.points(coords)

# fill the vtkImageData object
pb = ProgressBar(0, bins, c=4)
for iz in pb.range():
    pb.print()
    for iy in range(bins):
        for ix in range(bins): 

            pt = vector(ix, iy, iz)*fact
            closestPointsIDs = cloud.closestPoint(pt, N=5, returnIds=True)
            
            num, den = 0, 0
            for i in closestPointsIDs: # work out RBF manually on N points
                invdist = 1 / (mag2(coords[i] - pt) + 1e-06)
                num += scals[i] * invdist
                den += invdist       
            img.SetScalarComponentFromFloat(ix, iy, iz, 0, num/den) 

# vp.write(img, 'imgcube.tif') # or .vti

# set colors and transparencies along the scalar range
vol = Volume(img, c=['r','g','b'], alphas=[0.4, 0.8]) #vtkVolume
act = vp.points(coords/fact)

vp.show([vol, act, text(__doc__)], viewup='z')




