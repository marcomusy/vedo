"""Calculate the surface curvature of an opject by fitting a sphere to 
each vertex."""
from vedo import *
import numpy as np

msh = Ellipsoid()

printc(__doc__, invert=1)

plt = Plotter(N=4, axes=1)
plt.show(msh, at=0)

# Use built-in curvature method
msh1 = msh.clone().addCurvatureScalars(method=0).cmap('viridis').addScalarBar(title='Local curvature', horizontal=True, size=(100, None))
msh1.show(at=1, azimuth=30, elevation=30)

# Use sphere-fit curvature
msh2 = msh.clone()

# Set parameters and allocate arrays
radius = 1.5
curvature = np.zeros(msh2.N())
residues = np.zeros(msh2.N())

# iterate over surface points and fit sphere
for idx in range(msh2.N()):
    
    patch = Points(msh2.closestPoint(msh2.points()[idx], radius=radius))
    
    s = fitSphere(patch)
    curvature[idx] = 1/(s.radius)**2
    residues[idx] = s.residue
    
msh2.pointdata['Spherefit_Curvature'] = curvature
msh2.pointdata['Spherefit_Curvature_residue'] = residues
msh2.show(at=2)

msh2.cmap('viridis', msh2.pointdata['Spherefit_Curvature']).addScalarBar(title='Sphere-fitted curvature', horizontal=True, size=(100, None))
msh2.show(at=2)

# Show fit residues
msh3 = msh2.clone()
msh3.cmap('jet', msh2.pointdata['Spherefit_Curvature_residue']).addScalarBar(title='Fit residues', horizontal=True, size=(100, None))
msh3.show(at=3)
    
    