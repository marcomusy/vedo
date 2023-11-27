"""Calculate the surface curvature of an object
by fitting a sphere to each vertex."""
from vedo import printc, Ellipsoid, Plotter,fit_sphere
import numpy as np

msh = Ellipsoid()

printc(__doc__, invert=1)

plt = Plotter(N=4, axes=1)
plt.at(0).show(msh, "Original shape")

# Use built-in curvature method
msh1 = msh.clone().compute_curvature(method=0)
msh1.cmap('viridis').add_scalarbar()
plt.at(1).show(msh1, "Gaussian curvature", azimuth=30, elevation=30)

# Use sphere-fit curvature
msh2 = msh.clone()

# Set parameters and allocate arrays
radius = 1.5
curvature = np.zeros(msh2.npoints)
residues = np.zeros(msh2.npoints)

# iterate over surface points and fit sphere
for idx in range(msh2.npoints):

    patch = msh2.closest_point(msh2.vertices[idx], radius=radius)
    s = fit_sphere(patch)
    curvature[idx] = 1/(s.radius)**2
    residues[idx] = s.residue

msh2.pointdata['Spherefit_Curvature'] = curvature
msh2.pointdata['Spherefit_Curvature_Residue'] = residues
msh2.cmap('viridis', 'Spherefit_Curvature')
msh2.add_scalarbar()
plt.at(2).show(msh2, "Sphere-fitted curvature")

# Show fit residues
msh3 = msh2.clone()
msh3.cmap('jet', 'Spherefit_Curvature_Residue').add_scalarbar()
plt.at(3).show(msh3, 'Sphere-fitted curvature\nFit residues')

plt.interactive().close()

