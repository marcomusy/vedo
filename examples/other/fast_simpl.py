"""Use fast-simplification to decimate a mesh and transfer
data defined on the original vertices to the decimated ones."""
# https://github.com/pyvista/fast-simplification
# pip install fast-simplification
# Credits: Louis Pujol, see #992
import numpy as np
import fast_simplification as fs
import vedo

# Load a mesh and define a signal on vertices
mesh = vedo.Sphere()
points = mesh.vertices
faces = mesh.cells
signal = points[:, 0]
mesh.pointdata["signal"] = signal
mesh.cmap("rainbow").lw(1)

# Decimate the mesh and compute the mapping between the original vertices
# and the decimated ones with fast-simplification
points_decim, faces_decim, collapses = fs.simplify(
    points, faces, target_reduction=0.9, return_collapses=True
)
points_decim, faces_decim, index_mapping = fs.replay_simplification(
    points, faces, collapses
)

# Compute the average of the signal on the decimated vertices (scatter operation)
unique_values, counts = np.unique(index_mapping, return_counts=True)
a = np.zeros(len(unique_values), dtype=signal.dtype)
np.add.at(a, index_mapping, signal)  # scatter addition
a /= counts  # divide by the counts of each vertex index to get the average

# Create a new mesh with the decimated vertices and the averaged signal
decimated_mesh = vedo.Mesh([points_decim, faces_decim])
decimated_mesh.pointdata["signal"] = a
decimated_mesh.cmap("rainbow").lw(1)

vedo.show([[mesh, __doc__], decimated_mesh], N=2, axes=1).close()
