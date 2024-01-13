"""Use fast-simplification to decimate a mesh and transfer
data defined on the original faces to the decimated ones."""
# Credits: Louis Pujol
# https://github.com/pyvista/fast-simplification
# pip install fast-simplification
import numpy as np
import fast_simplification as fs
import vedo

# Load a mesh and define a signal on vertices
mesh = vedo.Sphere().lw(1)
points = mesh.vertices
faces = mesh.cells
signal = points[:, 2]
mesh.pointdata["signal"] = signal

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
decimated_mesh = vedo.Mesh([points_decim, faces_decim]).lw(1)
decimated_mesh.pointdata["signal"] = a

vedo.show(mesh, decimated_mesh, N=2, axes=1).close()
