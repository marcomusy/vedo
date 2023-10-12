"""Example usage of remove_outliers() and compute_clustering()"""
# Import the vedo library and numpy
from vedo import np, Points, show

# Generate 4 random sets of N points in 3D space
N = 2000
f = 0.6
noise1 = np.random.rand(N, 3) * f + np.array([1, 1, 0])
noise2 = np.random.rand(N, 3) * f + np.array([1, 0, 1.2])
noise3 = np.random.rand(N, 3) * f + np.array([0, 1, 1])
noise4 = np.random.randn(N, 3) * f / 8 + np.array([1, 1, 1])

# Create a Points object from the noisy point sets
noise4 = Points(noise4).remove_outliers(radius=0.05).vertices
pts = noise1.tolist() + noise2.tolist() + noise3.tolist() + noise4.tolist()
pts = Points(pts)

# Cluster the points to find back their original identity
clpts = pts.compute_clustering(radius=0.1).print()
# Set the color of the points based on their cluster ID using the 'jet' colormap
clpts.cmap("jet", "ClusterId")

show(clpts, __doc__, axes=1, viewup='z', bg='blackboard').close()
