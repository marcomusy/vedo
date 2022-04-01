"""Example usage of removeOutliers() and addClustering() methods"""
from vedo import np, Points, show

# generate 4 random sets of N points in space
N = 2000
f = 0.6
noise1 = np.random.rand(N, 3) * f + np.array([1, 1, 0])
noise2 = np.random.rand(N, 3) * f + np.array([1, 0, 1.2])
noise3 = np.random.rand(N, 3) * f + np.array([0, 1, 1])
noise4 = np.random.randn(N, 3) * f / 8 + np.array([1, 1, 1])

noise4 = Points(noise4).removeOutliers(radius=0.05).points()

# merge points to lose their identity
pts = noise1.tolist() + noise2.tolist() + noise3.tolist() + noise4.tolist()
pts = Points(pts)

# find back their identity by clustering
clpts = pts.addClustering(radius=0.1).print()
clpts.cmap("jet", "ClusterId")
# or
# clpts = pts.addConnection(radius=0.1).print()
# clpts.cmap("jet", "RegionLabels")

show(clpts, __doc__, axes=1, viewup='z', bg='blackboard').close()
