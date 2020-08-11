"""Example usage of removeOutliers()
and cluster() methods.
"""
from vedo import show, cluster, removeOutliers
import numpy as np


# generate 4 random sets of N points in space
N = 2000
f = 0.6
noise1 = np.random.rand(N, 3) * f + np.array([1, 1, 0])
noise2 = np.random.rand(N, 3) * f + np.array([1, 0, 1.2])
noise3 = np.random.rand(N, 3) * f + np.array([0, 1, 1])
noise4 = np.random.randn(N, 3) * f / 8 + np.array([1, 1, 1])

noise4 = removeOutliers(noise4, 0.05)

# merge points to lose their identity
pts = noise1.tolist() + noise2.tolist() + noise3.tolist() + noise4.tolist()

# find back their identity through clustering
cl = cluster(pts, radius=0.1)  # returns a vtkAssembly

show(cl, __doc__, axes=1, viewup='z')
