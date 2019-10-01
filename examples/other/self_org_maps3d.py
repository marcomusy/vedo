""" Self Organizing Maps on a 3D lattice """
# -----------------------------------------------------------------------------
# Copyright 2019 (C) Nicolas P. Rougier
# Released under a BSD two-clauses license
#
# References: Kohonen, Teuvo. Self-Organization and Associative Memory.
#             Springer, Berlin, 1984.
# https://github.com/rougier/ML-Recipes/blob/master/recipes/ANN/som.py
# -----------------------------------------------------------------------------
import numpy as np
import scipy.spatial

class SOM:

    def __init__(self, shape, distance):
        self.codebook = np.random.uniform(0, 1, shape)
        self.distance = distance / distance.max()

    def learn(self, samples, n_epoch=10000, sigma=(0.25, 0.01), lrate=(0.5, 0.01)):
        t = np.linspace(0, 1, n_epoch)
        lrate = lrate[0] * (lrate[1] / lrate[0]) ** t
        sigma = sigma[0] * (sigma[1] / sigma[0]) ** t
        I = np.random.randint(0, len(samples), n_epoch)
        samples = samples[I]

        for i in range(n_epoch):
            # Get random sample
            data = samples[i]

            # Get index of nearest node (minimum distance)
            winner = np.argmin(((self.codebook - data) ** 2).sum(axis=-1))

            # Gaussian centered on winner
            G = np.exp(-self.distance[winner] ** 2 / sigma[i] ** 2)

            # Move nodes towards sample according to Gaussian
            self.codebook -= lrate[i] * G[..., np.newaxis] * (self.codebook-data)

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    n = 16
    ls = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(ls, ls, ls)
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    D = scipy.spatial.distance.cdist(P, P)

    som = SOM((len(P), 3), D)

    samples = P

    som.learn(samples, n_epoch=7000, sigma=(0.5, 0.01), lrate=(0.5, 0.01))

    # Draw network
    x, y, z = [som.codebook[:, i].reshape(n, n, n) for i in range(3)]

    from vtkplotter import Points, Line, show
    Points(samples, c="lb", alpha=0.2)
    for k in [0, 8, 15]:

        for i in range(n):
            ptjs = []
            for j in range(n):
                ptjs.append((x[i, j, k], y[i, j, k], z[i, j, k]))
            Line(ptjs)  # create line through a serie of 3d points

        for j in range(n):
            ptjs = []
            for i in range(n):
                ptjs.append((x[i, j, k], y[i, j, k], z[i, j, k]))
            Line(ptjs)

    show(..., axes=8)
