"""Self organizing maps"""
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
from vedo import Sphere, Grid, Plotter, progressbar


class SOM:

    def __init__(self, shape, distance):
        self.codebook = np.random.uniform(0, 1, shape)
        self.distance = distance / distance.max()
        self.samples  = []

    def learn(self, n_epoch=10000, sigma=(0.25,0.01), lrate=(0.5,0.01)):

        t = np.linspace(0, 1, n_epoch)
        lrate = lrate[0] * (lrate[1] / lrate[0]) ** t
        sigma = sigma[0] * (sigma[1] / sigma[0]) ** t
        I = np.random.randint(0, len(self.samples), n_epoch)
        self.samples = self.samples[I]

        for i in progressbar(n_epoch):
            # Get random sample
            data = self.samples[i]

            # Get index of nearest node (minimum distance)
            winner = np.argmin(((self.codebook - data)**2).sum(axis=-1))

            # Gaussian centered on winner
            G = np.exp(-self.distance[winner]**2 / sigma[i]**2)

            # Move nodes towards sample according to Gaussian
            self.codebook -= lrate[i] * G[..., np.newaxis] * (self.codebook-data)

            # Draw network
            if i>500 and not i%20 or i==n_epoch-1:
                x, y, z = [self.codebook[:,i].reshape(n,n) for i in range(3)]
                grd.wireframe(False).lw(0.5).bc('blue9').flat()
                grdpts = grd.points
                for i in range(n):
                    for j in range(n):
                        grdpts[i*n+j] = (x[i,j], y[i,j], z[i,j])
                grd.points = grdpts
                if plt: plt.azimuth(1.0).render()

        if plt: plt.interactive().close()

        return [self.codebook[:,i].reshape(n,n) for i in range(3)]

# -------------------------------------------------------------------------------
if __name__ == "__main__":

    n = 20
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    P = np.c_[X.ravel(), Y.ravel()]
    D = scipy.spatial.distance.cdist(P, P)

    sphere = Sphere(res=90).cut_with_plane(origin=(0,-.3,0), normal='y')
    sphere.subsample(0.01).add_gaussian_noise(0.5).point_size(3)

    plt = Plotter(axes=6, interactive=False)
    grd = Grid(res=[n-1, n-1]).c('green2')
    plt.show(__doc__, sphere, grd)

    som = SOM((len(P), 3), D)
    som.samples = sphere.points.copy()
    som.learn(n_epoch=4000, sigma=(1, 0.01), lrate=(1, 0.01))
