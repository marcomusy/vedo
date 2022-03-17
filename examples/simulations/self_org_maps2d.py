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
from vedo import *


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
        pts = Points(self.samples, r=2, c='darkred')
        doc = Text2D(__doc__)

        pb = ProgressBar(0,n_epoch)
        for i in pb.range():
            pb.print("epochs")
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
                grd = Grid(res=[n-1,n-1], c='green2')
                grd.wireframe(False).lw(0.5).bc('blue9').flat()
                grdpts = grd.points()
                for i in range(n):
                    for j in range(n):
                        grdpts[i*n+j] = (x[i,j], y[i,j], z[i,j])
                grd.points(grdpts)
                plt = show(doc, pts, grd, axes=6, azimuth=2, interactive=False)
                if plt.escaped: break  # hit ESC

        plt.interactive().close()
        return [self.codebook[:,i].reshape(n,n) for i in range(3)]

# -------------------------------------------------------------------------------
if __name__ == "__main__":

    settings.allowInteraction = True

    n = 25
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    P = np.c_[X.ravel(), Y.ravel()]
    D = scipy.spatial.distance.cdist(P, P)

    s = Sphere(res=90).cutWithPlane(origin=(0,-.3,0), normal='y').subsample(0.01)

    som = SOM((len(P), 3), D)
    som.samples = s.points()
    som.learn(n_epoch=7000, sigma=(1, 0.01), lrate=(1, 0.01))
