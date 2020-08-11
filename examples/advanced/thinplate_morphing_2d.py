"""Take 2 clouds of points, source and target, and morph
source on target using thin plate splines as a model.
The fitting minimizes the distance to the target surface.
"""
from vedo import *
import scipy.optimize as opt
import numpy as np
np.random.seed(0)

class Morpher:
    def __init__(self):
        self.source = None
        self.morphed_source = None
        self.target = None
        self.bound = None
        self.method = "SLSQP"  # 'SLSQP', 'L-BFGS-B', 'TNC' ...
        self.fitTolerance = 1e-6
        self.params = []
        self.fitResult = None
        self.chi2 = 1.0e30
        self.ndf = None
        self.ptsource = []
        self.pttarget = []

    def _func(self, pars):
        self.params = np.array(pars)

        ptmoved = np.multiply((self.pttarget-self.ptsource).T, (1+self.params)).T + self.ptsource
        self.morphed_source = self.source.clone().thinPlateSpline(self.ptsource, ptmoved)

        d = self.morphed_source.points() - self.target.points()
        chi2 = np.sum(np.multiply(d,d))/self.ndf
        if chi2 < self.chi2:
            print("Emin ->", chi2)
            self.chi2 = chi2
        return chi2

    # ------------------------------------------------------- Fit
    def morph(self):

        print("\n..minimizing with " + self.method)
        self.morphed_source = self.source.clone()

        indexes = list(range(0,self.source.N(), int(self.source.N()/self.ndf)))
        self.ptsource = self.source.points()[indexes]
        self.pttarget = self.target.points()[indexes]
        self.ndf = len(indexes)

        bnds = [(-self.bound, self.bound)] * self.ndf
        x0 = [0.0] * self.ndf  # initial guess
        res = opt.minimize(self._func, x0,
                           bounds=bnds,
                           method=self.method,
                           tol=self.fitTolerance)
        # recalc for all pts:
        self._func(res["x"])
        print("\nFinal fit score", res["fun"])
        self.fitResult = res

    # ------------------------------------------------------- Visualization
    def draw_shapes(self):
        sb = self.source.bounds()
        x1, x2, y1, y2, z1, z2 = sb
        maxb = max(x2-x1, y2-y1)
        grid0 = Grid(self.source.centerOfMass(), sx=maxb, sy=maxb, resx=40, resy=40)
        T = self.morphed_source.getTransform()
        grid1 = grid0.alpha(0.3).wireframe(0).clone().applyTransform(T) # warp the grid

        text1 = Text2D(__doc__, c="k")
        text2 = Text2D("morphed vs target\nn.d.f.="+str(self.ndf), c="k")
        arrows = Arrows(self.ptsource, self.pttarget, alpha=0.5, s=1).c("k")

        self.morphed_source.pointSize(10).c('g')
        settings.interactorStyle = 7 # lock scene to 2D
        show(grid0, self.source, self.target, arrows, text1, at=0, N=2, axes=0)
        show(grid1, self.morphed_source, self.target, text2, at=1, zoom=1.2, interactive=1)


#################################
if __name__ == "__main__":

    mr = Morpher()

    # make up a random cloud and distort it
    pts_s = np.random.randn(100, 3)
    pts_t = np.array(pts_s) +np.sin(2*pts_s)/5
    pts_s[:,2] = 0 # let's make it 2D
    pts_t[:,2] = 0

    mr.source = Points(pts_s, r=10, c="g", alpha=1)
    mr.target = Points(pts_t, r=10, c="r", alpha=1).rotateZ(10) # add rotation too

    mr.bound = 1  # limits the parameter value
    mr.ndf   = 10 # fit degree (nr of degrees of freedom)

    mr.morph()

    #print("Result of parameter fit:\n", mr.params)
    #now mr.msource contains the modified/morphed source.
    mr.draw_shapes()
