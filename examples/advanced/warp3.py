"""Take 2 clouds of points, source and target, and morph
the plane using thin plate splines as a model.
The fitting minimizes the distance to a subset of the target cloud"""
from vedo import printc, Points, Grid, Arrows, Lines, Plotter
import scipy.optimize as opt
import numpy as np
np.random.seed(2)

class Morpher(Plotter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.source = None
        self.morphed_source = None
        self.target = None
        self.bound = None
        self.sigma = 1  # stiffness of the mesh
        self.method = "SLSQP"  # 'SLSQP', 'L-BFGS-B', 'TNC' ...
        self.fitTolerance = 1e-6
        self.fitResult = None
        self.chi2 = 1.0e30
        self.npts = None
        self.ptsource = []
        self.pttarget = []


    def _func(self, pars):
        shift = np.array(np.split(pars,2)).T # recreate the shift vectors
        z = np.zeros((self.npts,1))
        shift = np.append(shift, z, axis=1) # make them 3d
        self.morphed_source = self.source.clone().warp(self.ptsource,
                                                       self.ptsource + shift,
                                                       sigma=self.sigma,
                                                       mode="2d")
        d = self.morphed_source.points() - self.target.points()
        chi2 = np.sum(np.multiply(d,d))#/len(d)
        if chi2 < self.chi2:
            printc("new minimum ->", chi2)
            self.chi2 = chi2
        return chi2

    # ------------------------------------------------------- Fit
    def morph(self):
        print("\n..minimizing with " + self.method)
        self.morphed_source = self.source.clone()

        self.ptsource = self.source.points()[:self.npts] # pick the first npts points
        self.pttarget = self.target.points()[:self.npts]

        delta = self.pttarget - self.ptsource
        x0 = delta[:,(0,1)].T.ravel() # initial guess, a flat list of x and y shifts
        bnds = [(-self.bound, self.bound)] * (2*self.npts)
        res = opt.minimize(self._func, x0, bounds=bnds, method=self.method, tol=self.fitTolerance)
        self.fitResult = res
        # recalculate the last step:
        self._func(res["x"])

    # ------------------------------------------------------- Visualization
    def draw_shapes(self):
        sb = self.source.bounds()
        x1, x2, y1, y2, z1, z2 = sb
        maxb = max(x2-x1, y2-y1)
        grid0 = Grid(self.source.centerOfMass(), s=[maxb,maxb], res=[40,40])
        T = self.morphed_source.getTransform()
        grid1 = grid0.alpha(0.3).wireframe(0).clone().applyTransform(T) # warp the grid
        arrows = Arrows(self.ptsource, self.pttarget, alpha=0.5, s=3).c("k")
        lines = Lines(self.source, self.target).c('db')
        mlines = Lines(self.morphed_source, self.target).c('db')

        self.at(0).show(grid0, self.source, self.target, lines, arrows, __doc__)
        self.at(1).show(grid1, self.morphed_source, self.target, mlines,
                        f"morphed source (green) vs target (red)\nNDF = {2*self.npts}")


#################################
if __name__ == "__main__":

    # make up a source random cloud
    pts_s = np.random.randn(25, 2)
    pts_t = pts_s + np.sin(2*pts_s)/5 # and distort it

    mr = Morpher(N=2)
    mr.source = Points(pts_s, r=20, c="g", alpha=0.5)
    mr.target = Points(pts_t, r=10, c="r", alpha=1.0)

    mr.bound = 2  # limits the x and y shift
    mr.npts  = 6  # allow move only a subset of points (implicitly sets the NDF of the fit)
    mr.sigma = 1. # stiffness of the mesh (1=max stiffness)

    mr.morph()

    #now mr.msource contains the modified/morphed source.
    mr.draw_shapes()
    mr.interactive().close()
