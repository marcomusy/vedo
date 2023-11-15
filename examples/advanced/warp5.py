"""
Takes 2 shapes, source and target, and morphs source on target
this is obtained by fitting 18 parameters of a non linear,
quadratic, transformation defined in transform()
The fitting minimizes the distance to the target surface
using algorithms available in the scipy.optimize package.
"""
from vedo import dataurl, vector, mag2, mag
from vedo import Plotter, Sphere, Point, Text3D, Arrows, Mesh
import scipy.optimize as opt

print(__doc__)


class Morpher:
    def __init__(self):
        self.source = None
        self.target = None
        self.bound = 0.1
        self.method = "SLSQP"  # 'SLSQP', 'L-BFGS-B', 'TNC' ...
        self.tolerance = 0.0001
        self.subsample = 200  # pick only subsample pts
        self.allow_scaling = False
        self.params = []
        self.msource = None
        self.s_size = ([0, 0, 0], 1)  # ave position and ave size
        self.fitResult = None
        self.chi2 = 1.0e10
        self.plt = None

    # -------------------------------------------------------- fit function
    def transform(self, p):
        a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, c1, c2, c3, c4, c5, c6, s = self.params
        pos, sz = self.s_size[0], self.s_size[1]
        x, y, z = (p - pos) / sz * s  # bring to origin, norm and scale
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        xp = x + 2 * a1 * xy + a4 * xx + 2 * a2 * yz + a5 * yy + 2 * a3 * xz + a6 * zz
        yp = +2 * b1 * xy + b4 * xx + y + 2 * b2 * yz + b5 * yy + 2 * b3 * xz + b6 * zz
        zp = +2 * c1 * xy + c4 * xx + 2 * c2 * yz + c5 * yy + z + 2 * c3 * xz + c6 * zz
        p2 = vector(xp, yp, zp)
        p2 = (p2 * sz) + pos  # take back to original size and position
        return p2

    def _func(self, pars):
        self.params = pars
        #calculate chi2
        d2sum, n = 0.0, self.source.npoints
        srcpts = self.source.vertices
        rng = range(0, n, int(n / self.subsample))
        for i in rng:
            p1 = srcpts[i]
            p2 = self.transform(p1)
            tp = self.target.closest_point(p2)
            d2sum += mag2(p2 - tp)
        d2sum /= len(rng)
        if d2sum < self.chi2:
            if d2sum < self.chi2 * 0.99:
                print("Emin ->", d2sum)
            self.chi2 = d2sum
        return d2sum

    # ------------------------------------------------------- Fit
    def morph(self):
        def avesize(pts):  # helper fnc
            s, amean = 0, vector(0, 0, 0)
            for p in pts:
                amean = amean + p
            amean /= len(pts)
            for p in pts:
                s += mag(p - amean)
            return amean, s / len(pts)

        print("\n..minimizing with " + self.method)
        self.msource = self.source.clone()

        self.s_size = avesize(self.source.vertices)
        bnds = [(-self.bound, self.bound)] * 18
        x0 = [0.0] * 18  # initial guess
        x0 += [1.0]  # the optional scale
        if self.allow_scaling:
            bnds += [(1.0 - self.bound, 1.0 + self.bound)]
        else:
            bnds += [(1.0, 1.0)]  # fix scale to 1
        res = opt.minimize(self._func, x0,
                           bounds=bnds, method=self.method, tol=self.tolerance)
        # recalc for all pts:
        self.subsample = self.source.npoints
        self._func(res["x"])
        print("\nFinal fit score", res["fun"])
        self.fitResult = res

    # ------------------------------------------------------- Visualization
    def draw_shapes(self):

        newpts = []
        for p in self.msource.vertices:
            newp = self.transform(p)
            newpts.append(newp)
        self.msource.vertices = newpts

        arrs = []
        pos, sz = self.s_size[0], self.s_size[1]
        sphere0 = Sphere(pos, r=sz, res=10, quads=True).wireframe().c("gray")
        for p in sphere0.vertices:
            newp = self.transform(p)
            arrs.append([p, newp])
        hair = Arrows(arrs, s=0.3, c='jet').add_scalarbar()

        zero = Point(pos).c("black")
        x1, x2, y1, y2, z1, z2 = self.target.bounds()
        tpos = [x1, y2, z1]
        text1 = Text3D("source vs target",  tpos, s=sz/10).color("dg")
        text2 = Text3D("morphed vs target", tpos, s=sz/10).color("db")
        text3 = Text3D("deformation",       tpos, s=sz/10).color("dr")

        self.plt = Plotter(shape=[1, 3], axes=1)
        self.plt.at(2).show(sphere0, zero, text3, hair)
        self.plt.at(1).show(self.msource, self.target, text2)
        self.plt.at(0).show(self.source, self.target, text1, zoom=1.2)
        self.plt.interactive().close()


#################################
if __name__ == "__main__":

    mr = Morpher()
    mr.source = Mesh(dataurl+"270.vtk").color("g",0.4)
    mr.target = Mesh(dataurl+"290.vtk").color("b",0.3)
    mr.target.wireframe()
    mr.allow_scaling = True
    mr.bound = 0.4  # limits the parameter value

    mr.morph()

    print("Result of parameter fit:\n", mr.params)

    # now mr.msource contains the modified/morphed source.
    mr.draw_shapes()
