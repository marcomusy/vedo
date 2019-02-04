'''
Takes 2 shapes, source and target, and morphs source on target
this is obtained by fitting 18 parameters of a non linear, 
quadratic, transformation defined in transform()
The fitting minimizes the distance to the target surface
using algorithms available in the scipy.optimize package.
'''  
from __future__ import division, print_function
print(__doc__)

from vtkplotter import *
import scipy.optimize as opt


vp = Plotter(shape=[1,3], interactive=0, bg='w')

class Morpher:
    def __init__(self):
        self.source = None
        self.target = None
        self.bound  = 0.1
        self.method = 'SLSQP' #'SLSQP', 'L-BFGS-B', 'TNC' ...
        self.tolerance = 0.0001
        self.subsample = 200  #pick only subsample pts
        self.allowScaling = False
        self.params  = []

        self.msource = None  
        self.s_size  = ([0,0,0], 1) # ave position and ave size
        self.fitResult = None
        self.chi2   = 1.0e+10

    #-------------------------------------------------------- fit function
    def transform(self, p):
        a1,a2,a3,a4,a5,a6, b1,b2,b3,b4,b5,b6, c1,c2,c3,c4,c5,c6, s = self.params
        pos, sz = self.s_size[0], self.s_size[1]
        x, y, z = (p - pos)/sz*s #bring to origin, norm and scale
        xx, yy, zz, xy, yz, xz = x*x, y*y, z*z, x*y, y*z, x*z
        xp = x +2*a1*xy +a4*xx    +2*a2*yz +a5*yy    +2*a3*xz +a6*zz
        yp =   +2*b1*xy +b4*xx +y +2*b2*yz +b5*yy    +2*b3*xz +b6*zz
        zp =   +2*c1*xy +c4*xx    +2*c2*yz +c5*yy +z +2*c3*xz +c6*zz
        p2 = vector(xp,yp,zp)
        p2 = (p2 * sz) + pos #take back to original size and position
        return p2   

    def _func(self, pars):
        self.params = pars

        d2sum, n = 0.0, self.source.N()
        srcpts = self.source.polydata().GetPoints()
        mpts = self.msource.polydata().GetPoints()
        rng = range(0,n, int(n/self.subsample))
        for i in rng:
            p1 = srcpts.GetPoint(i)
            p2 = self.transform(p1)
            tp = self.target.closestPoint(p2)
            d2sum += mag2(p2-tp)
            mpts.SetPoint(i, p2)

        d2sum /= len(rng)
        if d2sum<self.chi2:
            if d2sum<self.chi2*0.99: 
                print ( 'Emin ->', d2sum )
            self.chi2 = d2sum
        return d2sum
   
    #------------------------------------------------------- Fit
    def morph(self):
        
        def avesize(pts): # helper fnc
            s, amean = 0, vector(0, 0, 0)
            for p in pts: amean = amean + p
            amean /= len(pts)
            for p in pts: s += mag(p - amean)
            return amean, s/len(pts)

        print ('\n..minimizing with '+self.method)
        self.msource = self.source.clone()

        self.s_size = avesize(self.source.coordinates())
        bnds = [(-self.bound, self.bound)]*18
        x0   = [0.0]*18   #initial guess
        x0  += [1.0]      # the optional scale
        if self.allowScaling: bnds += [(1.0-self.bound, 1.0+self.bound)]
        else: bnds += [(1.0, 1.0)] # fix scale to 1
        res = opt.minimize(self._func, x0, bounds=bnds,
                           method=self.method, tol=self.tolerance)
        #recalc for all pts:
        self.subsample = self.source.N()
        self._func(res['x'])
        print ('\nFinal fit score', res['fun'])
        self.fitResult = res

    #------------------------------------------------------- Visualization
    def draw_shapes(self):
        
        pos, sz = self.s_size[0], self.s_size[1]
        
        sphere0 = sphere(pos, c='gray', r=sz, alpha=.8, res=16).wire(1)
        sphere1 = sphere(pos, c='gray', r=sz, alpha=.2, res=16)

        hairsacts=[]
        for i in range(sphere0.N()):
            p = sphere0.point(i)
            newp = self.transform(p)
            sphere1.point(i, newp)
            hairsacts.append( arrow(p, newp, s=0.3, alpha=.5) )

        zero = point(pos, c='black')
        x1,x2, y1,y2, z1,z2 = self.target.polydata().GetBounds()
        tpos = [x1, y2, z1]
        text1 = text('source vs target',  tpos, s=sz/10, c='dg')
        text2 = text('morphed vs target', tpos, s=sz/10, c='dg')
        text3 = text('deformation',       tpos, s=sz/10, c='dr')

        vp.show([sphere0, sphere1, zero, text3] + hairsacts, at=2)
        vp.show([self.msource, self.target, text2], at=1) 
        vp.show([self.source, self.target, text1], at=0, zoom=1.2, interactive=1)


#################################
if __name__ == '__main__':

    mr = Morpher()
    mr.source = vp.load('data/270.vtk',  c='g', alpha=.4)
    mr.target = vp.load('data/290.vtk',  c='b', alpha=.3, wire=1)
    mr.allowScaling = True
    mr.bound = 0.4 # limits the parameter value

    mr.morph()

    print('Result of parameter fit:\n', mr.params)
    # now mr.msource contains the modified/morphed source.
    mr.draw_shapes()
    
    












