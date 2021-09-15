import vedo
import numpy as np

vedo.settings.useDepthPeeling = True

class Lens(vedo.Mesh):
    """A refractive object of arbitrary shape defined by an arbitrary mesh"""
    def __init__(self, actor, n=1.52):
        vedo.Mesh.__init__(self, actor.polydata(), "blue8", 0.5)
        self.computeNormals(cells=True, points=False)
        self.lighting('off')
        self.name = actor.name
        self.normals = self.celldata["Normals"]
        self.ref_index = n
        self.hits = []
        self.hits_type = [] # +1 if ray is entering, -1 if exiting

    def n_at(self, wave_length):
        # dispersion to be implemented, see:
        # https://en.wikipedia.org/wiki/Sellmeier_equation
        return self.ref_index


class MirrorSurface(vedo.Mesh):
    """A mirror surface defined by an arbitrary Mesh object"""
    def __init__(self, actor):
        vedo.Mesh.__init__(self, actor.polydata(), "blue8", 0.5)
        self.computeNormals(cells=True, points=True)
        self.name = actor.name
        self.normals = self.celldata["Normals"]
        self.color('silver').lw(0).wireframe(False).alpha(1).phong()
        self.ref_index = -1   
        self.hits = []
        self.hits_type = []


class Screen(vedo.Grid):
    """A simple read out screen plane"""
    def __init__(self, sizex, sizey):
        vedo.Grid.__init__(self, resx=1, resy=1, sx=sizex, sy=sizey)
        self.computeNormals(cells=True, points=False)
        self.name = "Screen"
        self.normals = self.celldata["Normals"]
        self.color('red3').lw(2).lighting('off').wireframe(False).alpha(0.2)
        self.ref_index = -1 # dummy
        self.hits = []
        self.hits_type = []


class Ray(object):
    """A ray of light"""
    def __init__(self, origin, direction=(0,0,1)):
        self.name = "Ray"
        self.p = np.asarray(origin) # current position
        self.n = 1.0003             # initial refr index, assume air
        self.v = np.asarray(direction)
        self.polarization = None
        self.dmax = 10
        self.maxiterations = 20
        self.tolerance = 0.001
        self.OBBTreeTolerance = 0 # automatic
        self.path = [origin]
    
    def _rotate(self, p, angle, axis):
        magv = vedo.utils.mag(axis)
        if not magv: return p
        a = np.cos(angle / 2)
        b, c, d = -axis * (np.sin(angle / 2) /magv)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = np.array([ [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                       [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                       [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]  ])
        return np.dot(R, p)

    def trace(self, items):
        """Trace the path of a single ray of light through the input list of lenses, mirrors etc."""
        for item in items:
            
            for i in range(self.maxiterations):

                hit_cids = item.intersectWithLine(self.p, 
                                                  self.p + self.v * self.dmax, 
                                                  returnIds=True, tol=self.OBBTreeTolerance)
                if not hit_cids: continue
                hit, cid = hit_cids[0] # grab the first hit, point and cell ID of the mesh

                n = item.normals[cid]
                w = np.cross(self.v, n) / vedo.utils.mag(self.v) / vedo.utils.mag(n)
                sintheta1 = vedo.utils.mag(w)
                theta1 = np.arcsin(sintheta1)
                inout = np.sign(np.dot(n, self.v)).astype(int)
                ri = item.ref_index / self.n
                if inout < 0:
                    ri = 1 / ri
                sintheta2 = ri * sintheta1
                if abs(sintheta2) > 1.0 or item.ref_index<0:
                    if item.ref_index<0:
                        theta1 *= inout # mirrors reflect on both sides
                    self.v = self._rotate(-self.v, -2*theta1, -w)
                    item.hits_type.append(0)
                else:
                    theta2 = np.arcsin(sintheta2)
                    self.v = self._rotate(self.v, -theta1+theta2, -w*inout)
                    item.hits_type.append(-inout)

                self.p = hit + self.v*self.tolerance  # update position
                self.path.append(hit)
                item.hits.append(hit)
                
        self.path = np.array(self.path)
        return self


    def asLine(self, min_hits=1, max_hits=1000):
        # Return a vedo.Line object if it has at least min_hits and < max_hits
        if min_hits < len(self.path) < max_hits:
            return vedo.Line(self.path).lw(1).c("blue3")
        return None


########################################################################################## test
if __name__ == "__main__":
            
    source = vedo.Disc(r1=0, r2=0.7, res=5).alpha(0.1).smooth()
    s = vedo.Sphere(r=2, res=50)
    shape = s.boolean("intersect", s.clone().z(3.5)).z(1.4)
    lens  = Lens(shape, n=1.5).color("orange9")
    screen= Screen(3,3).z(5)
    
    items = [lens, screen]
    raylines = []
    for p in source.points():
        ray = Ray(p).trace(items).asLine()
        raylines.append(ray)
    
    vedo.show("Test of  1/f = (n-1) \dot (1/R1-1/R2) \approx 1/2",
              source, *items, raylines, lens.boundaries().lw(1),
              azimuth=-90, zoom=1.2, size=(1100,700), axes=dict(zxGrid=True),
    ).close()


