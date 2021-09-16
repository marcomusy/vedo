import vedo
import numpy as np

vedo.settings.useDepthPeeling = True

class OpticalElement(object):
    # A base class
    def __init__(self):
        self.name = "OpticalElement"
        self.ref_index = None
        self.normals = []
        self._hits = []
        self._hits_type = [] # +1 if ray is entering, -1 if exiting

    def n_at(self, wave_length):
        # to be overridden to implement dispersion
        return self.ref_index
    
    @property
    def hits(self):
        """Photons coordinates hitting this element"""
        return np.array(self._hits)
    
    @property
    def hits_type(self):
        """Flag +1 if ray is entering, -1 if exiting"""
        return np.array(self._hits_type)
       

class Lens(vedo.Mesh, OpticalElement):
    """A refractive object of arbitrary shape defined by an arbitrary mesh"""
    def __init__(self, actor, ref_index="glass"):
        vedo.Mesh.__init__(self, actor.polydata(), "blue8", 0.5)
        OpticalElement.__init__(self)
        self.computeNormals(cells=True, points=False)
        self.lighting('off')
        self.name = actor.name
        self.normals = self.celldata["Normals"]
        self.ref_index = ref_index

    def n_at(self, wave_length): # in nanometers
        """This is where material dispersion law is implemented"""
        if self.ref_index == "glass":
            # Dispersion of a common borosilicate glass, see:
            # https://en.wikipedia.org/wiki/Sellmeier_equation
            B1 = 1.03961212
            B2 = 0.231792344
            C1 = 6.00069867e-03
            C2 = 2.00179144e-02
            l2 = (wave_length/1000.)**2
            n = np.sqrt(1 + B1 * l2/(l2-C1) + B2 * l2/(l2-C2))
            return n
        else:
            return self.ref_index


class MirrorSurface(vedo.Mesh, OpticalElement):
    """A mirror surface defined by an arbitrary Mesh object"""
    def __init__(self, actor):
        vedo.Mesh.__init__(self, actor.polydata(), "blue8", 0.5)
        OpticalElement.__init__(self)
        self.computeNormals(cells=True, points=True)
        self.name = actor.name
        self.normals = self.celldata["Normals"]
        self.color('silver').lw(0).wireframe(False).alpha(1).phong()
        self.ref_index = -1   


class Screen(vedo.Grid, OpticalElement):
    """A simple read out screen plane"""
    def __init__(self, sizex, sizey):
        vedo.Grid.__init__(self, resx=1, resy=1, sx=sizex, sy=sizey)
        OpticalElement.__init__(self)
        self.computeNormals(cells=True, points=False)
        self.name = "Screen"
        self.normals = self.celldata["Normals"]
        self.color('red3').lw(2).lighting('off').wireframe(False).alpha(0.2)
        self.ref_index = -2 # dummy


class Photon(object):
    """A photn to be tracked as a ray of light"""
    def __init__(self, origin=(0,0,0), direction=(0,0,1), wave_length=450):
        self.name = "Ray"
        self.p = np.asarray(origin) # current position
        self.n = 1.0003             # initial refr index, assume air
        self.v = np.asarray(direction)
        self.polarization = []      # not implemented
        self.attenuation = 0        # not implemented
        self.wave_length = wave_length
        self.dmax = 10
        self.maxiterations = 20
        self.tolerance = 0.001
        self.OBBTreeTolerance = 0 # automatic
        self.path = [self.p]
    
    def _rotate(self, p, angle, axis):
        magv = np.linalg.norm(axis)
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
        """Trace the path of a single photon through the input list of lenses, mirrors etc."""
        for item in items:
            
            for i in range(self.maxiterations):

                hit_cids = item.intersectWithLine(self.p, self.p + self.v * self.dmax, 
                                                  returnIds=True, tol=self.OBBTreeTolerance)
                if not hit_cids: continue
                hit, cid = hit_cids[0]  # grab the first hit, point and cell ID of the mesh

                n = item.normals[cid]
                w = np.cross(self.v/np.linalg.norm(self.v), n)
                sintheta1 = np.linalg.norm(w)
                theta1 = np.arcsin(sintheta1)
                inout = np.sign(np.dot(n, self.v))
                ref_index = item.n_at(self.wave_length)  # dispersion
                r = ref_index/self.n if inout>0 else self.n/ref_index
                sintheta2 = r * sintheta1  # Snell law
                if abs(sintheta2) > 1.0 or ref_index<0:
                    if ref_index<0:
                        theta1 *= inout  # mirrors must reflect on both sides
                    self.v = self._rotate(-self.v, 2*theta1, w)
                    item._hits_type.append(0)
                else:
                    theta2 = np.arcsin(sintheta2)
                    self.v = self._rotate(self.v, theta2-theta1, -w*inout)
                    item._hits_type.append(-inout)

                self.p = hit + self.v*self.tolerance  # update position
                self.path.append(hit)
                item._hits.append(hit)
                
        self.path = np.array(self.path)
        return self


    def asLine(self, min_hits=1, max_hits=1000, c=None):
        """Return a vedo.Line object if it has at least min_hits and less than max_hits"""
        if min_hits < len(self.path) < max_hits:
            if c is None:
                c = vedo.colors.colorMap(self.wave_length, "jet", 450, 750) /1.5
            return vedo.Line(self.path).lw(1).color(c)
        return None


######################################################################### test thin lenses
if __name__ == "__main__":
            
    source = vedo.Disc(r1=0, r2=0.7, res=4).points()  # numpy 3d points
    s = vedo.Sphere(r=2, res=50)                      # construct a thin lens:
    shape = s.boolean("intersect", s.clone().z(3.5)).z(1.4)
    lens  = Lens(shape, ref_index=1.52).color("orange9")
    screen= Screen(3,3).z(5)
    
    items = [lens, screen]
    lines = [Photon(pt).trace(items).asLine() for pt in source]  # list of vedo.Line

    vedo.show("Test of  1/f = (n-1) \dot (1/R1-1/R2) \approx 1/2",
              items, lines, lens.boundaries().lw(2),
              azimuth=-90, zoom=1.2, size=(1100,700), axes=dict(zxGrid=True),
    ).close()

    ####################################################################### test dispersion
    s = vedo.Cone(res=4).scale([1,1,0.4]).rotateY(-90).rotateX(45).pos(-0.5,0,1.5)
    prism = Lens(s, ref_index="glass").lw(1)
    screen= Screen(2,2).z(6)
    lines = []
    for wl in np.arange(450,750, 10):
        photon = Photon(direction=(-0.5,0,1), wave_length=wl)
        line = photon.trace([prism,screen]).asLine()
        lines.append(line)
    vedo.show("Test of chromatic dispersion", prism, screen, lines,
              zoom=1.5, size=(1100,700), axes=1
    ).close()

    
    
    

