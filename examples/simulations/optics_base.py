import vedo
import numpy as np

vedo.settings.useDepthPeeling = True

############################
class OpticalElement(object):
    # A base class
    def __init__(self):
        self.name = "OpticalElement"
        self.type = "undefined"
        self.normals = []
        self._hits = []
        self._hits_type = [] # +1 if ray is entering, -1 if exiting
        self.cellids = []

    def n_at(self, wave_length):
        # to be overridden to implement dispersion
        return self.ref_index

    @property
    def hits(self):
        """Ray coordinates hitting this element"""
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
        self.name = actor.name
        self.type = "lens"
        self.computeNormals(cells=True, points=False)
        self.lighting('off')
        self.normals = self.celldata["Normals"]
        self.ref_index = ref_index

    def n_at(self, wave_length): # in meters
        """This is where material dispersion law is implemented"""
        if self.ref_index == "glass":
            # Dispersion of a common borosilicate glass, see:
            # https://en.wikipedia.org/wiki/Sellmeier_equation
            B1 = 1.03961212
            B2 = 0.231792344
            C1 = 6.00069867e-03
            C2 = 2.00179144e-02
            l2 = (wave_length*1e+06)**2
            n = np.sqrt(1 + B1 * l2/(l2-C1) + B2 * l2/(l2-C2))
            return n
        else:
            return self.ref_index


class Mirror(vedo.Mesh, OpticalElement):
    """A mirror surface defined by an arbitrary Mesh"""
    def __init__(self, actor):
        vedo.Mesh.__init__(self, actor.polydata(), "blue8", 0.5)
        OpticalElement.__init__(self)
        self.computeNormals(cells=True, points=True)
        self.name = actor.name
        self.type = "mirror"
        self.normals = self.celldata["Normals"]
        self.color('silver').lw(0).wireframe(False).alpha(1).phong()

class Screen(vedo.Grid, OpticalElement):
    """A simple read out screen plane"""
    def __init__(self, sizex, sizey):
        vedo.Grid.__init__(self, res=[1,1], s=[sizex,sizey])
        OpticalElement.__init__(self)
        self.computeNormals(cells=True, points=False)
        self.name = "Screen"
        self.type = "screen"
        self.normals = self.celldata["Normals"]
        self.color('red3').lw(2).lighting('off').wireframe(False).alpha(0.2)

class Absorber(vedo.Grid, OpticalElement):
    """A simple non detecting absorber, not generating a hit."""
    def __init__(self, sizex, sizey):
        vedo.Grid.__init__(self, res=[100,100], s=[sizex,sizey])
        OpticalElement.__init__(self)
        self.computeNormals()
        self.name = "Absorber"
        self.type = "screen"
        self.normals = self.celldata["Normals"]
        self.color('k3').lw(1).lighting('default').wireframe(False).alpha(0.8)

class Detector(vedo.Mesh, OpticalElement):
    """A detector surface defined by an arbitrary Mesh"""
    def __init__(self, actor):
        vedo.Mesh.__init__(self, actor.polydata(), "k5", 0.5)
        OpticalElement.__init__(self)
        self.computeNormals()
        self.name = "Detector"
        self.type = "screen"
        self.normals = self.celldata["Normals"]
        self.color('k9').lw(2).lighting('off').wireframe(False).alpha(1)

    def count(self):
        """Count the hits on the detector cells and store them in cell array 'Counts'."""
        arr = np.zeros(self.NCells(), dtype=np.uint)
        for cid in self.cellids:
            arr[cid] += 1
        self.celldata["Counts"] = arr
        return self

    def integrate(self, pols):
        """Integrate the polarization vector and store
        the probability in cell array 'Probability'."""
        arr = np.zeros([self.NCells(), 3], dtype=np.float)
        for i, cid in enumerate(self.cellids):
            arr[cid] += pols[i]
        arr = np.power(np.linalg.norm(arr, axis=1), 2) / len(self.cellids)
        self.celldata["Probability"] = arr
        return self


###################################################
class Ray(object):
    """A photon to be tracked as a ray of light.
    wave_length in meters (so use e.g. 450.0e-09 m = 450 nm)"""
    def __init__(self, origin=(0,0,0), direction=(0,0,1),
                 wave_length=450.0e-09, phase=0, pol=(1,0,0), n=1.003):
        self.name = "Ray"
        self.p = np.asarray(origin) # current position
        self.v = np.asarray(direction)
        self.v = self.v / np.linalg.norm(self.v)
        self.wave_length = wave_length
        self.path = [self.p]
        self._amplitudes = [1.0]
        self._polarizations = [np.array(pol)]
        self.phase = phase
        self.dmax = 20
        self.maxiterations = 20
        self.tolerance = None  # will be computed automatically
        self.OBBTreeTolerance = 1e-05  # None = automatic
        self.ref_index = n

    @property
    def amplitudes(self):
        """
        Amplitudes/attenuations at each hit.
        It assumes random light polarization (natural light).
        """
        return np.array(self._amplitudes)

    @property
    def polarizations(self):
        """Exact polarization vector at each hit."""
        return np.array(self._polarizations)

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

    def _reflectance(self, r12, theta_i, theta_t, inout):
        """Fresnel law for probability to reflect at interface, r12=n1/n2.
        This can be used to compute how much of the main ray arrives at the screen.
        A list of amplitudes at each step is stored in ray.aplitudes.
        """
        if inout < 0: # need to check the sign
            ci = np.cos(theta_i)
            ct = np.cos(theta_t)
        else: # flip
            ct = np.cos(theta_i)
            ci = np.cos(theta_t)
            r12 = 1 / r12
        a = (r12*ci - ct) / (r12*ci + ct) # Rs
        b = (r12*ct - ci) / (r12*ct + ci) # Rp
        return (a*a + b*b)/2

    def trace(self, elements):
        """Trace the path of a single photon through the input list of lenses, mirrors etc."""

        for element in elements:

            self.tolerance = element.diagonalSize()/1000.

            for i in range(self.maxiterations):

                hit_cids = element.intersectWithLine(self.p, self.p + self.v * self.dmax,
                                                     returnIds=True, tol=self.OBBTreeTolerance)
                if not len(hit_cids):
                    break               # no hits
                hit, cid = hit_cids[0]  # grab the first hit, point and cell ID of the mesh
                d = np.linalg.norm(hit - self.p)
                if d < self.tolerance:
                    # it's picking itself.. get the second hit if it exists
                    if len(hit_cids) < 2: break
                    hit, cid = hit_cids[1]
                    d = np.linalg.norm(hit - self.p)

                n = element.normals[cid]
                w = np.cross(self.v, n)
                sintheta1 = np.linalg.norm(w)
                theta1 = np.arcsin(sintheta1)
                inout = np.sign(np.dot(n, self.v)) # ray entering of exiting
                ref_index = self.ref_index

                # polarization vector
                k = 2*np.pi / (self.wave_length/ref_index)
                pol = self._polarizations[-1]
                amp = self._amplitudes[-1] # this assumes random polarizations
                hit_type = -3

                if element.type == "screen":
                    if element.name == "Wall":
                        break
                    pol = pol * np.cos(k * d + self.phase)

                elif element.type == "mirror":
                    theta1 *= inout  # mirrors must reflect on both sides
                    self.v = self._rotate(-self.v, 2*theta1, w)
                    pol = pol * np.cos(k * d + self.phase + np.pi)
                    hit_type = -2

                elif element.type == "lens":
                    ref_index = element.n_at(self.wave_length)  # dispersion
                    r = ref_index/self.ref_index if inout>0 else self.ref_index/ref_index
                    sintheta2 = r * sintheta1  # Snell law
                    if abs(sintheta2) > 1.0:   # total internal reflection
                        self.v = self._rotate(-self.v, 2*theta1, w)
                        pol = pol * np.cos(k * d + self.phase + np.pi)
                        hit_type = -2
                    else:                      # refraction
                        theta2 = np.arcsin(sintheta2)
                        self.v = self._rotate(self.v, theta2-theta1, -w*inout)
                        amp = amp * (1-self._reflectance(r, theta1, theta2, inout))
                        pol = pol * np.cos(k * d + self.phase)
                        hit_type = -inout
                else:
                    print("Unknown element type", element.type)

                self._amplitudes.append(amp)
                self._polarizations.append(pol)

                self.path.append(hit)
                element._hits.append(hit)
                element._hits_type.append(hit_type)
                element.cellids.append(cid)
                if element.type == "screen":
                    break
                self.p = hit  # update position

        self.path = np.array(self.path)
        return self

    def asLine(self, min_hits=1, max_hits=1000, c=None, cmap_amplitudes="", vmin=0):
        """Return a vedo.Line object if it has at least min_hits and less than max_hits"""
        if min_hits < len(self.path) < max_hits:
            ln = vedo.Line(self.path).lw(1)
            if cmap_amplitudes:
                ln.cmap(cmap_amplitudes, self._amplitudes, vmin=vmin)
            elif c is None:
                c = vedo.colors.colorMap(self.wave_length, "jet", 450e-09, 750e-09) /1.5
                ln.color(c)
            else:
                ln.color(c)
            return ln
        return None




