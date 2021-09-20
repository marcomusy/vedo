import numpy as np
import vedo
from optics_base import Lens, Ray, MirrorSurface, Detector, Screen  # see file ./optics_base.py


###################################################################### thin lenses
s = vedo.Sphere(r=2, res=50)                      # construct a thin lens:
shape = s.boolean("intersect", s.clone().z(3.5)).z(1.4)
lens  = Lens(shape, ref_index=1.52).color("orange9")
screen= Screen(3,3).z(5)

elements = [lens, screen]
source = vedo.Disc(r1=0, r2=0.7, res=4).points()  # numpy 3d points
lines = [Ray(pt).trace(elements).asLine() for pt in source]  # list of vedo.Line

vedo.show("Test of  1/f = (n-1) \dot (1/R1-1/R2) \approx 1/2",
          elements, lines, lens.boundaries().lw(2),
          azimuth=-90, zoom=1.2, size=(1100,700), axes=dict(zxGrid=True)).close()


####################################################################### dispersion
s = vedo.Cone(res=4).scale([1,1,0.4]).rotateY(-90).rotateX(45).pos(-0.5,0,1.5)
prism = Lens(s, ref_index="glass").lw(1)
screen= Screen(2,2).z(6)
lines = []
for wl in np.arange(450,750, 10)*1e-09:
    ray = Ray(direction=(-0.5,0,1), wave_length=wl)
    line = ray.trace([prism,screen]).asLine()
    lines.append(line)
vedo.show("Test of chromatic dispersion", prism, screen, lines,
          zoom=1.5, size=(1100,700), axes=1).close()


################################################################ spherical mirrors
s1 = vedo.Sphere(r=7, res=50).cutWithPlane([0,0,6],'z').cutWithCylinder(invert=True)
s2 = vedo.Sphere(r=5, res=50).cutWithPlane([0,0,-2],'-z').cutWithCylinder().z(10)
m1 = MirrorSurface(s1)
m2 = MirrorSurface(s2)
screen = Screen(5,5).z(9)
elements = [m2, m1, m2,  m1, screen] ## NOTE ordering!
source= vedo.Disc(r1=1, r2=3, res=[20,60]).cutWithPlane().cutWithPlane(normal='y').z(1)
lines = [Ray(pt).trace(elements).asLine(2) for pt in source.points()]
vedo.show("Reflection from spherical mirrors", elements, lines, axes=1).close()


################################################################# parabolic mirror
s = vedo.Paraboloid(res=200).cutWithPlane([0,0,-0.4], 'z').scale([1,1,0.1]).z(1)
elements = [MirrorSurface(s), Screen(0.2,0.2).z(0.35)]
source= vedo.Disc(r1=.1, r2=.3, res=[10,30]).cutWithPlane().cutWithPlane(normal='y')
lines = [Ray(pt).trace(elements).asLine() for pt in source.points()]
vedo.show("Reflection from a parabolic mirror", elements, lines, axes=2, azimuth=-90).close()


# ################################################################# interference
s1 = vedo.Sphere(res=100).rotateY(90).cutWithPlane([0,0,0.9], normal='z').y(-.5)
s2 = vedo.Sphere(res=100).rotateY(90).cutWithPlane([0,0,0.9], normal='z').y(+.5)
src = vedo.merge(s1,s2).clean().computeNormals()
dirs = src.pointdata["Normals"]
screen= Screen(3,3).z(4)

grid = vedo.Grid(normal=[0,0,1], resx=40, resy=40, sx=4,sy=4)
detector = Detector(grid).z(3.5)

elements = [detector]
rays, lines, pols = [], [], []
for i,pt in enumerate(src.points()):
    ray = Ray(pt, direction=dirs[i], wave_length=1).trace(elements) # radio waves
    line = ray.asLine()
    if not i%20:
        lines.append(line)
    pols.append(ray.polarizations[-1])

# detector.count().cmap("bone_r", on='cells').addScalarBar("Counts")
detector.integrate(pols).cmap("brg", on='cells').addScalarBar("Prob.")

vedo.show("Interference on a detector surface", s1,s2, lines, elements,
          zoom=1.5, size=(1100,700), elevation=180, azimuth=90, axes=1).close()


