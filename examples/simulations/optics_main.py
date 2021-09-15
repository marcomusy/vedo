"""Simulation of an optical system with lenses and mirrors of arbitrary shapes and orientations
(points mark the exit locations of photons, many from internal reflections)"""
from vedo import Grid, Sphere, Cube, Cone, Points, show
from optics import Lens, Ray, MirrorSurface, Screen  # see file ./optics.py
import numpy as np
        
source = Grid(resx=16, resy=16).alpha(0.1)

# create meshes as normal vedo objects
s1 = Sphere(r=8).z(-8.1)
s2 = Sphere(r=8).z(+8.1)
shape1 = Sphere(r=0.9).cutWithPlane().cap().rotateY(-90).pos(0,0,0.5)
shape2 = Cube(side=2).triangulate().boolean('-', s1).boolean("-", s2).z(3)
shape3 = Cone().rotateY(-90).z(6)
shape4 = Cube().scale([1,1,0.2]).rotateY(60).z(8)
shape5 = Sphere(r=2).boolean("intersect", Sphere(r=2).z(3.5)).rotateX(10).pos(0.8,0,7.5)
shape6 = Grid(resx=1, resy=1).rotateY(-60).rotateX(30).pos(0,-1,11)

# build lenses with their refr. indices, and mirrors, using those meshes
lens1 = Lens(shape1, n=1.4).color("blue9")
lens2 = Lens(shape2).color("blue7")
lens3 = Lens(shape3).color("green9")
lens4 = Lens(shape4).color("purple9").lineWidth(1)
lens5 = Lens(shape5, n=1.5).color("orange9")
mirror= MirrorSurface(shape6)
screen= Screen(4,4).rotateY(20).pos(1,0,12)

# ray tracing
items = [lens1, lens2, lens3, lens4, lens5, mirror, screen]
raylines = []
for p in source.points():
    ray = Ray(p).trace(items).asLine(min_hits=8)
    raylines.append(ray)

# grab the coords of photons exiting the conic lens3 (hits_type=-1)
lens3.hits      = np.array(lens3.hits)
lens3.hits_type = np.array(lens3.hits_type)
cone_hits = Points(lens3.hits[lens3.hits_type==-1], r=8, c="green1")

# show everithing
show(source, items, raylines, lens5.boundaries().lw(1), cone_hits, __doc__,
     azimuth=-90, elevation=-89, zoom=2, size=(1500,700),
     axes=dict(xyGrid=False, 
               zShiftAlongY=1, 
               zAxisRotation=-45,
               xLabelRotation=90, 
               yLabelRotation=90,
               zLabelRotation=90,
               zTitleRotation=90,
               ytitle=" ",
               xyFrameLine=True,
               numberOfDivisions=10,
               textScale=0.6,
              ),
)
