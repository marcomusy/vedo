"""Simulation of an optical system with lenses and mirrors of arbitrary shapes and orientations
(points mark the exit locations of photons, many from internal total reflections)"""
from vedo import Grid, Sphere, Cube, Cone, Points, show
from optics_base import Lens, Photon, MirrorSurface, Screen  # see file ./optics_base.py
import numpy as np
        
source = Grid(resx=20, resy=20).points() # a numpy array of 3d points

# create meshes as ordinary vedo objects
sm = Sphere(r=8).z(-8.1)
sp = Sphere(r=8).z(+8.1)
shape1 = Sphere(r=0.9).cutWithPlane().cap().rotateY(-90).pos(0,0,0.5)
shape2 = Cube(side=2).triangulate().boolean('-', sm).boolean("-", sp).z(3)
shape3 = Cone().rotateY(-90).z(6)
shape4 = Cube().scale([1.5,1,0.2]).rotateY(60).pos(-0.5,0,8)
shape5 = Sphere(r=2).boolean("intersect", Sphere(r=2).z(3.5)).rotateX(10).pos(0.8,0,7.5)
shape6 = Grid(resx=1, resy=1).rotateY(-60).rotateX(30).pos(0,-1,11)

# build lenses (with their refractive indices), and mirrors, using those meshes
lens1 = Lens(shape1, ref_index=1.4).c("blue9") # constant refr. index
lens2 = Lens(shape2).c("blue7")
lens3 = Lens(shape3).c("green9")
lens4 = Lens(shape4).c("purple9").lw(1)
lens5 = Lens(shape5).c("orange9")
mirror= MirrorSurface(shape6)
screen= Screen(4,4).rotateY(20).pos(1,0,12)
items = [lens1, lens2, lens3, lens4, lens5, mirror, screen]

# generate photons and trace them through the optical elements
lines = []
for pt in source:
    λ = np.random.uniform(low=450, high=750)  # nanometers
    photon = Photon(pt, direction=(0,0,1), wave_length=λ)
    line = photon.trace(items).asLine(min_hits=8, c="blue3") # vedo.Line
    lines.append(line)

# grab the coords of photons exiting the conic lens3 (hits_type=-1)
cone_hits = Points(lens3.hits[lens3.hits_type==-1], r=8, c="green1")

# show everithing
show(__doc__, items, lines, lens5.boundaries().lw(2), cone_hits,
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
