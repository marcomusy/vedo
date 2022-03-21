"""Simulation of an optical system with lenses and mirrors of arbitrary shapes and orientations
(points mark the exit locations of photons, many from internal total reflection)"""
from vedo import Grid, Sphere, Cube, Cone, Points, show
from optics_base import Lens, Ray, Mirror, Screen  # see file ./optics_base.py
import numpy as np

# Create meshes as ordinary vedo objects
sm = Sphere(r=8).z(-8.1)
sp = Sphere(r=8).z(+8.1)
shape1 = Sphere(r=0.9, res=53).cutWithPlane().cap().rotateY(-90).pos(0,0,0.5)
shape2 = Cube(side=2).triangulate().boolean('-', sm).boolean("-", sp).z(3)
shape3 = Cone().rotateY(-90).z(6)
shape4 = Cube().scale([1.7,1,0.2]).rotateY(70).pos(-0.3,0,8)
shape5 = Sphere(r=2).boolean("intersect", Sphere(r=2).z(3.5)).rotateX(10).pos(0.8,0,7.5)
shape6 = Grid(res=[1,1]).rotateY(-60).rotateX(30).pos(0,-1,11)

# Build lenses (with their refractive indices), and mirrors, using those meshes
lens1 = Lens(shape1, ref_index=1.35).c("blue9") # constant refr. index
lens2 = Lens(shape2, ref_index="glass").c("blue7")
lens3 = Lens(shape3, ref_index="glass").c("green9")
lens4 = Lens(shape4, ref_index="glass").c("purple9").lineWidth(1)
lens5 = Lens(shape5, ref_index="glass").c("orange9")
mirror= Mirror(shape6)
screen= Screen(4,4).rotateY(20).pos(1,0,12)
elements = [lens1, lens2, lens3, lens4, lens5, mirror, screen]

# Generate photons and trace them through the optical elements
lines = []
source = Grid(res=[20,20]).points() # a numpy array
for pt in source:
    λ = np.random.uniform(low=450, high=750)*1e-09  # nanometers
    ray = Ray(pt, direction=(0,0,1), wave_length=λ)
    line = ray.trace(elements).asLine(min_hits=4, cmap_amplitudes="Blues") # vedo.Line
    lines.append(line)
lines = list(filter(None, lines)) # remove possible None to add a scalar bar to lines[0]
lines[0].addScalarBar("Ampl.")

# Grab the coords of photons exiting the conic lens3 (hits_type==-1)
cone_hits = Points(lens3.hits[lens3.hits_type==-1], r=8, c="green1")

# Show everything
show(__doc__, elements, lines, lens5.boundaries().lw(2), cone_hits,
     size=(1500,700), bg='k2', bg2='k9', zoom=2, azimuth=-90,
)
