"""Interpolate a 2D surface through a set of points"""
import numpy as np
from scipy.spatial import distance_matrix
from vedo import Grid, Points, Lines, show
np.random.seed(1)

def harmonic_shepard(pts, vals, radius):
    dists = distance_matrix(pts, pts) + radius
    rdists = 1.0 / dists
    sum_vals = np.sum(rdists * vals, axis=1)
    return sum_vals / np.sum(rdists, axis=1)

# Create a grid of points
surf = Grid(res=[25,25])

# Pick n random points on the surface
ids = np.random.randint(0, surf.npoints, 10)
pts = surf.vertices[ids]

# Create a set of random scalars
scals1 = np.random.randn(10) * 0.1 

ptss1 = pts.copy()
ptss1[:,2] = scals1   # assign scalars as z-coords
pts1 = Points(ptss1).color("red5").point_size(15)

# Compute an interpolated (smoothed) set of scalars
scals2 = harmonic_shepard(pts, scals1, radius=0.1)
ptss2 = pts.copy()
ptss2[:,2] = scals2
pts2 = Points(ptss2).color("purple5").point_size(15)

# Warp the surface to match the interpolated points
ptsource, pttarget = [], []
for pt in pts2.vertices:
    pt_surf = surf.closest_point(pt)
    ptsource.append(pt_surf)
    pttarget.append(pt)
warped = surf.warp(ptsource, pttarget, mode='2d')
warped.color("b4").lc('lightblue').wireframe(False).lw(1)

lines = Lines(pts1, pts2, lw=2)

show(pts1, pts2, lines, warped, __doc__, axes=1, viewup="z").close()

