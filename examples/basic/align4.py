"""Align a set of curves in space
with Procrustes method"""
from vedo import *

# Load splines from a file (returns a group of vedo.Lines, like a list)
splines = Assembly(dataurl+'splines.npy')

# Perform Procrustes alignment on the splines, allowing for non-rigid transformations
procus = procrustes_alignment(splines, rigid=False)

# Unpack the aligned splines from the Assembly object into a Python list
alignedsplines = procus.unpack()

# Obtain the mean spline and create a Line object with thicker width and blue color
mean = procus.info['mean']
lmean = Line(mean).z(0.001) # z-shift it to make it visible
lmean.linewidth(4).c('blue')

# Color the aligned splines based on their distance from the mean spline
for l in alignedsplines:
    darr = mag(l.vertices - mean)  # distance array
    l.cmap('hot_r', darr, vmin=0, vmax=0.007)

# Add the mean spline and script description to the list of aligned splines
alignedsplines += [lmean, __doc__]

# Show the original and aligned splines in two side-by-side views 
# with independent cameras
show([splines, alignedsplines], N=2, sharecam=False, axes=1).close()

