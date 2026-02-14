"""Align a set of curves in space using Procrustes analysis."""
from vedo import Assembly, dataurl, procrustes_alignment, Line, mag, show

# Load splines (an Assembly of curves).
splines = Assembly(dataurl + "splines.npy")

# Non-rigid Procrustes alignment.
aligned_group = procrustes_alignment(splines, rigid=False)

# Unpack aligned curves for per-curve coloring.
aligned_splines = aligned_group.unpack()

# Mean curve returned by Procrustes, slightly shifted to make it visible.
mean = aligned_group.info["mean"]
mean_line = Line(mean).z(0.001).linewidth(4).c("blue")

# Color each aligned spline by distance to mean shape.
for spline in aligned_splines:
    distances = mag(spline.coordinates - mean)
    spline.cmap("hot_r", distances, vmin=0, vmax=0.007)

aligned_splines += [mean_line, __doc__]

# Compare original (left) and aligned (right).
show([splines, aligned_splines], N=2, sharecam=False, axes=1).close()
