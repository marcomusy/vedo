"""Align two shapes and report the average squared residual distance."""
from vedo import Mesh, dataurl, mag2, printc, show

# Reference surface and curve to be aligned onto it.
limb = Mesh(dataurl + "270.vtk").c("gold")
rim1 = Mesh(dataurl + "270_rim.vtk").c("red5").lw(4)

# Align a clone of the rim with rigid transform only (no scaling).
rim2 = rim1.clone().align_to(limb, rigid=True).c("green5").lw(5)

# Compute mean squared distance of aligned points to the target surface.
sq_err_sum = 0.0
for p in rim2.coordinates:
    cpt = limb.closest_point(p)
    sq_err_sum += mag2(p - cpt)
average_squared_distance = sq_err_sum / rim2.npoints

printc("Average squared distance =", average_squared_distance, c="g")

show(limb, rim1, rim2, __doc__, axes=1).close()
