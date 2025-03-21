"""Align 2 shapes:
the red line to the yellow surface"""
from vedo import Mesh, dataurl, mag2, printc, show

# Load two mesh objects, a limb and a rim, and color them gold and red
limb = Mesh(dataurl + "270.vtk").c("gold")
rim1 = Mesh(dataurl + "270_rim.vtk").c("red5").lw(4)

# Make a clone copy of the rim and align it to the limb
# Using rigid=True does not allow scaling
rim2 = rim1.clone().align_to(limb, rigid=True).c("green5").lw(5)

# Calculate the average squared distance between the aligned rim and the limb
d = 0
for p in rim2.coordinates:
    cpt = limb.closest_point(p)
    d += mag2(p - cpt)  # square of residual distance
average_squared_distance = d / rim2.npoints

# Print the average squared distance between the aligned rim and the limb
printc("Average squared distance =", average_squared_distance, c="g")

show(limb, rim1, rim2, __doc__, axes=1).close()
