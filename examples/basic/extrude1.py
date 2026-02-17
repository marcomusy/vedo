"""Extruding a polygon along the z-axis"""
from vedo import Star, show

# Create a yellow star and rotate it around the x-axis
star = Star().color('y')

# Extrude the star along the z-axis, with a shift of 1,
#  a rotation of 10 degrees, a decrease in radius of 0.2,
epol = star.extrude(zshift=1, rotation=10, dr=-0.2, cap=False, res=1)

# Set the back color of the extruded polygon to violet
epol.bc('violet').lighting("default")

# Show the extruded polygon, the script docstring, axes,
# then close the window
show(epol, __doc__, axes=1, viewup='z').close()
