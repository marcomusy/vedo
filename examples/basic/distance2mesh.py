"""Compute the (signed) distance of one mesh to another"""
from vedo import Sphere, Cube, show

# Create a sphere object and position it at (10,20,30)
s1 = Sphere().pos(10,20,30)

# Create a cube object with color grey and scaled 
# along the x-axis by 2, and positioned at (14,20,30)
s2 = Cube(c='grey4').scale([2,1,1]).pos(14,20,30)

# Compute the Euclidean distance between the 2 surfaces
# and set the color of the sphere based on the distance
s1.distance_to(s2, signed=False)
s1.cmap('hot').add_scalarbar('Signed\nDistance')

# Show the sphere, the cube, the script docstring, axes,
# then close the window
show(s1, s2, __doc__ , axes=1, size=(1000,500), zoom=1.5).close()
