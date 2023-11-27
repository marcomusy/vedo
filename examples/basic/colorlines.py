"""Color lines by a scalar
Click the lines to get their lengths"""
from vedo import *

# Define the points for the first line
pts1 = [(sin(x/8), cos(x/8), x/5) for x in range(25)]

# Create the first line and color it black
l1 = Line(pts1).c('black')

# Create the second line by cloning the first and rotating it
l2 = l1.clone().rotate_z(180).shift(1,0,0)

# Calculate a scalar value for each line segment as 
# the distance between the corresponding points on the two lines
dist = mag(l1.vertices - l2.vertices)

# Color the lines based on the scalar value using the 'Accent' colormap,
#  and add a scalar bar to the plot
lines = Lines(l1, l2, lw=8)
lines.celldata["distance"] = dist
lines.cmap('Accent').add_scalarbar('length')

# Define a callback function to print the length of the clicked line segment
def clickfunc(evt):
    if evt.object:
        # Get the ID of the closest point on the clicked line segment
        idl = evt.object.closest_point(evt.picked3d, return_cell_id=True)
        # Print the length of the line segment with 3 decimal places
        print('clicked line', idl, 'length =', dist[idl])

# Create a plotter with the mouse click callback function and show the lines
plt = Plotter(axes=1, bg2='lightblue')
plt.add_callback('mouse click', clickfunc)
plt.show(l1,l2, lines, __doc__, viewup='z')
plt.close()
