"""Draw color arrow glyphs"""
from vedo import *

# Create two spheres with different radii, wireframes, 
# and colors, and set the position of one of them
s1 = Sphere(r=10, res=8).wireframe().c('white')
s2 = Sphere(r=20, res=8).wireframe().c('white',0.1).pos(0,4,0)

# Get the coordinates of the vertices of each sphere
coords1 = s1.vertices 
coords2 = s2.vertices

# --- color can be a colormap which maps arrow sizes
# Define a title for the first set of arrows,
#  and create an Arrows object with coordinates and a colormap for color
t1 = 'Color arrows by size\nusing a color map'
a1 = Arrows(coords1, coords2, c='coolwarm', alpha=0.4)
a1.add_scalarbar(c='w')

# --- get a list of random rgb colors
# Generate a list of random RGB colors for each arrow 
# based on an array of integers, and define a title for the second set of arrows
nrs = np.random.randint(0, 10, len(coords1))
cols = get_color(nrs)
t2 = 'Color arrows by an array\nand scale them by half'
a2 = Arrows(coords1, coords2, c=cols)

# Display two groups of objects on two renderers: the two spheres, 
# the Arrows object with a colormap for color and a scalar bar, 
# and the title for the first set of arrows on one renderer; 
# the two spheres, the Arrows object with random RGB colors, 
# and the title for the second set of arrows on another renderer
show([(s1, s2, a1, t1), (s1, s2, a2, t2)], N=2, bg='bb', bg2='lb').close()
