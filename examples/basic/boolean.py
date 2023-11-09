from vedo import *

# Enable depth peeling for rendering transparency
settings.use_depth_peeling = True

# Declare an instance of the Plotter class with 2 rows and 2 columns of renderers,
# and disable interactive mode, so that the program can continue running
plt = Plotter(shape=(2, 2), interactive=False, axes=3)

# Create two sphere meshes
s1 = Sphere(pos=[-0.7, 0, 0]).c("red5",0.5)
s2 = Sphere(pos=[0.7, 0, 0]).c("green5",0.5)

# Show the spheres on the first renderer, and display the docstring as the title
plt.at(0).show(s1, s2, __doc__)

# Perform a boolean intersection operation between the two spheres,
# set the color to magenta, and show the result on the second renderer
b1 = s1.boolean("intersect", s2).c('magenta')
plt.at(1).show(b1, "intersect", resetcam=False)

# Perform a boolean union operation between the two spheres,
# set the color to blue, add a wireframe, and show the result on the third renderer
b2 = s1.boolean("plus", s2).c("blue").wireframe(True)
plt.at(2).show(b2, "plus", resetcam=False)

# Perform a boolean difference operation between the two spheres,
# compute the normals, add a scalarbar, and show the result on the fourth renderer
b3 = s1.boolean("minus", s2).compute_normals().add_scalarbar(c='white')
plt.at(3).show(b3, "minus", resetcam=False)

# Enable interactive mode, and close the plot
plt.interactive().close()
