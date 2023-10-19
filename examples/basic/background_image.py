"""
Set a jpeg background image
on a separate rendering layer
"""
from vedo import *

# Create a plotter object with 4 subrenderers 
# and individual camera for each one
plt = Plotter(
    N=4,
    sharecam=False, # each subrenderer has its own camera
    bg=dataurl+"images/tropical.jpg", # set the background image
)

# Load a 3D model of a flamingo and rotate it so it is upright
a1 = Cube().rotate_z(20)

# Display a docstring on the second subrenderer
plt.at(2).show(__doc__)

# Zoom in on the background image to fill the window
plt.background_renderer.GetActiveCamera().Zoom(1.8)

# Display a logo on the first subrenderer
plt.at(0).show(VedoLogo(distance=2))

# Display the flamingo model on the fourth subrenderer
plt.at(3).show(a1)

# Allow the plot to be interacted with and then close it
plt.interactive().close()
