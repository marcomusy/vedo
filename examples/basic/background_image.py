"""
Set a JPEG image as a background layer while rendering 3D content on top.
"""
from vedo import Plotter, dataurl, Cube, VedoLogo

# Four independent sub-renderers sharing the same background image layer.
plt = Plotter(
    N=4,
    sharecam=False,
    bg=dataurl + "images/tropical.jpg",
)

# Foreground actor.
cube = Cube().rotate_z(20)

plt.at(2).show(__doc__)

# Camera controls for the dedicated background renderer.
plt.background_renderer.GetActiveCamera().Zoom(1.8)

plt.at(0).show(VedoLogo(distance=2))
plt.at(3).show(cube)
plt.interactive().close()
