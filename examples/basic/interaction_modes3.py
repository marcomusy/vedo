"""Interaction mode to fly over a surface.
- Press arrows to move the camera in the plane of the surface.
- "t" and "g" will move the camera up and down along z.
- "x" and "X" will reset the camera to the default position towards +/-x.
- "y" and "Y" will reset the camera to the default position towards +/-y.
- "." and "," will rotate azimuth to the right or left.
- "r" will reset the camera to the default position."""
from vedo import settings, ParametricShape, Text2D, Axes, Plotter
from vedo.interactor_modes import FlyOverSurface

settings.enable_default_keyboard_callbacks = False
settings.enable_default_mouse_callbacks = False

surf = ParametricShape("RandomHills").cmap("Spectral")

mode = FlyOverSurface()
txt = Text2D(__doc__, c="k", font="Antares", s=0.8)

plt = Plotter(size=(1200, 600))
plt.user_mode(mode)
plt.show(surf, Axes(surf), txt, elevation=-90, zoom=2, axes=14)
plt.close()
