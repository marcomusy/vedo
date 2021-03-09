"""
Show a cube for each available texture name.
Any jpg file can be used as texture.
"""
from vedo import settings, Plotter, Cube, Text2D
from vedo.settings import textures, textures_path

print(__doc__)
print('textures_path:', settings.textures_path)
print('textures:', settings.textures)

settings.immediateRendering = False
vp = Plotter(N=len(settings.textures), axes=0)

for i, name in enumerate(settings.textures):
    if i>30: break
    cb = Cube().texture(name)
    vp.show(cb, name, at=i, azimuth=1)

vp.show(interactive=True)
