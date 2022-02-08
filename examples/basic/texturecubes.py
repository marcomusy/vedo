"""
Show a cube for each available texture name.
Any jpg file can be used as texture.
"""
from vedo import settings, Plotter, Cube
from vedo import textures, textures_path

print(__doc__)
print('textures_path:', textures_path)
print('textures:', textures)

settings.immediateRendering = False
plt = Plotter(N=len(textures), axes=0)

for i, name in enumerate(textures):
    if i>30: break
    cb = Cube().texture(name)
    plt.show(cb, name, at=i, azimuth=1)

plt.show(interactive=True).close()
