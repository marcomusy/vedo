"""
Show a cube for each available texture name.
Any jpg file can be used as texture.
"""
from vtkplotter import Plotter, Cube, Text2D
from vtkplotter.settings import textures, textures_path

print(__doc__)
print('textures_path:', textures_path)
print('textures:', textures)

vp = Plotter(N=len(textures), axes=0)

for i, name in enumerate(textures):
    if i>30: break
    cb = Cube().texture(name)
    tname = Text2D(name, pos=3)
    vp.show(cb, tname, at=i)

vp.show(interactive=1)
