'''
Show a cube for each available texture name
any jpg file can be used as texture.
'''
from vtkplotter import Plotter, Cube, Text
from vtkplotter.utils import textures, textures_path

print(__doc__)
print(textures_path)
print(textures)

vp = Plotter(N=len(textures), axes=0)

for i, txt in enumerate(textures):
    cb = Cube(texture=txt) 
    tname = Text(txt, pos=3)
    vp.show([cb, tname], at=i)

vp.camera.Elevation(70)
vp.camera.Azimuth(10)
vp.show(interactive=1)
