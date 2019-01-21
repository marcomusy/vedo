# Show a cube for each available texture name
# any jpg file can be used as texture.
#
from vtkplotter import Plotter, cube, text
from vtkplotter.utils import textures, textures_path


print(textures_path)
print(textures)

vp = Plotter(N=len(textures), axes=0)

for i, txt in enumerate(textures):
    cb = cube(texture=txt) 
    tname = text(txt, pos=3)
    vp.show([cb, tname], at=i)

vp.camera.Elevation(70)
vp.camera.Azimuth(10)
vp.show(interactive=1)
