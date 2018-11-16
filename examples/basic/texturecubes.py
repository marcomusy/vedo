# Show a cube for each available texture name
# any jpg file can be used as texture.
#
from vtkplotter import Plotter
from vtkplotter.utils import textures, textures_path


print(textures_path)
print(textures)

vp = Plotter(N=len(textures), axes=0)

for i, txt in enumerate(textures):
    cube = vp.cube(texture=txt) 
    tname = vp.text(txt, pos=3)
    vp.show([cube, tname], at=i)

vp.camera.Elevation(70)
vp.camera.Azimuth(10)
vp.show(interactive=1)
