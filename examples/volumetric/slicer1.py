"""Use sliders to slice volume
Click button to change colormap"""
from vedo import dataurl, Volume, show, Text2D
from vedo.applications import SlicerPlotter

filename = dataurl+'embryo.slc'
#filename = dataurl+'embryo.tif'
#filename = dataurl+'vase.vti'

vol = Volume(filename)#.print()

plt = SlicerPlotter( vol,
                     bg='white', bg2='lightblue',
                     cmaps=("gist_ncar_r","jet","Spectral_r","hot_r","bone_r"),
                     useSlider3D=False,
                   )

#Can now add any other object to the Plotter scene:
#plt += Text2D('some message', font='arial')

plt.show().close()