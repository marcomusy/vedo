"""Use sliders to slice volume
Click button to change colormap
"""
from vedo import datadir, load, show, Text2D
from vedo.applications import Slicer

filename = datadir+'embryo.slc'
#filename = datadir+'embryo.tif'
#filename = datadir+'vase.vti'

vol = load(filename)#.printInfo()

plt = Slicer(vol,
             bg='white', bg2='lightblue',
             cmaps=("gist_ncar_r","jet","Spectral_r","hot_r","bone_r"),
             useSlider3D=False,
             )

#Can now add any other object to the Plotter scene:
#plt += Text2D('some message', font='arial')

plt.show()