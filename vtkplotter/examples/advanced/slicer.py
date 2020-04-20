"""Use sliders to slice volume
Click button to change colormap
"""
from vtkplotter import datadir,load,Text2D
from vtkplotter.applications import slicer

filename = datadir+'embryo.slc'
#filename = datadir+'embryo.tif'
#filename = datadir+'vase.vti'

vol = load(filename)#.printInfo()

plt = slicer(vol,
             bg='white', bg2='lightblue',
             cmaps=("gist_ncar_r","jet","Spectral_r","hot_r","bone_r"),
             useSlider3D=False,
             )

#Can now add any other object to the Plotter scene:
#plt += Text2D('some message', font='arial')

plt.show()

#####################################################################################
#####################################################################################
exit()   ############################################################################
#####################################################################################
#####################################################################################



# THE FOLLOWING CODE DOES THE SAME AND IT IS MEANT TO ILLUSTRATE
# HOW THE slicer() METHOD WORKS INTERNALLY:

from vtkplotter import *

################################ prepare the scene
#vol = Volume(10*np.random.randn(300,250,200)+100) # test
vol = load(filename)#.printInfo()
box = vol.box().wireframe().alpha(0) # make an invisible box
vp = show(box, axes=1, bg='white', bg2='lightblue', size=(850,700),
          title=filename, interactive=False, newPlotter=True)
vp.showInset(vol, pos=(1,1), size=0.3, draggable=False)

################# inits
visibles = [None, None, None]
cmaps = ["gist_ncar_r","jet","Spectral_r","hot_r","gist_earth_r","bone_r"]
cmap = cmaps[0]
dims = vol.dimensions()
i_init = int(dims[2]/2)
msh = vol.zSlice(i_init).pointColors(cmap=cmap).lighting('plastic')
msh.addScalarBar(pos=(0.04,0.0), horizontal=True, titleFontSize=0)
vp.renderer.AddActor(msh)
visibles[2] = msh


################# the 2D slider
def sliderfunc_z(widget, event):
    i = int(widget.GetRepresentation().GetValue())
    msh = vol.zSlice(i).pointColors(cmap=cmap).lighting('plastic')
    vp.renderer.RemoveActor(visibles[2])
    if i and i<dims[2]: vp.renderer.AddActor(msh)
    visibles[2] = msh

vp.addSlider2D(sliderfunc_z, 0, dims[2], title='Z', value=i_init,
               pos=[(0.9,0.04), (0.9,0.3)], showValue=False, c='db')


################# the colormap button
def buttonfunc():
    global cmap
    bu.switch()
    cmap = bu.status()
    for mesh in visibles:
        if mesh:
            mesh.pointColors(cmap=cmap)
    vp.renderer.RemoveActor(mesh.scalarbar)
    mesh.scalarbar = addons.addScalarBar(mesh, pos=(0.04,0.0),
                                         horizontal=True, titleFontSize=0)
    vp.renderer.AddActor(mesh.scalarbar)

bu = vp.addButton(buttonfunc,
    pos=(0.27, 0.005),
    states=cmaps,
    c=["db"]*len(cmaps), bc=["lb"]*len(cmaps),  # colors of states
    size=14,
    bold=True,
)


################# show the first mesh and start interacting
vp.show(msh, interactive=True)
