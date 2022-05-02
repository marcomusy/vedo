"""Create slicers for multiple datasets"""
from vedo import Plotter, Text2D, load, printc, dataurl

volumes = load([dataurl+'vase.vti', dataurl+'embryo.slc', dataurl+'head.vti'])

cmaps = ['hot_r', 'gist_ncar_r', 'bone_r']
alphas = [0, 0, 0.15, 0, 0] # transparency of the grey volume
sliderstart, sliderstop = [0.025,0.04], [0.025,0.4] # slider positioning

######################################################################
def slicerfunc(index, data):

    vol = data.mode(1).c('k',alphas)
    dims = vol.dimensions()
    box = vol.box().alpha(0.5)
    vmin, vmax = vol.scalarRange()
    msh = vol.zSlice(0).cmap(cmaps[index], vmin=vmin, vmax=vmax)
    sb = msh.lighting('off').addScalarBar3D()
    zb = vol.zbounds()
    visibles = [msh]
    txt = Text2D('..'+data.filename[-30:], font='Calco')
    plt.at(index).show(vol, msh, sb, box, txt)

    def func(widget, event):
        i = int(widget.GetRepresentation().GetValue())
        plt.renderer = widget.GetCurrentRenderer()
        plt.resetcam = False
        msh = vol.zSlice(i).lighting('off')
        msh.cmap(cmaps[index], vmin=vmin, vmax=vmax)
        plt.remove(visibles[0], render=False)
        if 0 < i < dims[2]:
            zlev = zb[1]/(zb[1]-zb[0])*i + zb[0]
            plt.add([msh, sb.z(zlev)])
        visibles[0] = msh
    return func

######################################################################
plt = Plotter(shape=(1,3), sharecam=False, bg2='lightcyan')

for index, data in enumerate(volumes):
    plt.addSlider2D(slicerfunc(index, data),
                    0, data.dimensions()[2], value=0,
                    pos=(sliderstart, sliderstop))

printc("Right click to rotate, use slider to slice along z.", box='-')
plt.interactive().close()
