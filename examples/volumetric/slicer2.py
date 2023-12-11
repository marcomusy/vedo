"""Slice multiple datasets"""
from vedo import Plotter, Text2D, load, dataurl, ScalarBar3D

volumes = [dataurl+'vase.vti', dataurl+'embryo.slc', dataurl+'head.vti']
volumes = load(volumes)
cmaps = ['hot_r', 'gist_ncar_r', 'bone_r']

########################################################################
def initfunc(iren, vol):

    vol.mode(1).cmap('k').alpha([0, 0, 0.15, 0, 0])
    txt = Text2D(data.filename[-20:], font='Calco')
    plt.at(iren).show(vol, vol.box(), txt)

    def func(widget, event):
        zs = int(widget.value)
        widget.title = f"z-slice = {zs}"
        msh = vol.zslice(zs)
        msh.cmap(cmaps[iren]).lighting("off")
        msh.name = "slice"
        sb = ScalarBar3D(msh, c='k')
        # sb = sb.clone2d("bottom-right", 0.08)
        plt.renderer = widget.renderer  # make it the current renderer
        plt.remove("slice", "ScalarBar3D").add(msh, sb)

    return func  # this is the actual function returned!


########################################################################
plt = Plotter(shape=(1, len(volumes)), sharecam=False, bg2='lightcyan')

for iren, data in enumerate(volumes):
    plt.add_slider(
        initfunc(iren, data), #func
        0, data.dimensions()[2],
        value=0,
        show_value=False,
        pos=[(0.1,0.1), (0.25,0.1)],
    )

plt.interactive().close()
