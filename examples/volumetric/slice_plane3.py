"""Interactively slice a Volume along a plane.
Middle button + drag to slide the plane along the arrow"""
from vedo import *

normal = [0, 0, 1]
cmap = "gist_stern_r"

def func(w, _):
    c, n = pcutter.origin, pcutter.normal
    vslice = vol.slice_plane(c, n, autocrop=True).cmap(cmap)
    vslice.name = "Slice"
    plt.at(1).remove("Slice").add(vslice)


vol = Volume(dataurl + "embryo.slc").cmap(cmap)
vslice = vol.slice_plane(vol.center(), normal).cmap(cmap)
vslice.name = "Slice"

plt = Plotter(axes=0, N=2, bg="k", bg2="bb", interactive=False)
plt.at(0).show(vol, __doc__, zoom=1.5)

pcutter = PlaneCutter(
    vslice,
    normal=normal,
    alpha=0,
    c="white",
    padding=0,
    can_translate=False,
    can_scale=False,
)
pcutter.add_observer("interaction", func)
plt.at(1).add(pcutter)

plt.interactive()
plt.close()
