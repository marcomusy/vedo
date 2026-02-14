"""Interactively slice a Volume along a plane.
Middle button + drag to slide the plane along the arrow"""
from vedo import *

normal = [0, 0, 1]
cmap = "gist_stern_r"

def func(w, _):
    # Re-slice at current cutter origin/normal.
    c, n = pcutter.origin, pcutter.normal
    vslice = vol.slice_plane(c, n, autocrop=True).cmap(cmap)
    vslice.name = "Slice"
    plt.at(1).remove("Slice").add(vslice)


vol = Volume(dataurl + "embryo.slc").cmap(cmap)
vslice = vol.slice_plane(vol.center(), normal).cmap(cmap)
vslice.name = "Slice"

plt = Plotter(axes=0, N=2, bg="k", bg2="bb")

pcutter = PlaneCutter(
    vslice,
    normal=normal,
    alpha=0,
    c="white",
    padding=0,
)
pcutter.add_observer("interaction", func)
plt.at(0).add(vol, __doc__)
plt.at(1).add(pcutter, vol.box())

pcutter.on() # enable the cutter after adding it to the plotter

plt.show(zoom=1.2)
plt.interactive()
plt.close()
