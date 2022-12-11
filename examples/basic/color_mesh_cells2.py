"""Colorize a mesh cell by clicking on it"""
from vedo import Mesh, Plotter, dataurl

def func(evt):
    msh = evt.actor
    if not msh:
        return
    pt = evt.picked3d
    idcell = msh.closest_point(pt, return_cell_id=True)
    m.cellcolors[idcell] = [255,0,0,200] #RGBA 

m = Mesh(dataurl + "panther.stl").c("blue7")
m.force_opaque().linewidth(1)

plt = Plotter()
plt.add_callback("mouse click", func)
plt.show(m, __doc__, axes=1).close()

