from vedo import *

def callb(evt):
    msh = evt.actor
    if not msh:
        return
    pt = evt.picked3d
    idcell = msh.closest_point(pt, return_cell_id=True)
    msh.cellcolors[idcell] = [255,0,0,255] # red, opaque
    
m = Mesh(dataurl + "290.vtk")
m.decimate().smooth().compute_normals()
m.compute_quality().cmap("Blues", on="cells")

plt = Plotter()
plt.add_callback("mouse click", callb)
plt.show(m, m.labels("cellid"))
plt.close()

