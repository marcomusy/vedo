from vedo import *

def callb(evt):
    msh = evt.object
    if not msh:
        return
    pt = evt.picked3d
    idcell = msh.closest_point(pt, return_cell_id=True)
    # msh.cellcolors[idcell] = [255,0,0,255] # red, opaque
    cols = msh.cellcolors.copy()
    cols[idcell] = [0,255,0,255] # green, opaque
    msh.cellcolors = cols
    plt.render()
    
m = Mesh(dataurl + "290.vtk")
m.decimate().smooth().compute_normals()
m.compute_quality().cmap("Blues", on="cells")

print(m.cellcolors)

plt = Plotter()
plt.add_callback("mouse click", callb)
plt.show(m, m.labels("cellid"))
plt.close()

