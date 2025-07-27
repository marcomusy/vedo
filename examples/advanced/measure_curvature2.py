import numpy as np
from vedo import Mesh, Plotter, Points, Point, Arrows
from vedo import dataurl, project_point_on_variety, progressbar


#################################################################
def onclick(event):
    if not event.object:
        return
    pid = event.object.closest_point(event.picked3d, return_point_id=True)
    ids = msh1.find_adjacent_vertices(pid, depth=depth, adjacency_list=adlist)
    print(f"Clicked point {pid} with {len(ids)} adjacent vertices", end='')
    print(f" -> curvatures: {curvs_g[pid]:.3f}, {curvs_m[pid]:.3f}")
    bpts = pts1[ids]
    _, res = project_point_on_variety(
        pts1[pid], bpts, degree=2, compute_surface=True, compute_curvature=True
    )
    vpts = Points(bpts, r=3).rename("onclick")
    vpt = Point(pts1[pid], c="green4").rename("onclick")
    res[0].color("green7").alpha(1).rename("onclick")
    plt.remove("onclick").add(vpts, vpt, res[0]).render()


################################################################
msh1 = Mesh(dataurl + "270_flank.vtk").normalize()
# msh1.subdivide()
msh1.smooth()
msh1.compute_normals()
vrange = [-10, 10]
depth = 5  # how many layers of adjacent vertices to consider

# Compute adjacency list for the mesh vertices
adlist = msh1.compute_adjacency()

################################################################
# Compute curvature at all points by fitting a quadratic surface
curvs_g = []
curvs_m = []
pts1 = msh1.points
for i in progressbar(range(msh1.npoints)):
    ids = msh1.find_adjacent_vertices(i, depth=depth, adjacency_list=adlist)
    bpts = msh1.points[ids]
    _, res = project_point_on_variety(pts1[i], bpts, degree=2, compute_curvature=True)
    curvs_g.append(res[4])
    curvs_m.append(res[5])

msh1.pointdata["Gauss_Curvature"] = curvs_g
msh1.cmap("coolwarm", "Gauss_Curvature", vmin=vrange[0], vmax=vrange[1]).add_scalarbar()
msh1.lighting("glossy")

msh2 = msh1.clone()
msh2.pointdata["Mean_Curvature"] = curvs_m
msh2.cmap("Reds", "Mean_Curvature", vmin=0, vmax=3*vrange[1]).add_scalarbar()

msh3 = msh1.clone().lighting("off").alpha(0.5)
msh3.pointdata.select("Gauss_Curvature")
isolines = msh3.isolines(n=10, vmin=vrange[0], vmax=vrange[1]).c("black")

vgrad = msh3.gradient("Gauss_Curvature")
vgrad /= np.linalg.norm(vgrad, axis=1)[:, np.newaxis]  # normalize
arrows = Arrows(msh3.points, msh3.points + vgrad / 40, c="black")

plt = Plotter(N=3, axes=4, size=(2400, 800))
plt.add_callback("mouse click", onclick)
plt.at(0).show("Gaussian curvature\nCLICK on a point to see its curvature", msh1)
plt.at(1).show("Mean curvature", msh2)
plt.at(2).show("Gradient Vectors and Isolines", msh3, arrows, isolines)
plt.interactive().close()
