"""Estimate Gaussian and mean curvature by local quadratic surface fitting."""

import numpy as np
from vedo import Mesh, Plotter, Points, Point, Arrows
from vedo import dataurl, project_point_on_variety, progressbar

DEPTH  = 5   # neighbourhood depth (layers of adjacent vertices)
DEGREE = 2   # polynomial degree for the local surface fit

################################################################
def onclick(event):
    if not event.object:
        return
    pid  = event.object.closest_point(event.picked3d, return_point_id=True)
    ids  = msh.find_adjacent_vertices(pid, depth=DEPTH, adjacency_list=adlist)
    bpts = msh.points[ids]
    _, poly, grid = project_point_on_variety(
        msh.points[pid], bpts,
        degree=DEGREE, return_grid=True,
        normal=msh.vertex_normals[pid],
    )
    gauss, mean = poly[3], poly[4]
    grid.color("green7").alpha(0.9).rename("pick")
    plt.remove("pick").add(
        Points(bpts, r=3, c="yellow5").rename("pick"),
        Point(msh.points[pid], c="green4").rename("pick"),
        grid,
    ).render()
    print(f"vertex {pid:5d}  neighbours={len(ids):3d}  K={gauss:.3f}  H={mean:.3f}")


################################################################
msh = Mesh(dataurl + "290.vtk")
msh.subdivide()
msh.normalize().smooth().compute_normals()
adlist = msh.compute_adjacency()

################################################################
gauss_vals, mean_vals = [], []
for i in progressbar(range(msh.npoints)):
    ids = msh.find_adjacent_vertices(i, depth=DEPTH, adjacency_list=adlist)
    _, poly, _ = project_point_on_variety(
        msh.points[i], msh.points[ids],
        degree=DEGREE,
        normal=msh.vertex_normals[i],
    )
    gauss_vals.append(poly[3])
    mean_vals.append(poly[4])

gauss_vals = np.array(gauss_vals)
mean_vals  = np.array(mean_vals)

vmax_g = np.max(np.abs(gauss_vals))
vmax_m = np.max(np.abs(mean_vals))
vmax_g = 12
vmax_m = 4

msh_m = msh.clone()
msh_m.pointdata["Mean_Curvature"] = mean_vals
msh_m.cmap("PuOr_r", "Mean_Curvature", vmin=-vmax_m, vmax=vmax_m).add_scalarbar()
msh_m.lighting("glossy")

msh_g = msh.clone()
msh_g.pointdata["Gauss_Curvature"] = gauss_vals
msh_g.cmap("coolwarm", "Gauss_Curvature", vmin=-vmax_g, vmax=vmax_g).add_scalarbar()
msh_g.lighting("glossy")

msh_iso = msh_g.clone().lighting("off").alpha(0.5)
isolines = msh_iso.isolines(n=10, vmin=-vmax_g, vmax=vmax_g).c("black")
vgrad = msh_iso.gradient("Gauss_Curvature")
vgrad /= np.linalg.norm(vgrad, axis=1, keepdims=True)
arrows = Arrows(msh_iso.points, msh_iso.points + vgrad / 40, c="black")

################################################################
plt = Plotter(N=3, axes=4, size=(2400, 800))
plt.add_callback("mouse click", onclick)
plt.at(0).show("Mean curvature  (click a point)", msh_m)
plt.at(1).show("Gaussian curvature", msh_g)
plt.at(2).show("Gaussian curvature: gradient & isolines", msh_iso, arrows, isolines)
plt.interactive().close()
