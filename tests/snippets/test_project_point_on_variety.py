import vedo

# Example usage
########################################################################
mesh = vedo.Mesh(vedo.dataurl + "bunny.obj").subdivide().scale(100)
mesh.wireframe().alpha(0.1)

pt = mesh.coordinates[30]
points = mesh.closest_point(pt, n=200)

pt_trans, poly, grid = vedo.project_point_on_variety(
    pt, points, degree=3, return_grid=True
)
vpoints = vedo.Points(points, r=6, c="yellow2")

plotter = vedo.Plotter(size=(1200, 800))
plotter += mesh, vedo.Point(pt), vpoints, grid, f"Residue: {pt - pt_trans}"
plotter.show(axes=1).close()
