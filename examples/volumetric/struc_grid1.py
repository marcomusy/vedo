# A StructuredGrid is a dataset where edges of the hexahedrons are 
# not necessarily parallel to the coordinate axes.
# It can be thought of as a tessellation of a block of 3D space,
# similar to a `RectilinearGrid`
# except that the cells are not necessarily cubes, they can have different
# orientations but are connected in the same way as a `RectilinearGrid`.

from vedo import *

# a noisy geometry
cx = np.sqrt(np.linspace(100, 400, 10))
cy = np.linspace(30, 40, 20)
cz = np.linspace(40, 50, 30)
x, y, z = np.meshgrid(cx, cy, cz) + np.random.normal(0, 0.01, (20, 10, 30))

# sgrid1 = StructuredGrid(dataurl + "structgrid.vts")
sgrid1 = StructuredGrid([x, y, z])
sgrid1.cmap("viridis", sgrid1.vertices[:, 0]+np.sin(sgrid1.vertices[:, 1]))
print(sgrid1)

sgrid2 = sgrid1.clone().cut_with_plane(normal=(-1,1,1), origin=[14,34,44])
msh2 = sgrid2.tomesh(shrink=0.9).linewidth(1).cmap("viridis")

show(
    [["StructuredGrid", sgrid1], ["Shrinked Mesh", msh2]],
    N=2, axes=1, viewup="z",
)
