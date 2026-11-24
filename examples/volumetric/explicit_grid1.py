"""Model a fault with an explicit structured grid.
The cells keep their regular (i, j, k) indices across the fault,
even though cells on opposite sides no longer share points there."""

import numpy as np
from vedo import ExplicitStructuredGrid, show


nx, ny, nz = 10, 6, 4  # Number of cells along each axis
fault = nx // 2
gap = 0.2  # Leave a small gap so the fault faces are visible.

# Add an extra plane of points at the fault and shift the right block.
pi, pj, pk = np.indices((nx + 2, ny + 1, nz + 1))
right = pi > fault
x = pi - (1 - gap) * right
z = pk + 1.4 * right
coordinates = np.column_stack([axis.ravel(order="F") for axis in (x, pj, z)])
point_ids = np.arange(len(coordinates)).reshape(pi.shape, order="F")

# List cells in structured order, with i varying fastest.
i, j, k = [axis.ravel(order="F") for axis in np.indices((nx, ny, nz))]
point_i = i + (i >= fault)  # Use the duplicated plane on the right.

# VTK hexahedron corners, first on the bottom face and then on the top.
di, dj, dk = np.array([
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]).T
cells = point_ids[point_i[:, None] + di, j[:, None] + dj, k[:, None] + dk]

grid = ExplicitStructuredGrid(coordinates, cells=cells, dims=(nx + 1, ny + 1, nz + 1))

# Give each layer the same color on both sides to highlight the offset.
palette = np.array(
    [(163, 106, 71), (224, 178, 107), (110, 151, 133), (224, 201, 151)],
)
colors = np.repeat(palette, nx * ny, axis=0)
grid.cellcolors = colors
grid.properties.EdgeVisibilityOn()
grid.properties.SetEdgeColor(0.28, 0.23, 0.20)
show(grid, __doc__, viewup="z", axes=1).close()
