from vedo import *

grid = Grid().wireframe(False)
square = Rectangle().extrude(0.5).scale(0.4).rotate_z(20).shift(0,0,-.1)
square.alpha(0.3)

centers = grid.cell_centers().coordinates
ids = square.inside_points(centers, return_ids=True)

arr = np.zeros(centers.shape[0]).astype(np.uint8)
arr[ids] = 1
grid.celldata["myflag"] = arr
grid.cmap("rainbow", "myflag", on='cells')

show(grid, square, axes=8)