from vedo import *

settings.default_font = "VictorMono"

r = 0.2

points_2d = np.random.rand(10000, 2) - 0.5
values = np.random.rand(10000)
vmin, vmax = values.min(), values.max()

pcloud1 = Points(points_2d, c='k', r=5).rotate_x(30)
pcloud2 = pcloud1.clone().cut_with_cylinder(r=r, invert=True)

cyl = Cylinder(r=r, height=1, res=360).alpha(0.2)
dists = pcloud1.distance_to(cyl, signed=True)
mask = dists < 0
print("The boolean mask is", mask)

pcloud1.pointdata['values'] = values
pcloud1.pointdata['MASK'] = mask

pts1 = pcloud1.clone().point_size(5)
pts1.cut_with_scalar(0.5, 'MASK')

pts1.cmap('bwr', 'values', vmin=vmin, vmax=vmax).add_scalarbar3d(title='values')
# pts1.cmap('RdYlBu', 'MASK').add_scalarbar3d(title='MASK')
pts1.scalarbar.rotate_x(90)

grid = Grid(res=[100,100]).rotate_x(30)
grid.interpolate_data_from(pcloud1, n=3)
grid.cut_with_cylinder(r=r, invert=True)
grid.cmap('bwr', 'values', vmin=vmin, vmax=vmax).wireframe(False).lw(0)
grid.add_scalarbar3d(title='interpolated values').scalarbar.rotate_x(90)

show(cyl, grid, axes=True)
