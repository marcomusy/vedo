from vedo import *

line = Line([0,0,0], [1,1,1], res=100)
seeds = Points([[10,10,10], [100,100,100]])

################################################

vol = Volume(dataurl+'embryo.tif')
tm = TetMesh(dataurl+'limb.vtu')
rg = RectilinearGrid(dataurl+'RectilinearGrid.vtr')

################################################

print("\n -- TEST METHOD add_ids() -------------------")
print(vol.add_ids())
print(tm.add_ids())
print(rg.add_ids())

print("\n -- TEST METHOD average_size() -------------------")
print(vol.average_size())
print(tm.average_size())
print(rg.average_size())

print("\n--- TEST METHOD bounds() -------------------")
print(vol.bounds())
print(tm.bounds())
print(rg.bounds())

print("\n -- TEST METHOD cell_centers() -------------------")
print(vol.cell_centers().coordinates)
print(tm.cell_centers().coordinates)
print(rg.cell_centers().coordinates)

print("\n -- TEST METHOD cells() -------------------")
print(vol.cells) # NORMALLY THIS GIVES WARNING
print(tm.cells)
print(rg.cells) # NORMALLY THIS GIVES WARNING

print("\n -- TEST METHOD center_of_mass() -------------------")
print(vol.center_of_mass())
print(tm.center_of_mass())
print(rg.center_of_mass())

print("\n--- TEST METHOD compute_cell_size() -------------------")
print(vol.compute_cell_size())
print(tm.compute_cell_size())
print(rg.compute_cell_size())

# print("\n--- TEST METHOD compute_streamlines() -------------------")
# print(vol.compute_streamlines(seeds))
# print(tm.compute_streamlines([[100,0,0], [1000,100,1]]))
# print(rg.compute_streamlines([[0,0,0], [1,1,1]]))

print("\n--- TEST METHOD copy_data_from() -------------------")
print(vol.clone().copy_data_from(vol))
print(tm.clone().copy_data_from(tm))
print(rg.clone().copy_data_from(rg))

# print("\n--- TEST METHOD divergence() -------------------")
# print(vol.divergence())
# print(tm.divergence())
# print(rg.divergence())

print("\n--- TEST METHOD find_cells_along_line() -------------------")
print(vol.find_cells_along_line([0,0,0], [1000,1000,1000]))
print(tm.find_cells_along_line([0,0,0], [100,1,1]))
print(rg.find_cells_along_line([0,0,0], [10,1,1]))

print("\n--- TEST METHOD find_cells_in_bounds() -------------------")
print(vol.find_cells_in_bounds(Sphere().bounds()))
print(tm.find_cells_in_bounds(Sphere().bounds()))
print(rg.find_cells_in_bounds(Sphere().bounds()))

print("\n--- TEST METHOD integrate_data() -------------------")
print(vol.integrate_data())
print(tm.integrate_data())
print(rg.integrate_data())

print("\n--- TEST METHOD interpolate_data_from() -------------------")
print(vol.interpolate_data_from(vol, n=1))
print(tm.interpolate_data_from(vol, n=1))
print(rg.interpolate_data_from(vol, n=1))

print("\n--- TEST METHOD map_cells_to_points() -------------------")
print(vol.clone().map_cells_to_points())
print(tm.clone().map_cells_to_points())
print(rg.clone().map_cells_to_points())

print("\n--- TEST METHOD map_points_to_cells() -------------------")
print(vol.clone().map_points_to_cells())
print(tm.clone().map_points_to_cells())
print(rg.clone().map_points_to_cells())

print("\n--- TEST METHOD lines() -------------------")
print(vol.lines)
print(tm.lines)
print(rg.lines)

print("\n--- TEST METHOD lines_as_flat_array() -------------------")
print(vol.lines_as_flat_array)
print(tm.lines_as_flat_array)
print(rg.lines_as_flat_array)

print("\n--- TEST METHOD mark_boundaries() -------------------")
print(vol.mark_boundaries())
print(tm.mark_boundaries())
print(rg.mark_boundaries())

print("\n--- TEST METHOD memory_address() -------------------")
print(vol.memory_address())
print(tm.memory_address())
print(rg.memory_address())

print("\n--- TEST METHOD memory_size() -------------------")
print(vol.memory_size())
print(tm.memory_size())
print(rg.memory_size())

print("\n--- TEST METHOD modified() -------------------")
print(vol.modified())
print(tm.modified())
print(rg.modified())

print("\n--- TEST METHOD npoints() -------------------")
print(vol.npoints)
print(tm.npoints)
print(rg.npoints)

print("\n--- TEST METHOD ncells() -------------------")
print(vol.ncells)
print(tm.ncells)
print(rg.ncells)

print("\n--- TEST METHOD probe() -------------------")
print(line.probe(vol))
print(line.probe(tm))
print(line.probe(rg))

print("\n--- TEST METHOD resample_data_from() -------------------")
print(vol.clone().resample_data_from(vol))
print(tm.clone().resample_data_from(tm))
print(rg.clone().resample_data_from(rg))

print("\n--- TEST METHOD smooth_data() -------------------")
print(vol.smooth_data())
print(tm.smooth_data())
print(rg.smooth_data())

print("\n--- TEST METHOD shrink() -------------------")
print(tm.shrink())

print("\n--- TEST METHOD to_mesh() -------------------")
print(vol.tomesh())
print(tm.tomesh())
print(rg.tomesh())

print("\n--- TEST METHOD write() -------------------")
print(vol.write("test.vti"))
print(tm.write("test.vtu"))
print(rg.write("test.vtr"))

print("\n--- TEST METHOD cut_with_mesh() -------------------")
print(tm.cut_with_mesh(Ellipsoid().scale(5)))
print(rg.cut_with_mesh(Ellipsoid().scale(5)))

print("\n--- TEST METHOD cut_with_plane() -------------------")
print(tm.cut_with_plane(normal=(1,1,0), origin=(500,0,0)))
print(rg.cut_with_plane(normal=(1,1,0), origin=(0,0,0)))

print("\n--- TEST METHOD extract_cells_by_type() -------------------")
print(tm.extract_cells_by_type("tetra"))

print("\n--- TEST METHOD isosurface() -------------------")
print(vol.isosurface())
print(tm.isosurface())
rg.map_cells_to_points()
print(rg.isosurface())

