#!/usr/bin/env python3
# Visualization example with polyscope (pip install polyscope)
# https://polyscope.run/py/
import vedo
import polyscope

m = vedo.load(vedo.dataurl+'embryo.tif').isosurface().extractLargestRegion()
# m = vedo.load(vedo.dataurl+'man.vtk')

polyscope.set_program_name("vedo using polyscope")
polyscope.set_verbosity(0)
polyscope.set_up_dir("z_up")
polyscope.init()
ps_mesh = polyscope.register_surface_mesh('My vedo mesh',
                                          m.points(), m.faces(),
                                          color=[0.5,0,0],
                                          smooth_shade=True,
                                         )
ps_mesh.add_scalar_quantity("heights", m.points()[:,2], defined_on='vertices')
ps_mesh.set_material("wax") # wax, mud, jade, candy
polyscope.show()

vedo.show(m, axes=11)