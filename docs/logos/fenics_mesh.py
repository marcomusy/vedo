from vtkplotter import *

mytext='FEniCS'

tf = Text(mytext, pos=[0,0,0], depth=0.5).triangle().subdivide(3).clean(0.0025)

bx = Box([1.35, 0.27, -0.0], 5.8, 1.5, .2).triangle().subdivide(4, method=1).clean(0.0025)

show(tf, bx)

printc('implicitModeller', mytext, "this takes time")
imp = implicitModeller(mergeActors(tf, bx),
                       distance=0.04,
                       outer=True,
#                       res=[50,20,10],
                       res=[110,40,20],
                       bounds=[-1.0, 10.0, -1.0, 2.0, -.5, .5],
                       maxdist=0.25,
        )

imp.write(mytext+'.stl', binary=False)
show(imp)

#### tetralize
import mshr
import dolfin

surface = mshr.Surface3D(mytext+'.stl')
polyhedral_domain = mshr.CSGCGALDomain3D(surface)
dolfin.info(polyhedral_domain, True)

generator = mshr.CSGCGALMeshGenerator3D()
generator.parameters["mesh_resolution"] = 40.0
mesh = generator.generate(polyhedral_domain)

dolfin.File(mytext + ".xml") << mesh
printc('saved', mytext + ".xml")

#from vtkplotter.dolfin import plot
#plot(mesh)
