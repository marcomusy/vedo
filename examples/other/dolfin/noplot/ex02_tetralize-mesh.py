"""
Tetrahedral meshes generation with 
package mshr and dolfin.

You can then visualize the file with:
> vtkplotter shuttle.xml
"""
import mshr, dolfin
from vtkplotter import load, printc, show, Text, datadir

fname = "shuttle.stl"

actor = load(datadir + fname)
surface = mshr.Surface3D(datadir + fname)

# add a cylinder
cyl = mshr.Cylinder(dolfin.Point(-1.4, 0, 0), dolfin.Point(-1.0, 0, 0), 0.5, 0.7)
totdomain = surface + cyl

polyhedral_domain = mshr.CSGCGALDomain3D(totdomain)
dolfin.info(polyhedral_domain, True)

generator = mshr.CSGCGALMeshGenerator3D()

#### Try automatic
generator.parameters["mesh_resolution"] = 35.0

#### OR: set your parameters, e.g.
# r =  actor.diagonalSize()
# mesh_res = 30.
# cell_size = r/mesh_res*2
# generator.parameters["edge_size"]       = cell_size
# generator.parameters["facet_angle"]     = 20.0
# generator.parameters["facet_size"]      = cell_size
# generator.parameters["facet_distance"]  = cell_size/2.
# generator.parameters["cell_size"]       = cell_size
# generator.parameters["cell_radius_edge_ratio"] = 3.
# generator.parameters["detect_sharp_features"] = True
# generator.parameters["odt_optimize"]    = True
# generator.parameters["lloyd_optimize"]  = True
# generator.parameters["perturb_optimize"]= True

print("------------------------------- parameters")
for k in generator.parameters.keys():
    print(k, "=", generator.parameters[k])
print("------------------------------------------")

mesh = generator.generate(polyhedral_domain)

xmlname = fname.split(".")[0] + ".xml"

dolfin.File(xmlname) << mesh

printc(mesh, "saved to " + xmlname, c="g")

show(mesh, Text(__doc__))
