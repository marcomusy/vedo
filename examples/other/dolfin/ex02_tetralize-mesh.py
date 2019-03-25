"""
Tetrahedral meshes generation with 
package mshr and dolfin.

You can then visualize the file with:
> vtkplotter shuttle.xml
"""
import mshr
import dolfin
from vtkplotter.dolfin import datadir, plot

fname = "shuttle.stl"

surface = mshr.Surface3D(datadir + fname)

# add a cylinder
cyl = mshr.Cylinder(dolfin.Point(-1.4, 0, 0), 
	                dolfin.Point(-1.0, 0, 0), 0.5, 0.7)
totdomain = surface + cyl

polyhedral_domain = mshr.CSGCGALDomain3D(totdomain)
dolfin.info(polyhedral_domain, True)

generator = mshr.CSGCGALMeshGenerator3D()

#### Try automatic
generator.parameters["mesh_resolution"] = 35.0
mesh = generator.generate(polyhedral_domain)

xmlname = fname.split(".")[0] + ".xml"
dolfin.File(xmlname) << mesh
print(mesh, "saved to " + xmlname)


##################################
plot(mesh, text=__doc__)
##################################

