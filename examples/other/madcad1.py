# Example of usage of the madcad library
# See https://pymadcad.readthedocs.io/en/latest/index.html
import vedo
from madcad import *

##########################################################################
points = [O, X, X + Z, 2 * X + Z, 2 * (X + Z), X + 2 * Z, X + 5 * Z, 5 * Z]
section = Wire(points).segmented().flip() 
rev = revolution(2 * pi, (O, Z), section)
rev.mergeclose()
vedo.show("Revolution of a wire", rev, axes=7).close()


##########################################################################
# m = screw(10, 20)
# m["part"].option(color=vec3(70, 130, 180) / 255) # RGB
# vedo.show("A blue screw", m, axes=1).close()

##########################################################################
# Obtain two different shapes that has noting to to with each other
m1 = brick(width=vec3(2))
m2 = m1.transform(vec3(0.5, 0.3, 0.4)).transform(quat(0.7 * vec3(1, 1, 0)))
# Remove the volume of the second to the first
diff = difference(m1, m2)
vedo.show("Boolean difference", diff, axes=14).close()


##########################################################################
cube = brick(width=vec3(2))
bevel(
   cube,
   [(0, 1), (1, 2), (2, 3), (0, 3), (1, 5), (0, 4)],  # Edges to smooth
   ("width", 0.3),  # Cutting description, known as 'cutter'
)
vedo.show("A bevel cube", cube, axes=1).close()


##########################################################################
square_profile = square((O, Z), 5).flip()
primitives = [
    ArcCentered(( 5 * X,  Y),      O, 10 * X),
    ArcCentered((15 * X, -Y), 10 * X, 20 * X),
]
# Generate a path
path = web(primitives)
path.mergeclose()
m = tube(square_profile, path)

vmesh = vedo.utils.madcad2vedo(m)  # <-- convert to vedo.Mesh
print(vmesh)

scalar = vmesh.vertices[:, 0]
vmesh.cmap("rainbow", scalar, on="points").add_scalarbar(title="x-value")
vedo.show("Generating a path", vmesh, axes=7).close()

##########################################################################
c1 = Circle((vec3(0), Z), 1)
c2 = Circle((2 * X, X), 0.5)
c3 = (Circle((2 * Y, Y), 0.5), "tangent", 2)
e1 = extrusion(2 * Z, web(c1))

m = junction(e1, c2, c3, tangents="normal")
vm = vedo.utils.madcad2vedo(m)
vedo.show(vm, e1, axes=1, viewup="z").close()
