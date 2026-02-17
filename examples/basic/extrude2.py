"""Linear extrusion"""
from vedo import Circle, show

# Build a planar polygon to extrude.
base = Circle(res=8).wireframe(False).c("gold")

# Vector extrusion: displacement is zshift * direction.
vec_extruded = (
    base.extrude_linear(shift=0.7, direction=(0.4, 0.0, -1.0), cap=True)
    .c("tomato")
    .pos(-0.8, 0, 0)
)

# Normal extrusion: uses mesh normals instead of a fixed vector.
nor_extruded = (
    base.compute_normals()
    .extrude_linear(shift=0.25, use_normal=True, cap=True)
    .c("seagreen")
    .pos(0.8, 0, 0)
)

show(
    base.wireframe(True), 
    vec_extruded, nor_extruded,
    __doc__, axes=1, viewup="z"
).close()
