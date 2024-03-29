# Example usage with pygmsh package:
# https://github.com/nschloe/pygmsh
import pygmsh  # pip install pygmsh
from vedo import TetMesh, SphereCutter, Plotter

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_min = 0.1
    geom.characteristic_length_max = 0.1
    rectangle = geom.add_rectangle([-1.0, -1.0, 0.0], 2.0, 2.0)
    disk1 = geom.add_disk([-1.2, 0.0, 0.0], 0.5)
    disk2 = geom.add_disk([+1.2, 0.0, 0.0], 0.5)
    disk3 = geom.add_disk([0.0, -0.9, 0.0], 0.5)
    disk4 = geom.add_disk([0.0, +0.9, 0.0], 0.5)
    flat = geom.boolean_difference(
        geom.boolean_union([rectangle, disk1, disk2]),
        geom.boolean_union([disk3, disk4]),
    )
    geom.extrude(flat, [0, 0, 0.3])
    msh = geom.generate_mesh()


lines, triangles, tetras, vertices = msh.cells

vmsh = TetMesh([msh.points, tetras.data]).tomesh(fill=True)

plt = Plotter(axes=1, interactive=False)
plt.show(
    vmsh,
    "Drag the sphere,\nright-click&drag to zoom",
)
cutter = SphereCutter(vmsh)
plt.add(cutter)
plt.interactive()
plt.close()
