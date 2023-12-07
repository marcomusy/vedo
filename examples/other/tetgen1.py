"""Use tetgenpy to tetrahedralize a cube."""
try:
    import tetgenpy
except ImportError:
    print("tetgenpy not installed, try: pip install tetgenpy")
import vedo

# Tetrahedralize unit cube, define points
points = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
]

# Define facets, here they are hexa faces
facets = [
    [1, 0, 2, 3],
    [0, 1, 5, 4],
    [2, 0, 4, 6],
    [1, 3, 7, 5],
    [3, 2, 6, 7],
    [4, 5, 7, 6],
]

# Prepare TetgenIO - input for tetgen
tetgen_in = tetgenpy.TetgenIO()

# Set points, facets, and facet_markers.
# facet_markers can be useful for setting boundary conditions
tetgen_in.setup_plc(
    points=points,
    facets=facets,
    facet_markers=[[i] for i in range(1, len(facets) + 1)],
)

# Tetgen's tetraheralize function with switches
tetgen_out = tetgenpy.tetrahedralize("qa.05", tetgen_in)

# Unpack output
# print(tetgen_out.points())
# print(tetgen_out.tetrahedra())
# print(tetgen_out.trifaces())
# print(tetgen_out.trifacemarkers())

plt = vedo.Plotter().add_ambient_occlusion(0.1)
tmesh = vedo.TetMesh(tetgen_out).shrink().color("pink7")
plt.show(tmesh, __doc__, axes=14).close()

# Or simply:
# vedo.show(tetgen_out, axes=14).close()

# Save to file
# tmesh.write("tetramesh.vtu")

