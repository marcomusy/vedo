"""Read and show meshio objects"""
import meshio
from vedo import download, show, Mesh

fpath = download('https://vedo.embl.es/examples/data/shuttle.obj')
mesh = meshio.read(fpath)

# vedo understands meshio format for polygonal data:
# show(mesh, __doc__, axes=7)

# explicitly convert it to a vedo.Mesh object:
m = Mesh(mesh).lineWidth(1).color('tomato').print()
show(m, __doc__, axes=7).close()
