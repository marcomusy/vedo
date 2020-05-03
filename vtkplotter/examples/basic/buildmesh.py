"""Manually build a mesh from points and faces"""
from vtkplotter import Mesh, printc, show

verts = [(50,50,50), (70,40,50), (50,40,80), (80,70,50)]
faces = [(0,1,2), (2,1,3), (1,0,3)]
# (the first triangle face is formed by vertex 0, 1 and 2)

# Build the polygonal Mesh object:
m = Mesh([verts, faces])
m.backColor('violet').lineColor('tomato').lineWidth(2)

# retrieve them as numpy arrays
printc('points():\n', m.points(), c=3)
printc('faces(): \n', m.faces(),  c=3)

show(m, __doc__, viewup='z', axes=8)
