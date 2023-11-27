"""Generate a polygonal Mesh from a contour line"""
from vedo import dataurl, Assembly, Line, show
from vedo.pyplot import histogram

shapes = Assembly(dataurl + "timecourse1d.npy")  # group of lines
shape = shapes[56]  # pick one
cmap = "RdYlBu"

# Generate the Mesh from the line
msh = shape.generate_mesh(invert=True)
msh.smooth()           # make the triangles more uniform
msh.compute_quality()  # add a measure of triangle quality
msh.cmap(cmap)

contour = Line(shape).c("red4").lw(5)
labels = contour.labels("id")

histo = histogram(
    msh.celldata["Quality"],
    xtitle="triangle mesh quality",
    aspect=25/9,
    c=cmap,
).clone2d("bottom-right")

show(contour, labels, msh, histo, __doc__).close()
