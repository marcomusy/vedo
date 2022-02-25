"""Generate a polygonal Mesh from a contour line"""
from vedo import dataurl, load, Line, show
from vedo.pyplot import histogram

shapes = load(dataurl+'timecourse1d.npy') # list of lines
shape = shapes[56].mirror().rotateZ(-90)
cmap = "RdYlBu"

msh = shape.tomesh()       # Generate the Mesh from the line
msh.smooth()               # make the triangles more uniform
msh.addQuality()           # add a measure of triangle quality
msh.cmap(cmap, on="cells").addScalarBar3D()

contour = Line(shape).c('red4').lw(5)
labels  = contour.labels('id')

histo = histogram(msh.celldata['Quality'],
                  xtitle='triangle mesh quality',
                  aspect=3/4,
                  c=cmap,
)

show([(contour, labels, msh, __doc__), histo], N=2, sharecam=0).close()
