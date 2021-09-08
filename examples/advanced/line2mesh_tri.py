"""Generate a polygonal Mesh
from a contour line"""
from vedo import *
from vedo.pyplot import histogram

shapes = load(dataurl+'timecourse1d.npy') # list of lines
shape = shapes[56].mirror().rotateZ(-90)
cmap = "RdYlBu"

msh = shape.tomesh()       # Generate the Mesh from the line
msh.smooth()               # make the triangles more uniform
msh.addQuality()           # add a measure of triangle quality
msh.cmap(cmap, on="cells").addScalarBar3D()

contour = Line(shape).c('red4').lw(5)
histo = histogram(msh.celldata['Quality'], aspect=3/4,
                  c=cmap, xtitle='triangle mesh quality')
show([(contour, contour.labels('id'), msh, __doc__),
      histo,
     ], N=2, sharecam=False,
).close()