"""Generate a polygonal Mesh
from a contour line."""
from vedo import *
from vedo.pyplot import histogram

shapes = load(datadir+'timecourse1d.npy') # list of lines
shape = shapes[56].mirror().rotateZ(-90)

msh = shape.tomesh()       # Generate the Mesh from the line
msh.smoothLaplacian()      # make the triangles more uniform
msh.addQuality().addScalarBar3D() # Measure triangle quality

contour = Line(shape).c('red4').lw(5)
histo = histogram(msh.getCellArray('Quality'), aspect=3/4,
                  c='RdYlBu', xtitle='triangle mesh quality')
show([
      [contour, contour.labels('id'), msh, __doc__],
      histo,
     ], N=2, sharecam=False,
)