"""Create a scatter plot to overlay
three different distributions"""
from vedo import *
from numpy.random import randn

### first cloud in blue, place it at z=0:
x = randn(2000) * 3
y = randn(2000) * 2
pts1 = Points([x,y], c="blue", alpha=0.5).z(0.0)
bra1 = Brace([-7,-8], [7,-8],
             comment='whole population', s=0.4, c='b')

### second cloud in red
x = randn(1200) + 4
y = randn(1200) + 2
pts2 = Points([x,y], c="red", alpha=0.5).z(0.1)
bra2 = Brace([8,2,0.3], [6,5,0.3], comment='red zone',
             angle=180, justify='bottom-center', c='r')

### third cloud with a black marker
x = randn(20) + 4
y = randn(20) - 4
mark = Marker('*', s=0.25)
pts3 = Glyph([x,y], mark, c='k').z(0.2)
bra3 = Brace([8,-6], [8,-2], comment='my stars').z(0.3)

# some text message
msg = Text3D("preliminary\nresults!", font='Quikhand', s=1.5)
msg.c('black').rotateZ(20).pos(-10,3,.2)

show(pts1, pts2, pts3, msg, bra1, bra2, bra3, __doc__,
     axes=1, zoom=1.2, viewup="2d",
).close()
