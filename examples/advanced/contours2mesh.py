"""Form a surface mesh by joining countour lines"""
from vedo import Circle, Ribbon, merge, show

cs = []
for i in range(-10, 10):
    r = 10 / (i * i + 10)
    c = Circle(r=r).rotateY(i*2).z(i/10).x(i/20)
    c.lineWidth(3).lineColor('blue5')
    cs.append(c)

# create the mesh by merging the ribbon strips
rbs = []
for i in range(len(cs) - 1):
    rb = Ribbon(cs[i], cs[i+1], closed=True, res=(150,5))
    rbs.append(rb)
mesh = merge(rbs).clean().cap().color('limegreen')

cs.append(__doc__)

show([cs, mesh], N=2, axes=1, elevation=-40, bg2='lb').close()
