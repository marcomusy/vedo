"""Create an animated logo"""
from vedo import *
from vedo.pyplot import histogram

exa = Polygon().scale(4.1).pos(5.25, 4.8, 0).off()
his = histogram([-1, 1], [-1, 1], mode='hex').unpack()

exah, cmh = [], []
for h in his:
    cm = h.centerOfMass()
    if exa.isInside(cm):
        h.c('green').shrink(0.9).addShadow(z=-.4)
        exah.append(h)
        cmh.append(cm)

v1 = vector(9.4, 5.2, 0)
v2 = vector(9.4, 2.7, 0)
t1 = Text("EMBL",  v1, c="k",  s=1.5, depth=0)
t2 = Text("European Molecular\nBiology Laboratory", v2, c="dg", s=0.6, depth=0)

show(exa, exah, t1, t2, axes=0, interactive=0, elevation=-50)
for ti in reversed(range(100)):
    t = ti / 100.
    for j, h in enumerate(exah):
        cx, cy, _ = cmh[j] - [4,5,0]
        x = t*-4+(1-t)*6
        g = exp(-(cx-x)**2/.5)*2
        h.z(g)
        t1.pos([sin(t)*-10, 0, -0.41] + v1).alpha((1-t)**2)
        t2.pos([sin(t)*-15, 0, -0.41] + v2).alpha((1-t)**4)
        exah[13].c('red')
    show(exa, exah, t1, t2, resetcam=0, elevation=t, azimuth=-0.02)

interactive()
