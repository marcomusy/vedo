"""Create an animated logo"""
from vedo import *
from vedo.pyplot import histogram

settings.use_parallel_projection = True

exa = Polygon().scale(4.1).pos(5.25, 4.8, 0).off()
his = histogram([-1, 1], [-1, 1], mode='hex').unpack()

exah, cmh = [], []
for h in his:
    cm = h.center_of_mass()
    if exa.is_inside(cm):
        h.c('green').shrink(0.9).add_shadow(plane='z', point=-.4)
        exah.append(h)
        cmh.append(cm)

v1 = vector(9.4, 5.2, 0)
v2 = vector(9.4, 2.7, 0)
t1 = Text3D("EMBL",  v1, c="k", font="VTK", s=1.5, depth=0)
t2 = Text3D("European Molecular\nBiology Laboratory", v2, font="VTK", vspacing=1.75, c="dg", s=0.6)

plt = show(exa, exah, t1, t2, axes=0, interactive=0, elevation=-50)
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
    plt.show(exa, exah, t1, t2, resetcam=0, elevation=t, azimuth=-0.02)

plt.interactive()
