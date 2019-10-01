"""
Sharpe lab animated logo.
"""
from vtkplotter import *

exa = Polygon().scale(4.1).pos(5.25, 4.8, 0).off()
box = Box([10, 5, 0], 20, 20, 15).alpha(0)
his = hexHistogram([-1, 1], [-1, 1]).getActors()

exah, cmh = [], []
for h in his:
    cm = h.centerOfMass()
    if exa.isInside(cm):
        h.c('green').shrink(0.94)
        exah.append(h)
        cmh.append(cm)
exah[13].c('red')

t1 = Text("Sharpe Lab",     (9.1, 5.0, -0.1), c="k")
t2 = Text("EMBL Barcelona", (9.2, 3.4, -0.1), c="dg")

for ti in reversed(range(100)):
    t = ti / 100.
    for j, h in enumerate(exah):
         cx, cy, _ = cmh[j] - [4,5,0]
         h.pos(cos(cy*t) *t*2, sin(cx*t)*t*2, t*cx/2).alpha((1-t)**3)
         t1.alpha((1-t)**4)
         t2.scale([(1-t)*0.67, (1-t)*0.75, (1-t)*0.75]).alpha((1-t)**2)
    show(box, exa, exah, t1, t2,
         resetcam=0, elevation=0, bg="w", axes=0, interactive=0)

interactive()
