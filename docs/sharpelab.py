"""
Create an animated logo.
"""
from vtkplotter import *

exa = Polygon().scale(4.1).pos(5.25, 4.8, 0).off()
box = Box([10, 5, 0], 20, 20, 15).alpha(0)
his = histogram2D([-1, 1], [-1, 1]).getActors()

exah, cmh = [], []
for h in his:
    cm = h.centerOfMass()
    if exa.isInside(cm):
        h.shrink(0.94)#.addShadow(z=-1)
        exah.append(h)
        cmh.append(cm)
exah[13].c('red')

v1 = vector(9.1, 5.3, -0.1)
v2 = vector(9.2, 3.4, -0.1)
t1 = Text("Sharpe Lab",  v1, c="k").scale([.9,1,1])
t2 = Text("EMBL Barcelona", v2, c="dg")

Plotter(bg="w", axes=0, interactive=0)
def run(rng):
	for ti in rng:
	    t = ti / 100.
	    for j, h in enumerate(exah):
	        cx, cy, _ = cmh[j] - [4,5,0]
	        h.pos(cos(cy*t) *t*2, sin(cx*t)*t*2, t*cx/2).alpha((1-t)**3)
	        #h.shadow.alpha(t**4)
	        t1.alpha((1-t)**4)
	        t2.scale([(1-t)*0.52, (1-t)*0.6, (1-t)*0.6]).alpha((1-t)**2)
	    show(box, exa, exah, t1, t2, resetcam=0, elevation=0)
import time
run(reversed(range(100)))
time.sleep(2)
run(range(100))
time.sleep(.2)
run(reversed(range(100)))
interactive()
