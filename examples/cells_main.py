from __future__ import division, print_function
from cell import Cell
import plotter 

vp = plotter.vtkPlotter(verbose=0, interactive=0, axes=3)

c1 = Cell('cr', 'r', [1,0,0])
c2 = Cell('cg', 'g', [0,1,0])
c3 = Cell('cb', 'b', [0,0,1])
cells = [c1,c2,c3]
for c in cells: c.build(vp)

#time loop
pb = vp.ProgressBar(0,10, 0.01, c=1)
for t in pb.range():
    newcells = []    
    for c in cells:
        if c.shouldDie(t): 
            c.remove()
            continue
        newcells.append(c)
        if c.shouldDivide(t): 
            newc = c.split(t)
            newcells.append(newc)
    cells = newcells

    #move cells around following some potential
    for i, ci in enumerate(cells):
        vtot = [0,0,0]
        for j, cj in enumerate(cells):
            if i==j : continue
            v, d = ci.dist(cj)
            f = -v * (d-0.5)/100
            vtot += f 
        ci.addPos(vtot)

    # use show() instead of render() because there can be
    # various new actors in the scene
    vp.show(resetcam=1)   
    vp.camera.Azimuth(.4)   # move camera at each loop
    vp.camera.Elevation(.01) 
    pb.print('Ncells='+str(len(cells)))
        
vp.show(interactive=1)
