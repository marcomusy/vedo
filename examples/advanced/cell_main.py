# Simulation of bacteria types that divide at a given rate
# As they divide they occupy more and more space
from __future__ import division, print_function
from vtkplotter import Plotter, ProgressBar
from cell import Cell, Colony

vp = Plotter(verbose=0, interactive=0, axes=3)

# place vtkCamera at a specific position 
# (get these numbers by pressing Shift-C)
vp.camera.SetPosition( [2.5, 2.5, 5.5] )
vp.camera.SetFocalPoint( [0.4, 0.4, 0.4] )
vp.camera.SetParallelScale( 1.8 )
vp.camera.SetViewUp( [-0.1, 1, -0.3] )

# Let's start with creating 3 colonies of 1 cell each 
# of types: red, green and blue, in different positions in space
# and with 3 differents rates of division (tdiv in hours)
c1 = Colony( [ Cell([1,0,0], tdiv= 8) ], c='b')
c2 = Colony( [ Cell([0,1,0], tdiv= 9) ], c='g')
c3 = Colony( [ Cell([0,0,1], tdiv=10) ], c='r')
colonies = [c1, c2, c3]

# time goes from 0 to 90 hours 
pb = ProgressBar(0, 90, step=0.1, c=1)
for t in pb.range():
    msg = '[Nb,Ng,Nr,t] = '
    vp.actors = [] # clean up the list of actors

    for colony in colonies:

        newcells = []    
        for cell in colony.cells:

            if cell.dieAt(t): continue
            if cell.divideAt(t): 
                newc = cell.split() # make daughter cell
                vp.line(cell.pos, newc.pos, c='k', lw=3, alpha=.5)
                newcells.append(newc)
            newcells.append(cell)
        colony.cells = newcells

        pts = [c.pos for c in newcells] # draw all points at once
        vp.points(pts, c=colony.color, r= 5, alpha=.80) # nucleus
        vp.points(pts, c=colony.color, r=15, alpha=.05) # halo
        msg += str(len(colony.cells)) + ','

    pb.print(msg+str(int(t)))
    vp.show(resetcam=0)

# draw the oriented ellipsoid that contains 50% of the cells
for colony in colonies: 
    pts = [c.pos for c in colony.cells]
    vp.pca(pts, pvalue=0.5, c=colony.color, pcaAxes=0, alpha=.3,
           legend='1/rate='+str(colony.cells[0].tdiv)+'h')
vp.show(resetcam=0, interactive=1)
