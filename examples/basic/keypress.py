# This example shows how to implement a custom function that is triggered by 
# pressing a keyboard button when the rendering window is in interactive mode.
# Every time a key is pressed a cube is randomly added to the scene 
# and some info is printed.
# 
from vtkplotter import Plotter, printc 
from numpy.random import randn

##############################################################################
def myfnc(key, vplt):
    printc('You just pressed: '+key, c='r')
    if key != 'c': return

    cb = vplt.cube(pos=randn(3,1), length=.2, alpha=.5, legend='cube') 
    vplt.render(cb)
    
    printc('current #actors : '+str(len(vplt.actors)), c=2)
    printc('clicked renderer: '+str(vplt.clickedRenderer), c=2) 
    if vplt.clickedActor: 
        printc('clicked actor   : '+vplt.clickedActor.legend, c=4)
        printc('clicked 3D point coordinates: '+str(vplt.picked3d), c=4)
    
##############################################################################

vp = Plotter(verbose=0)

vp.keyPressFunction = myfnc # make it known to Plotter class

vp.load('data/shapes/bunny.obj', alpha=.5).normalize()

printc('\nPress c to execute myfnc()', c=1)

vp.show()