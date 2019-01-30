'''
This example shows how to implement a custom function that is triggered by 
 pressing a keyboard button when the rendering window is in interactive mode.
 Every time a key is pressed the picked point of the mesh is used 
 to add a sphere and some info is printed.
'''
from vtkplotter import Plotter, printc, sphere, text


##############################################################################
def myfnc(key):
    if not vp.clickedActor or key != 'c':
        printc('click an actor and press c.', c='r')
        return
    printc('clicked actor   :', vp.clickedActor.legend, c=4)
    printc('clicked 3D point:', vp.picked3d, c=4)
    printc('clicked renderer:', vp.clickedRenderer, c=2) 
    
    vp.add(sphere(pos=vp.picked3d, r=.05, c='v'))
    vp.show()
    
    
##############################################################################

vp = Plotter(verbose=0)

vp.keyPressFunction = myfnc # make it known to Plotter class

vp.load('data/shapes/bunny.obj').normalize()

printc('\nPress c to execute myfnc()', c=1)

vp.add(text(__doc__))

vp.show()