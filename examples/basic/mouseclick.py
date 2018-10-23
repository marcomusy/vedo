# Mouse click event example
# click of the mouse causes a call to a custom function
#
from vtkplotter import Plotter, printc


def onLeftClick(actor):
    printc('Left   button pressed on', [actor], c='g')

def onMiddleClick(actor):
    printc('Middle button pressed on', [actor], c='y')

def onRightClick(actor):
    printc('Right  button pressed on', [actor], c='r')


vp = Plotter(verbose=0)

vp.load('data/shapes/skyscraper.obj')

vp.mouseLeftClickFunction   = onLeftClick
vp.mouseMiddleClickFunction = onMiddleClick
vp.mouseRightClickFunction  = onRightClick

printc("Click object to trigger function call", invert=1, box='-')
vp.show()

