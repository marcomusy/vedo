"""Mouse click and other type of events
will trigger a call to a custom function"""
from vedo import printc, Plotter, Cube, datadir

printc("Click object to trigger a function call", invert=1)


def onLeftClick(mesh):
    printc("Left button pressed on", [mesh], c=mesh.color())

def onEvent(iren, event):
    printc(event, 'happened at position', iren.GetEventPosition())


plt = Plotter(axes=11)

# load some mesh:
plt.load(datadir+"teapot.vtk").c("gold")
plt.load(datadir+"mug.ply").rotateX(90).scale(8).pos(2,0,-.7).c("silver")

# simplified way to create an observer with ready access to the clicked mesh:
plt.mouseLeftClickFunction = onLeftClick

# a more general way, see:
# https://vtk.org/doc/nightly/html/classvtkCommand.html
# E.g.: KeyPressEvent, RightButtonPressEvent, MouseMoveEvent, ..etc
plt.addCallback('InteractionEvent', onEvent)

plt += __doc__
plt.show()
