"""Mouse click and other type of events
will trigger a call to a custom function"""
from vedo import printc, Plotter, Mesh, dataurl

printc("Click object to trigger a function call", invert=1)

# callback functions
def on_left_click(event):
    if not event.object:
        return
    printc("Left button pressed on", [event.object], c=event.object.color())
    printc(event)  # dump the full event info

def on_drag(event):
    printc(event.name, 'happened at mouse position', event.picked2d)

######################
tea = Mesh(dataurl+"teapot.vtk").c("gold")
mug = Mesh(dataurl+"mug.ply").rotate_x(90).scale(8).pos(2,0,-.7).c("red3")

plt = Plotter(axes=11)
plt.add_callback('LeftButtonPress', on_left_click)
plt.add_callback('Interaction', on_drag) # mouse dragging triggers this
plt.show(tea, mug, __doc__).close()
