"""Start typing a color name then press return
E.g. pink4"""
from vedo import settings, dataurl, Plotter, Mesh

settings.enableDefaultKeyboardCallbacks = False

def kfunc(evt):
    global msg
    evt.keyPressed = evt.keyPressed.replace("period", ".")
    if evt.keyPressed == "BackSpace" and msg:
        msg = msg[:-1]
        evt.keyPressed = ''
    elif evt.keyPressed == "Return":
        bfunc()
        return
    elif evt.keyPressed == "Escape":
        plt.close()

    if len(evt.keyPressed) > 1:
        return

    msg += f"{evt.keyPressed}"
    bu.actor.SetInput(msg)
    plt.render()

def bfunc():
    mesh.color(msg)
    plt.render()


plt = Plotter(axes=1)
plt.interactor.RemoveObservers("CharEvent") # might be needed

msg = ""
plt.addCallback("key press", kfunc)

bu = plt.addButton(
    bfunc,
    pos=(0.7, 0.05),  # x,y fraction from bottom left corner
    states=["input box"],
    c=["w"],
    bc=["dg"],        # colors of states
    font="courier",   # arial, courier, times
    size=45,
    bold=True,
)

mesh = Mesh(dataurl+"magnolia.vtk").c("v").flat()

plt.show(mesh, __doc__).close()

