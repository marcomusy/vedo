"""Start typing a color name then press return
E.g. pink4"""
from vedo import settings, dataurl, Plotter, Mesh

settings.enableDefaultKeyboardCallbacks = False

def kfunc(evt):
    global msg
    evt.keypress = evt.keypress.replace("period", ".")
    if evt.keypress == "BackSpace" and msg:
        msg = msg[:-1]
        evt.keypress = ''
    elif evt.keypress == "Return":
        bfunc()
        return
    elif evt.keypress == "Escape":
        plt.close()

    if len(evt.keypress) > 1:
        return

    msg += f"{evt.keypress}"
    bu.actor.SetInput(msg)
    plt.render()

def bfunc():
    mesh.color(msg)
    plt.render()


plt = Plotter(axes=1)
plt.interactor.RemoveObservers("CharEvent") # might be needed

msg = ""
plt.add_callback("key press", kfunc)

bu = plt.add_button(
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

