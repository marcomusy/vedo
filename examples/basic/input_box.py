"""Start typing a color name then press return
E.g. pink4"""
from vedo import settings, dataurl, Plotter, Mesh

settings.enable_default_keyboard_callbacks = False

def kfunc(evt):
    global msg
    evt.keypress = evt.keypress.replace("period", ".")
    if evt.keypress == "BackSpace" and msg:
        msg = msg[:-1]
        evt.keypress = ''
    elif evt.keypress == "Return":
        bfunc(0)
        return
    elif evt.keypress == "Escape":
        plt.close()

    if len(evt.keypress) > 1:
        return

    msg += f"{evt.keypress}"
    bu.text(msg)
    plt.render()

def bfunc(obj, ename=""):
    mesh.color(msg)
    plt.render()


plt = Plotter(axes=1)
plt.remove_callback("CharEvent") # might be needed

msg = ""
plt.add_callback("key press", kfunc)

bu = plt.add_button(
    bfunc,
    pos=(0.5, 0.1),  # x,y fraction from bottom left corner
    states=["input box"],
    c=["w"],
    bc=["dg"],        # colors of states
    font="courier",   # arial, courier, times
    size=45,
    bold=True,
)

mesh = Mesh(dataurl+"magnolia.vtk").c("v").flat()

plt.show(mesh, __doc__).close()

