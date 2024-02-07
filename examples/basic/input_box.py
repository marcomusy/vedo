"""Start typing a color name for the mesh.
E.g.: pink4
(Press 'Esc' to exit)"""
from vedo import settings, dataurl, get_color_name
from vedo import Plotter, Mesh, Text2D


def kfunc(evt):
    key = evt.keypress.lower()
    field_txt = field.text().strip() # strip leading/trailing spaces

    if key == "backspace" and field_txt:
        key = ""
        field_txt = field_txt[:-1]
    elif key == "escape":
        plt.close()
        return
    elif len(key) > 1:
        return

    color_name = field_txt + key
    field.text(f"{color_name:^12}").frame(color_name, lw=8)
    mesh.color(color_name)
    msg.text(get_color_name(color_name))
    plt.render()


settings["enable_default_keyboard_callbacks"] = False

mesh = Mesh(dataurl+"magnolia.vtk").color("black").flat()

field = Text2D("black", pos="bottom-center",s=3, font="Meson", bg="k2", c="w", alpha=1)
msg = Text2D(pos="top-right", s=2, font="Quikhand", c="k1", bg="k7", alpha=1)

plt = Plotter()
plt.add_callback("key press", kfunc)
plt.show(mesh, field, msg, __doc__).close()
