from vedo import *


def func(widget, e):
    x = widget.value
    m = msh.clone()
    ids = m.find_cells_in_bounds(xbounds=(-10,x))
    m.delete_cells(ids)
    plt.remove("frog").add(m).render()


msh = Mesh("data/frog.obj").texture("data/frog.jpg")
msh.name = "frog"

plt = Plotter(axes=1)
plt.add_slider(func, xmin=-6, xmax=3, value=-6)
plt.show(msh)
plt.close()