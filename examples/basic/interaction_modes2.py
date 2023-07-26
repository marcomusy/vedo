"""Use the mouse to select objects and vertices in a mesh.
Middle-click and drag to interact with the scene."""
from vedo import *

settings.enable_default_mouse_callbacks = False

def mode_select(objs):
    print("Selected objects:", objs)
    d0 = mode.start_x, mode.start_y  # display coords
    d1 = mode.end_x,   mode.end_y

    frustum = plt.pick_area(d0, d1)
    infru = frustum.inside_points(mesh, return_ids=False)
    color = np.random.randint(0, 10)
    infru.ps(10).c(color)
    plt.add(frustum, infru).render()

mesh = Mesh(dataurl+"cow.vtk").c("k5").lw(1)

mode = interactor_modes.BlenderStyle()
mode.callback_select = mode_select

plt = Plotter()
plt.show(mesh, __doc__, axes=1, mode=mode)