"""Compute 3D world coordinates from 2D screen pixel coordinates
(hover mouse to place the points)"""
from vedo import *

settings.default_font = "Ubuntu"
settings.use_depth_peeling = True


def func(evt):                   # this is the callback function
    i = evt.at                   # the renderer nr. which is being hit
    pt2d = evt.picked2d          # 2D screen coordinate
    # passing a list of meshes will force the points to be placed on any of them
    pt3d = plt.at(i).compute_world_coordinate(pt2d, objs=[objs[i]])
    if mag(pt3d) < 0.01:
        return
    newpt = Point(pt3d).color(i)
    txt.text(f'2D coords: {pt2d}\n3D coords: {pt3d}\nNpt = {len(plt.objects)}')
    txt.color(i)                  # update text and color on the fly
    plt.at(i).add(newpt).render() # add new point and render i


# create an empty text (to be updated in the callback)
txt = Text2D("", s=1.4, font='Brachium', c='white', bg='green8')

# create two polygonal meshes
mesh1 = TessellatedBox()
mesh2 = ParametricShape('ConicSpiral')
mesh2.c('indigo1').lc('grey9').lw(1)
objs = [mesh1, mesh2]

plt = Plotter(N=2, bg='blackboard', axes=1, sharecam=False)
plt.add_callback('mouse move', func)

plt.at(0).show(mesh1, __doc__, viewup='z')
plt.at(1).show(mesh2, txt, zoom=1.4)
plt.interactive().close()

