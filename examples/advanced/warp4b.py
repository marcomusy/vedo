"""Morphological alignment of 3D surfaces.
Pick a point on the source surface, 
then pick the corresponding point on the target surface.
Pick at least 4 point pairs.
Press c to clear the selection.
Press d to delete the last selection.
Press q to quit."""
from vedo import Plotter, Mesh, Points, Text2D, Axes, settings, dataurl

################################################
settings.default_font = "Calco"
settings.enable_default_mouse_callbacks = False

################################################
def update():
    source_pts = Points(sources, r=12, c="purple5")
    target_pts = Points(targets, r=12, c="purple5")
    source_pts.name = "source_pts"
    target_pts.name = "target_pts"
    slabels = source_pts.labels2d("id", c="purple3")
    tlabels = target_pts.labels2d("id", c="purple3")
    slabels.name = "source_pts"
    tlabels.name = "target_pts"
    plt.at(0).remove("source_pts").add(source_pts, slabels)
    plt.at(1).remove("target_pts").add(target_pts, tlabels)
    plt.render()

    if len(sources) == len(targets) and len(sources) > 3:
        warped = source.clone().warp(sources, targets)
        warped.name = "warped"
        # wpoints = points.clone().apply_transform(warped.transform)
        plt.at(2).remove("warped").add(warped)
        plt.render()

def click(evt):
    if evt.object == source:
        sources.append(evt.picked3d)
        source.pickable(False)
        target.pickable(True)
        msg0.text("--->")
        msg1.text("now pick a target point")
    elif evt.object == target:
        targets.append(evt.picked3d)
        source.pickable(True)
        target.pickable(False)
        msg0.text("now pick a source point")
        msg1.text("<---")
    update()

def keypress(evt):
    global sources, targets
    if evt.keypress == "c":
        sources.clear()
        targets.clear()
        plt.at(0).remove("source_pts")
        plt.at(1).remove("target_pts")
        plt.at(2).remove("warped")
        msg0.text("CLEARED! Pick a point here")
        msg1.text("")
        source.pickable(True)
        target.pickable(False)
        update()
    elif evt.keypress == "d":
        n = min(len(sources), len(targets))
        sources = sources[:n-1]
        targets = targets[:n-1]
        msg0.text("Last point deleted! Pick a point here")
        msg1.text("")
        source.pickable(True)
        target.pickable(False)
        update()
    elif evt.keypress == "q":
        plt.close()
        exit()
        
################################################
target = Mesh(dataurl + "290.vtk").cut_with_plane(origin=(1,0,0))
target.pickable(False).c("yellow5")
ref = target.clone().pickable(False).alpha(0.75)

source = Mesh(dataurl + "limb_surface.vtk")
source.pickable(True).c("k5").alpha(1)

clicked = []
sources = []
targets = []

msg0 = Text2D("Pick a point on the surface", c='white', alpha=1, bg="blue4", pos="bottom-center")
msg1 = Text2D("", c='white', bg="blue4", alpha=1, pos="bottom-center")

plt = Plotter(N=3, axes=0, sharecam=0, size=(2490, 810))
plt.add_callback("click", click)
plt.add_callback("keypress", keypress)
plt.at(0).show(source, msg0, __doc__)
plt.at(1).show(f"Reference {target.filename}", msg1, target)
cam1 = plt.camera  # will share the same camera btw renderers 1 and 2
plt.at(2).show("Morphing Output", ref, Axes(ref), camera=cam1, bg="k9")
plt.interactive()
plt.close()

