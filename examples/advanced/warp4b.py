# Same as warp4b.py but using the applications.MorphPlotter class
from vedo import Mesh, settings, dataurl
from vedo.applications import MorphPlotter


####################################################################################
# THIS IS IMPLEMENTED IN vedo.applications.MorphPlotter, shown here for reference
####################################################################################
# from vedo import Plotter, Points, Text2D, Axes
# class MorphPlotter(Plotter):
#     
#     def __init__(self, source, target, **kwargs):
#         kwargs.update(dict(N=3, sharecam=0))
#         super().__init__(**kwargs)
#
#         self.source = source.pickable(True)
#         self.target = target.pickable(False)
#         self.clicked = []
#         self.sources = []
#         self.targets = []
#         self.msg0 = Text2D("Pick a point on the surface",
#                            pos="bottom-center", c='white', bg="blue4", alpha=1)
#         self.msg1 = Text2D(pos="bottom-center", c='white', bg="blue4", alpha=1)
#         instructions = (
#             "Morphological alignment of 3D surfaces.\n"
#             "Pick a point on the source surface, then\n"
#             "pick the corresponding point on the target surface\n"
#             "Pick at least 4 point pairs. Press:\n"
#             "- c to clear the selection.\n"
#             "- d to delete the last selection.\n"
#             "- q to quit."
#         )
#         self.instructions = Text2D(instructions, s=0.7, bg="blue4", alpha=0.1)
#         self.at(0).add(source, self.msg0, self.instructions).reset_camera()
#         self.at(1).add(f"Reference {target.filename}", self.msg1, target)
#         cam1 = self.camera  # save camera at 1
#         self.at(2).add("Morphing Output", target, Axes(target)).background("k9")
#         self.camera = cam1  # use the same camera of renderer1
#
#         self.callid1 = self.add_callback("on key press", self.on_keypress)
#         self.callid2 = self.add_callback("on click", self.on_click)
#         self._interactive = True
#
#     def update(self):
#         source_pts = Points(self.sources).color("purple5").ps(12)
#         target_pts = Points(self.targets).color("purple5").ps(12)
#         source_pts.name = "source_pts"
#         target_pts.name = "target_pts"
#         slabels = self.source_pts.labels2d("id", c="purple3")
#         tlabels = self.target_pts.labels2d("id", c="purple3")
#         slabels.name = "source_pts"
#         tlabels.name = "target_pts"
#         self.at(0).remove("source_pts").add(source_pts, slabels)
#         self.at(1).remove("target_pts").add(target_pts, tlabels)
#         self.render()
#
#         if len(self.sources) == len(self.targets) and len(self.sources) > 3:
#             warped = self.source.clone().warp(self.sources, self.targets)
#             warped.name = "warped"
#             self.at(2).remove("warped").add(warped)
#             self.render()
#
#     def on_click(self, evt):
#         if evt.object == source:
#             self.sources.append(evt.picked3d)
#             self.source.pickable(False)
#             self.target.pickable(True)
#             self.msg0.text("--->")
#             self.msg1.text("now pick a target point")
#         elif evt.object == self.target:
#             self.targets.append(evt.picked3d)
#             self.source.pickable(True)
#             self.target.pickable(False)
#             self.msg0.text("now pick a source point")
#             self.msg1.text("<---")
#         self.update()
#
#     def on_keypress(self, evt):
#         if evt.keypress == "c":
#             self.sources.clear()
#             self.targets.clear()
#             self.at(0).remove("source_pts")
#             self.at(1).remove("target_pts")
#             self.at(2).remove("warped")
#             self.msg0.text("CLEARED! Pick a point here")
#             self.msg1.text("")
#             self.source.pickable(True)
#             self.target.pickable(False)
#             self.update()
#         elif evt.keypress == "d":
#             n = min(len(self.sources), len(self.targets))
#             self.sources = self.sources[:n-1]
#             self.targets = self.targets[:n-1]
#             self.msg0.text("Last point deleted! Pick a point here")
#             self.msg1.text("")
#             self.source.pickable(True)
#             self.target.pickable(False)
#             self.update()            
################################################################################

settings.default_font = "Calco"
settings.enable_default_mouse_callbacks = False

source = Mesh(dataurl+"limb_surface.vtk").color("k5")
source.rotate_y(90).rotate_z(-60).rotate_x(40)
target = Mesh(dataurl+"290.vtk").cut_with_plane(origin=(1,0,0)).color("yellow5")

plt = MorphPlotter(source, target, size=(2490, 850))
plt.show()
plt.close()

