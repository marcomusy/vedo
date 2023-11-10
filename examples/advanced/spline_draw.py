from vedo import dataurl, Image 
from vedo.applications import SplinePlotter  # ready to use class!

pic = Image(dataurl + "images/embryo.jpg")

plt = SplinePlotter(pic)
plt.show(mode="image", zoom='tightest')

if plt.line:
    print("Npts =", len(plt.cpoints), "NSpline =", plt.line.npoints)


#####################################################################
# This is a simplified version of vedo.applications.SplinePlotter 
#####################################################################
# from vedo import printc, precision, Plotter, Spline, Points, Text2D
#
# class MySplinePlotter(Plotter):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.cpoints = []
#         self.points = None
#         self.spline = None

#     def on_left_click(self, evt):
#         if not evt.object:
#             return
#         p = evt.picked3d + [0, 0, 1]
#         self.cpoints.append(p)
#         self.update()
#         printc("Added point:", precision(p[:2], 4), c="g")

#     def on_right_click(self, evt):
#         if evt.object and len(self.cpoints) > 0:
#             self.cpoints.pop()  # pop removes the last point
#             self.update()
#             printc("Deleted last point", c="r")

#     def on_key_press(self, evt):
#         if evt.keypress == "c":
#             self.cpoints = []
#             self.remove(self.spline, self.points).render()
#             printc("==== Cleared all points ====", c="r", invert=True)

#     def update(self):
#         self.remove([self.spline, self.points])  # remove old points and spline
#         self.points = Points(self.cpoints).ps(10).c("purple5")
#         self.points.pickable(False)  # avoid picking the same point
#         if len(self.cpoints) > 2:
#             self.spline = Spline(self.cpoints, closed=False).c("yellow5").lw(3)
#             self.add(self.points, self.spline)
#         else:
#             self.add(self.points)

# plt = MySplinePlotter(axes=True, bg="blackboard")
# plt.add_callback("key press", plt.on_key_press)
# plt.add_callback("left mouse click", plt.on_left_click)
# plt.add_callback("right mouse click", plt.on_right_click)
# plt.show(pic, mode="image", zoom=1.2)
# plt.close()
