"""Interactively cut a mesh by drawing free-hand a spline in space"""
# The tool can also be invoked from command line e.g.: > vedo --edit mesh.ply
import vedo
from vedo.applications import FreeHandCutPlotter

#### This class is a simplified version of the above, shown here as an example: #######
#
# class FreeHandCutPlotter(vedo.Plotter):
#     def __init__(self, mesh):
#         vedo.Plotter.__init__(self)
#         self.mesh = mesh
#         self.drawmode = False
#         self.cpoints = []
#         self.points = None
#         self.spline = None
#         self.msg  = "Right-click and move to draw line\n"
#         self.msg += "Second right-click to stop drawing\n"
#         self.msg += "Press z to cut mesh"
#         self.txt2d = vedo.Text2D(self.msg, pos='top-left', font="Bongas")
#         self.txt2d.c("white").background("green4", alpha=1)
#         self.addCallback('KeyPress', self.onKeyPress)
#         self.addCallback('RightButton', self.onRightClick)
#         self.addCallback('MouseMove', self.onMouseMove)

#     def onRightClick(self, evt):
#         self.drawmode = not self.drawmode  # toggle mode

#     def onMouseMove(self, evt):
#         if self.drawmode:
#             self.remove([self.points, self.spline])
#             cpt = self.computeWorldPosition(evt.picked2d) # make this 2d-screen point 3d
#             self.cpoints.append(cpt)
#             self.points = vedo.Points(self.cpoints, r=8).c('black')
#             if len(self.cpoints) > 2:
#                 self.spline = vedo.Line(self.cpoints, closed=True).lw(5).c('red5')
#                 self.add([self.points, self.spline])

#     def onKeyPress(self, evt):
#         if evt.keyPressed == 'z' and self.spline:       # cut mesh with a ribbon-like surface
#             vedo.printc("Cutting the mesh please wait..", invert=True)
#             tol = self.mesh.diagonalSize()/2            # size of ribbon
#             pts = self.spline.points()
#             n = vedo.fitPlane(pts, signed=True).normal  # compute normal vector to points
#             rib = vedo.Ribbon(pts - tol*n, pts + tol*n, closed=True)
#             self.mesh.cutWithMesh(rib)
#             self.remove([self.spline, self.points]).render()
#             self.cpoints, self.points, self.spline = [], None, None

#     def start(self, **kwargs):
#         return self.show(self.txt2d, self.mesh, **kwargs)
#
######################################################################################
vedo.settings.useParallelProjection = True  # to avoid perspective artifacts

msh = vedo.Volume(vedo.dataurl+'embryo.tif').isosurface().color('gold', 0.25) # Mesh

plt = FreeHandCutPlotter(msh).addHoverLegend()
#plt.init(some_list_of_initial_pts) #optional!
plt.start(axes=1, bg2='lightblue').close()
