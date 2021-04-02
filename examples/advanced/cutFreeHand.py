"""Interactively cut a mesh by drawing free-hand a spline in space"""
import vedo

class FreeHandCutPlotter(vedo.Plotter):

    def __init__(self, mesh, title="Mesh free-hand cutter", invert=False):
        vedo.Plotter.__init__(self, title=title)

        self.mesh = mesh
        self.invert = invert  # flip selection area inside-out
        self.click = False
        self.clicker = 5
        self.cpoints = []
        self.points = None
        self.spline = None

        self.msg  = "Left-click and hold to rotate\n"
        self.msg += "Right-click and move to draw line\n"
        self.msg += "Second right-click to stop drawing\n"
        self.msg += "Press c to clear points\n"
        self.msg += "Press s to cut mesh"
        self.txt2d = vedo.Text2D(self.msg, pos='bottom-left', font=1)
        self.txt2d.c("white").background("green4", alpha=1).frame()

        self.addCallback('KeyPress', self.onKeyPress)
        self.addCallback('RightButton', self.onRightClick)
        self.addCallback('MouseMove', self.onMouseMove)

    def onRightClick(self, evt):
        self.click = not self.click  # toggle mode
        self.txt2d.background('red5') if self.click else self.txt2d.background('green4')

    def onMouseMove(self, evt):
        self.clicker += 1
        if self.click and self.clicker > 5:
            self.clicker = 0
            self.remove([self.points, self.spline])
            cpt = self.computeWorldPosition(evt.picked2d) # make this 2d-screen point 3d
            self.cpoints.append(cpt)
            self.points = vedo.Points(self.cpoints, r=8).c('black')
            if len(self.cpoints) > 2:
                self.spline = vedo.Line(self.cpoints, closed=True).lw(5).c('red5')
                self.txt2d.background('red5')
                self.add([self.points, self.spline])

    def onKeyPress(self, evt):
        if evt.keyPressed == 's' and self.spline:       # cut mesh
            self.txt2d.background('red8').text("\n...working...\n")
            self.render()
            # Cut self.mesh with a ribbon-like surface:
            tol = self.mesh.diagonalSize()/4            # size of ribbon (not shown)
            pts = self.spline.points()
            n = vedo.fitPlane(pts, signed=True).normal  # compute normal vector to points
            rb = vedo.Ribbon(pts - tol*n, pts + tol*n, closed=True)
            mcut = self.mesh.clone().cutWithMesh(rb, invert=self.invert).extractLargestRegion()
            self.txt2d.background('red5') if self.click else self.txt2d.background('green4')
            self.txt2d.text(self.msg)                   # put back original message
            self.remove([self.mesh, self.spline, self.points]).add(mcut)
            self.mesh = mcut                            # discard old mesh by overwriting it
            self.cpoints, self.points, self.spline = [], None, None

        if evt.keyPressed == 'c':                       # clear all points
            self.remove([self.spline, self.points]).render()
            self.cpoints, self.points, self.spline = [], None, None

    def start(self):
        return self.show(self.txt2d, self.mesh, __doc__, axes=1)


######################################################################################
if __name__ == "__main__":

    vedo.settings.useParallelProjection = True
    mesh = vedo.Volume(vedo.datadir+'embryo.tif').isosurface().color('gold', 0.5)
    plt = FreeHandCutPlotter(mesh)
    plt.start()
