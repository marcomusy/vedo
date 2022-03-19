# Morph one shape into another interactively (can work in 3d too!)
#
from vedo import Plotter, Axes, dataurl, load, printc, merge
from vedo.shapes import Text2D, Points, Lines, Arrows2D, Grid

class Morpher:

    def __init__(self, mesh1, mesh2, n):  ############################### init

        self.n = n  # desired nr. of intermediate shapes
        self.mode = '2d'
        self.mesh1 = mesh1
        self.mesh2 = mesh2
        self.merged_meshes = merge(mesh1, mesh2)
        self.mesh1.lw(4).c('grey2').pickable(False)
        self.mesh2.lw(4).c('grey1').pickable(False)

        self.arrow_starts = []
        self.arrow_stops  = []
        self.dottedln = None
        self.toggle = False
        self.instructions = ("Click to add arrows interactively on the left panel\n"
                             "right-click to remove last arrow. Then press:\n"
                             "- m to morph the plane\n"
                             "- c to clear\n"
                             "- g to generate interpolation")
        self.msg1 = Text2D(self.instructions, pos='top-left', font="VictorMono", bg='g2', alpha=0.6)
        self.msg2 = Text2D('[output will show here]', pos='top-left', font="VictorMono")

        sz = self.merged_meshes.diagonalSize()
        self.plane1 = Grid(s=[sz,sz], res=[50,50]).pos(self.merged_meshes.centerOfMass())
        self.plane1.wireframe(False).alpha(1).lineWidth(0.1).c('white').lc('grey5')
        self.plane2 = self.plane1.clone().pickable(False)

        self.plotter = Plotter(N=2, bg='light blue', size=(2000,1000))
        self.plotter.addCallback('left click', self.onleftclick)
        self.plotter.addCallback('right click', self.onrightclick)
        self.plotter.addCallback('key press', self.onkeypress)

    def start(self):  ################################################ show stuff
        paxes = Axes(self.plane1, xyGrid=0, textScale=0.6)
        self.plotter.at(0).show(self.plane1, paxes, self.msg1, self.mesh1, self.mesh2)
        self.plotter.at(1).show(self.plane2, self.msg2, mode='image')
        if len(self.arrow_starts):
            self.draw(True)
            self.draw(False)
            self.msg1.text(self.instructions)
        self.plotter.show(interactive=True, zoom=1.3).close()

    def draw(self, toggle=None):  #################################### update scene
        if toggle is None:
            toggle = self.toggle
        if toggle:
            self.msg1.text("Choose start point or press:\nm to morph the shapes\ng to interpolate")
            self.plotter.at(0).remove("displacementArrows")
            if not len(self.arrow_starts): return
            arrows = Arrows2D(self.arrow_starts, self.arrow_stops, c='red4')
            arrows.name = "displacementArrows"
            self.plotter.add(arrows)
        else:
            self.msg1.text("Click to choose an end point")
            self.plotter.at(0).remove("displacementPoints")
            points = Points(self.arrow_starts, r=15, c='green3', alpha=0.5)
            points.name = "displacementPoints"
            self.plotter.add(points)

    def onleftclick(self, evt):  ############################################ add points
        msh = evt.actor
        if not msh or msh.name!="Grid": return
        pt = self.merged_meshes.closestPoint(evt.picked3d) # get the closest pt on the line
        self.arrow_stops.append(pt) if self.toggle else self.arrow_starts.append(pt)
        self.draw()
        self.toggle = not self.toggle

    def onrightclick(self, evt):  ######################################## remove points
        if not self.arrow_starts: return
        self.arrow_starts.pop()
        if not self.toggle:
            self.arrow_stops.pop()
        self.plotter.at(0).clear().addRendererFrame()
        self.plotter.add([self.plane1, self.msg1, self.mesh1, self.mesh2], render=False)
        self.draw(False)
        self.draw(True)

    def onkeypress(self, evt):  ###################################### MORPH & GENERATE
        if evt.keyPressed == 'm': ##--------- morph mesh1 based on the existing arrows
            if len(self.arrow_starts) != len(self.arrow_stops):
                printc("You must select your end point first!", c='y')
                return

            output = [self.mesh1.clone().c('grey4'), self.mesh2, self.msg2]
            warped_plane = self.plane1.clone().pickable(False)
            warped_plane.warp(self.arrow_starts, self.arrow_stops, mode=self.mode)
            output.append(warped_plane + Axes(warped_plane, xyGrid=0, textScale=0.6))

            mw = self.mesh1.clone().applyTransform(warped_plane.transform).c('red4')
            output.append(mw)

            T_inv = warped_plane.transform.GetInverse()
            a = Points(self.arrow_starts, r=10).applyTransform(warped_plane.transform)
            b = Points(self.arrow_stops,  r=10).applyTransform(warped_plane.transform)
            self.dottedln = Lines(a,b, res=self.n).applyTransform(T_inv).pointSize(5)
            output.append(self.dottedln)

            self.msg1.text(self.instructions)
            self.msg2.text("Morphed output:")
            self.plotter.at(1).clear().addRendererFrame().add(output)

        elif evt.keyPressed == 'g':  ##------- generate intermediate shapes
            if not self.dottedln:
                return
            intermediates = []
            allpts = self.dottedln.points()
            allpts = allpts.reshape(len(self.arrow_starts), self.n+1, 3)
            for i in range(self.n + 1):
                pi = allpts[:,i,:]
                m_nterp = self.mesh1.clone().warp(self.arrow_starts, pi, mode=self.mode).c('b3').lw(1)
                intermediates.append(m_nterp)
            self.msg2.text("Morphed output + Interpolation:")
            self.plotter.at(1).add(intermediates)
            self.dottedln = None

        elif evt.keyPressed == 'c':  ##------- clear all
            self.arrow_starts = []
            self.arrow_stops  = []
            self.toggle = False
            self.dottedln = None
            self.msg1.text(self.instructions)
            self.msg2.text("[output will show here]")
            self.plotter.at(0).clear()
            self.plotter.add([self.plane1, self.msg1, self.mesh1, self.mesh2])
            self.plotter.at(1).clear().addRendererFrame()
            self.plotter.add([self.plane2, self.msg2])


######################################################################################## MAIN
if __name__ == "__main__":
    outlines = load(dataurl+"timecourse1d.npy") # load a set of 2d shapes
    mesh1 = outlines[25]
    mesh2 = outlines[35].scale(1.3).shift(-2,0,0)
    morpher = Morpher(mesh1, mesh2, 10)  # generate 10 intermediate outlines
    morpher.start()


