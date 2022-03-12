from vedo import dataurl, printc, precision
from vedo import Plotter, Picture, Spline, Points, Text2D

#########################################################################
class SplinePlotter(Plotter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cpoints = []
        self.points = None
        self.spline = None

    def onLeftClick(self, evt):
        if not evt.actor: return
        p = evt.picked3d + [0,0,1]
        self.cpoints.append(p)
        self.update()
        printc("Added point:", precision(p[:2],4), c='g')

    def onRightClick(self, evt):
        if evt.actor and len(self.cpoints):
            self.cpoints.pop() # pop removes from the list the last pt
            self.update()
            printc("Deleted last point", c="r")

    def update(self):
        self.remove([self.spline, self.points])  # remove old points and spline
        self.points = Points(self.cpoints).ps(10).c('purple5')
        self.points.pickable(False)  # avoid picking the same point
        if len(self.cpoints) > 2:
            self.spline = Spline(self.cpoints, closed=False).c('yellow5').lw(3)
            self.add([self.points, self.spline])
        else:
            self.add(self.points)

    def keyPress(self, evt):
        if evt.keyPressed == 'c':
            self.cpoints = []
            self.remove([self.spline, self.points]).render()
            printc("==== Cleared all points ====", c="r", invert=True)


##############################################################################
if __name__ == "__main__":

    filename = dataurl+"images/Mouse-_embryo_E11.5.jpg"
    pic = Picture(filename, channels=(0,1,2))  # keep rgb but drop alpha channel

    t = """Click to add a point
    Right-click to remove it
    Drag mouse to change constrast
    Press c to clear points"""
    instrucs = Text2D(t, pos='bottom-left', c='white', bg='green', font='Quikhand')

    plt = SplinePlotter(axes=True, bg='blackboard')
    plt.addCallback('KeyPress', plt.keyPress)
    plt.addCallback('LeftButtonPress', plt.onLeftClick)
    plt.addCallback('RightButtonPress', plt.onRightClick)
    plt.show(filename, pic, instrucs, mode='image', zoom=1.2)
    plt.close()


