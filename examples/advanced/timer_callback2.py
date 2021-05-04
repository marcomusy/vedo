# Create a class which wraps the vedo.Plotter class and adds a timer callback
# Credits: Nicolas Antille, https://github.com/nantille
# Check out the simpler example: timer_callback1.py
import vedo


class Viewer:

    def __init__(self, *args, **kwargs):
        self.dt = kwargs.pop("dt", 100) # update every dt milliseconds
        self.timerId = None
        self.isplaying = False
        self.counter = 0 # frame counter

        self.plotter = vedo.Plotter(*args, **kwargs) # setup the Plotter object
        self.timerevt = self.plotter.addCallback('timer', self.handle_timer)

    def initialize(self):
        # initialize here extra elements like buttons etc..
        self.button = self.plotter.addButton(self._buttonfunc,
                                             states=["\u23F5 Play  ","\u23F8 Pause"],
                                             font="Kanopus", size=32 )
        return self

    def show(self, *args, **kwargs):
        plt = self.plotter.show(*args, **kwargs)
        return plt

    def _buttonfunc(self):
        if self.timerId is not None:
            self.plotter.timerCallback("destroy", self.timerId)
        if not self.isplaying:
            self.timerId = self.plotter.timerCallback("create", dt=100)
        self.button.switch()
        self.isplaying = not self.isplaying

    def handle_timer(self, event):
        #####################################################################
        ### Animate your stuff here                                       ###
        #####################################################################
        #print(event)               # info about what was clicked and more
        #print(self.plotter.actors) # to access object from the internal list
        earth.rotateZ(1)            # rotate the Earth by 1deg
        moon.color(self.counter)    # change color to the Moon
        txt2d.text("Moon color is:").color(self.counter).background(self.counter,0.1)
        txt2d.text(vedo.getColorName(self.counter), "top-center")
        txt2d.text("..press q to quit", "bottom-right")
        self.plotter.render()
        self.counter += 1


viewer = Viewer(axes=1, dt=150).initialize()

earth  = vedo.Earth()
moon   = vedo.Sphere(r=0.1).x(1.5).color('k7')
txt2d  = vedo.CornerAnnotation().font("Kanopus")

viewer.show(earth, moon, txt2d, viewup='z').close()
