"""The butterfly effect with cylindrical mirrors and a laser"""
# Original idea from "The Action Lab": https://www.youtube.com/watch?v=kBow0kTVn3s
#
from vedo import Plotter, Grid, Cylinder, merge
from optics_base import Ray, Mirror, Detector  # see file ./optics_base.py

grid = Grid(res=[3,4])  # pick a few points in space to place cylinders
pts = grid.points().tolist() + grid.cellCenters().tolist()

# Create the mirror by merging many (y-scaled) cylinders into a single mesh object
cyls = [Cylinder(p, r=0.065, height=0.2, res=720).scale([1,1.5,1]) for p in pts]
mirror = Mirror(merge(cyls)).color("silver")

# Create a detector surface as a thin cylinder surrounding the mirror
sd = Cylinder(r=1, height=0.3, cap=False).cutWithPlane([0,-0.95,0], normal='y')
detector = Detector(sd)


def slider(widget, event):        ### callback to shift the beam along x
    dx = widget.GetRepresentation().GetValue()
    ray = Ray([dx,-1.2,-0.1], direction=(0,1,0.02))
    ray.maxiterations = 1000      # max nr. of reflections
    ray.trace([mirror, detector]) # cumpute trajectory
    detector.count().cmap("Reds", on='cells', vmax=10)
    line = ray.asLine().lineWidth(4).c('green5')
    if plt.actors[-1].name == "Line":
        plt.pop()                 # remove the last Line
    plt.add(line)                 # add the new one

plt = Plotter(axes=1, bg='peachpuff', bg2='blue9')
plt.addSlider2D(slider, -0.07, 0.07, value=0, pos=5, title="beam shift")
plt.show(mirror, detector, __doc__, elevation=-30)
plt.close()

