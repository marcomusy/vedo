from vtkplotter import Plotter, load, Plane, datadir

vp = Plotter()

cow = vp.load(datadir+"cow.byu", c="grey", alpha=0.7)
vp += Plane(pos=[0, -3.6, 0], normal=[0, 1, 0], sx=20, texture="grass")
vp.show(viewup='y')

# vp.light() returns a vtkLight object with focal Point, fp, to actor cow
# fp can also be explicitly set as fp=[x,y,z]
l = vp.addLight(pos=[-6, 6, 6], focalPoint=cow, deg=12, showsource=1)

# can be switched on/off this way
#l.SwitchOff()

vp.show()
