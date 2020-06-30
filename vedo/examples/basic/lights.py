from vedo import Plotter, Plane, datadir

vp = Plotter()

cow = vp.load(datadir+"cow.vtk").c("grey").scale(4).rotateX(-90)
vp += Plane(pos=[0, -3.6, 0], normal=[0, 1, 0], sx=20).texture("grass")
vp.show(viewup='y', interactive=0)

# vp.light() returns a vtkLight object with focal Point, fp, to mesh cow
# fp can also be explicitly set as fp=[x,y,z]
l = vp.addLight(pos=[-6, 6, 6], focalPoint=cow, deg=12, showsource=1)

# can be switched on/off this way
#l.SwitchOff()

vp.show(interactive=1)
