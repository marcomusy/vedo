import vtkplotter

vp = vtkplotter.Plotter(bg='black', axes=0)

vp.earth(r=6371) # km
vp.sphere(pos=[384402,0,0], r=1737, c='gray') # moon to its real scale

vp.show(zoom=0.8) 

