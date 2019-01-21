import vtkplotter

vp = vtkplotter.Plotter(bg='black', axes=0, verbose=0)

# moon to real scale:
e = vtkplotter.shapes.earth(r=6371) # km
m = vtkplotter.shapes.sphere(pos=[384402,0,0], r=1737, c='gray') 
vp.show([e,m], zoom=0.8) 

