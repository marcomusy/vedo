from vtkplotter import *

styles = ['default', 'metallic', 'plastic', 'shiny', 'reflective']

a = load(datadir+"beethoven.ply").subdivide()

for i,s in enumerate(styles):
    show(a.clone().lighting(s), Text(s), at=i, N=len(styles))

interactive()
