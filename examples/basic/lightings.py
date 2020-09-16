from vedo import *

styles = ['default', 'metallic', 'plastic', 'shiny', 'glossy', 'ambient', 'off']

a = load(datadir+"beethoven.ply").c('gold').subdivide()

for i,s in enumerate(styles):
    show(a.clone(deep=False).lighting(s), Text2D(s),
    	 at=i, N=len(styles), bg='bb')

interactive()
