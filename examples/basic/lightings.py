from vedo import *

styles = ['default', 'metallic', 'plastic', 'shiny', 'glossy', 'ambient', 'off']

msh = Mesh(dataurl+"beethoven.ply").c('gold').subdivide()

for i,s in enumerate(styles):
    msh_copy = msh.clone(deep=False).lighting(s)
    show(msh_copy, s, at=i, N=len(styles), bg='bb')

interactive().close()
