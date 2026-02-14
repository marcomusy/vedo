"""Compare predefined lighting styles on the same mesh."""
from vedo import dataurl, Mesh, Plotter

styles = ['default', 'metallic', 'plastic', 'shiny', 'glossy', 'ambient', 'off']

msh = Mesh(dataurl+"beethoven.ply").c('gold').subdivide()

plt = Plotter(N=len(styles), bg='bb')

for i, style in enumerate(styles):
    msh_copy = msh.clone(deep=False).lighting(style)
    plt.at(i).show(msh_copy, style)

plt.interactive().close()
