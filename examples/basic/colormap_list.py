from vedo import *

n = 256
i, grids, vnames1, vnames2 = 0, [], [], []

for name in colors.cmaps_names:
    if '_r' in name:
        continue # skip reversed maps
    cols = colorMap(range(n), name)

    # make a strip of n cells and assing them individual colors
    gr = Grid(s=[50,1], res=[n,1]).cellIndividualColors(cols*255)
    gr.lineWidth(0).wireframe(False).y(-i*1.2)
    grids.append([gr, gr.box().c('grey')])

    tx1 = Text3D('\rightarrow '+name, justify='left-center', s=0.75, font=2)
    tx1.pos(gr.xbounds(1), gr.y()).c('w')
    tx2 = tx1.clone(deep=False).c('k')
    vnames1.append(tx1)
    vnames2.append(tx2)
    i += 1

printc("Try picking a color by pressing Shift-i", invert=True)

plt = Plotter(N=2, size=(1300,1000))
plt.at(0).show(grids, vnames1, bg='blackboard')
plt.at(1).show(grids, vnames2, bg='white', mode='image', zoom='tight')
plt.interactive().close()
