from vedo import *

n = 256
i, grids, vnames1, vnames2 = 0, [], [], []

for name in colors.cmaps_names:
    if '_r' in name:
        continue # skip reversed maps
    cols = colorMap(range(n), name)

    # make a strip of n cells and assing them individual colors
    gr = Grid(sx=50, resx=n, sy=1, resy=1).cellIndividualColors(cols)
    gr.lineWidth(0).wireframe(False).y(-i*1.2)
    grids.append([gr, gr.box().c('grey')])

    tx1 = Text3D('\rightarrow '+name, justify='left-center', s=0.75, font=2)
    tx1.pos(gr.xbounds(1), gr.y()).c('w')
    tx2 = tx1.clone(deep=False).c('k')
    vnames1.append(tx1)
    vnames2.append(tx2)
    i += 1

printc("Try picking a color by pressing Shift-i", invert=1)
show(grids, vnames1, at=0, N=2, size=(1300,1000), bg='blackboard',
     title="Color Maps with n="+str(n)+" colors")
show(grids, vnames2, at=1, bg='white', zoom=1.75, interactive=True).close()
