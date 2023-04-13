from vedo import *

# Set the number of colors to generate
n = 256

# Initialize some variables
i, grids, vnames1, vnames2 = 0, [], [], []

# Loop over all available colormap names
for name in colors.cmaps_names:
    # Skip reversed maps
    if '_r' in name:
        continue
    
    # Generate a list of n RGB color values for the colormap
    cols = color_map(range(n), name)

    gr = Grid(s=[50,1], res=[n,1])
    gr.cellcolors = cols*255
    gr.linewidth(0).wireframe(False).y(-i*1.2)
    grids.append([gr, gr.box().c('grey')])

    # Add a text label with the colormap name to the left of the strip
    tx1 = Text3D(':rightarrow '+name, justify='left-center', s=0.75, font=2)
    tx1.pos(gr.xbounds(1), gr.y()).c('w')
    tx2 = tx1.clone(deep=False).c('k')
    vnames1.append(tx1)
    vnames2.append(tx2)
    i += 1

printc("Try picking a color by pressing Shift-i", invert=True)

# Create a plotter with two renderers
plt = Plotter(N=2, size=(1300,1000))

# Show the grids with the white text labels
plt.at(0).show(grids, vnames1, bg='blackboard')
plt.at(1).show(grids, vnames2, bg='white', mode='image', zoom='tight')

# Enable interactivity and display the plot, then close it
plt.interactive().close()
