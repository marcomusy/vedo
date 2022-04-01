# A plot(mode="bars") example. Useful to plot categories.
from vedo import precision, Text3D, colorMap, settings
from vedo.pyplot import plot

settings.defaultFont = "Meson"

counts  = [1946, 8993, 3042, 1190, 1477,    0,    0]
percent = [11.68909178, 54.01850072, 18.27246516,  7.14800577,  8.87193657, 0, 0]
labels  = ['<100', '100-250', '250-500', '500-750', '750-1000', '1000-2000', '>2000']
colors  = colorMap(range(len(counts)), "hot")

# plot() will return a PlotBars object
fig = plot(
    [counts, labels, colors],
    mode="bars",
    ylim=(0,10_000),
    aspect=16/9,
    title='Clusters in lux range',
    axes=dict(
        xLabelRotation=30,
        xLabelSize=0.02,
    ),
)

for i in range(len(percent)):
    val = precision(percent[i], 3)+'%'
    txt = Text3D(val, pos=(fig.centers[i], counts[i]), justify="bottom-center", c="blue2")
    fig += txt.scale(200).shift(0,170,0)

fig.show(size=(1000,750), zoom='tight').close()
