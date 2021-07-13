# A plot(mode="bars") example. Useful to plot categories.
from vedo import precision, Text3D, colorMap, settings
from vedo.pyplot import plot

settings.defaultFont = "Meson"

counts  = [1946, 8993, 3042, 1190, 1477,    0,    0]
percent = [11.68909178, 54.01850072, 18.27246516,  7.14800577,  8.87193657, 0, 0]
labels  = ['<100', '100-250', '250-500', '500-750', '750-1000', '1000-2000', '>2000']
colors  = colorMap(range(len(counts)), "hot")

plt = plot([counts, labels, colors],
            mode="bars",
            ylim=(0,10500),
            aspect=4/3,
            axes=dict(htitle="Clusters in lux range",
                      hTitleItalic=False,
                      xLabelRotation=35,
                      xLabelSize=0.02,
                      tipSize=0, # axes arrow tip size
                     ),
)

for i in range(len(percent)):
    val = precision(percent[i], 3)+'%'
    txt = Text3D(val, pos=(plt.centers[i], counts[i]), justify="bottom-center", c="blue2")
    plt += txt.scale(200).shift(0,150,0)

plt.show(size=(1000,750), zoom=1.3, viewup='2d').close()
