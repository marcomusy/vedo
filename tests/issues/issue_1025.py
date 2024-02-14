from vedo import *

colors = [
    "black",
    "lightgreen",
    "orange",
    "yellow",
    "green",
    "lightblue",
    "pink",
    "red",
    "cyan",
    "yellow",
    "blue",
    "tomato",
    "violet",
    "brown",
]
n = len(colors)

x = np.linspace(0, n, n+1)
# x = np.sqrt(np.linspace(0, n, n+1))
vals = x.copy()[1:]
print(vals)

xyz = np.concatenate((x[:, None], np.ones((n+1, 2))), axis=-1)
xyz_segments = np.stack((xyz[:-1, :], xyz[1:, :]), axis=1)

lines0 = Lines(xyz_segments, lw=8)
lines0.cmap(colors, vals, on="cells")

lines1 = Lines(xyz_segments, lw=8).shift(0,-0.2)
table = [(x, color) for x, color in zip(vals, colors)]
clut = build_lut(table, vmin=vals[0], interpolate=0)
print("table:", table)
lines1.cmap(clut, vals, on="cells")


pts = Points(xyz, c='black', r=8)

plt = Plotter()
plt += [pts, pts.labels2d('id'), lines0, lines1]
plt.show("Colors must match", size=(1200,300), zoom=3.5).close()
