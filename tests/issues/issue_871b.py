from vedo import *

settings.default_font = "Theemim"

pts = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0.5)]
data = [1, 10, 100, 1000, 10000]
scalarbar = None

line = Line(pts, c="k", lw=10)
line.pointdata["mydata"] = data

line.cmap("jet", "mydata", logscale=True)

# automatic add scalarbar
# line.add_scalarbar(title="mydata", size=(100,800))
# line.add_scalarbar3d(title="mydata", nlabels=4)
#
# Or manual add scalarbar
# scalarbar = ScalarBar(line, title="mydata", size=(100,800))
scalarbar = ScalarBar3D(line, title="mydata", 
                        c='black', nlabels=4, label_format=":.1e")
# modify the text of the scalarbar
for e in scalarbar.unpack():
    if isinstance(e, Text3D):
        txt = e.text().replace(".0e+0", " x10^")
        if "x10" in txt: # skip the title
            e.text(txt)  # update new text
            e.scale(0.02)

plt = Plotter()
plt += [line, line.labels("mydata", scale=.02), scalarbar]
plt.show()