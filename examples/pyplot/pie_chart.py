from vedo import settings, show
from vedo.pyplot import pie_chart

settings.default_font = "Komika"

title     = "A pie chart plot"
fractions = [0.1, 0.2, 0.3, 0.1, 0.3]
colors    = [ 1,   2,   3,   4, 'white']
labels    = ["stuff_1 ", "stuff_2 ", "comp^A ", "comp^B ", ""]

pc = pie_chart(fractions, c=colors, labels=labels, title=title)
pc2d = pc.clone2d("top-left", size=0.975, ontop=False)

show(pc2d).close()
