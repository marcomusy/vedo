from vedo.pyplot import donut

title     = "A donut plot"
fractions = [0.1, 0.2, 0.3, 0.1, 0.3]
colors    = [ 1,   2,   3,   4, 'white']
labels    = ["stuff_1 ", "stuff_2 ", "comp^A ", "comp^B ", ""]

dn = donut(fractions, c=colors, labels=labels, title=title)

dn.show().close()
