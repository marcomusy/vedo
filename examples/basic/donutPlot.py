from vtkplotter import donutPlot

title     = "A donut plot"
fractions = [0.1, 0.2, 0.3, 0.1, 0.3]
colors    = [ 1,   2,   3,   4, 'white']
labels    = ["stuff1", "stuff2", "compA", "compB", ""]

dn = donutPlot(fractions, c=colors, labels=labels, title=title)

dn.show(axes=None, bg='w')
