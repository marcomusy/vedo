"""Read a data from ascii file and make a simple analysis
visualizing 3 of the 5 dimensions of the dataset"""
import numpy as np
from vtkplotter import *
from vtkplotter.pyplot import histogram


################################### Read the csv data:
delimiter=','
with open(datadir+'genes.csv', "r") as f:
    lines = f.readlines()
data = []
for i,lns in enumerate(lines):
    if i==0:
        names = lns.split(delimiter) # read header
        continue
    ln = lns.split(delimiter)
    vals = [float(x) for x in ln]
    data.append(vals)
data = np.array(data)

print("Print first 5 rows:")
print(names)
print(data[:5])
print("Number of rows:", len(data))
##################################################

# extract the columns into separate vectors:
genes = data.T
g0, g1, g2, g3, g4 = genes # unpack
n0, n1, n2, n3, n4 = names

# now create and show histograms of the gene expressions
h0 = histogram(g0, xtitle=n0, c=0)
h1 = histogram(g1, xtitle=n1, c=1)
h2 = histogram(g2, xtitle=n2, c=2)
h3 = histogram(g3, xtitle=n3, c=3, logscale=True)
show(h0, h1, h2, h3, shape=(1,4))

# this is where you choose what variables to show as 3D points
pts = np.c_[g4,g2,g3] # form an array of 3d points from the columns
axs = dict(xtitle='gene4', ytitle='gene2', ztitle='gene3')

pts_1 = pts[g0>0]            # select only points that have g0>0
print("after selection nr. of points is", len(pts_1))
p_1 = Points(pts_1, r=4, c='red')   # create the vtkplotter object

pts_2 = pts[(g0<0) & (g1>.5)] # select excluded points that have g1>0.5
p_2 = Points(pts_2, r=8, c='green') # create the vtkplotter object

# Show the two clouds superposed on a new plotter window:
show(p_1, p_2, __doc__, axes=axs, newPlotter=True)






