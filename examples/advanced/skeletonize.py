"""Using 1D Moving Least Squares to skeletonize a surface"""
from vedo import dataurl, Points, Plotter

N = 9    # nr of iterations
f = 0.2  # fraction of neighbours

pcl = Points(dataurl+"man.vtk").subsample(0.02)

plt = Plotter(N=N, axes=1)
for i in range(N):
    pcl = pcl.clone().smoothMLS1D(f=f).color(i)
    plt.at(i).show(f"iteration {i}", pcl, elevation=-5)

plt.interactive().close()
