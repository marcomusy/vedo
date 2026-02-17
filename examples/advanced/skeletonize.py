"""Using 1D Moving Least Squares to skeletonize a surface"""
from vedo import dataurl, Points, Plotter

N = 9    # nr of iterations
f = 0.2  # fraction of neighbours

# Initial sparse cloud from a surface.
pcl = Points(dataurl+"man.vtk").subsample(0.02)

plt = Plotter(N=N, axes=1)
for i in range(N):
    # Repeated MLS-1D steps progressively collapse toward a curve skeleton.
    pcl = pcl.clone().smooth_mls_1d(f=f).color(i)
    plt.at(i).show(f"iteration {i}", pcl, elevation=-8)

plt.interactive().close()
