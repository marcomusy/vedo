"""Probe a voxel dataset at specified points
and plot a histogram of the values"""
from vedo import np, dataurl, Points, Volume, Axes, show
from vedo.pyplot import histogram

vol = Volume(dataurl + 'embryo.slc')
vol_axes = Axes(vol)

pts = np.random.rand(5000, 3)*256
mpts = Points(pts).probe(vol).point_size(3)
mpts.print()

# valid = mpts.pointdata['ValidPointMask']
scalars = mpts.pointdata['SLCImage']
his = histogram(scalars, xtitle='Probed voxel value', xlim=(5,100))

show([(vol, vol_axes, mpts, __doc__), his], N=2, sharecam=False).close()
