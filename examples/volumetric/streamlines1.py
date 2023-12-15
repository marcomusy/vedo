"""Streamlines originating from a set of seed points in space
subjected to a vectorial field defined on a small set of points."""
from vedo import *
import pandas as pd

data = "https://raw.githubusercontent.com/plotly/datasets/master/vortex.csv"
df = pd.read_csv(data)
pts  = np.c_[df['x'], df['y'], df['z']]
wind = np.c_[df['u'], df['v'], df['w']]

vpts = Points(pts, r=10)
vpts.pointdata["Wind"] = wind

# Convert points to a volume to create a domain for the streamlines
vol = vpts.tovolume(kernel='shepard', n=4, dims=(20,20,20))
vol_pts = vol.coordinates
iwind = vol.pointdata["Wind"] # interpolated wind
arrs = Arrows(vol_pts, vol_pts + iwind*0.5, alpha=0.2)

# Subsample the points to use as seed points
seeds = vpts.clone().subsample(0.2)

# Compute stream lines with Runge-Kutta integration
# vol.pointdata.select("Wind") # in case there are other vectors
streamlines = vol.compute_streamlines(seeds.vertices)
streamlines.pointdata["wind_intensity"] = mag(streamlines.pointdata["Wind"])
streamlines.cmap("Reds").add_scalarbar()
print(streamlines)
show(seeds, arrs, streamlines, __doc__, axes=1, viewup='z').close()

# Create a tube around the streamlines
streamtubes = Tubes(streamlines, r=0.01, vary_radius_by_scalar=True)
streamtubes.cmap("Reds").add_scalarbar()
print(streamtubes)
show(streamtubes, __doc__, axes=1, viewup='z').close()
