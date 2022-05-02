"""Streamlines originating from a set of seed points in space
subjected to a vectorial field defined on a small set of points.
This field is interpolated to a user-defined bounding box."""
from vedo import *
import pandas as pd

data = "https://raw.githubusercontent.com/plotly/datasets/master/vortex.csv"
df = pd.read_csv(data)
pts  = np.c_[df['x'], df['y'], df['z']]
wind = np.c_[df['u'], df['v'], df['w']]

domain = Points(pts)
domain.pointdata["Wind"] = wind

seeds = domain.clone().subsample(0.2) # these are the seed points

# Compute stream lines with Runge-Kutta integration, we
# extrapolate the field defined on points to a bounding box
streamlines = StreamLines(
    domain,
    seeds,
    maxPropagation=100,
    extrapolateToBox=dict(bounds=[-20,20, -15,15, -20,20]),
)
streamlines.lw(5).cmap("Blues", "Wind").addScalarBar()

show(streamlines, __doc__, axes=1, viewup='z').close()
