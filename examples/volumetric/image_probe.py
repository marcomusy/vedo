"""Probe image intensities along a set of radii"""
from vedo import Picture, dataurl, Circle, Lines, show
from vedo.pyplot import plot
import numpy as np

pic = Picture(dataurl+'images/spheroid.jpg')
cpt = [580,600,0]
circle = Circle(cpt, r=500, res=36).wireframe()

pts = circle.points()                 # 3d coords of the points of the circle
centers = np.zeros_like(pts) + cpt    # create the same amount of center coords
lines = Lines(centers, pts, res=50)   # create Lines with 50 pts of resolution each

msh = pic.tomesh()                    # transform the picture into a quad mesh
lines.interpolateDataFrom(msh, N=3)   # interpolate all msh data onto the lines
rgb = lines.pointdata['RGBA']         # extract the rgb intensities
intensities = np.sum(rgb, axis=1)     # sum the rgb values into one single intensty
intensities_ray = np.split(intensities, 36)  # split array so we can index any radius
mean_intensity = np.mean(intensities_ray, axis=0)  # compute the average intensity

# add some optional plotting here:
fig = plot(
    mean_intensity,
    lc='black', lw=5, spline=True,
    xtitle='radial distance', ytitle='intensity', aspect=16/9,
)
for i in range(0,36, 3):
    fig += plot(intensities_ray[i], lc=i, lw=1, like=fig)
fig.scale(21).shift(60,-800)          # scale up and move plot below the image

show(msh, circle, lines, fig, __doc__, size=(625,1000), zoom=1.5)
