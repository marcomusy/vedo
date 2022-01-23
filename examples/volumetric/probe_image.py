"""Probe image intensities along a set of radii"""
from vedo import Picture, Circle, Lines, show
from vedo.pyplot import plot
import numpy as np

pic = Picture("https://aws1.discourse-cdn.com/business4/uploads/imagej/original/3X/a/d/adaa1f1f3e88f8285b502ed863be494b6af6f46d.jpeg")
cpt = [580,600,0]
circle = Circle(cpt, r=500, res=36).wireframe()

pts = circle.points()                 # 3d coords of the points of the circle
centers = np.zeros_like(pts) + cpt    # same amount of center coords
lines = Lines(centers, pts, res=50)   # create Lines with 50 pts each

msh = pic.tomesh()                    # transform the picture into a quad mesh
lines.interpolateDataFrom(msh, N=4)   # interpolate all msh data onto the lines
rgb = lines.pointdata['RGBA']         # extract the rgb intensities
intensities = np.sum(rgb, axis=1)     # sum the rgb values into one single intensty
intensities_ray = np.split(intensities, 36)  # split array so we can index any radius
mean_intensity = np.mean(intensities_ray, axis=0)  # compute the average intensity

# do some optional plotting here:
plt = plot(mean_intensity, lc='black', lw=5,
           xtitle='radial distance', ytitle='intensity', aspect=1, spline=True)
for i in range(0,36,3):
    plt += plot(intensities_ray[i], lc=i, lw=1)

show([
      [msh, circle, lines, __doc__],  # show this on the first renderer
      ["...plot the result:", plt],   # show this on the second renderer
     ], N=2, sharecam=False,          # nr of rendering subwindows, camera is not shared
)
