# Transform a picture into a mesh
from vedo import Picture, dataurl, show
import numpy as np

pic = Picture(dataurl+"images/dog.jpg").smooth(5)
msh = pic.tomesh()  # make a quad-mesh out of it

# build a scalar array with intensities
rgb = msh.pointdata["RGBA"]
intensity = np.sum(rgb, axis=1)
intensityz = np.zeros_like(rgb)
intensityz[:,2] = intensity / 10

# set the new vertex points
pts = msh.points() + intensityz
msh.points(pts)

# more cosmetics
msh.triangulate().smooth()
msh.lighting("default").lineWidth(0)
msh.cmap("bone", "RGBA").addScalarBar()

msht = pic.clone().threshold(100).lineWidth(0)

show([[pic, "A normal jpg image.."],
      [msh, "..becomes a polygonal Mesh"],
      [msht, "Thresholding also generates a Mesh"]
     ], N=3, axes=1, zoom=1.1, elevation=-20, bg='black').close()
