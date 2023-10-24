# Transform a image into a mesh
from vedo import Image, dataurl, show
import numpy as np

pic = Image(dataurl+"images/dog.jpg").smooth(5)
msh = pic.tomesh()  # make a quad-mesh out of it

# build a scalar array with intensities
rgb = msh.pointdata["RGBA"]
intensity = np.sum(rgb, axis=1)
intensityz = np.zeros_like(rgb)
intensityz[:,2] = intensity / 10

# set the new vertex points
msh.vertices += intensityz

# more cosmetics
msh.triangulate().smooth()
msh.lighting("default").linewidth(0)
msh.cmap("bone", "RGBA").add_scalarbar()

msht = pic.clone().threshold(100).linewidth(0)

show([[pic, "A normal jpg image.."],
      [msh, "..becomes a polygonal Mesh"],
      [msht, "Thresholding also generates a Mesh"]
     ], N=3, axes=1, zoom=1.1, elevation=-20, bg='black').close()
