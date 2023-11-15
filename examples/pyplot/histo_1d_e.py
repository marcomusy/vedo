"""Plot a histogram of the distance of each point 
of a sphere to the oceans mesh. The distance is used to
threshold the sphere and create continents."""
from vedo import *
from vedo.pyplot import histogram

# Download the oceans mesh
oceans = Mesh(dataurl + "oceans.vtk").c("blue9")
size = oceans.average_size()

# Create a sphere and compute the distance to the oceans mesh
sphere = IcoSphere(subdivisions=5).scale(size*1.01)
dists = sphere.distance_to(oceans)

# Create a histogram of the distance
histo = histogram(dists, logscale=True, c="gist_earth")
histo+= Arrow2D([200,1], [200,0]).z(1).c("red5")

# Threshold the sphere to create continents
continents = sphere.threshold("Distance", above=20.0)
continents.cmap("gist_earth").linewidth(1)

show(oceans, continents, histo.clone2d(), __doc__)
