"""pygeodesic library to compute geodesic distances"""
# pip install pygeodesic
import pygeodesic.geodesic as geodesic
import vedo

m = vedo.Mesh(vedo.dataurl+"bunny.obj").c("green9")

geoalg = geodesic.PyGeodesicAlgorithmExact(m.points(), m.faces())

# Use source and target point ids
distance, path = geoalg.geodesicDistance(639, 834)
distances, _   = geoalg.geodesicDistances([639, 1301]) # any of the two

line = vedo.Line(path).c("k").lw(4)
m.cmap("Set2", distances, name="GeodesicDistance")

vedo.show(m, line, __doc__, axes=1)

