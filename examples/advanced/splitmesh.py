"""
Split a mesh by connectivity and order the pieces 
by increasing area.
"""
print(__doc__)
from vtkplotter import splitByConnectivity, load, show

blobs = load("data/embryo.tif", threshold=80)

# search up to the 40th subunit of mesh, return a list
# of length 40, but only keep the largest 10:
sblobs = splitByConnectivity(blobs, maxdepth=40)[0:9]

sblobs[0].alpha(0.5)  # make the largest part transparent

show([blobs, sblobs], N=2, axes=1)
