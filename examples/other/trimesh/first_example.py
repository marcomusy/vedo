"""
trimesh to vtkplotter interoperability
"""

# Install trimesh with:
# sudo apt install python3-rtree
# pip install rtree shapely
# conda install trimesh

import trimesh
from vtkplotter import download, trimesh2vtk, show

url = 'https://raw.githubusercontent.com/mikedh/trimesh/master/models/'
filename = download(url + 'machinist.XAML')
mesh = trimesh.load(filename)

actor = trimesh2vtk(mesh) # returns a Actor(vtkActor) object from Trimesh

# Any of these will work:
show(mesh) # conversion is on the fly (don't need 'actor')

# or
#actor.show()

# or
#show(actor)

# or
mesh.show() # this is the trimesh built-in visualizer
