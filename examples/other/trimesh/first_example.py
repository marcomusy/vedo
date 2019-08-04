"""
trimesh to vtkplotter interoperability
"""
# Install trimesh with:
# sudo apt install python3-rtree
# pip install rtree shapely
# conda install trimesh

import trimesh
import vtkplotter
from vtkplotter.trimesh import trimesh2vtk, vtk2trimesh

url = 'https://raw.githubusercontent.com/mikedh/trimesh/master/models/'
filename = vtkplotter.download(url + 'machinist.XAML')

mesh = trimesh.load(filename)

actor = trimesh2vtk(mesh) # returns a Actor(vtkActor) object from Trimesh
vtkplotter.show(mesh) # vtkplotter visualizer (conversion is on the fly)

trimsh_reconverted = vtk2trimesh(actor)
trimsh_reconverted.show() # this is the trimesh built-in visualizer

