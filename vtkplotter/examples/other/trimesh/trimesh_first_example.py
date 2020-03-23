"""
trimesh to vtkplotter interoperability
"""
# Install trimesh with:
# sudo apt install python3-rtree
# pip install rtree shapely
# conda install trimesh

import trimesh
import vtkplotter
from vtkplotter import trimesh2vtk, vtk2trimesh

url = 'https://raw.githubusercontent.com/mikedh/trimesh/master/models/'
filename = vtkplotter.download(url + 'machinist.XAML')

mesh = trimesh.load(filename)

vmesh = trimesh2vtk(mesh) # returns a Mesh(vtkActor) object from Trimesh
vtkplotter.show(mesh) # vtkplotter visualizer (conversion is on the fly)

trimsh_reconverted = vtk2trimesh(vmesh)
trimsh_reconverted.show() # this is the trimesh built-in visualizer

