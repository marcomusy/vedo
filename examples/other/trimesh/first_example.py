"""
trimesh to vedo interoperability
"""
# Install trimesh with:
# sudo apt install python3-rtree
# pip install rtree shapely
# conda install trimesh

import trimesh
import vedo
from vedo import trimesh2vedo

url = 'https://raw.githubusercontent.com/mikedh/trimesh/master/models/'
filename = vedo.download(url + 'machinist.XAML')

mesh = trimesh.load(filename)

vedo.show(mesh) # vedo visualizer (conversion is on the fly)


# explicit conversion
vmesh = trimesh2vedo(mesh) # returns a vedo.Mesh(vtkActor) object
trimsh_reconverted = vmesh.to_trimesh()

try:
    trimsh_reconverted.show() # this is the trimesh built-in visualizer
except:
    pass
