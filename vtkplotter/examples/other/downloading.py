# Download a large file from a URL link and unzip it
#
from vtkplotter import download, gunzip, show

fgz = download('https://vtkplotter.embl.es/examples/truck.vtk.gz') # 200MB

filename = gunzip(fgz)
print('gunzip-ped to temporary file:', filename)

show(filename)
