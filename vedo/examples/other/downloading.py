# Download a large file from a URL link and unzip it
#
from vedo import download, gunzip, show

fgz = download('https://vedo.embl.es/examples/truck.vtk.gz') # 200MB

filename = gunzip(fgz)
print('gunzip-ped to temporary file:', filename)

show(filename)
