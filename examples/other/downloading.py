"""Download a file from a URL address
https://vedo.embl.es/examples/panther.stl.gz
and unzip it on the fly"""
from vedo import load, show

# use force=True to discard any previous cached downloads
mesh = load('https://vedo.embl.es/examples/panther.stl.gz', force=False)

show(mesh, __doc__, axes=True)
