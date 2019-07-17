"""Using normal vtk commands to load a vtkImageData
then use vtkplotter to show the resulting 3d images.

mode=0, composite rendering
mode=1, maximum-projection rendering
"""
import vtk
from vtkplotter import datadir, load

# Create the reader for the data.
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(datadir+"vase.vti")
reader.Update()
img = reader.GetOutput() # vtkImageData object

# NB: the above lines could be reduced to:
#img = load(datadir+"vase.vti").imagedata()

#################################
from vtkplotter import Volume, show, Text

# can set colors and transparencies along the scalar range
# from minimum to maximum value. In this example voxels with
# the smallest value will be completely transparent (and white)
# while voxels with highest value of the scalar will get alpha=0.8
# and color will be=(0,0,1)
vol1 = Volume(img, mode=0) # composite rendering
vol1.color(["white", "fuchsia", "dg", (0,0,1)])
#vol1.color('jet') # a matplotlib colormap name is also accepted
vol1.alpha([0.0, 0.2, 0.3, 0.8])

# a transparency for the GRADIENT of the scalar can also be set:
# in this case when the scalar is ~constant the gradient is ~zero
# and the voxel are made transparent:
vol1.alphaGradient([0.0, 0.5, 0.9])

# mode = 1 is maximum-projection volume rendering
vol2 = load(datadir+"vase.vti").mode(1).addPos(60,0,0)

# show command creates and returns an instance of class Plotter
show(vol1, vol2, Text(__doc__), bg="w", axes=1)
