"""
Using normal vtk commands to load a xml vti file
then use vtkplotter to show the resulting 3d image.
"""
import vtk

# Create the reader for the data.
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("data/vase.vti")
reader.Update()
img = reader.GetOutput()

# specify the data array in the file to process
# img.GetPointData().SetActiveAttribute('SLCImage', 0)


#################################
from vtkplotter import Volume, load, show, Text

# can set colors and transparencies along the scalar range
vol = Volume(img, c=["gray", "fuchsia", "dg", (0, 0, 1)], alphas=[0.1, 0.2, 0.3, 0.8])

# load command returns an isosurface (vtkActor) of the 3d image
iso = load("data/vase.vti", threshold=140).wire(True).alpha(0.1)

# show command creates and returns an instance of class Plotter
show(vol, iso, Text(__doc__), verbose=0, bg="w")
