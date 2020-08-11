"""Generate the isosurfaces corresponding to a set of thresholds.
(These surfaces are not separate meshes).
"""
from vtk import vtkQuadric, vtkSampleFunction

# Quadric definition. This is a type of implicit function.
quadric = vtkQuadric()
quadric.SetCoefficients(0.5, 1, 0.2, 0, 0.1, 0, 0, 0.2, 0, 0)

# the vtkSampleFunction evaluates quadric over a volume
sample = vtkSampleFunction()
sample.SetSampleDimensions(40, 40, 40)
sample.SetImplicitFunction(quadric)
sample.Update()

img = sample.GetOutput()  # vtkImageData
print("Scalar Range", img.GetScalarRange(), "\ntry press shift-x.")

########################
from vedo import show, Volume

# generate an isosurface the volume for each thresholds
ts = [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]

# Use c=None to use the default vtk color map. isos is of type Mesh
isos = Volume(img).isosurface(threshold=ts)

show(isos, __doc__)
