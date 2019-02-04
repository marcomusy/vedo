# Volumetric Examples directory
In this directory you will find examples with volumes and voxel data (`vtkImageData`).
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples/volumetric
python example.py  # on mac OSX try 'pythonw' instead
```
(_click thumbnail image to get to the python script_)

|    |    |
|:-------------:|:-----|
| [![imageOperations](https://user-images.githubusercontent.com/32848391/50739032-6b2a7c80-11da-11e9-82fb-495c803ea9bf.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/imageOperations.py)<br/> `imageOperations.py` |  Perform simple mathematical operations between 3d images.<br>Possible operations are: +, -, /, 1/x, sin, cos, exp, log, abs,  sqrt, min, max, atan, atan2, median, mag, dot, gradient, divergence, laplacian.|
|    |    |
| [![interpolateVolume](https://user-images.githubusercontent.com/32848391/50739033-6b2a7c80-11da-11e9-86fd-6026b22737df.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/interpolateVolume.py)<br/> `interpolateVolume.py` | Generate a voxel dataset (`vtkImageData`) by interpolating a scalar which is only known on a small set of points. Interpolation is based on RBF (Radial Basis Function).|
|    |    |
| [![isosurfaces1](https://user-images.githubusercontent.com/32848391/52141625-975ce000-2656-11e9-91fc-291e072fc4c1.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/isosurfaces1.py)<br/> `isosurfaces1.py` | Generate the isosurfaces corresponding to an input set of thresholds on volumetric data.|
|    |    |
| [![isosurfaces2](https://user-images.githubusercontent.com/32848391/51558920-ec436e00-1e80-11e9-9d96-aa9b7c72d58b.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/isosurfaces2.py)<br/> `isosurfaces2.py` | Generate the isosurfaces corresponding to an input set of thresholds on volumetric data.|
|    |    |
| [![probeLine](https://user-images.githubusercontent.com/32848391/48198460-3aa0a080-e359-11e8-982d-23fadf4de66f.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/probeLine.py)<br/> `probeLine.py` | Find and visualize the scalar value along a line intersecting a volume. |
|    |    |
| [![probePlane](https://user-images.githubusercontent.com/32848391/48198461-3aa0a080-e359-11e8-8c29-18f287f105e6.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/probePlane.py)<br/> `probePlane.py` | Slice a `vtkImageData` (voxel dataset) with a plane and visualize the scalar value on it.|
|    |    |
| [![readStructuredPoints](https://user-images.githubusercontent.com/32848391/48198462-3b393700-e359-11e8-8272-670bd5f2db42.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/readStructuredPoints.py)<br/> `readStructuredPoints.py` | Example of reading a vtk file containing `vtkStructuredPoints` data (representing the orbitals of the electron in the hydrogen atom). The dataset contains an existing scalar array named `probability_density` which is transformed into a color map.|
|    |    |
| [![readVolumeAsIsoSurface](https://user-images.githubusercontent.com/32848391/50739035-6b2a7c80-11da-11e9-8687-4e4d46ff6df0.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/readVolumeAsIsoSurface.py)<br/> `readVolumeAsIsoSurface.py` | Generate an isosurface from voxel data.<br>A tiff stack is a set of image slices in z. The scalar value (intensity of white) is used to create an isosurface by fixing a threshold. |
|    |    |
| [![readVolume](https://user-images.githubusercontent.com/32848391/50739034-6b2a7c80-11da-11e9-9c86-1b25b1b77f42.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/readVolume.py)<br/> `readVolume.py` | Load a`vtkImageData` object, make a `vtkVolume` and show it along with a mesh surface (the red sphere). |
|    |    |
| [![read_vti](https://user-images.githubusercontent.com/32848391/50739036-6bc31300-11da-11e9-89b3-04a75187f812.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/read_vti.py)<br/> `read_vti.py` | Using typical vtk commands to load a xml `.vti` file, then use vtkplotter to show the resulting 3D image.|
|    |    |
| [![signedDistance](https://user-images.githubusercontent.com/32848391/50739037-6bc31300-11da-11e9-82b7-dd4ae11076ae.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/signedDistance.py)<br/> `signedDistance.py` | A mixed example with class `vtkSignedDistance`: generate a scalar field by the signed distance from a polydata, save it to a `.tif` file, then extract an isosurface from the 3D image. |
