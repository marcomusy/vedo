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
|    |    |
| [![interpolateVolume](https://user-images.githubusercontent.com/32848391/50739033-6b2a7c80-11da-11e9-86fd-6026b22737df.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/interpolateVolume.py)<br/> `interpolateVolume.py` | Generate a voxel dataset (`vtkImageData`) by interpolating a scalar which is only known on a scattered set of points. Interpolation is based on RBF (Radial Basis Function).|
|    |    |
| [![probeLine](https://user-images.githubusercontent.com/32848391/48198460-3aa0a080-e359-11e8-982d-23fadf4de66f.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/probeLine.py)<br/> `probeLine.py` | Find and visualize the scalar value along a line intersecting a volume. |
|    |    |
| [![probePlane](https://user-images.githubusercontent.com/32848391/48198461-3aa0a080-e359-11e8-8c29-18f287f105e6.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/probePlane.py)<br/> `probePlane.py` | Intersect a `vtkImageData` (voxel dataset) with a plane and visualize the scalar value on it.|
|    |    |
| [![readStructuredPoints](https://user-images.githubusercontent.com/32848391/48198462-3b393700-e359-11e8-8272-670bd5f2db42.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/readStructuredPoints.py)<br/> `readStructuredPoints.py` | Example of reading a vtk file containing `vtkStructuredPoints` data (representing the orbitals of the electron in the hydrogen atom). The dataset contains an existing scalar array named 'probability_density' which is transformed into a color map.<br>The list of existing arrays can be found by selecting an actor and pressing i in the rendering window.|
|    |    |
| [![readVolumeAsIsoSurface](https://user-images.githubusercontent.com/32848391/50739035-6b2a7c80-11da-11e9-8687-4e4d46ff6df0.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/readVolumeAsIsoSurface.py)<br/> `readVolumeAsIsoSurface.py` | Example to read volumetric data.<br>A tiff stack is a set of image slices in z. The scalar value (intensity of white) is used to create an isosurface by fixing a threshold.<br> In this example the level of white is in the range 0=black -> 150=white. If `threshold=None` this is set to 1/3 of the scalar range.<br> Setting `connectivity` to True discards the small isolated pieces of surface and only keeps the largest connected surface.<br> Smoothing applies a gaussian `smoothing` with a standard deviation which is expressed in units of pixels.<br> Backface color is set to violet (`bc='v'`) to spot where the vtk reconstruction is (by mistake!) inverting the normals to the surface.<br> If the spacing of the tiff stack is uneven in *xyz*, this can be corrected by setting scaling factors with `scaling=[xfac,yfac,zfac]` |
|    |    |
| [![readVolume](https://user-images.githubusercontent.com/32848391/50739034-6b2a7c80-11da-11e9-9c86-1b25b1b77f42.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/readVolume.py)<br/> `readVolume.py` | Load a`vtkImageData` object, make a `vtkVolume` and show it along with a mesh surface (the red sphere). |
|    |    |
| [![read_vti](https://user-images.githubusercontent.com/32848391/50739036-6bc31300-11da-11e9-89b3-04a75187f812.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/read_vti.py)<br/> `read_vti.py` | Using typical vtk commands to load a xml `.vti` file, then use vtkplotter to show the resulting 3D image.|
|    |    |
| [![signedDistance](https://user-images.githubusercontent.com/32848391/50739037-6bc31300-11da-11e9-82b7-dd4ae11076ae.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/signedDistance.py)<br/> `signedDistance.py` | A mixed example with class `vtkSignedDistance`: generate a scalar field by the signed distance from a polydata, save it to a `.tif` file, then extract an isosurface from the 3D image. |
