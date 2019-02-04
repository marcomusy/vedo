# vtkplotter
[![Downloads](https://pepy.tech/badge/vtkplotter)](https://pepy.tech/project/vtkplotter)
[![Downloads](https://pepy.tech/badge/vtkplotter/week)](https://pepy.tech/project/vtkplotter)
[![lics](https://img.shields.io/badge/license-MIT-blue.svg)](https://en.wikipedia.org/wiki/MIT_License)
[![pythvers](https://img.shields.io/badge/python-2.7%7C3.6-brightgreen.svg)](https://pypi.org/project/vtkplotter)
[![gdocs](https://img.shields.io/badge/docs%20by-gendocs-blue.svg)](https://gendocs.readthedocs.io/en/latest)

A python module for scientific visualization, analysis and animation of 3D objects 
and point clouds based on [VTK](https://www.vtk.org/) 
and [numpy](http://www.numpy.org/).


## Download and Install:
```bash
pip install vtkplotter
```

## Documentation
Automatically generated documentation can be found [**here**](https://vtkplotter.embl.es).


## Hello World example
In your python script, load a simple `3DS` file and display it:
```python
from vtkplotter import show

show('data/shapes/flamingo.3ds') 
```
![flam](https://user-images.githubusercontent.com/32848391/50738813-58af4380-11d8-11e9-84ce-53579c1dba65.png)


## Command-line usage
```bash
vtkplotter meshfile.vtk 
# valid formats: [vtu,vts,vtp,ply,obj,stl,3ds,xml,neutral,gmsh,pcd,xyz,txt,byu,tif,slc,vti,mhd,png,jpg]
```
to visualize multiple files or files time-sequences try `-n` or `-s` options. Use `-h` for help.<br> 
Voxel-data (_vti, slc, tiff_) files can also be visualized with options `-g` and `--slicer`,
e.g.:
```bash
vtkplotter -g -c blue examples/data/embryo.slc  # (3D scan of a mouse embryo)
vtkplotter --slicer   examples/data/embryo.slc    
```
![e2](https://user-images.githubusercontent.com/32848391/50738810-58af4380-11d8-11e9-8fc7-6c6959207224.jpg)


## Examples Gallery
A get-started tutorial script is available for download:
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples
python tutorial.py  
```
**Many more examples can be found in directories** _(scroll down to see the screenshots):_ <br>
[**examples/basic**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic)<br>
[**examples/advanced**](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced)<br>
[**examples/volumetric**](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric)<br>
[**examples/others**](https://github.com/marcomusy/vtkplotter/blob/master/examples/other).<br>
_(Scroll down to see the screenshots)._

|    |    |
|:-------------:|:-----|
| ![rabbit](https://user-images.githubusercontent.com/32848391/50738808-5816ad00-11d8-11e9-9854-c952be6fb941.jpg)  | Apply a *Moving Least Squares* algorithm to obtain a smooth surface from a to a large cloud of scattered points in space ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py)) <br />  `python advanced/moving_least_squares2D.py` |
|    |    |
| ![gyro](https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif) | Simulation of a gyroscope hanging from a spring ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py)) <br /> `python advanced/gyroscope1.py`|
|    |    |
|  ![ruth](https://user-images.githubusercontent.com/32848391/43984362-5c545a0e-9d00-11e8-8ce5-572b96bb91d1.gif)  | Simulation of [Rutherford scattering](https://en.wikipedia.org/wiki/Rutherford_scattering) of charged particles on a fixed target ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/particle_simulator.py))<br /> `python advanced/particle_simulator.py`   |
|    |    |
| ![qsine2](https://user-images.githubusercontent.com/32848391/47751431-06aae880-dc92-11e8-9fcf-6659123edbfa.gif)   | Quantum-tunnelling effect integrating the Schroedinger equation with 4th order Runge-Kutta method. The animation shows the evolution of a particle in a box hitting a sinusoidal potential barrier. ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/tunnelling2.py)) <br /> `python advanced/tunnelling2.py`   |
|    |    |
| ![turing](https://user-images.githubusercontent.com/32848391/40665257-1412a30e-635d-11e8-9536-4c73bf6bdd92.gif)   |  Visualizing a Turing system of reaction-diffusion between two molecules ([script](https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/turing.py)) <br /> `python advanced/turing.py`  |
<br />



### Mesh format conversion
The command `vtkconvert` can be used to convert multiple files from a format to a different one:
```
Usage: vtkconvert [-h] [-to] [files [files ...]]
allowed targets formats: [vtk, vtp, vtu, vts, ply, stl, byu, xml]

Example: > vtkconvert myfile.vtk -to ply
```
