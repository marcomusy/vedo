# _FEniCS/Dolfin_ examples
In this directory you will find a bunch of examples of to visualize meshes in conjunction with 
[FEniCS/Dolfin](https://fenicsproject.org/) package.
It emulates the functionality of the `plot()` command of *matplotlib*.
To gain more control on the property of the shown objects see the analogous examples in 
[noplot](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/noplot).

To run the examples:
```bash
git clone https://github.com/marcomusy/vtkplotter.git
cd vtkplotter/examples/other/dolfin
python example.py  # on mac OSX try 'pythonw' instead
```
(_click thumbnail image to get to the python script_)

|    |    |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----|
| [![showmesh](https://user-images.githubusercontent.com/32848391/53026243-d2d31900-3462-11e9-9dde-518218c241b6.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/ex01_show-mesh.py)<br/>`ex01_show-mesh.py`               | Show dolfin meshes in different ways. |
|    |    |
| [![submesh](https://user-images.githubusercontent.com/32848391/56675428-4e984e80-66bc-11e9-90b0-43dde7e4cc29.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/demo_submesh.py)<br/> `demo_submesh.py`                   | How to extract matching sub meshes from a common mesh.  |
|    |    |
| [![pi_estimate](https://user-images.githubusercontent.com/32848391/56675429-4e984e80-66bc-11e9-9217-a0652a8e74fe.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/pi_estimate.py)<br/> `pi_estimate.py`                 | Estimate _pi_ by integrating a circle surface. Latex formulas can be added to the renderer directly. |
|    |    |
| [![tet_mesh](https://user-images.githubusercontent.com/32848391/53026244-d2d31900-3462-11e9-835a-1fa9d66d3dae.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/ex02_tetralize-mesh.py)<br/> `ex02_tetralize-mesh.py`    | Tetrahedral meshes generation with package _mshr_.  |
|    |    |
| [![poisson](https://user-images.githubusercontent.com/32848391/54925524-bec18200-4f0e-11e9-9eab-29fd61ef3b8e.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/ex03_poisson.py)<br/> `ex03_poisson.py`                   | Solving Poisson equation with Dirichlet conditions. |
|    |    |
| [![mixpoisson](https://user-images.githubusercontent.com/32848391/53045761-b220b880-348e-11e9-840f-94c5c0e86668.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/ex04_mixed-poisson.py)<br/> `ex04_mixed-poisson.py`    | Solving Poisson equation using a mixed (two-field) formulation. |
|    |    |
| [![nonmatching](https://user-images.githubusercontent.com/32848391/53044916-95838100-348c-11e9-928c-eefe8ba2e8ce.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/ex05_non-matching-meshes.py)<br/> `ex05_non-matching-meshes.py` | Interpolate functions between finite element spaces on non-matching meshes. |
|    |    |
| [![elasticity1](https://user-images.githubusercontent.com/32848391/53026245-d2d31900-3462-11e9-9db4-96211569d114.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/ex06_elasticity1.py)<br/> `ex06_elasticity1.py`       | Solving an elasticity problem. Show mesh displacements with arrows. |
|    |    |
| [![elasticity2](https://user-images.githubusercontent.com/32848391/53026246-d36baf80-3462-11e9-96a5-8eaf0bb0f9a4.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/ex06_elasticity2.py)<br/> `ex06_elasticity2.py`       | Solving an elasticity problem. Use scalars and vectors to colorize mesh displacements with different color maps. |
|    |    |
| [![stokes](https://user-images.githubusercontent.com/32848391/53044917-95838100-348c-11e9-9a94-aa10b8f1658c.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/ex07_stokes-iterative.py)<br/> `ex07_stokes-iterative.py`  | Stokes equations with an iterative solver. |
|    |    |
| [![elastodyn](https://user-images.githubusercontent.com/32848391/54932788-bd4a8680-4f1b-11e9-9326-33645171a45e.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/elastodynamics.py)<br/> `elastodynamics.py`             | Perform time integration of transient elastodynamics using the generalized-alpha method. |
|    |    |
| [![stokes](https://user-images.githubusercontent.com/32848391/55098209-aba0e480-50bd-11e9-8842-42d3f0b2d9c8.png)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/stokes.py)<br/> `stokes.py`                                | Solve 2D navier-stokes equations with boundary conditions. |
|    |    |
| [![ft02](https://user-images.githubusercontent.com/32848391/55499287-ed91d380-5645-11e9-8e9a-e31e2e3b1649.jpg)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/ft02_poisson_membrane.py)<br/> `ft02_poisson_membrane.py`    | Deflection of a membrane by a gaussian load. |
|    |    |
| [![ft04](https://user-images.githubusercontent.com/32848391/55578167-88a5ae80-5715-11e9-84ea-bdab54099887.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/ft04_heat_gaussian.py)<br/> `ft04_heat_gaussian.py`          | Diffusion of a Gaussian hill on a square domain. |
|    |    |
| [![cahn](https://user-images.githubusercontent.com/32848391/56664730-edb34b00-66a8-11e9-9bf3-73431f2a98ac.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/demo_cahn-hilliard.py)<br/> `demo_cahn-hilliard.py`          | Solution of a particular nonlinear time-dependent fourth-order equation, known as the Cahn-Hilliard equation. |
|    |    |
| [![navier-stokes_lshape](https://user-images.githubusercontent.com/32848391/56671156-6bc91f00-66b4-11e9-8c58-e6b71e2ad1d0.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/navier-stokes_lshape.py)<br/> `navier-stokes_lshape.py`  |  Solve the incompressible Navier-Stokes equations on an L-shaped domain using Chorin's splitting method. |
|    |    |
| [![turing_pattern](https://user-images.githubusercontent.com/32848391/56056437-77cfeb00-5d5c-11e9-9887-828e5745d547.gif)](https://github.com/marcomusy/vtkplotter/blob/master/examples/other/dolfin/turing_pattern.py)<br/> `turing_pattern.py`        | Solve a reaction-diffusion problem on a 2D domain.  |
