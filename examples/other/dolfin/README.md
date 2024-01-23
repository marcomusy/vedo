# _FEniCS/Dolfin_ examples
In this directory you will find a bunch of examples of to visualize meshes in conjunction with 
[FEniCS/Dolfin](https://fenicsproject.org/) package.

The `plot()` function emulates the *matplotlib* functionality.

Install `mshr` with
```
conda install conda-forge::mshr
```

To gain more control on the property of the shown objects one can access the output of the `plot()`
method and change their properties, e.g.:
```python
plt = plot(u_solution)
msh = plt.actors[0]
msh.color('blue').alpha(0.5).cut_with_plane()  # etc
plt.show()
```
