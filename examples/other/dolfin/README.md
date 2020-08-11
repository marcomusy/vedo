# _FEniCS/Dolfin_ examples
In this directory you will find a bunch of examples of to visualize meshes in conjunction with 
[FEniCS/Dolfin](https://fenicsproject.org/) package.
It emulates the functionality of the `plot()` command of *matplotlib*.

Run any of the examples with:

`vedo -ir example.py`

To gain more control on the property of the shown objects one can access the output of the `plot()`
method and chenge their properties, e.g.:
```python
plt = plot(u)
msh = plt.actors[0]
msh.color('blue').alpha(0.5).cutWithPlane() # etc
plt.show()
```
