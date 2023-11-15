from vedo import *
from vedo.pyplot import matrix


def func(event):
    if not event.object:
        return

    if event.object.name == "Sphere":
        sph = Sphere()
        arr = np.random.rand(sph.npoints)*np.random.rand()
        sph.cmap("Blues", arr)
        # sph.add_scalarbar(title="Elevation", c="k")
        sph.add_scalarbar3d(title="Elevation", c="k")
        plt.remove("Sphere").add(sph).render()

    if event.object.name == "Matrix":
        arr = np.eye(n, m)/2 + np.random.randn(n, m)*0.1
        mat = matrix(arr, scale=0.15).scale(2).y(2)
        plt.remove("Matrix").add(mat).render()

sph = Sphere()

n, m = (6, 5)
mat = matrix(
    np.eye(n, m)/2 + np.random.randn(n, m)*0.1,
    scale=0.15,  # size of bin labels; set it to 0 to remove labels
).scale(2).y(2)

plt = Plotter()
plt.add_callback("mouse left click", func)
plt.show(sph, mat, 'click to change random data')

