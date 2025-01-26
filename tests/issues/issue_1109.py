import numpy as np
from vedo import Volume, show

# make up some fake data
X, Y, Z = np.mgrid[:4, :4, :2]
scalar_field = (X - 2) ** 2 + (Y - 2) ** 2 + (Z - 2) ** 2

vol = Volume(scalar_field.astype(int))
spacing = vol.spacing() # get the voxel size
# print("spacing", spacing)
# print('numpy array from Volume:', vol.pointdata)
# print('input_scalars', vol.pointdata['input_scalars'])

# extract a z-slice at index k=1
zslice = vol.zslice(k=1)
zslice.cmap("hot_r").lw(1).alpha(0.9).add_scalarbar3d()
# print("input_scalars", zslice.pointdata["input_scalars"])

# create a set of points at the cell centers and associate the scalar value
# corresponding to the bottom left corner of each voxel
cc = zslice.cell_centers().shift([-spacing[0] / 2, -spacing[1] / 2, 0])
cc.resample_data_from(zslice)

zslice.compute_normals()

zslice2 = zslice.clone()
zslice2.celldata["pixel_value"] = cc.pointdata["input_scalars"]
print(zslice2.celldata["pixel_value"])

lego = vol.legosurface(vmin=0, vmax=10).wireframe()

show(
    [
        (vol, lego, vol.cell_centers()),
        (
            lego,
            cc,
            zslice,
            zslice.labels("id"),
            zslice.labels("cellid"),
        ),
        zslice2,
    ],
    N=3,
    axes=1,
)
