"""Stream tubes originating from a probing grid of points.
Data is from CFD analysis of airflow in an office with
ventilation and a burning cigarette."""
from vedo import *
from off_furniture import furniture

fpath = download(dataurl + 'office.binary.vtk')
sgrid = loadStructuredGrid(fpath)
ugrid = UnstructuredGrid(sgrid) # convert to unstructured grid which vedo supports

# Create a grid of points and use it as integration seeds
seeds = Grid(res=[2,3], c="white").rotate_y(90).pos(2,2,1)

streamlines = ugrid.compute_streamlines(seeds, initial_step_size=0.01, max_propagation=15)
streamlines.cmap("Reds").add_scalarbar3d(c='white')
streamlines.scalarbar = streamlines.scalarbar.clone2d("center-right", size=0.15)
print(streamlines)

show(streamlines, seeds, furniture(), __doc__, axes=1, bg='bb').close()
