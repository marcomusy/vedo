"""Stream tubes originating from a probing grid of points.
Data is from CFD analysis of airflow in an office with
ventilation and a burning cigarette"""
from vedo import *
from off_furniture import furniture

fpath = download(dataurl + 'office.binary.vtk')
sgrid = loadStructuredGrid(fpath)

# Create a grid of points and use those as integration seeds
seeds = Grid(res=[2,3], c="gray").rotate_y(90).pos(2,2,1)

# Now we will generate multiple streamlines in the data
slines = StreamLines(
    sgrid,
    seeds,
    initial_step_size=0.01,
    max_propagation=15,
    tubes=dict(radius=0.005, mode=2, ratio=1),
)
slines.cmap("Reds").add_scalarbar3d(c='white')
slines.scalarbar.x(5) # reposition scalarbar at x=5

show(slines, seeds, furniture(), __doc__, axes=1, bg='bb').close()
