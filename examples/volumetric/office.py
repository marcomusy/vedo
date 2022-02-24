"""Stream tubes originating from a probing grid of points.
Data is from CFD analysis of airflow in an office with
ventilation and a burning cigarette"""
from vedo import *
from off_furniture import furniture

# We read a data file the is a CFD analysis of airflow in an office
# (with ventilation and a burning cigarette).
fpath = download('https://vedo.embl.es/examples/data/office.binary.vtk')
sgrid = loadStructuredGrid(fpath)

# Create a grid of points and use those as integration seeds
seeds = Grid(pos=[2,2,1], normal=[1,0,0], res=[2,3], c="gray")

# Now we will generate multiple streamlines in the data.
# We select the integration order to use (RungeKutta order 4) and
# associate it with the streamer. We integrate in the forward direction.
slines = streamLines(sgrid, seeds,
                     integrator="rk4",
                     direction="forward",
                     initialStepSize=0.01,
                     maxPropagation=15,
                     tubes={"radius":0.004, "varyRadius":2, "ratio":1},
)
slines.addScalarBar3D(c='w')
slines.scalarbar.x(5) # reposition scalarbar at x=5

show(slines, seeds, furniture(), __doc__, axes=1, bg='bb').close()
