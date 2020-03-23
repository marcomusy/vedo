"""
Stream tubes originating from a probing grid of points.
Data is from CFD analysis of airflow in an office with
ventilation and a burning cigarette.
"""
# see original script at:
# https://github.com/Kitware/VTK/blob/master/Examples/
#  VisualizationAlgorithms/Python/officeTube.py
from vtkplotter import *
from office_furniture import furniture


# We read a data file the is a CFD analysis of airflow in an office
# (with ventilation and a burning cigarette).
sgrid = loadStructuredGrid(datadir + "office.binary.vtk")


# Now we will generate multiple streamlines in the data. We create a
# grid of points of points and then use those as integration seeds.
seeds = Grid(pos=[2,2,1], normal=[1,0,0], resx=2, resy=3, c="gray")

# We select the integration order to use (RungeKutta order 4) and
# associate it with the streamer. We integrate in the forward direction.
slines = streamLines(
                    sgrid, seeds,
                    integrator="rk4",
                    direction="forward",
                    initialStepSize=0.01,
                    maxPropagation=15,
                    tubes={"radius":0.004, "varyRadius":2, "ratio":1},
                    )

comment = Text2D(__doc__, c="w")
show(slines, seeds, furniture(), comment, axes=0, bg='bb')
