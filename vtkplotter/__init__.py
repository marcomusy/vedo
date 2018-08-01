
from vtkplotter.plotter import Plotter
from vtkplotter.colors import colorMap
from vtkplotter.vtkio import printc, ProgressBar
from vtkplotter.utils import vector, mag, norm, arange
from numpy import sin, cos, sqrt, exp, log, dot, cross, array

class vtkPlotter:
    def __init__(self, shape=(1,1), N=None, size='auto', maxscreensize=(1100,1800), 
                 title='vtkPlotter', bg='w', bg2=None, axes=1, projection=False,
                 sharecam=True, verbose=True, interactive=None):
        print('\nPlease use Plotter instead of obsolete vtkPlotter class.')
        print('For example:')
        print('   import vtkplotter')
        print('   vp = vtkplotter.Plotter()  # ***NOT plotter.vtkPlotter()***')
        print('   vp.sphere()')
        print('   vp.show()\n')
        exit(0)
