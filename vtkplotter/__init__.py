__author__  = "Marco Musy"
__license__ = "MIT" 
__maintainer__ = "M. Musy, G. Dalmasso"
__email__   = "marco.musy@embl.es"
__status__  = "dev"
__website__ = "https://github.com/marcomusy/vtkplotter"


from vtkplotter.plotter import Plotter
from vtkplotter.plotter import __version__
from vtkplotter.colors import colorMap, printc
from vtkplotter.vtkio import ProgressBar
from vtkplotter.utils import vector, mag, norm, arange

from numpy import sin, cos, sqrt, exp, log, dot, cross, array
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class vtkPlotter:
    def __init__(self, shape=(1,1), N=None, size='auto', maxscreensize=(1100,1800), 
                 title='vtkPlotter', bg='w', bg2=None, axes=1, projection=False,
                 sharecam=True, verbose=True, interactive=None):
        import sys
        print('\nPlease use Plotter instead of obsolete vtkPlotter class.')
        print('For example:')
        print('   import vtkplotter')
        print('   vp = vtkplotter.Plotter()  # ***NOT plotter.vtkPlotter()***')
        print('   vp.sphere()')
        print('   vp.show()\n')
        sys.exit(0)
