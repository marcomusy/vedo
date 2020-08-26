from __future__ import division, print_function
import numpy as np
import vtk
import vedo
import vedo.colors as colors
import vedo.docs as docs
import vedo.settings as settings
import vedo.utils as utils
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, numpy_to_vtkIdTypeArray

__doc__ = (
    """Base classes. Do not instantiate these."""
    + docs._defs
)

__all__ = ['Base3DProp',
           'BaseActor',
           'BaseGrid',
           "probePoints",
           "probeLine",
           "probePlane",
           "streamLines",
           ]


###############################################################################
# classes
class Base3DProp(object):
    """
    """
    def __init__(self):

        self.filename = ""
        self.name = ""
        self.trail = None
        self.trailPoints = []
        self.trailSegmentSize = 0
        self.trailOffset = None
        self.shadow = None
        self.shadowX = None
        self.shadowY = None
        self.shadowZ = None
        self.axes = None
        self.picked3d = None
        self.units = None
        self.top = np.array([0,0,1])
        self.base = np.array([0,0,0])
        self.info = dict()
        self._time = 0
        self._legend = None
        self.renderedAt = set()
        self.transform = None
        self._set2actcam = False


    def pickable(self, value=None):
        """Set/get pickable property of mesh."""
        if value is None:
            return self.GetPickable()
        else:
            self.SetPickable(value)
            return self

    def time(self, t=None):
        """Set/get object's absolute time."""
        if t is None:
            return self._time
        self._time = t
        return self  # return itself to concatenate methods

    def origin(self, x=None, y=None, z=None):
        """Set/get object's origin.
        Relevant to control the scaling with `scale()` and rotations.
        Has no effect on position."""
        if x is None:
            return np.array(self.GetOrigin())
        if z is None:  # assume o_x is of the form (x,y,z)
            if y is not None: # assume x and y are given so z=0
                z=0
            else: # assume o_x is of the form (x,y,z)
                x, y, z = x
        self.SetOrigin(x, y, z)
        return self

    def pos(self, x=None, y=None, z=None):
        """Set/Get object position."""
        if x is None:
            return np.array(self.GetPosition())
        if z is None:  # assume p_x is of the form (x,y,z)
            if y is not None: # assume x and y are given so z=0
                z=0
            else: # assume p_x is of the form (x,y,z)
                x, y, z = x
        self.SetPosition(x, y, z)

        if self.trail:
            self.updateTrail()
        if self.shadow:
            self._updateShadow()
        return self  # return itself to concatenate methods

    def addPos(self, dp_x=None, dy=None, dz=None):
        """Add vector to current object position."""
        p = np.array(self.GetPosition())
        if dz is None:  # assume dp_x is of the form (x,y,z)
            self.SetPosition(p + dp_x)
        else:
            self.SetPosition(p + [dp_x, dy, dz])
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self._updateShadow()
        return self

    def x(self, position=None):
        """Set/Get object position along x axis."""
        p = self.GetPosition()
        if position is None:
            return p[0]
        self.SetPosition(position, p[1], p[2])
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self._updateShadow()
        return self

    def y(self, position=None):
        """Set/Get object position along y axis."""
        p = self.GetPosition()
        if position is None:
            return p[1]
        self.SetPosition(p[0], position, p[2])
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self._updateShadow()
        return self

    def z(self, position=None):
        """Set/Get object position along z axis."""
        p = self.GetPosition()
        if position is None:
            return p[2]
        self.SetPosition(p[0], p[1], position)
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self._updateShadow()
        return self

    def rotate(self, angle, axis=(1, 0, 0), axis_point=(0, 0, 0), rad=False):
        """Rotate around an arbitrary `axis` passing through `axis_point`."""
        if rad:
            anglerad = angle
        else:
            anglerad = np.deg2rad(angle)
        axis = utils.versor(axis)
        a = np.cos(anglerad / 2)
        b, c, d = -axis * np.sin(anglerad / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )
        rv = np.dot(R, self.GetPosition() - np.array(axis_point)) + axis_point

        if rad:
            angle *= 180.0 / np.pi
        # this vtk method only rotates in the origin of the object:
        self.RotateWXYZ(angle, axis[0], axis[1], axis[2])
        self.SetPosition(rv)
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                           self.shadow.GetProperty().GetColor(),
                           self.shadow.GetProperty().GetOpacity())
        return self

    def rotateX(self, angle, rad=False):
        """Rotate around x-axis. If angle is in radians set ``rad=True``."""
        if rad:
            angle *= 180 / np.pi
        T = vtk.vtkTransform()
        self.ComputeMatrix()
        T.SetMatrix(self.GetMatrix())
        T.PostMultiply()
        T.RotateX(angle)
        self.SetOrientation(T.GetOrientation())
        self.SetPosition(T.GetPosition())
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                           self.shadow.GetProperty().GetColor(),
                           self.shadow.GetProperty().GetOpacity())
        return self

    def rotateY(self, angle, rad=False):
        """Rotate around y-axis. If angle is in radians set ``rad=True``."""
        if rad:
            angle *= 180 / np.pi
        T = vtk.vtkTransform()
        self.ComputeMatrix()
        T.SetMatrix(self.GetMatrix())
        T.PostMultiply()
        T.RotateY(angle)
        self.SetOrientation(T.GetOrientation())
        self.SetPosition(T.GetPosition())
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                           self.shadow.GetProperty().GetColor(),
                           self.shadow.GetProperty().GetOpacity())
        return self

    def rotateZ(self, angle, rad=False):
        """Rotate around z-axis. If angle is in radians set ``rad=True``."""
        if rad:
            angle *= 180 / np.pi
        T = vtk.vtkTransform()
        self.ComputeMatrix()
        T.SetMatrix(self.GetMatrix())
        T.PostMultiply()
        T.RotateZ(angle)
        self.SetOrientation(T.GetOrientation())
        self.SetPosition(T.GetPosition())
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                           self.shadow.GetProperty().GetColor(),
                           self.shadow.GetProperty().GetOpacity())
        return self


    def orientation(self, newaxis=None, rotation=0, rad=False):
        """
        Set/Get object orientation.

        :param rotation: If != 0 rotate object around newaxis.
        :param rad: set to True if angle is in radians.

        |gyroscope2| |gyroscope2.py|_
        """
        if rad:
            rotation *= 180.0 / np.pi
        if self.top is None or self.base is None:
            initaxis = (0,0,1)
        else:
            initaxis = utils.versor(self.top - self.base)
        if newaxis is None:
            return initaxis
        newaxis = utils.versor(newaxis)
        pos = np.array(self.GetPosition())
        crossvec = np.cross(initaxis, newaxis)
        angle = np.arccos(np.dot(initaxis, newaxis))
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.Translate(-pos)
        if rotation:
            T.RotateWXYZ(rotation, initaxis)
        T.RotateWXYZ(np.rad2deg(angle), crossvec)
        T.Translate(pos)
        self.SetUserTransform(T)
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                           self.shadow.GetProperty().GetColor(),
                           self.shadow.GetProperty().GetOpacity())
        return self


    def scale(self, s=None, absolute=False):
        """Set/get object's scaling factor.

        :param float, list s: scaling factor(s).
        :param bool absolute: if True previous scaling factors are ignored.

        .. note:: if `s=(sx,sy,sz)` scale differently in the three coordinates."""
        if s is None:
            return np.array(self.GetScale())
        if absolute:
            self.SetScale(s)
        else:
            self.SetScale(np.multiply(self.GetScale(), s))
        return self


    def getTransform(self):
        """
        Check if ``info.transform`` exists and returns a ``vtkTransform``.
        Otherwise return current user transformation (where the object is currently placed).
        """
        if self.transform:
            return self.transform
        else:
            T = self.GetMatrix()
            tr = vtk.vtkTransform()
            tr.SetMatrix(T)
            return tr

    def setTransform(self, T):
        """
        Transform object position and orientation.
        """
        if isinstance(T, vtk.vtkMatrix4x4):
            self.SetUserMatrix(T)
        else:
            try:
                self.SetUserTransform(T)
            except TypeError:
                colors.printc('\times Error in setTransform():',
                              'consider transformPolydata() instead.', c='r')
        return self


    def on(self):
        """Switch on  object visibility. Object is not removed."""
        self.VisibilityOn()
        return self

    def off(self):
        """Switch off object visibility. Object is not removed."""
        self.VisibilityOff()
        return self

    def box(self, scale=1, pad=0):
        """Return the bounding box as a new ``Mesh``.

        :param float scale: box size can be scaled by a factor
        :param float,list pad: a constant pad can be added (can be a list [padx,pady,padz])

        .. hint:: |latex.py|_
        """
        b = self.GetBounds()
        from vedo.shapes import Box
        if not utils.isSequence(pad):
            pad=[pad,pad,pad]
        pos = (b[0]+b[1])/2, (b[3]+b[2])/2, (b[5]+b[4])/2
        length, width, height = b[1]-b[0], b[3]-b[2], b[5]-b[4]
        bx = Box(pos,
                 length*scale+pad[0], width*scale+pad[1], height*scale+pad[2],
                 c='gray')
        if hasattr(self, 'GetProperty'): # could be Assembly
            if isinstance(self.GetProperty(), vtk.vtkProperty): # could be volume
                pr = vtk.vtkProperty()
                pr.DeepCopy(self.GetProperty())
                bx.SetProperty(pr)
        bx.flat().lighting('off')
        bx.wireframe()
        return bx

    def useBounds(self, ub=True):
        """Instruct the current camera to either take into account or ignore
        the object bounds when resetting."""
        self.SetUseBounds(ub)
        return self

    def bounds(self):
        """Get the object bounds.
        Returns a list in format [xmin,xmax, ymin,ymax, zmin,zmax]."""
        return self.GetBounds()

    def xbounds(self):
        """Get the bounds [xmin,xmax]."""
        b = self.GetBounds()
        return (b[0], b[1])

    def ybounds(self):
        """Get the bounds [ymin,ymax]."""
        b = self.GetBounds()
        return (b[2], b[3])

    def zbounds(self):
        """Get the bounds [zmin,zmax]."""
        b = self.GetBounds()
        return (b[4], b[5])

    def diagonalSize(self):
        """Get the length of the diagonal of mesh bounding box."""
        b = self.GetBounds()
        return np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])* 2 + (b[5]-b[4])**2)

    def maxBoundSize(self):
        """Get the maximum size in x, y or z of the bounding box."""
        b = self.GetBounds()
        return max(b[1]-b[0], b[3]-b[2], b[5]-b[4])

    def minBoundSize(self):
        """Get the minimum size in x, y or z of the bounding box."""
        b = self.GetBounds()
        xm = b[1] - b[0]
        ym = b[3] - b[2]
        zm = b[5] - b[4]
        m = 0
        if xm: m=xm
        if ym and m>ym: m=ym
        if zm and m>zm: m=zm
        return m

    def printInfo(self):
        """Print information about an object."""
        utils.printInfo(self)
        return self

    def buildAxes(self, **kargs):
        """Draw axes for the input object or for a specified range.
        Returns an ``Assembly`` object.

        - `xtitle`,                ['x'], x-axis title text
        - `xrange`,               [None], x-axis range in format (xmin, ymin), default is automatic.
        - `numberOfDivisions`,    [None], approximate number of divisions on the longest axis
        - `axesLineWidth`,           [1], width of the axes lines
        - `gridLineWidth`,           [1], width of the grid lines
        - `reorientShortTitle`,   [True], titles shorter than 2 letter are placed horizontally
        - `titleDepth`,              [0], extrusion fractional depth of title text
        - `xyGrid`,               [True], show a gridded wall on plane xy
        - `yzGrid`,               [True], show a gridded wall on plane yz
        - `zxGrid`,               [True], show a gridded wall on plane zx
        - `zxGrid2`,             [False], show zx plane on opposite side of the bounding box
        - `xyGridTransparent`    [False], make grid plane completely transparent
        - `xyGrid2Transparent`   [False], make grid plane completely transparent on opposite side box
        - `xyPlaneColor`,       ['gray'], color of the plane
        - `xyGridColor`,        ['gray'], grid line color
        - `xyAlpha`,              [0.15], grid plane opacity
        - `xyFrameLine`,          [None], add a frame for the plane
        - `showTicks`,            [True], show major ticks
        - `digits`,               [None], use this number of significant digits in scientific notation
        - `titleFont`,              [''], font for axes titles
        - `labelFont`,              [''], font for numeric labels
        - `textScale`,             [1.0], global scaling factor for text elements (titles, labels)
        - `xTitlePosition`,       [0.32], title fractional positions along axis
        - `xTitleOffset`,         [0.05], title fractional offset distance from axis line
        - `xTitleJustify`, ["top-right"], title justification
        - `xTitleRotation`,          [0], add a rotation of the axis title
        - `xTitleBox`,           [False], add a box around title text
        - `xLineColor`,      [automatic], color of the x-axis
        - `xTitleColor`,     [automatic], color of the axis title
        - `xTitleBackfaceColor`,  [None],  color of axis title on its backface
        - `xTitleSize`,          [0.025], size of the axis title
        - 'xTitleItalic',            [0], a bool or float to make the font italic
        - `xHighlightZero`,       [True], draw a line highlighting zero position if in range
        - `xHighlightZeroColor`, [autom], color of the line highlighting the zero position
        - `xTickLength`,         [0.005], radius of the major ticks
        - `xTickThickness`,     [0.0025], thickness of the major ticks along their axis
        - `xMinorTicks`,             [1], number of minor ticks between two major ticks
        - `xValuesAndLabels`          [], assign custom tick positions and labels [(pos1, label1), ...]
        - `xLabelColor`,     [automatic], color of numeric labels and ticks
        - `xLabelPrecision`,         [2], nr. of significative digits to be shown
        - `xLabelSize`,          [0.015], size of the numeric labels along axis
        - 'xLabelRotation',          [0], rotate clockwise [1] or anticlockwise [-1] by 90 degrees
        - 'xFlipText',           [False], flip axis title and numeric labels orientation
        - `xLabelOffset`,        [0.025], offset of numeric labels
        - `tipSize`,              [0.01], size of the arrow tip
        - `limitRatio`,           [0.04], below this ratio don't plot small axis

        :Example:

            .. code-block:: python

                from vedo import Box, show
                b = Box(pos=(1,2,3), length=8, width=9, height=7).alpha(0)
                bax = b.buildAxes(c='k')  # returns Assembly object
                show(b, bax)

        |customAxes| |customAxes.py|_

        |customIndividualAxes| |customIndividualAxes.py|_
        """
        from vedo.addons import buildAxes
        a = buildAxes(self, **kargs)
        self.axes = a
        return a

#    def delete(self):
#        """
#        """
#        settings.collectable_actors.remove(self)
#        del self


    def show(self, **options):
        """
        Create on the fly an instance of class ``Plotter`` or use the last existing one to
        show one single object.

        This is meant as a shortcut. If more than one object needs to be visualised
        please use the syntax `show(mesh1, mesh2, volume, ..., options)`.

        :param bool newPlotter: if set to `True`, a call to ``show`` will instantiate
            a new ``Plotter`` object (a new window) instead of reusing the first created.
            See e.g.: |readVolumeAsIsoSurface.py|_

        :return: the current ``Plotter`` class instance.

        .. note:: E.g.:

            .. code-block:: python

                from vedo import *
                s = Sphere()
                s.show(at=1, N=2)
                c = Cube()
                c.show(at=0, interactive=True)
        """
        from vedo.plotter import show
        return show(self, **options)


########################################################################################
class BaseActor(Base3DProp):
    """Adds functionality to ``Mesh``, ``Assembly``,
    ``Volume`` and ``Picture`` objects.

    .. warning:: Do not use this class to instance objects, use the above ones.
    """
    def __init__(self):
        Base3DProp.__init__(self)

        self.scalarbar = None
        self._mapper = None

        self.flagText = None
        self._caption = None


    def mapper(self, newMapper=None):
        """Return the ``vtkMapper`` data object, or update it with a new one."""
        if newMapper:
            self.SetMapper(newMapper)
            if self._mapper:
                iptdata = self._mapper.GetInput()
                if iptdata:
                    newMapper.SetInputData(self._mapper.GetInput())
            self._mapper = newMapper
            self._mapper.Modified()
        return self._mapper

    def inputdata(self):
        """Return the VTK input data object."""
        if self._mapper:
            return self._mapper.GetInput()
        return self.GetMapper().GetInput()


    def N(self):
        """Retrieve number of points. Shortcut for `NPoints()`."""
        return self.inputdata().GetNumberOfPoints()

    def NPoints(self):
        """Retrieve number of points. Same as `N()`."""
        return self.inputdata().GetNumberOfPoints()

    def NCells(self):
        """Retrieve number of cells."""
        return self.inputdata().GetNumberOfCells()

    def cellCenters(self):
        """Get the coordinates of the cell centers.

        |delaunay2d| |delaunay2d.py|_
        """
        vcen = vtk.vtkCellCenters()
        if hasattr(self, "polydata"):
            vcen.SetInputData(self.polydata())
        else:
            vcen.SetInputData(self.inputdata())
        vcen.Update()
        return vtk_to_numpy(vcen.GetOutput().GetPoints().GetData())

    def findCellsWithin(self, xbounds=(), ybounds=(), zbounds=(), c=None):
        """
        Find cells that are within specified bounds.
        Setting a color will add a vtk array to colorize these cells.
        """
        if len(xbounds) == 6:
            bnds = xbounds
        else:
            bnds = list(self.bounds())
            if len(xbounds) == 2:
                bnds[0] = xbounds[0]
                bnds[1] = xbounds[1]
            if len(ybounds) == 2:
                bnds[2] = ybounds[0]
                bnds[3] = ybounds[1]
            if len(zbounds) == 2:
                bnds[4] = zbounds[0]
                bnds[5] = zbounds[1]

        cellIds = vtk.vtkIdList()
        self.cell_locator = vtk.vtkCellTreeLocator()
        self.cell_locator.SetDataSet(self.polydata())
        self.cell_locator.BuildLocator()
        self.cell_locator.FindCellsWithinBounds(bnds, cellIds)

        if c is not None:
            cellData = vtk.vtkUnsignedCharArray()
            cellData.SetNumberOfComponents(3)
            cellData.SetName('CellsWithinBoundsColor')
            cellData.SetNumberOfTuples(self.polydata(False).GetNumberOfCells())
            defcol = np.array(self.color())*255
            for i in range(cellData.GetNumberOfTuples()):
                cellData.InsertTuple(i, defcol)
            self.polydata(False).GetCellData().SetScalars(cellData)
            self._mapper.ScalarVisibilityOn()
            flagcol = np.array(colors.getColor(c))*255

        cids = []
        for i in range(cellIds.GetNumberOfIds()):
            cid = cellIds.GetId(i)
            if c is not None:
                cellData.InsertTuple(cid, flagcol)
            cids.append(cid)

        return np.array(cids)


    def lighting(self, style='', ambient=None, diffuse=None,
                 specular=None, specularPower=None, specularColor=None):
        """
        Set the ambient, diffuse, specular and specularPower lighting constants.

        :param str,int style: preset style,
            option presets are `[metallic, plastic, shiny, glossy, ambient, off]`
        :param float ambient: ambient fraction of emission [0-1]
        :param float diffuse: emission of diffused light in fraction [0-1]
        :param float specular: fraction of reflected light [0-1]
        :param float specularPower: precision of reflection [1-100]
        :param color specularColor: color that is being reflected by the surface

        |wikiphong|

        |specular| |specular.py|_
        """
        pr = self.GetProperty()

        if style:
            if style=='off':
                pr.SetInterpolationToFlat()
                pr.LightingOff()
                return self

            if hasattr(pr, "GetColor"):  # could be Volume
                c = pr.GetColor()
            else:
                c = (1,1,0.99)
            mpr = self._mapper
            if hasattr(mpr, 'GetScalarVisibility') and mpr.GetScalarVisibility():
                c = (1,1,0.99)
            if style=='metallic': pars = [0.1, 0.3, 1.0, 10, c]
            elif style=='plastic' : pars = [0.3, 0.4, 0.3,  5, c]
            elif style=='shiny'   : pars = [0.2, 0.6, 0.8, 50, c]
            elif style=='glossy'  : pars = [0.1, 0.7, 0.9, 90, (1,1,0.99)]
            elif style=='ambient' : pars = [0.8, 0.1, 0.0,  0, (1,1,1)]
            elif style=='default' : pars = [0.1, 1.0, 0.05, 5, c]
            else:
                colors.printc("Error in lighting(): Available styles are", c='r')
                colors.printc("[default,metallic,plastic,shiny,glossy,ambient,off]", c='r')
                raise RuntimeError()
            pr.SetAmbient(pars[0])
            pr.SetDiffuse(pars[1])
            pr.SetSpecular(pars[2])
            pr.SetSpecularPower(pars[3])
            if hasattr(pr, "GetColor"): pr.SetSpecularColor(pars[4])

        if ambient is not None: pr.SetAmbient(ambient)
        if diffuse is not None: pr.SetDiffuse(diffuse)
        if specular is not None: pr.SetSpecular(specular)
        if specularPower is not None: pr.SetSpecularPower(specularPower)
        if specularColor is not None: pr.SetSpecularColor(colors.getColor(specularColor))
        return self

    def printHistogram(self, bins=10, height=10, logscale=False, minbin=0,
                       horizontal=False, char=u"\U00002589",
                       c=None, bold=True, title='Histogram'):
        """
        Ascii histogram printing.
        Input can also be ``Volume`` or ``Mesh``.
        Returns the raw data before binning (useful when passing vtk objects).

        :param int bins: number of histogram bins
        :param int height: height of the histogram in character units
        :param bool logscale: use logscale for frequencies
        :param int minbin: ignore bins before minbin
        :param bool horizontal: show histogram horizontally
        :param str char: character to be used
        :param str,int c: ascii color
        :param bool char: use boldface
        :param str title: histogram title

        :Example:
            .. code-block:: python

                from vedo import printHistogram
                import numpy as np
                d = np.random.normal(size=1000)
                data = printHistogram(d, c='blue', logscale=True, title='my scalars')
                data = printHistogram(d, c=1, horizontal=1)
                print(np.mean(data)) # data here is same as d

            |printhisto|
        """
        utils.printHistogram(self, bins, height, logscale, minbin,
                             horizontal, char, c, bold, title)
        return self


    def c(self, color=False):
        """
        Shortcut for `color()`.
        If None is passed as input, will use colors from current active scalars.
        """
        return self.color(color)


    def getArrayNames(self):
        from vtk.numpy_interface import dataset_adapter
        wrapped = dataset_adapter.WrapDataObject(self.GetMapper().GetInput())
        return {"PointData":wrapped.PointData.keys(),
                "CellData":wrapped.CellData.keys() }


    def getPointArray(self, name=0):
        """Return point array content as a ``numpy.array``.
        This can be identified either as a string or by an integer number.

        Getting an array also makes it the active one, if more than one is present.
        """
        if hasattr(self, '_polydata') and self._polydata:
            data = self._polydata.GetPointData()
            if isinstance(name, int):
                name = data.GetArrayName(name)
            arr = data.GetArray(name)
            if not arr:
                return None

            self._mapper.ScalarVisibilityOn()
            if settings.autoResetScalarRange:
                self._mapper.SetScalarRange(arr.GetRange())

        else:
            indata = self.inputdata()
            if indata:
                data = indata.GetPointData()
                if isinstance(name, int):
                    name = data.GetArrayName(name)
                arr = data.GetArray(name)
                if not arr:
                    return None

        data.SetActiveScalars(name)
        self._mapper.SetScalarModeToUsePointData()
        return vtk_to_numpy(arr)


    def getCellArray(self, name=0):
        """Return cell array content as a ``numpy.array``.
        This can be identified either as a string or by an integer number.

        Getting an array also makes it the active one, if more than one is present.
        """
        if hasattr(self, '_polydata') and self._polydata:
            data = self._polydata.GetCellData()
            if isinstance(name, int):
                name = data.GetArrayName(name)
            arr = data.GetArray(name)
            if not arr:
                return None

            self._mapper.ScalarVisibilityOn()
            if settings.autoResetScalarRange:
                self._mapper.SetScalarRange(arr.GetRange())

        else:
            indata = self.inputdata()
            if indata:
                data = indata.GetCellData()
                if isinstance(name, int):
                    name = data.GetArrayName(name)
                arr = data.GetArray(name)
                if not arr:
                    return None

        data.SetActiveScalars(name)
        self._mapper.SetScalarModeToUseCellData()
        return vtk_to_numpy(arr)


    def addIDs(self, asfield=False):
        """
        Generate point and cell ids.

        :param bool asfield: flag to control whether to generate scalar or field data.
        """
        ids = vtk.vtkIdFilter()
        ids.SetInputData(self._polydata)
        ids.PointIdsOn()
        ids.CellIdsOn()
        if asfield:
            ids.FieldDataOn()
        else:
            ids.FieldDataOff()
        ids.Update()
        return self._update(ids.GetOutput())

    def selectPointArray(self, name):
        """Make this point array the active one. Name can be a string or integer."""
        data = self.inputdata().GetPointData()
        if isinstance(name, int):
            name = data.GetArray(name)
        data.SetActiveScalars(name)
        if hasattr(self._mapper, 'SetArrayName'):
            self._mapper.SetArrayName(name)
        self._mapper.SetScalarModeToUsePointData()
        return self

    def selectCellArray(self, name):
        """Make this cell array the active one. Name can be a string or integer."""
        data = self.inputdata().GetCellData()
        if isinstance(name, int):
            name = data.GetArray(name)
        data.SetActiveScalars(name)
        if hasattr(self._mapper, 'SetArrayName'):
            self._mapper.SetArrayName(name)
        self._mapper.SetScalarModeToUseCellData()
        return self

    def addPointArray(self, input_array, name):
        """
        Add point array and assign it a name.

        |mesh_coloring| |mesh_coloring.py|_
        """
        data = self.inputdata()

        if isinstance(input_array, vtk.vtkDataArray):
            data.GetPointData().AddArray(input_array)
            input_array.SetName(name)
            data.GetPointData().SetActiveScalars(name)
            self._mapper.ScalarVisibilityOn()
            self._mapper.SetScalarRange(input_array.GetRange())
            if hasattr(self._mapper, 'SetArrayName'):
                self._mapper.SetArrayName(name)
            self._mapper.SetScalarModeToUsePointData()
            return self

        if len(input_array) != data.GetNumberOfPoints():
            colors.printc('Error in addPointArray(): Number of inputs != nr. of points',
                          len(input_array), data.GetNumberOfPoints(), c='r')
            raise RuntimeError()

        nparr = np.ascontiguousarray(input_array)
        if len(nparr.shape)==1: # scalars
            varr = numpy_to_vtk(nparr, deep=True)
            varr.SetName(name)
            data.GetPointData().AddArray(varr)
            data.GetPointData().SetActiveScalars(name)
            self._mapper.SetScalarModeToUsePointData()
            if hasattr(self._mapper, 'ScalarVisibilityOn'): # could be volume mapper
                self._mapper.ScalarVisibilityOn()
                self._mapper.SetScalarRange(varr.GetRange())

        elif len(nparr.shape)==2: # vectors or higher dim ntuples
            varr = vtk.vtkFloatArray()
            varr.SetNumberOfComponents(nparr.shape[1])
            varr.SetName(name)
            for v in nparr:
                varr.InsertNextTuple(v)
            data.GetPointData().AddArray(varr)
            if nparr.shape[1] == 3:
                data.GetPointData().SetActiveVectors(name)
        else:
            colors.printc('Error in addPointArray(): cannot deal with shape:',
                          nparr.shape, c='r')
            return self

        if hasattr(self._mapper, 'SetArrayName'):
            self._mapper.SetArrayName(name)
        self._mapper.SetScalarModeToUsePointData()
        return self

    def addCellArray(self, input_array, name):
        """
        Add Cell array and assign it a name.

        |mesh_coloring| |mesh_coloring.py|_
        """
        data = self.inputdata()

        if isinstance(input_array, vtk.vtkDataArray):
            data.GetCellData().AddArray(input_array)
            input_array.SetName(name)
            data.GetCellData().SetActiveScalars(name)
            self._mapper.ScalarVisibilityOn()
            self._mapper.SetScalarRange(input_array.GetRange())
            if hasattr(self._mapper, 'SetArrayName'):
                self._mapper.SetArrayName(name)
            self._mapper.SetScalarModeToUseCellData()
            return self

        if len(input_array) != data.GetNumberOfCells():
            colors.printc('Error in addCellArray(): Number of inputs != nr. of Cells',
                          len(input_array), data.GetNumberOfCells(), c='r')
            raise RuntimeError()

        nparr = np.ascontiguousarray(input_array)
        if len(nparr.shape)==1: # scalars
            varr = numpy_to_vtk(nparr, deep=True)
            varr.SetName(name)
            data.GetCellData().AddArray(varr)
            data.GetCellData().SetActiveScalars(name)
            if hasattr(self._mapper, 'ScalarVisibilityOn'): # could be volume mapper
                self._mapper.ScalarVisibilityOn()
                self._mapper.SetScalarRange(varr.GetRange())

        elif len(nparr.shape)==2: # vectors or higher dim ntuples
            varr = vtk.vtkFloatArray()
            varr.SetNumberOfComponents(nparr.shape[1])
            varr.SetName(name)
            for v in nparr:
                varr.InsertNextTuple(v)
            data.GetCellData().AddArray(varr)
            if nparr.shape[1] == 3:
                data.GetCellData().SetActiveVectors(name)
        else:
            colors.printc('Error in addCellArray(): cannot deal with shape:',
                          nparr.shape, c='r')
            return self

        if hasattr(self._mapper, 'SetArrayName'):
            self._mapper.SetArrayName(name)
        self._mapper.SetScalarModeToUseCellData()
        return self


    def addPointScalars(self, scalars, name):
        """addPointScalars is OBSOLETE: use addPointArray."""
        colors.printc("WARNING - addPointScalars is OBSOLETE: use addPointArray.", c='y', box='-')
        return self.addPointArray(scalars, name)
    def addPointVectors(self, vectors, name):
        """addPointVectors is OBSOLETE: use addPointArray."""
        colors.printc("WARNING - addPointVectors is OBSOLETE: use addPointArray.", c='y', box='-')
        return self.addPointArray(vectors, name)
    def addCellScalars(self, scalars, name):
        """addCellScalars is OBSOLETE: use addCellArray."""
        colors.printc("WARNING - addCellScalars is OBSOLETE: use addCellArray.", c='y', box='-')
        return self.addCellArray(scalars, name)
    def addCellVectors(self, vectors, name):
        """addCellVectors is OBSOLETE: use addCellArray."""
        colors.printc("WARNING - addCellVectors is OBSOLETE: use addCellArray.", c='y', box='-')
        return self.addCellArray(vectors, name)


    def gradient(self, arrname=None, on='points'):
        """
        Compute and return the gradiend of a scalar field as a numpy array.

        :param str arrname: name of the existing scalar field
        :param str on: either 'points' or 'cells'

        |isolines| |isolines.py|_
        """
        gra = vtk.vtkGradientFilter()
        if on.startswith('p'):
            varr = self.inputdata().GetPointData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        else:
            varr = self.inputdata().GetCellData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        if not arrname:
            if self.GetScalars():
                arrname = varr.GetScalars().GetName()
            else:
                colors.printc('Error in gradient: no scalars found for', on, c='r')
                raise RuntimeError
        gra.SetInputData(self.inputdata())
        gra.SetInputScalars(tp, arrname)
        gra.SetResultArrayName('Gradients')
        gra.Update()
        if on.startswith('p'):
            gvecs = vtk_to_numpy(gra.GetOutput().GetPointData().GetArray('Gradients'))
        else:
            gvecs = vtk_to_numpy(gra.GetOutput().GetCellData().GetArray('Gradients'))
        return gvecs


    def mapCellsToPoints(self):
        """
        Transform cell data (i.e., data specified per cell)
        into point data (i.e., data specified at each vertex).
        The method of transformation is based on averaging the data values
        of all cells using a particular point.
        """
        c2p = vtk.vtkCellDataToPointData()
        c2p.SetInputData(self.inputdata())
        c2p.Update()
        self._mapper.SetScalarModeToUsePointData()
        return self._update(c2p.GetOutput())

    def mapPointsToCells(self):
        """
        Transform point data (i.e., data specified per point)
        into cell data (i.e., data specified per cell).
        The method of transformation is based on averaging the data values
        of all points defining a particular cell.

        |mesh_map2cell| |mesh_map2cell.py|_
        """
        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(self.inputdata())
        p2c.Update()
        self._mapper.SetScalarModeToUseCellData()
        return self._update(p2c.GetOutput())


    def addScalarBar(self,
                     pos=(0.8,0.05),
                     title="",
                     titleXOffset=0,
                     titleYOffset=15,
                     titleFontSize=12,
                     nlabels=None,
                     c=None,
                     horizontal=False,
                     useAlpha=True,
    ):
        """
        Add a 2D scalar bar for the specified obj.

        .. hint:: |mesh_coloring| |mesh_coloring.py|_ |scalarbars.py|_
        """
        import vedo.addons as addons
        self.scalarbar = addons.addScalarBar(self,
                                             pos,
                                             title,
                                             titleXOffset,
                                             titleYOffset,
                                             titleFontSize,
                                             nlabels,
                                             c,
                                             horizontal,
                                             useAlpha,
                                             )
        return self


    def addScalarBar3D(self,
        pos=None,
        sx=None,
        sy=None,
        title='',
        titleFont="",
        titleXOffset = -1.5,
        titleYOffset = 0.0,
        titleSize =  1.5,
        titleRotation = 0.0,
        nlabels=9,
        labelFont="",
        labelOffset = 0.375,
        italic=0,
        c=None,
        useAlpha=True,
        drawBox=True,
    ):
        """
        Draw a 3D scalar bar.

        ``obj`` input can be:
            - a list of numbers,
            - a list of two numbers in the form `(min, max)`,
            - a ``Mesh`` already containing a set of scalars associated to vertices or cells,
            - if ``None`` the last object in the list of actors will be used.

        Return an ``Assembly`` object.

        :param float sx: thickness of scalarbar
        :param float sy: length of scalarbar
        :param str title: scalar bar title
        :param float titleXOffset: horizontal space btw title and color scalarbar
        :param float titleYOffset: vertical space offset
        :param float titleSize: size of title wrt numeric labels
        :param float titleRotation: title rotation in degrees
        :param int nlabels: number of numeric labels
        :param float labelOffset: space btw numeric labels and scale
        :param bool,float italic: use italic font for title and labels
        :param bool useAlpha: render transparency of the color bar, otherwise ignore
        :param bool drawBox: draw a box around the colorbar (useful with useAlpha=True)

        |mesh_coloring| |mesh_coloring.py|_
        """
        import vedo.addons as addons
        self.scalarbar = addons.addScalarBar3D(self,
                                                pos,
                                                sx,
                                                sy,
                                                title,
                                                titleFont,
                                                titleXOffset,
                                                titleYOffset,
                                                titleSize,
                                                titleRotation,
                                                nlabels,
                                                labelFont,
                                                labelOffset,
                                                italic,
                                                c,
                                                useAlpha,
                                                drawBox,
                                                )
        return self.scalarbar


    def write(self, filename, binary=True):
        """Write object to file."""
        import vedo.io as io
        return io.write(self, filename, binary)


########################################################################################
class BaseGrid(BaseActor):

    def __init__(self):

        BaseActor.__init__(self)

        self._data = None
        self.useCells = True
        #-----------------------------------------------------------

    def _update(self, data):
        self._data = data
        self._mapper.SetInputData(self.tomesh().polydata())
        self._mapper.Modified()
        return self

    def tomesh(self, fill=True, shrink=1.0):
        """
        Build a polygonal Mesh from the current Grid object.

        If fill=True, the interior faces of all the cells are created.
        (setting a `shrink` value slightly smaller than the default 1.0
        can avoid flickering due to internal adjacent faces).
        If fill=False, only the boundary faces will be generated.
        """
        from vedo.mesh import Mesh
        gf = vtk.vtkGeometryFilter()
        if fill:
            sf = vtk.vtkShrinkFilter()
            sf.SetInputData(self._data)
            sf.SetShrinkFactor(shrink)
            sf.Update()
            gf.SetInputData(sf.GetOutput())
            gf.Update()
        else:
            gf.SetInputData(self._data)
            gf.Update()
        poly = gf.GetOutput()

        msh = Mesh(poly).flat()
        msh.scalarbar = self.scalarbar
        lut = utils.ctf2lut(self)
        if lut:
            msh._mapper.SetLookupTable(lut)
        if self.useCells:
            msh._mapper.SetScalarModeToUseCellData()
        else:
            msh._mapper.SetScalarModeToUsePointData()
        #msh._mapper.SetScalarRange(msh._mapper.GetScalarRange())
        # print(msh._mapper.GetScalarRange(), lut.GetRange())
        # msh._mapper.SetScalarRange()
        # msh.selectCellArray('chem_0')
        return msh

    def points(self, pts=None, transformed=True, copy=False):
        """
        Set/Get the vertex coordinates of the mesh.
        Argument can be an index, a set of indices
        or a complete new set of points to update the mesh.

        :param bool transformed: if `False` ignore any previous transformation
            applied to the mesh.
        :param bool copy: if `False` return the reference to the points
            so that they can be modified in place, otherwise a copy is built.
        """
        if pts is None: ### getter

            vpts = self._data.GetPoints()
            if vpts:
                if copy:
                    return np.array(vtk_to_numpy(vpts.GetData()))
                else:
                    return vtk_to_numpy(vpts.GetData())
            else:
                return np.array([])

        elif (utils.isSequence(pts) and not utils.isSequence(pts[0])) or isinstance(pts, (int, np.integer)):
            #passing a list of indices or a single index
            return vtk_to_numpy(self.polydata(transformed).GetPoints().GetData())[pts]

        else:           ### setter

            if len(pts) == 3 and len(pts[0]) != 3:
                # assume plist is in the format [all_x, all_y, all_z]
                pts = np.stack((pts[0], pts[1], pts[2]), axis=1)
            vpts = self._data.GetPoints()
            vpts.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
            self._data.GetPoints().Modified()
            # reset mesh to identity matrix position/rotation:
            self.PokeMatrix(vtk.vtkMatrix4x4())
            return self


    def cells(self):
        """
        Get the cells connectivity ids as a numpy array.
        The output format is: [[id0 ... idn], [id0 ... idm],  etc].
        """
        arr1d = vtk_to_numpy(self._data.GetCells().GetData())
        if arr1d is None:
            return []

        #Get cell connettivity ids as a 1D array. vtk format is:
        #[nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        i = 0
        conn = []
        n = len(arr1d)
        if n:
            while True:
                cell = [arr1d[i+k] for k in range(1, arr1d[i]+1)]
                conn.append(cell)
                i += arr1d[i]+1
                if i >= n:
                    break
        return conn


    def color(self, col):
        """
        Assign a color or a set of colors along the range of the scalar value.
        A single constant color can also be assigned.
        Any matplotlib color map name is also accepted, e.g. ``volume.color('jet')``.

        E.g.: say that your cells scalar runs from -3 to 6,
        and you want -3 to show red and 1.5 violet and 6 green, then just set:

        ``volume.color(['red', 'violet', 'green'])``
        """
        smin, smax = self._data.GetScalarRange()
        ctf = self.GetProperty().GetRGBTransferFunction()
        ctf.RemoveAllPoints()
        self._color = col

        if utils.isSequence(col):
            for i, ci in enumerate(col):
                r, g, b = colors.getColor(ci)
                x = smin + (smax - smin) * i / (len(col) - 1)
                ctf.AddRGBPoint(x, r, g, b)
                #colors.printc('\tcolor at', round(x, 1),
                #              '\tset to', colors.getColorName((r, g, b)), c='w', bold=0)
        elif isinstance(col, str):
            if col in colors.colors.keys() or col in colors.color_nicks.keys():
                r, g, b = colors.getColor(col)
                ctf.AddRGBPoint(smin, r,g,b) # constant color
                ctf.AddRGBPoint(smax, r,g,b)
            elif colors._mapscales:
                for x in np.linspace(smin, smax, num=64, endpoint=True):
                    r,g,b = colors.colorMap(x, name=col, vmin=smin, vmax=smax)
                    ctf.AddRGBPoint(x, r, g, b)
        elif isinstance(col, int):
            r, g, b = colors.getColor(col)
            ctf.AddRGBPoint(smin, r,g,b) # constant color
            ctf.AddRGBPoint(smax, r,g,b)
        else:
            colors.printc("ugrid.color(): unknown input type:", col, c='r')
        return self

    def alpha(self, alpha):
        """
        Assign a set of tranparencies along the range of the scalar value.
        A single constant value can also be assigned.

        E.g.: say alpha=(0.0, 0.3, 0.9, 1) and the scalar range goes from -10 to 150.
        Then all cells with a value close to -10 will be completely transparent, cells at 1/4
        of the range will get an alpha equal to 0.3 and voxels with value close to 150
        will be completely opaque.

        As a second option one can set explicit (x, alpha_x) pairs to define the transfer function.
        E.g.: say alpha=[(-5, 0), (35, 0.4) (123,0.9)] and the scalar range goes from -10 to 150.
        Then all cells below -5 will be completely transparent, cells with a scalar value of 35
        will get an opacity of 40% and above 123 alpha is set to 90%.
        """
        smin, smax = self._data.GetScalarRange()
        otf = self.GetProperty().GetScalarOpacity()
        otf.RemoveAllPoints()
        self._alpha = alpha

        if utils.isSequence(alpha):
            alpha = np.array(alpha)
            if len(alpha.shape)==1: # user passing a flat list e.g. (0.0, 0.3, 0.9, 1)
                for i, al in enumerate(alpha):
                    xalpha = smin + (smax - smin) * i / (len(alpha) - 1)
                    # Create transfer mapping scalar value to opacity
                    otf.AddPoint(xalpha, al)
            elif len(alpha.shape)==2: # user passing [(x0,alpha0), ...]
                otf.AddPoint(smin, alpha[0][1])
                for xalpha, al in alpha:
                    # Create transfer mapping scalar value to opacity
                    otf.AddPoint(xalpha, al)
                otf.AddPoint(smax, alpha[-1][1])
            #colors.printc("alpha at", round(xalpha, 1), "\tset to", al)

        else:
            otf.AddPoint(smin, alpha) # constant alpha
            otf.AddPoint(smax, alpha)

        return self

    def alphaUnit(self, u=None):
        """
        Defines light attenuation per unit length. Default is 1.
        The larger the unit length, the further light has to travel to attenuate the same amount.

        E.g., if you set the unit distance to 0, you will get full opacity.
        It means that when light travels 0 distance it's already attenuated a finite amount.
        Thus, any finite distance should attenuate all light.
        The larger you make the unit distance, the more transparent the rendering becomes.
        """
        if u is None:
            return self.GetProperty().GetScalarOpacityUnitDistance()
        else:
            self.GetProperty().SetScalarOpacityUnitDistance(u)
            return self


    def shrink(self, fraction=0.8):
        """Shrink the individual cells to improve visibility."""
        sf = vtk.vtkShrinkFilter()
        sf.SetInputData(self._data)
        sf.SetShrinkFactor(fraction)
        sf.Update()
        return self._update(sf.GetOutput())

    def isosurface(self, threshold=None, largest=False):
        """Return an ``Mesh`` isosurface extracted from the ``Volume`` object.

        :param float,list threshold: value or list of values to draw the isosurface(s)
        :param bool largest: if True keep only the largest portion of the mesh

        |isosurfaces| |isosurfaces.py|_
        """
        from vedo.mesh import Mesh
        scrange = self._data.GetScalarRange()
        cf = vtk.vtkContourFilter()
        cf.SetInputData(self._data)
        cf.UseScalarTreeOn()
        cf.ComputeNormalsOn()

        if utils.isSequence(threshold):
            cf.SetNumberOfContours(len(threshold))
            for i, t in enumerate(threshold):
                cf.SetValue(i, t)
        else:
            if threshold is None:
                threshold = (2 * scrange[0] + scrange[1]) / 3.0
            cf.SetValue(0, threshold)

        cf.Update()
        poly = cf.GetOutput()

        if largest:
            conn = vtk.vtkPolyDataConnectivityFilter()
            conn.SetExtractionModeToLargestRegion()
            conn.SetInputData(poly)
            conn.Update()
            poly = conn.GetOutput()

        a = Mesh(poly, c=None).phong()
        a._mapper.SetScalarRange(scrange[0], scrange[1])
        return a


    def legosurface(self, vmin=None, vmax=None, invert=False, cmap='afmhot_r'):
        """
        Represent a ``Volume`` as lego blocks (voxels).
        By default colors correspond to the volume's scalar.
        Returns an ``Mesh``.

        :param float vmin: the lower threshold, voxels below this value are not shown.
        :param float vmax: the upper threshold, voxels above this value are not shown.
        :param str cmap: color mapping of the scalar associated to the voxels.

        |legosurface| |legosurface.py|_
        """
        from vedo.mesh import Mesh
        dataset = vtk.vtkImplicitDataSet()
        dataset.SetDataSet(self._data)
        window = vtk.vtkImplicitWindowFunction()
        window.SetImplicitFunction(dataset)

        srng = list(self._data.GetScalarRange())
        if vmin is not None:
            srng[0] = vmin
        if vmax is not None:
            srng[1] = vmax
        tol = 0.00001*(srng[1]-srng[0])
        srng[0] -= tol
        srng[1] += tol
        window.SetWindowRange(srng)

        extract = vtk.vtkExtractGeometry()
        extract.SetInputData(self._data)
        extract.SetImplicitFunction(window)
        extract.SetExtractInside(invert)
        extract.ExtractBoundaryCellsOff()
        extract.Update()

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(extract.GetOutput())
        gf.Update()

        a = Mesh(gf.GetOutput()).lw(0.1).flat()
        scalars = a.getPointArray()
        if scalars is None:
            print("Error in legosurface(): no scalars found!")
            return a
        a.cmap(cmap, scalars, vmin=srng[0], vmax=srng[1])
        a.mapPointsToCells()
        return a


    def cutWithPlane(self, origin=(0,0,0), normal=(1,0,0)):
        """
        Cut the mesh with the plane defined by a point and a normal.

        :param origin: the cutting plane goes through this point
        :param normal: normal of the cutting plane
        """
        strn = str(normal)
        if strn   ==  "x": normal = (1, 0, 0)
        elif strn ==  "y": normal = (0, 1, 0)
        elif strn ==  "z": normal = (0, 0, 1)
        elif strn == "-x": normal = (-1, 0, 0)
        elif strn == "-y": normal = (0, -1, 0)
        elif strn == "-z": normal = (0, 0, -1)
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        clipper = vtk.vtkClipDataSet()
        clipper.SetInputData(self._data)
        clipper.SetClipFunction(plane)
        clipper.GenerateClipScalarsOff()
        clipper.GenerateClippedOutputOff()
        clipper.SetValue(0)
        clipper.Update()
        cout = clipper.GetOutput()
        return self._update(cout)


    def cutWithBoundingBox(self, box):
        """
        Cut the grid with the specified bounding box.

        Parameter box has format [xmin, xmax, ymin, ymax, zmin, zmax].
        If a Mesh is passed, its bounding box is used.

        Example:

            .. code-block:: python

                from vedo import *
                tetmesh = TetMesh(datadir+'limb_ugrid.vtk')
                tetmesh.color('rainbow')
                cu = Cube(side=500).x(500) # any Mesh works
                tetmesh.cutWithBox(cu).show(axes=1)
        """
        bc = vtk.vtkBoxClipDataSet()
        bc.SetInputData(self._data)
        if isinstance(box, vtk.vtkProp):
            box = box.GetBounds()
        bc.SetBoxClip(*box)
        bc.Update()
        cout = bc.GetOutput()
        return self._update(cout)


    def cutWithMesh(self, mesh, invert=False, wholeCells=False, onlyBoundary=False):
        """
        Cut a UGrid, TetMesh or Volume mesh with a Mesh.

        :param bool invert: if True return cut off part of the input TetMesh.
        """
        polymesh = mesh.polydata()
        ug = self._data

        ippd = vtk.vtkImplicitPolyDataDistance()
        ippd.SetInput(polymesh)

        if wholeCells or onlyBoundary:
            clipper = vtk.vtkExtractGeometry()
            clipper.SetInputData(ug)
            clipper.SetImplicitFunction(ippd)
            clipper.SetExtractInside(not invert)
            clipper.SetExtractBoundaryCells(False)
            if onlyBoundary:
                clipper.SetExtractBoundaryCells(True)
                clipper.SetExtractOnlyBoundaryCells(True)
        else:
            signedDistances = vtk.vtkFloatArray()
            signedDistances.SetNumberOfComponents(1)
            signedDistances.SetName("SignedDistances")
            for pointId in range(ug.GetNumberOfPoints()):
                p = ug.GetPoint(pointId)
                signedDistance = ippd.EvaluateFunction(p)
                signedDistances.InsertNextValue(signedDistance)
            ug.GetPointData().SetScalars(signedDistances)
            clipper = vtk.vtkClipDataSet()
            clipper.SetInputData(ug)
            clipper.SetInsideOut(not invert)
            clipper.SetValue(0.0)

        clipper.Update()
        cug = clipper.GetOutput()

        if ug.GetCellData().GetScalars(): # not working
            scalname = ug.GetCellData().GetScalars().GetName()
            if scalname: # not working
                if self.useCells:
                    self.selectCellArray(scalname)
                else:
                    self.selectPointArray(scalname)

        self._update(cug)
        return self


    def tetralize(self, tetsOnly=True):
        """Tetralize the grid.
        If tetsOnly=True will cull all 1D and 2D cells from the output.

        Return a TetMesh.

        Example:

            .. code-block:: python

                from vedo import *
                ug = loadUnStructuredGrid(datadir+'ugrid.vtk')
                tmesh = tetralize(ug)
                tmesh.write('ugrid.vtu').show(axes=1)
        """
        from vedo.tetmesh import tetralize
        return tetralize(self._data, tetsOnly)


    def extractCellsByID(self, idlist, usePointIDs=False):
        """Return a new UGrid composed of the specified subset of indices."""
        from vedo.ugrid import UGrid
        selectionNode = vtk.vtkSelectionNode()
        if usePointIDs:
            selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
            contcells = vtk.vtkSelectionNode.CONTAINING_CELLS()
            selectionNode.GetProperties().Set(contcells, 1)
        else:
            selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        vidlist = numpy_to_vtkIdTypeArray(np.array(idlist).astype(np.int64))
        selectionNode.SetSelectionList(vidlist)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        es = vtk.vtkExtractSelection()
        es.SetInputData(0, self._data)
        es.SetInputData(1, selection)
        es.Update()
        tm_sel = UGrid(es.GetOutput())
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.GetProperty())
        tm_sel.SetProperty(pr)

        #assign the same transformation to the copy
        tm_sel.SetOrigin(self.GetOrigin())
        tm_sel.SetScale(self.GetScale())
        tm_sel.SetOrientation(self.GetOrientation())
        tm_sel.SetPosition(self.GetPosition())
        tm_sel._mapper.SetLookupTable(utils.ctf2lut(self))
        return tm_sel



############################################################################### funcs
def _getinput(obj):
    if isinstance(obj, (vtk.vtkVolume, vtk.vtkActor)):
        return obj.GetMapper().GetInput()
    else:
        return obj

def probePoints(dataset, pts):
    """
    Takes a ``Volume`` (or any other vtk data set)
    and probes its scalars at the specified points in space.

    Note that a mask is also output with valid/invalid points which can be accessed
    with `mesh.getPointArray('vtkValidPointMask')`.
    """
    if isinstance(pts, vedo.pointcloud.Points):
        pts = pts.points()

    def readPoints():
        output = src.GetPolyDataOutput()
        points = vtk.vtkPoints()
        for p in pts:
            x, y, z = p
            points.InsertNextPoint(x, y, z)
        output.SetPoints(points)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(len(pts))
        for i in range(len(pts)):
            cells.InsertCellPoint(i)
        output.SetVerts(cells)

    src = vtk.vtkProgrammableSource()
    src.SetExecuteMethod(readPoints)
    src.Update()
    img = _getinput(dataset)
    probeFilter = vtk.vtkProbeFilter()
    probeFilter.SetSourceData(img)
    probeFilter.SetInputConnection(src.GetOutputPort())
    probeFilter.Update()
    poly = probeFilter.GetOutput()
    pm = vedo.mesh.Mesh(poly)
    pm.name = 'probePoints'
    return pm

def probeLine(dataset, p1, p2, res=100):
    """
    Takes a ``Volume``  (or any other vtk data set)
    and probes its scalars along a line defined by 2 points `p1` and `p2`.

    Note that a mask is also output with valid/invalid points which can be accessed
    with `mesh.getPointArray('vtkValidPointMask')`.

    :param int res: nr of points along the line

    |probeLine1| |probeLine1.py|_ |probeLine2.py|_
    """
    line = vtk.vtkLineSource()
    line.SetResolution(res)
    line.SetPoint1(p1)
    line.SetPoint2(p2)
    img = _getinput(dataset)
    probeFilter = vtk.vtkProbeFilter()
    probeFilter.SetSourceData(img)
    probeFilter.SetInputConnection(line.GetOutputPort())
    probeFilter.Update()
    poly = probeFilter.GetOutput()
    lnn = vedo.mesh.Mesh(poly)
    lnn.name = 'probeLine'
    return lnn

def probePlane(dataset, origin=(0, 0, 0), normal=(1, 0, 0)):
    """
    Takes a ``Volume`` (or any other vtk data set)
    and probes its scalars on a plane defined by a point and a normal.
    """
    img = _getinput(dataset)
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)
    planeCut = vtk.vtkCutter()
    planeCut.SetInputData(img)
    planeCut.SetCutFunction(plane)
    planeCut.Update()
    poly = planeCut.GetOutput()
    cutmesh = vedo.mesh.Mesh(poly)
    cutmesh.name = 'probePlane'
    return cutmesh


def interpolateToStructuredGrid(mesh, kernel=None, radius=None,
                               bounds=None, nullValue=None, dims=None):
    """
    Generate a volumetric dataset (vtkStructuredData) by interpolating a scalar
    or vector field which is only known on a scattered set of points or mesh.
    Available interpolation kernels are: shepard, gaussian, voronoi, linear.

    :param str kernel: interpolation kernel type [shepard]
    :param float radius: radius of the local search
    :param list bounds: bounding box of the output vtkStructuredGrid object
    :param list dims: dimensions of the output vtkStructuredGrid object
    :param float nullValue: value to be assigned to invalid points
    """
    if isinstance(mesh, vtk.vtkPolyData):
        output = mesh
    else:
        output = mesh.polydata()

    if dims is None:
        dims = (20,20,20)

    if bounds is None:
        bounds = output.GetBounds()

    # Create a probe volume
    probe = vtk.vtkStructuredGrid()
    probe.SetDimensions(dims)

    points = vtk.vtkPoints()
    points.Allocate(dims[0] * dims[1] * dims[2])
    deltaZ = (bounds[5]-bounds[4]) / (dims[2] - 1)
    deltaY = (bounds[3]-bounds[2]) / (dims[1] - 1)
    deltaX = (bounds[1]-bounds[0]) / (dims[0] - 1)
    for k in range(dims[2]):
        z = bounds[4] + k * deltaZ
        kOffset = k * dims[0] * dims[1]
        for j in range(dims[1]):
            y = bounds[2] + j * deltaY
            jOffset = j * dims[0]
            for i  in range(dims[0]):
                x = bounds[0] + i * deltaX
                offset = i + jOffset + kOffset
                points.InsertPoint(offset, [x,y,z])
    probe.SetPoints(points)

    if radius is None:
        radius = min(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])/3

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(output)
    locator.BuildLocator()

    if kernel == 'gaussian':
        kern = vtk.vtkGaussianKernel()
        kern.SetRadius(radius)
    elif kernel == 'voronoi':
        kern = vtk.vtkVoronoiKernel()
    elif kernel == 'linear':
        kern = vtk.vtkLinearKernel()
        kern.SetRadius(radius)
    else:
        kern = vtk.vtkShepardKernel()
        kern.SetPowerParameter(2)
        kern.SetRadius(radius)

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(probe)
    interpolator.SetSourceData(output)
    interpolator.SetKernel(kern)
    interpolator.SetLocator(locator)
    if nullValue is not None:
        interpolator.SetNullValue(nullValue)
    else:
        interpolator.SetNullPointsStrategyToClosestPoint()
    interpolator.Update()
    return interpolator.GetOutput()


def streamLines(domain, probe,
                activeVectors='',
                integrator='rk4',
                direction='forward',
                initialStepSize=None,
                maxPropagation=None,
                maxSteps=10000,
                stepLength=None,
                extrapolateToBoundingBox=(),
                surfaceConstrain=False,
                computeVorticity=True,
                ribbons=None,
                tubes={},
                scalarRange=None,
                lw=None,
    ):
    """
    Integrate a vector field on a domain (a Mesh or other vtk datasets types)
    to generate streamlines.

    The integration is performed using a specified integrator (Runge-Kutta).
    The length of a streamline is governed by specifying a maximum value either
    in physical arc length or in (local) cell length.
    Otherwise, the integration terminates upon exiting the field domain.

    :param domain: the vtk object that contains the vector field
    :param str activeVectors: name of the vector array
    :param Mesh,list probe: the Mesh that probes the domain. Its coordinates will
        be the seeds for the streamlines, can also be an array of positions.
    :param str integrator: Runge-Kutta integrator, either 'rk2', 'rk4' of 'rk45'
    :param float initialStepSize: initial step size of integration
    :param float maxPropagation: maximum physical length of the streamline
    :param int maxSteps: maximum nr of steps allowed
    :param float stepLength: length of step integration.
    :param dict extrapolateToBoundingBox:
        Vectors defined on a surface are extrapolated to the entire volume defined by its bounding box

        - kernel, (str) - interpolation kernel type [shepard]
        - radius (float)- radius of the local search
        - bounds, (list) - bounding box of the output Volume
        - dims, (list) - dimensions of the output Volume object
        - nullValue, (float) - value to be assigned to invalid points

    :param bool surfaceConstrain: force streamlines to be computed on a surface
    :param bool computeVorticity: Turn on/off vorticity computation at streamline points
        (necessary for generating proper stream-ribbons)
    :param int ribbons: render lines as ribbons by joining them.
        An integer value represent the ratio of joining (e.g.: ribbons=2 groups lines 2 by 2)
    :param dict tubes: dictionary containing the parameters for the tube representation:

            - ratio, (int) - draws tube as longitudinal stripes
            - res, (int) - tube resolution (nr. of sides, 12 by default)
            - maxRadiusFactor (float) - max tube radius as a multiple of the min radius
            - varyRadius, (int) - radius varies based on the scalar or vector magnitude:

                - 0 - do not vary radius
                - 1 - vary radius by scalar
                - 2 - vary radius by vector
                - 3 - vary radius by absolute value of scalar

    :param list scalarRange: specify the scalar range for coloring

    .. hint:: Examples: |streamlines1.py|_ |streamribbons.py|_ |office.py|_ |streamlines2.py|_

        |streamlines2| |office| |streamribbons| |streamlines1|
    """

    if isinstance(domain, vtk.vtkActor):
        if len(extrapolateToBoundingBox):
            grid = interpolateToStructuredGrid(domain, **extrapolateToBoundingBox)
        else:
            grid = domain.polydata()
    else:
        grid = domain

    if activeVectors:
        grid.GetPointData().SetActiveVectors(activeVectors)

    b = grid.GetBounds()
    size = (b[5]-b[4] + b[3]-b[2] + b[1]-b[0])/3
    if initialStepSize is None:
        initialStepSize = size/100.
    if maxPropagation is None:
        maxPropagation = size

    if utils.isSequence(probe):
        pts = np.array(probe)
        if pts.shape[1] == 2: # make it 3d
            pts = np.c_[pts, np.zeros(len(pts))]
    else:
        pts = probe.clean().points()
    src = vtk.vtkProgrammableSource()
    def readPoints():
        output = src.GetPolyDataOutput()
        points = vtk.vtkPoints()
        for x, y, z in pts:
            points.InsertNextPoint(x, y, z)
        output.SetPoints(points)
    src.SetExecuteMethod(readPoints)
    src.Update()

    st = vtk.vtkStreamTracer()
    st.SetInputDataObject(grid)
    st.SetSourceConnection(src.GetOutputPort())

    st.SetInitialIntegrationStep(initialStepSize)
    st.SetComputeVorticity(computeVorticity)
    st.SetMaximumNumberOfSteps(maxSteps)
    st.SetMaximumPropagation(maxPropagation)
    st.SetSurfaceStreamlines(surfaceConstrain)
    if stepLength:
        st.SetMaximumIntegrationStep(stepLength)

    if 'f' in direction:
        st.SetIntegrationDirectionToForward()
    elif 'back' in direction:
        st.SetIntegrationDirectionToBackward()
    elif 'both' in direction:
        st.SetIntegrationDirectionToBoth()

    if integrator == 'rk2':
        st.SetIntegratorTypeToRungeKutta2()
    elif integrator == 'rk4':
        st.SetIntegratorTypeToRungeKutta4()
    elif integrator == 'rk45':
        st.SetIntegratorTypeToRungeKutta45()
    else:
        colors.printc("Error in streamlines, unknown integrator", integrator, c='r')

    st.Update()
    output = st.GetOutput()

    if ribbons:
        scalarSurface = vtk.vtkRuledSurfaceFilter()
        scalarSurface.SetInputConnection(st.GetOutputPort())
        scalarSurface.SetOnRatio(int(ribbons))
        scalarSurface.SetRuledModeToPointWalk()
        scalarSurface.Update()
        output = scalarSurface.GetOutput()

    if len(tubes):
        streamTube = vtk.vtkTubeFilter()
        streamTube.SetNumberOfSides(12)
        streamTube.SetRadius(tubes['radius'])

        if 'res' in tubes:
            streamTube.SetNumberOfSides(tubes['res'])

        # max tube radius as a multiple of the min radius
        streamTube.SetRadiusFactor(50)
        if 'maxRadiusFactor' in tubes:
            streamTube.SetRadius(tubes['maxRadiusFactor'])

        if 'ratio' in tubes:
            streamTube.SetOnRatio(int(tubes['ratio']))

        if 'varyRadius' in tubes:
            streamTube.SetVaryRadius(int(tubes['varyRadius']))

        streamTube.SetInputData(output)
        vname = grid.GetPointData().GetVectors().GetName()
        streamTube.SetInputArrayToProcess(1, 0, 0,
                                          vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                          vname)
        streamTube.Update()
        sta = vedo.mesh.Mesh(streamTube.GetOutput(), c=None)

        scals = grid.GetPointData().GetScalars()
        if scals:
            sta.mapper().SetScalarRange(scals.GetRange())
        if scalarRange is not None:
            sta.mapper().SetScalarRange(scalarRange)

        sta.GetProperty().BackfaceCullingOn()
        sta.phong()
        return sta

    sta = vedo.mesh.Mesh(output, c=None)

    if lw is not None and len(tubes)==0 and not ribbons:
        sta.lw(lw)
        sta._mapper.SetResolveCoincidentTopologyToPolygonOffset()
        sta.lighting('off')

    scals = grid.GetPointData().GetScalars()
    if scals:
        sta.mapper().SetScalarRange(scals.GetRange())
    if scalarRange is not None:
        sta.mapper().SetScalarRange(scalarRange)
    return sta



###################################################################################
# def extractCellsByType(obj, types=(7,)):    ### VTK9 only
#     """Extract cells of a specified type.
#     Given an input vtkDataSet and a list of cell types, produce an output
#     containing only cells of the specified type(s).
#     Find `here <https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html>`_
#     the list of possible cell types.
#     """
#     ef = vtk.vtkExtractCellsByType()
#     for ct in types:
#         ef.AddCellType(ct)
#     ef.Update()
#     return Mesh(ef.GetOutput())
