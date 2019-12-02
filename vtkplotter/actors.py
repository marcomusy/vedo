from __future__ import division, print_function

import numpy as np
import os
import vtk
import vtkplotter.colors as colors
import vtkplotter.docs as docs
import vtkplotter.settings as settings
import vtkplotter.utils as utils
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

__doc__ = (
    """
Submodule extending the ``vtkActor``, ``vtkVolume``
and ``vtkImageActor`` objects functionality.
"""
    + docs._defs
)

__all__ = [
    'Prop', # only for docs
    'Actor',
    'Assembly',
    'Picture',
    'Volume',
    'merge',
    'collection',
]


# functions
def collection():
    """
    Return the list of objects which have been created so far,
    without having to assign them a name.
    Useful in loops.

    :Example:
        .. code-block:: python

            from vtkplotter import Cone, collection, show
            for i in range(10):
                Cone(pos=[3*i, 0, 0], axis=[i, i-5, 0])
            show(collection())

            # in python3 you can simply use ellipses (three points symbol):
            show(...)
    """
    return settings.collectable_actors


def merge(*actors):
    """
    Build a new actor formed by the fusion of the polygonal meshes of the input objects.
    Similar to Assembly, but in this case the input objects become a single mesh.

    .. hint:: |thinplate_grid.py|_ |value-iteration.py|_

        |thinplate_grid| |value-iteration|
    """
    acts = []
    for a in utils.flatten(actors):
        if isinstance(a, vtk.vtkAssembly):
            acts += a.getActors()
        elif a:
            acts += [a]

    if len(acts) == 1:
        return acts[0].clone()
    elif len(acts) == 0:
        return None

    polylns = vtk.vtkAppendPolyData()
    for a in acts:
        polylns.AddInputData(a.polydata())
    polylns.Update()
    pd = polylns.GetOutput()
    return Actor(pd)


# classes
class Prop(object):
    """Adds functionality to ``Actor``, ``Assembly``,
    ``Volume`` and ``Picture`` objects.

    .. warning:: Do not use this class to instance objects, use the above ones.
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
        self.units = None
        self.top = None
        self.base = None
        self.info = dict()
        self._time = 0
        self._legend = None
        self.scalarbar = None
        self.renderedAt = set()
        self.picked3d = None
        self.cmap = None
        self.flagText = None
        self._mapper = None

    def inputdata(self):
        """Return the VTK input data object."""
        if self._mapper:
            return self._mapper.GetInput()
        return self.GetMapper().GetInput()

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

    def N(self):
        """Retrieve number of points. Shortcut for `NPoints()`."""
        return self.inputdata().GetNumberOfPoints()

    def NPoints(self):
        """Retrieve number of points. Same as `N()`."""
        return self.inputdata().GetNumberOfPoints()

    def NCells(self):
        """Retrieve number of cells."""
        return self.inputdata().GetNumberOfCells()


    def show(self, **options):
        """
        Create on the fly an instance of class ``Plotter`` or use the last existing one to
        show one single object.

        Allowed input objects are: ``Actor`` or ``Volume``.

        This is meant as a shortcut. If more than one object needs to be visualised
        please use the syntax `show([actor1, actor2,...], options)`.

        :param bool newPlotter: if set to `True`, a call to ``show`` will instantiate
            a new ``Plotter`` object (a new window) instead of reusing the first created.
            See e.g.: |readVolumeAsIsoSurface.py|_
        :return: the current ``Plotter`` class instance.

        .. note:: E.g.:

            .. code-block:: python

                from vtkplotter import *
                s = Sphere()
                s.show(at=1, N=2)
                c = Cube()
                c.show(at=0, interactive=True)
        """
        from vtkplotter.plotter import show
        return show(self, **options)


    def pickable(self, value=None):
        """Set/get pickable property of actor."""
        if value is None:
            return self.GetPickable()
        else:
            self.SetPickable(value)
            return self


    def legend(self, txt=None):
        """Set/get ``Actor`` legend text.

        :param str txt: legend text.

        Size and positions can be modified by setting attributes
        ``Plotter.legendSize``, ``Plotter.legendBC`` and ``Plotter.legendPos``.

        .. hint:: |fillholes.py|_
        """
        if txt:
            self._legend = txt
        else:
            return self._legend
        return self

    def flag(self, text=None):
        """Add a flag label which becomes visible when hovering the object with mouse.
        Can be later disabled by setting `flag(False)`.
        """
        if text is None:
            if self.filename:
                text = self.filename.split('/')[-1]
            elif self.name:
                text = self.name
            else:
                text = ""
        self.flagText = text
        return self


    def time(self, t=None):
        """Set/get actor's absolute time."""
        if t is None:
            return self._time
        self._time = t
        return self  # return itself to concatenate methods

    def pos(self, x=None, y=None, z=None):
        """Set/Get actor position."""
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
        """Add vector to current actor position."""
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
        """Set/Get actor position along x axis."""
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
        """Set/Get actor position along y axis."""
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
        """Set/Get actor position along z axis."""
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
        """Rotate ``Actor`` around an arbitrary `axis` passing through `axis_point`."""
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
        # this vtk method only rotates in the origin of the actor:
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
        """Rotate around x-axis. If angle is in radians set ``rad=True``.

        NB: actor.rotateX(12).rotateY(14) will rotate FIRST around Y THEN around X.
        """
        if rad:
            angle *= 180 / np.pi
        self.RotateX(angle)
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                           self.shadow.GetProperty().GetColor(),
                           self.shadow.GetProperty().GetOpacity())
        return self

    def rotateY(self, angle, rad=False):
        """Rotate around y-axis. If angle is in radians set ``rad=True``.

        NB: actor.rotateX(12).rotateY(14) will rotate FIRST around Y THEN around X.
        """
        if rad:
            angle *= 180.0 / np.pi
        self.RotateY(angle)
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                           self.shadow.GetProperty().GetColor(),
                           self.shadow.GetProperty().GetOpacity())
        return self

    def rotateZ(self, angle, rad=False):
        """Rotate around z-axis. If angle is in radians set ``rad=True``.

        NB: actor.rotateX(12).rotateZ(14) will rotate FIRST around Z THEN around X.
        """
        if rad:
            angle *= 180.0 / np.pi
        self.RotateZ(angle)
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                           self.shadow.GetProperty().GetColor(),
                           self.shadow.GetProperty().GetOpacity())
        return self

#    def rotateX(self, angle, rad=False):
#        """Rotate around x-axis. If angle is in radians set ``rad=True``."""
#        if rad:
#            angle *= 180 / np.pi
#        ipos = np.array(self.GetPosition())
#        self.SetPosition(0,0,0)
#        T = vtk.vtkTransform()
#        T.SetMatrix(self.GetMatrix())
#        T.PostMultiply()
#        T.RotateX(angle)
#        T.Translate(ipos)
#        self.SetUserTransform(T)
#        if self.trail:
#            self.updateTrail()
#        if self.shadow:
#            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
#                           self.shadow.GetProperty().GetColor(),
#                           self.shadow.GetProperty().GetOpacity())
#        return self
#    def origin(self, o=None):
#        """Set/get actor's origin coordinates. Default is (0,0,0).
#        Can be used to define an offset."""
#        if o is None:
#            return np.array(self.GetOrigin())
#        self.SetOrigin(o)
#        return self  # return itself to concatenate methods


    def orientation(self, newaxis=None, rotation=0, rad=False):
        """
        Set/Get actor orientation.

        :param rotation: If != 0 rotate actor around newaxis.
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


    def scale(self, s=None):
        """Set/get actor's scaling factor.

        :param s: scaling factor(s).
        :type s: float, list

        .. note:: if `s==(sx,sy,sz)` scale differently in the three coordinates."""
        if s is None:
            return np.array(self.GetScale())
        self.SetScale(s)
        return self

    def addShadow(self, x=None, y=None, z=None, c=(0.5, 0.5, 0.5), alpha=1):
        """
        Generate a shadow out of an ``Actor`` on one of the three Cartesian planes.
        The output is a new ``Actor`` representing the shadow.
        This new actor is accessible through `actor.shadow`.
        By default the actor is placed on the bottom/back wall of the bounding box.

        :param float x,y,z: identify the plane to cast the shadow to ['x', 'y' or 'z'].
            The shadow will lay on the orthogonal plane to the specified axis at the
            specified value of either x, y or z.

        |shadow|  |shadow.py|_

            |airplanes| |airplanes.py|_
        """
        if x is not None:
            self.shadowX = x
            shad = self.clone().projectOnPlane('x').x(x)
        elif y is not None:
            self.shadowY = y
            shad = self.clone().projectOnPlane('y').y(y)
        elif z is not None:
            self.shadowZ = z
            shad = self.clone().projectOnPlane('z').z(z)
        else:
            print('Error in addShadow(): must set x, y or z to a float!')
            return self
        shad.c(c).alpha(alpha).wireframe(False)
        shad.flat().backFaceCulling()
        shad.GetProperty().LightingOff()
        self.shadow = shad
        return self

    def _updateShadow(self):
        p = self.GetPosition()
        if self.shadowX is not None:
            self.shadow.SetPosition(self.shadowX, p[1], p[2])
        elif self.shadowY is not None:
            self.shadow.SetPosition(p[0], self.shadowY, p[2])
        elif self.shadowZ is not None:
            self.shadow.SetPosition(p[0], p[1], self.shadowZ)
        return self

    def addTrail(self, offset=None, maxlength=None, n=50, c=None, alpha=None, lw=2):
        """Add a trailing line to actor.
        This new actor is accessible through `actor.trail`.

        :param offset: set an offset vector from the object center.
        :param maxlength: length of trailing line in absolute units
        :param n: number of segments to control precision
        :param lw: line width of the trail

        .. hint:: See examples: |trail.py|_  |airplanes.py|_

            |trail|
        """
        if maxlength is None:
            maxlength = self.diagonalSize() * 20
            if maxlength == 0:
                maxlength = 1

        if self.trail is None:
            pos = self.GetPosition()
            self.trailPoints = [None] * n
            self.trailSegmentSize = maxlength / n
            self.trailOffset = offset

            ppoints = vtk.vtkPoints()  # Generate the polyline
            poly = vtk.vtkPolyData()
            ppoints.SetData(numpy_to_vtk([pos] * n))
            poly.SetPoints(ppoints)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(n)
            for i in range(n):
                lines.InsertCellPoint(i)
            poly.SetPoints(ppoints)
            poly.SetLines(lines)
            mapper = vtk.vtkPolyDataMapper()

            if c is None:
                if hasattr(self, "GetProperty"):
                    col = self.GetProperty().GetColor()
                else:
                    col = (0.1, 0.1, 0.1)
            else:
                col = colors.getColor(c)

            if alpha is None:
                alpha = 1
                if hasattr(self, "GetProperty"):
                    alpha = self.GetProperty().GetOpacity()

            mapper.SetInputData(poly)
            tline = Actor(poly, c=col, alpha=alpha)
            tline.SetMapper(mapper)
            tline.GetProperty().SetLineWidth(lw)
            self.trail = tline  # holds the vtkActor
        return self

    def updateTrail(self):
        currentpos = np.array(self.GetPosition())
        if self.trailOffset:
            currentpos += self.trailOffset
        lastpos = self.trailPoints[-1]
        if lastpos is None:  # reset list
            self.trailPoints = [currentpos] * len(self.trailPoints)
            return
        if np.linalg.norm(currentpos - lastpos) < self.trailSegmentSize:
            return

        self.trailPoints.append(currentpos)  # cycle
        self.trailPoints.pop(0)

        tpoly = self.trail.polydata()
        tpoly.GetPoints().SetData(numpy_to_vtk(self.trailPoints))
        return self

    def print(self):
        """Print  ``Actor``, ``Assembly``, ``Volume`` or ``Image`` infos."""
        utils.printInfo(self)
        return self

    def on(self):
        """Switch on actor visibility. Object is not removed."""
        self.VisibilityOn()
        return self

    def off(self):
        """Switch off actor visibility. Object is not removed."""
        self.VisibilityOff()
        return self

    def lighting(self, style='', ambient=None, diffuse=None,
                 specular=None, specularPower=None, specularColor=None,
                 enabled=True):
        """
        Set the ambient, diffuse, specular and specularPower lighting constants.

        :param str,int style: preset style, can be `[metallic, plastic, shiny, glossy, ambient]`
        :param float ambient: ambient fraction of emission [0-1]
        :param float diffuse: emission of diffused light in fraction [0-1]
        :param float specular: fraction of reflected light [0-1]
        :param float specularPower: precision of reflection [1-100]
        :param color specularColor: color that is being reflected by the surface
        :param bool enabled: enable/disable all surface light emission

        |wikiphong|

        |specular| |specular.py|_
        """
        pr = self.GetProperty()

        if style:
            if hasattr(pr, "GetColor"):  # could be Volume
                c = pr.GetColor()
            else:
                c = (1,1,0.99)
            mpr = self._mapper
            if hasattr(mpr, 'GetScalarVisibility') and mpr.GetScalarVisibility():
                c = (1,1,0.99)
            if   style=='metallic': pars = [0.1, 0.3, 1.0, 10, c]
            elif style=='plastic' : pars = [0.3, 0.4, 0.3,  5, c]
            elif style=='shiny'   : pars = [0.2, 0.6, 0.8, 50, c]
            elif style=='glossy'  : pars = [0.1, 0.7, 0.9, 90, (1,1,0.99)]
            elif style=='ambient' : pars = [1.0, 0.0, 0.0,  0, (1,1,1)]
            elif style=='default' : pars = [0.1, 1.0, 0.05, 5, c]
            else:
                colors.printc("Error in lighting(): Available styles are", c=1)
                colors.printc(" [default, metallic, plastic, shiny, glossy, ambient]", c=1)
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
        if not enabled: pr.LightingOff()
        return self

    def box(self, scale=1):
        """Return the bounding box as a new ``Actor``.

        :param float scale: box size can be scaled by a factor

        .. hint:: |latex.py|_
        """
        b = self.GetBounds()
        from vtkplotter.shapes import Box
        pos = (b[0]+b[1])/2, (b[3]+b[2])/2, (b[5]+b[4])/2
        length, width, height = b[1]-b[0], b[3]-b[2], b[5]-b[4]
        oa = Box(pos, length*scale, width*scale, height*scale, c='gray')
        if isinstance(self.GetProperty(), vtk.vtkProperty):
            pr = vtk.vtkProperty()
            pr.DeepCopy(self.GetProperty())
            oa.SetProperty(pr)
            oa.wireframe()
        return oa

    def bounds(self):
        """Get the bounds of the data object."""
        return self.GetMapper().GetInput().GetBounds()

    def printHistogram(self, bins=10, height=10, logscale=False, minbin=0,
                       horizontal=False, char=u"\U00002589",
                       c=None, bold=True, title='Histogram'):
        """
        Ascii histogram printing.
        Input can also be ``Volume`` or ``Actor``.
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

                from vtkplotter import printHistogram
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

    def printInfo(self):
        """Print information about a vtk object."""
        utils.printInfo(self)
        return self

    def c(self, color=False):
        """
        Shortcut for `color()`.
        If None is passed as input, will use colors from current active scalars.
        """
        return self.color(color)


    def getTransform(self):
        """
        Check if ``info['transform']`` exists and returns it.
        Otherwise return current user transformation
        (where the object is currently placed).
        """
        if "transform" in self.info.keys():
            T = self.info["transform"]
            return T
        else:
            T = self.GetMatrix()
            tr = vtk.vtkTransform()
            tr.SetMatrix(T)
            return tr

    def setTransform(self, T):
        """
        Transform object position and orientation wrt to its polygonal mesh,
        which remains unmodified.
        """
        if isinstance(T, vtk.vtkMatrix4x4):
            self.SetUserMatrix(T)
        else:
            try:
                self.SetUserTransform(T)
            except TypeError:
                colors.printc('~times Error in setTransform():',
                              'consider transformPolydata() instead.', c=1)
        return self


    def getArrayNames(self):
        from vtk.numpy_interface import dataset_adapter
        wrapped = dataset_adapter.WrapDataObject(self.GetMapper().GetInput())
        return {"PointData":wrapped.PointData.keys(), "CellData":wrapped.CellData.keys()}

    def getPointArray(self, name=0):
        """Return point array content as a ``numpy.array``.
        This can be identified either as a string or by an integer number."""
        data = None
        if hasattr(self, 'poly') and self.poly:
            data = self.poly
        elif hasattr(self, '_image') and self._image:
            data = self._image
        return vtk_to_numpy(data.GetPointData().GetArray(name))

    def getCellArray(self, name=0):
        """Return cell array content as a ``numpy.array``."""
        data = None
        if hasattr(self, 'poly') and self.poly:
            data = self.poly
        elif hasattr(self, '_image') and self._image:
            data = self._image
        return vtk_to_numpy(data.GetCellData().GetArray(name))

    def addPointScalars(self, scalars, name):
        """
        Add point scalars and assigning it a name.

        |mesh_coloring| |mesh_coloring.py|_
        """
        data = self.inputdata()
        if len(scalars) != data.GetNumberOfPoints():
            colors.printc('~times addPointScalars(): Number of scalars != nr. of points',
                          len(scalars), data.GetNumberOfPoints(), c=1)
            raise RuntimeError()

        arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
        arr.SetName(name)
        data.GetPointData().AddArray(arr)
        data.GetPointData().SetActiveScalars(name)
        self._mapper.SetArrayName(name)
        if settings.autoResetScalarRange:
            self._mapper.SetScalarRange(np.min(scalars), np.max(scalars))
        self._mapper.SetScalarModeToUsePointData()
        self._mapper.ScalarVisibilityOn()
        return self

    def addCellScalars(self, scalars, name):
        """
        Add cell scalars and assigning it a name.
        """
        data = self.inputdata()
        if isinstance(scalars, str):
            scalars = vtk_to_numpy(data.GetPointData().GetArray(scalars))

        if len(scalars) != data.GetNumberOfCells():
            colors.printc("~times addCellScalars() Number of scalars != nr. of cells",
                          len(scalars), data.GetNumberOfCells(), c=1)
            raise RuntimeError()

        arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
        arr.SetName(name)
        data.GetCellData().AddArray(arr)
        data.GetCellData().SetActiveScalars(name)
        self._mapper.SetArrayName(name)
        if settings.autoResetScalarRange:
            self._mapper.SetScalarRange(np.min(scalars), np.max(scalars))
        self._mapper.SetScalarModeToUseCellData()
        self._mapper.ScalarVisibilityOn()
        return self

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
        if isinstance(self, Volume):
            return self.updateVolume(c2p.GetOutput())
        else:
            return self.updateMesh(c2p.GetOutput())

    def mapPointsToCells(self):
        """
        Transform point data (i.e., data specified per point)
        into cell data (i.e., data specified per cell).
        The method of transformation is based on averaging the data values
        of all points defining a particular cell.

        |mesh_map2cell| |mesh_map2cell.py|_
        """
        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(self.polydata(False))
        p2c.Update()
        self._mapper.SetScalarModeToUseCellData()
        if isinstance(self, Volume):
            return self.updateVolume(p2c.GetOutput())
        else:
            return self.updateMesh(p2c.GetOutput())


####################################################
# Actor inherits from vtkActor and Prop
class Actor(vtk.vtkFollower, Prop):
    """
    Build an instance of object ``Actor`` derived from ``vtkActor``.

    Input can be ``vtkPolyData``, ``vtkActor``, or a python list of [vertices, faces].

    If input is any of ``vtkUnstructuredGrid``, ``vtkStructuredGrid`` or ``vtkRectilinearGrid``
    the goemetry is extracted.
    In this case the original VTK data structures can be accessed with: ``actor.inputdata()``.

    Finally input can be a list of vertices and their connectivity (faces of the polygonal mesh).
    For point clouds - e.i. no faces - just substitute the `faces` list with ``None``.

    E.g.: `Actor( [ [[x1,y1,z1],[x2,y2,z2], ...],  [[0,1,2], [1,2,3], ...] ] )`

    :param c: color in RGB format, hex, symbol or name
    :param float alpha: opacity value
    :param bool wire:  show surface as wireframe
    :param bc: backface color of internal surface
    :param str texture: jpg file name or surface texture name
    :param bool computeNormals: compute point and cell normals at creation

    .. hint:: A mesh can be built from vertices and their connectivity. See e.g.:

        |buildmesh| |buildmesh.py|_
    """

    def __init__(
        self,
        inputobj=None,
        c=None,
        alpha=1,
        computeNormals=False,
    ):
        vtk.vtkActor.__init__(self)
        Prop.__init__(self)

        self._polydata = None
        self._mapper = vtk.vtkPolyDataMapper()

        self._mapper.SetInterpolateScalarsBeforeMapping(settings.interpolateScalarsBeforeMapping)

        if settings.usePolygonOffset:
            self._mapper.SetResolveCoincidentTopologyToPolygonOffset()
            pof, pou = settings.polygonOffsetFactor, settings.polygonOffsetUnits
            self._mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(pof, pou)

        inputtype = str(type(inputobj))
        # print('inputtype',inputtype)

        if inputobj is None:
            self._polydata = vtk.vtkPolyData()
        elif "Actor" in inputtype:
            polyCopy = vtk.vtkPolyData()
            polyCopy.DeepCopy(inputobj.GetMapper().GetInput())
            self._polydata = polyCopy
            self._mapper.SetInputData(polyCopy)
            self._mapper.SetScalarVisibility(inputobj.GetMapper().GetScalarVisibility())
            pr = vtk.vtkProperty()
            pr.DeepCopy(inputobj.GetProperty())
            self.SetProperty(pr)
        elif "PolyData" in inputtype:
            if inputobj.GetNumberOfCells() == 0:
                carr = vtk.vtkCellArray()
                for i in range(inputobj.GetNumberOfPoints()):
                    carr.InsertNextCell(1)
                    carr.InsertCellPoint(i)
                inputobj.SetVerts(carr)
            self._polydata = inputobj  # cache vtkPolyData and mapper for speed
        elif "structured" in inputtype.lower() or "RectilinearGrid" in inputtype:
            if settings.visibleGridEdges:
                gf = vtk.vtkExtractEdges()
                gf.SetInputData(inputobj)
            else:
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(inputobj)
            gf.Update()
            self._polydata = gf.GetOutput()
        elif "trimesh" in inputtype:
            tact = utils.trimesh2vtk(inputobj, alphaPerCell=False)
            self._polydata = tact.polydata()
        elif utils.isSequence(inputobj):
            if len(inputobj) == 2: # assume [vertices, faces]
                self._polydata = utils.buildPolyData(inputobj[0], inputobj[1])
            else:
                self._polydata = utils.buildPolyData(inputobj, None)
        elif hasattr(inputobj, "GetOutput"): # passing vtk object
            if hasattr(inputobj, "Update"): inputobj.Update()
            self._polydata = inputobj.GetOutput()
        else:
            colors.printc("Error: cannot build Actor from type:\n", inputtype, c=1)
            raise RuntimeError()

        self.SetMapper(self._mapper)

        if settings.computeNormals is not None:
            computeNormals = settings.computeNormals

        if self._polydata:
            if computeNormals:
                pdnorm = vtk.vtkPolyDataNormals()
                pdnorm.SetInputData(self._polydata)
                pdnorm.ComputePointNormalsOn()
                pdnorm.ComputeCellNormalsOn()
                pdnorm.FlipNormalsOff()
                pdnorm.ConsistencyOn()
                pdnorm.Update()
                self._polydata = pdnorm.GetOutput()

            if self._mapper:
                self._mapper.SetInputData(self._polydata)

        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None
        self._bfprop = None  # backface property holder
        self._scals_idx = 0  # index of the active scalar changed from CLI
        self._ligthingnr = 0

        prp = self.GetProperty()
        prp.SetInterpolationToPhong()

        if settings.renderPointsAsSpheres:
            if hasattr(prp, 'RenderPointsAsSpheresOn'):
                prp.RenderPointsAsSpheresOn()

        if settings.renderLinesAsTubes:
            if hasattr(prp, 'RenderLinesAsTubesOn'):
                prp.RenderLinesAsTubesOn()

        # set the color by c or by scalar
        if self._polydata:

            arrexists = False

            if c is None:
                ptdata = self._polydata.GetPointData()
                cldata = self._polydata.GetCellData()
                exclude = ['normals', 'tcoord']

                if cldata.GetNumberOfArrays():
                    for i in range(cldata.GetNumberOfArrays()):
                        iarr = cldata.GetArray(i)
                        if iarr:
                            icname = iarr.GetName()
                            if icname and all(s not in icname.lower() for s in exclude):
                                cldata.SetActiveScalars(icname)
                                self._mapper.ScalarVisibilityOn()
                                self._mapper.SetScalarModeToUseCellData()
                                self._mapper.SetScalarRange(iarr.GetRange())
                                arrexists = True
                                break # stop at first good one

                # point come after so it has priority
                if ptdata.GetNumberOfArrays():
                    for i in range(ptdata.GetNumberOfArrays()):
                        iarr = ptdata.GetArray(i)
                        if iarr:
                            ipname = iarr.GetName()
                            if ipname and all(s not in ipname.lower() for s in exclude):
                                ptdata.SetActiveScalars(ipname)
                                self._mapper.ScalarVisibilityOn()
                                self._mapper.SetScalarModeToUsePointData()
                                self._mapper.SetScalarRange(iarr.GetRange())
                                arrexists = True
                                break

            if arrexists == False:
                if c is None:
                    c = "gold"
                c = colors.getColor(c)
                prp.SetColor(c)
                prp.SetAmbient(0.1)
                prp.SetDiffuse(1)
                prp.SetSpecular(.05)
                prp.SetSpecularPower(5)
                self._mapper.ScalarVisibilityOff()

        if alpha is not None:
            prp.SetOpacity(alpha)


    ###############################################
    def __add__(self, actors):
        if isinstance(actors, list):
            alist = [self]
            for l in actors:
                if isinstance(l, vtk.vtkAssembly):
                    alist += l.getActors()
                else:
                    alist += l
            return Assembly(alist)
        elif isinstance(actors, vtk.vtkAssembly):
            actors.AddPart(self)
            return actors
        return Assembly([self, actors])

    def __str__(self):
        utils.printInfo(self)
        return ""

    def updateMesh(self, polydata):
        """
        Overwrite the polygonal mesh of the actor with a new one.
        """
        self._polydata = polydata
        self._mapper.SetInputData(polydata)
        self._mapper.Modified()
        return self


    def getPoint(self, i):
        """
        Retrieve specific `i-th` point coordinates in mesh.
        Actor transformation is reset to its mesh position/orientation.

        :param int i: index of vertex point.

        .. warning:: if used in a loop this can slow down the execution by a lot.

        .. seealso:: ``actor.getPoints()``
        """
        poly = self.polydata(True)
        p = [0, 0, 0]
        poly.GetPoints().GetPoint(i, p)
        return np.array(p)

    def setPoint(self, i, p):
        """
        Set specific `i-th` point coordinates in mesh.
        Actor transformation is reset to its original mesh position/orientation.

        :param int i: index of vertex point.
        :param list p: new coordinates of mesh point.

        .. warning:: if used in a loop this can slow down the execution by a lot.

        .. seealso:: ``actor.Points()``
        """
        poly = self.polydata(False)
        poly.GetPoints().SetPoint(i, p)
        poly.GetPoints().Modified()
        poly.GetPoints().GetData().Modified()
        # reset actor to identity matrix position/rotation:
        self.PokeMatrix(vtk.vtkMatrix4x4())
        return self

    def getPoints(self, transformed=True, copy=False):
        """
        Return the list of vertex coordinates of the input mesh.
        Same as `actor.coordinates()`.

        :param bool transformed: if `False` ignore any previous transformation
            applied to the mesh.
        :param bool copy: if `False` return the reference to the points
            so that they can be modified in place.
        """
        poly = self.polydata(transformed)
        if copy:
            return np.array(vtk_to_numpy(poly.GetPoints().GetData()))
        else:
            return vtk_to_numpy(poly.GetPoints().GetData())

    def setPoints(self, pts):
        """
        Set specific points coordinates in mesh. Input is a python list.
        Actor transformation is reset to its mesh position/orientation.

        :param list pts: new coordinates of mesh vertices.
        """
        vpts = self._polydata.GetPoints()
        vpts.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
        self._polydata.GetPoints().Modified()
        # reset actor to identity matrix position/rotation:
        self.PokeMatrix(vtk.vtkMatrix4x4())
        return self

    def coordinates(self, transformed=True, copy=False):
        """
        Return the list of vertex coordinates of the input mesh.
        Same as `actor.getPoints()`.

        :param bool transformed: if `False` ignore any previous transformation
            applied to the mesh.
        :param bool copy: if `False` return the reference to the points
            so that they can be modified in place, otherwise a copy is built.
        """
        return self.getPoints(transformed, copy)

    def faces(self):
        """Get cell connettivity ids as a python ``list``.
        The output format is: [[id0 ... idn], [id0 ... idm],  etc].

        Same as `getConnectivity()`.
        """
        return self.getConnectivity()

    def getPolygons(self):
        """Get cell connettivity ids as a 1D array. The vtk format is:
            [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        """
        return vtk_to_numpy(self.polydata().GetPolys().GetData())

    def getConnectivity(self):
        """Get cell connettivity ids as a python ``list``. The format is:
            [[id0 ... idn], [id0 ... idm],  etc].

        Same as `faces()`.
        """
        arr1d = self.getPolygons()
        if len(arr1d) == 0:
            arr1d = vtk_to_numpy(self.polydata().GetStrips().GetData())

        #conn = arr1d.reshape(ncells, int(len(arr1d)/len(arr1d)))
        #return conn[:, 1:]
        # instead of:

        i = 0
        conn = []
        n = len(arr1d)
        for idummy in range(n):
            cell = []
            for k in range(arr1d[i]):
                cell.append(arr1d[i+k+1])
            conn.append(cell)
            i += arr1d[i]+1
            if i >= n:
                break
        return conn # cannot always make a numpy array of it!

    def addScalarBar(self,
                     pos=(0.8,0.05),
                     title="",
                     titleXOffset=0,
                     titleYOffset=15,
                     titleFontSize=12,
                     nlabels=10,
                     c=None,
                     horizontal=False,
                     vmin=None, vmax=None,
    ):
        """
        Add a 2D scalar bar to actor.

        |mesh_bands| |mesh_bands.py|_
        """
        import vtkplotter.addons as addons
        self.scalarbar = addons.addScalarBar(self,
                 pos,
                 title,
                 titleXOffset,
                 titleYOffset,
                 titleFontSize,
                 nlabels,
                 c,
                 horizontal,
                 vmin, vmax,
                 )
        return self

    def addScalarBar3D(
        self,
        pos=(0, 0, 0),
        normal=(0, 0, 1),
        sx=0.1,
        sy=2,
        title='',
        titleXOffset = -1.4, # space btw title and scale
        titleYOffset = 0.0,
        titleSize =  1.5,
        titleRotation = 0.0,
        nlabels=9,
        precision=3,
        labelOffset = 0.4,  # space btw numeric labels and scale
        c=None,
        alpha=1,
        cmap=None,
    ):
        """
        Draw a 3D scalar bar to actor.

        |mesh_coloring| |mesh_coloring.py|_
        """
        import vtkplotter.addons as addons
        self.scalarbar = addons.addScalarBar3D(self,
                                                pos,
                                                normal,
                                                sx, sy,
                                                title,
                                                titleXOffset,
                                                titleYOffset,
                                                titleSize,
                                                titleRotation,
                                                nlabels,
                                                precision,
                                                labelOffset,
                                                c, alpha, cmap)
        return self.scalarbar


    def texture(self, tname,
                tcoords=None,
                interpolate=True,
                repeat=True,
                edgeClamp=False,
                ):
        """Assign a texture to actor from image file or predefined texture `tname`.
        If tname is ``None`` texture is disabled.

        :param bool interpolate: turn on/off linear interpolation of the texture map when rendering.
        :param bool repeat: repeat of the texture when tcoords extend beyond the [0,1] range.
        :param bool edgeClamp: turn on/off the clamping of the texture map when
            the texture coords extend beyond the [0,1] range.
            Only used when repeat is False, and edge clamping is supported by the graphics card.
        """
        pd = self.polydata(False)
        if tname is None:
            pd.GetPointData().SetTCoords(None)
            pd.GetPointData().Modified()
            return self

        if isinstance(tname, vtk.vtkTexture):
            tu = tname
        else:
            if tcoords is not None:
                if not isinstance(tcoords, np.ndarray):
                    tcoords = np.array(tcoords)
                if tcoords.ndim != 2:
                    colors.printc('tcoords must be a 2-dimensional array', c=1)
                    return self
                if tcoords.shape[0] != pd.GetNumberOfPoints():
                    colors.printc('Error in texture(): nr of texture coords must match nr of points', c=1)
                    return self
                if tcoords.shape[1] != 2:
                    colors.printc('Error in texture(): vector must have 2 components', c=1)
                tarr = numpy_to_vtk(tcoords)
                tarr.SetName('TCoordinates')
                pd.GetPointData().SetTCoords(tarr)
                pd.GetPointData().Modified()
            else:
                if not pd.GetPointData().GetTCoords():
                    tmapper = vtk.vtkTextureMapToPlane()
                    tmapper.AutomaticPlaneGenerationOn()
                    tmapper.SetInputData(pd)
                    tmapper.Update()
                    tc = tmapper.GetOutput().GetPointData().GetTCoords()
                    pd.GetPointData().SetTCoords(tc)
                    pd.GetPointData().Modified()

            fn = settings.textures_path + tname + ".jpg"
            if os.path.exists(tname):
                fn = tname
            elif not os.path.exists(fn):
                colors.printc("~sad File does not exist or texture", tname,
                              "not found in", settings.textures_path, c="r")
                colors.printc("~pin Available built-in textures:", c="m", end=" ")
                for ff in os.listdir(settings.textures_path):
                    colors.printc(ff.split(".")[0], end=" ", c="m")
                print()
                return self

            fnl = fn.lower()
            if ".jpg" in fnl or ".jpeg" in fnl:
                reader = vtk.vtkJPEGReader()
            elif ".png" in fnl:
                reader = vtk.vtkPNGReader()
            elif ".bmp" in fnl:
                reader = vtk.vtkBMPReader()
            else:
                colors.printc("Error in texture(): supported files, PNG, BMP or JPG", c="r")
                return self
            reader.SetFileName(fn)
            reader.Update()

            tu = vtk.vtkTexture()
            tu.SetInputData(reader.GetOutput())
            tu.SetInterpolate(interpolate)
            tu.SetRepeat(repeat)
            tu.SetEdgeClamp(edgeClamp)

        self.GetProperty().SetColor(1, 1, 1)
        self._mapper.ScalarVisibilityOff()
        self.SetTexture(tu)
        self.Modified()
        return self


    def deletePoints(self, indices):
        """Delete a list of vertices identified by their index.

        |deleteMeshPoints| |deleteMeshPoints.py|_
        """
        cellIds = vtk.vtkIdList()
        self._polydata.BuildLinks()
        for i in indices:
            self._polydata.GetPointCells(i, cellIds)
            for j in range(cellIds.GetNumberOfIds()):
                self._polydata.DeleteCell(cellIds.GetId(j))  # flag cell

        self._polydata.RemoveDeletedCells()
        self._mapper.Modified()
        return self

    def computeNormals(self, points=True, cells=True):
        """Compute cell and vertex normals for the actor's mesh.

        .. warning:: Mesh gets modified, output can have a different nr. of vertices.
        """
        poly = self.polydata(False)
        pdnorm = vtk.vtkPolyDataNormals()
        pdnorm.SetInputData(poly)
        pdnorm.SetComputePointNormals(points)
        pdnorm.SetComputeCellNormals(cells)
        pdnorm.FlipNormalsOff()
        pdnorm.ConsistencyOn()
        pdnorm.Update()
        return self.updateMesh(pdnorm.GetOutput())

    def reverse(self, cells=True, normals=False):
        """
        Reverse the order of polygonal cells
        and/or reverse the direction of point and cell normals.
        Two flags are used to control these operations:

        - `cells=True` reverses the order of the indices in the cell connectivity list.

        - `normals=True` reverses the normals by multiplying the normal vector by -1
            (both point and cell normals, if present).
        """
        poly = self.polydata(False)
        rev = vtk.vtkReverseSense()
        if cells:
            rev.ReverseCellsOn()
        else:
            rev.ReverseCellsOff()
        if normals:
            rev.ReverseNormalsOn()
        else:
            rev.ReverseNormalsOff()
        rev.SetInputData(poly)
        rev.Update()
        return self.updateMesh(rev.GetOutput())

    def alpha(self, opacity=None):
        """Set/get actor's transparency. Same as `actor.opacity()`."""
        if opacity is None:
            return self.GetProperty().GetOpacity()

        self.GetProperty().SetOpacity(opacity)
        bfp = self.GetBackfaceProperty()
        if bfp:
            if opacity < 1:
                self._bfprop = bfp
                self.SetBackfaceProperty(None)
            else:
                self.SetBackfaceProperty(self._bfprop)
        return self

    def opacity(self, alpha=None):
        """Set/get actor's transparency. Same as `actor.alpha()`."""
        return self.alpha(alpha)

    def wireframe(self, value=True):
        """Set actor's representation as wireframe or solid surface.
        Same as `wireframe()`."""
        if value:
            self.GetProperty().SetRepresentationToWireframe()
        else:
            self.GetProperty().SetRepresentationToSurface()
        return self

    def flat(self):
        """Set surface interpolation to Flat.

        |wikiphong|
        """
        self.GetProperty().SetInterpolationToFlat()
        return self

    def phong(self):
        """Set surface interpolation to Phong."""
        self.GetProperty().SetInterpolationToPhong()
        return self

    def gouraud(self):
        """Set surface interpolation to Gouraud."""
        self.GetProperty().SetInterpolationToGouraud()
        return self

    def backFaceCulling(self, value=True):
        """Set culling of polygons based on orientation
        of normal with respect to camera."""
        self.GetProperty().SetBackfaceCulling(value)
        return self

    def frontFaceCulling(self, value=True):
        """Set culling of polygons based on orientation of normal with respect to camera."""
        self.GetProperty().SetFrontfaceCulling(value)
        return self

    def pointSize(self, ps=None):
        """Set/get actor's point size of vertices. Same as `ps()`"""
        if ps is not None:
            if isinstance(self, vtk.vtkAssembly):
                cl = vtk.vtkPropCollection()
                self.GetActors(cl)
                cl.InitTraversal()
                a = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
                a.GetProperty().SetRepresentationToPoints()
                a.GetProperty().SetPointSize(ps)
            else:
                self.GetProperty().SetRepresentationToPoints()
                self.GetProperty().SetPointSize(ps)
        else:
            return self.GetProperty().GetPointSize()
        return self

    def ps(self, pointSize=None):
        """Set/get actor's point size of vertices. Same as `pointSize()`"""
        return self.pointSize(pointSize)

    def color(self, c=False):
        """
        Set/get actor's color.
        If None is passed as input, will use colors from active scalars.
        Same as `c()`.
        """
        if c is False:
            return np.array(self.GetProperty().GetColor())
        elif c is None:
            self._mapper.ScalarVisibilityOn()
            return self
        elif isinstance(c, str):
            if c in colors._mapscales_cmaps:
                self.cmap = c
                if self._polydata.GetPointData().GetScalars():
                    aname = self._polydata.GetPointData().GetScalars().GetName()
                    if aname: self.pointColors(aname, cmap=c)
                if self._polydata.GetCellData().GetScalars():
                    aname = self._polydata.GetCellData().GetScalars().GetName()
                    if aname: self.cellColors(aname, cmap=c)
                self._mapper.ScalarVisibilityOn()
                return self
        self._mapper.ScalarVisibilityOff()
        cc = colors.getColor(c)
        self.GetProperty().SetColor(cc)
        if self.trail:
            self.trail.GetProperty().SetColor(cc)
        return self


    def backColor(self, bc=None):
        """
        Set/get actor's backface color.
        """
        backProp = self.GetBackfaceProperty()

        if bc is None:
            if backProp:
                return backProp.GetDiffuseColor()
            return self

        if self.GetProperty().GetOpacity() < 1:
            colors.printc("~noentry backColor(): only active for alpha=1", c="y")
            return self

        if not backProp:
            backProp = vtk.vtkProperty()

        backProp.SetDiffuseColor(colors.getColor(bc))
        backProp.SetOpacity(self.GetProperty().GetOpacity())
        self.SetBackfaceProperty(backProp)
        self._mapper.ScalarVisibilityOff()
        return self

    def bc(self, backColor=False):
        """Shortcut for `actor.backColor()`. """
        return self.backColor(backColor)

    def lineWidth(self, lw=None):
        """Set/get width of mesh edges. Same as `lw()`."""
        if lw is not None:
            if lw == 0:
                self.GetProperty().EdgeVisibilityOff()
                return self
            self.GetProperty().EdgeVisibilityOn()
            self.GetProperty().SetLineWidth(lw)
        else:
            return self.GetProperty().GetLineWidth()
        return self

    def lw(self, lineWidth=None):
        """Set/get width of mesh edges. Same as `lineWidth()`."""
        return self.lineWidth(lineWidth)

    def lineColor(self, lc=None):
        """Set/get color of mesh edges. Same as `lc()`."""
        if lc is not None:
            self.GetProperty().EdgeVisibilityOn()
            self.GetProperty().SetEdgeColor(colors.getColor(lc))
        else:
            return self.GetProperty().GetEdgeColor()
        return self

    def lc(self, lineColor=None):
        """Set/get color of mesh edges. Same as `lineColor()`."""
        return self.lineColor(lineColor)

    def clean(self, tol=None):
        """
        Clean actor's polydata. Can also be used to decimate a mesh if ``tol`` is large.
        If ``tol=None`` only removes coincident points.

        :param tol: defines how far should be the points from each other
            in terms of fraction of the bounding box length.

        |moving_least_squares1D| |moving_least_squares1D.py|_

            |recosurface| |recosurface.py|_
        """
        poly = self.polydata(False)
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.PointMergingOn()
        cleanPolyData.ConvertLinesToPointsOn()
        cleanPolyData.ConvertPolysToLinesOn()
        cleanPolyData.SetInputData(poly)
        if tol:
            cleanPolyData.SetTolerance(tol)
        cleanPolyData.Update()
        return self.updateMesh(cleanPolyData.GetOutput())

    def quantize(self, binSize):
        """
        The user should input binSize and all {x,y,z} coordinates
        will be quantized to that absolute grain size.

        Example:
            .. code-block:: python

                from vtkplotter import Paraboloid
                Paraboloid().lw(0.1).quantize(0.1).show()
        """
        poly = self.polydata(False)
        qp = vtk.vtkQuantizePolyDataPoints()
        qp.SetInputData(poly)
        qp.SetQFactor(binSize)
        qp.Update()
        return self.updateMesh(qp.GetOutput())

    def xbounds(self):
        """Get the actor bounds `[xmin,xmax]`."""
        b = self.polydata(True).GetBounds()
        return (b[0], b[1])

    def ybounds(self):
        """Get the actor bounds `[ymin,ymax]`."""
        b = self.polydata(True).GetBounds()
        return (b[2], b[3])

    def zbounds(self):
        """Get the actor bounds `[zmin,zmax]`."""
        b = self.polydata(True).GetBounds()
        return (b[4], b[5])

    def averageSize(self):
        """Calculate the average size of a mesh.
        This is the mean of the vertex distances from the center of mass."""
        cm = self.centerOfMass()
        coords = self.coordinates(copy=False)
        if not len(coords):
            return 0
        s, c = 0.0, 0.0
        n = len(coords)
        step = int(n / 10000.0) + 1
        for i in np.arange(0, n, step):
            s += utils.mag(coords[i] - cm)
            c += 1
        return s / c

    def diagonalSize(self):
        """Get the length of the diagonal of actor bounding box."""
        b = self.polydata().GetBounds()
        return np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)

    def maxBoundSize(self):
        """Get the maximum dimension in x, y or z of the actor bounding box."""
        b = self.polydata(True).GetBounds()
        return max(abs(b[1] - b[0]), abs(b[3] - b[2]), abs(b[5] - b[4]))

    def centerOfMass(self):
        """Get the center of mass of actor.

        |fatlimb| |fatlimb.py|_
        """
        cmf = vtk.vtkCenterOfMass()
        cmf.SetInputData(self.polydata(True))
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)

    def volume(self, value=None):
        """Get/set the volume occupied by actor."""
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        v = mass.GetVolume()
        if value is not None:
            if not v:
                colors.printc("~bomb Volume is zero: cannot rescale.", c=1, end="")
                colors.printc(" Consider adding actor.triangle()", c=1)
                return self
            self.scale(value / v)
            return self
        else:
            return v

    def area(self, value=None):
        """Get/set the surface area of actor.

        .. hint:: |largestregion.py|_
        """
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        ar = mass.GetSurfaceArea()
        if value is not None:
            if not ar:
                colors.printc("~bomb Area is zero: cannot rescale.", c=1, end="")
                colors.printc(" Consider adding actor.triangle()", c=1)
                return self
            self.scale(value / ar)
            return self
        else:
            return ar

    def closestPoint(self, pt, N=1, radius=None, returnIds=False):
        """
        Find the closest point(s) on a mesh given from the input point `pt`.

        :param int N: if greater than 1, return a list of N ordered closest points.
        :param float radius: if given, get all points within that radius.
        :param bool returnIds: return points IDs instead of point coordinates.

        .. hint:: |align1.py|_ |fitplanes.py|_  |quadratic_morphing.py|_

            |align1| |quadratic_morphing|

        .. note:: The appropriate kd-tree search locator is built on the
            fly and cached for speed.
        """
        poly = self.polydata(True)

        if N > 1 or radius:
            plocexists = self.point_locator
            if not plocexists or (plocexists and self.point_locator is None):
                point_locator = vtk.vtkPointLocator()
                point_locator.SetDataSet(poly)
                point_locator.BuildLocator()
                self.point_locator = point_locator

            vtklist = vtk.vtkIdList()
            if N > 1:
                self.point_locator.FindClosestNPoints(N, pt, vtklist)
            else:
                self.point_locator.FindPointsWithinRadius(radius, pt, vtklist)
            if returnIds:
                return [int(vtklist.GetId(k)) for k in range(vtklist.GetNumberOfIds())]
            else:
                trgp = []
                for i in range(vtklist.GetNumberOfIds()):
                    trgp_ = [0, 0, 0]
                    vi = vtklist.GetId(i)
                    poly.GetPoints().GetPoint(vi, trgp_)
                    trgp.append(trgp_)
                return np.array(trgp)

        clocexists = self.cell_locator
        if not clocexists or (clocexists and self.cell_locator is None):
            cell_locator = vtk.vtkCellLocator()
            cell_locator.SetDataSet(poly)
            cell_locator.BuildLocator()
            self.cell_locator = cell_locator

        trgp = [0, 0, 0]
        cid = vtk.mutable(0)
        dist2 = vtk.mutable(0)
        subid = vtk.mutable(0)
        self.cell_locator.FindClosestPoint(pt, trgp, cid, subid, dist2)
        if returnIds:
            return int(cid)
        else:
            return np.array(trgp)


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
        #self.cell_locator.SetNumberOfCellsPerNode(2)
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


    def distanceToMesh(self, actor, signed=False, negate=False):
        '''
        Computes the (signed) distance from one mesh to another.

        |distance2mesh| |distance2mesh.py|_
        '''
        poly1 = self.polydata()
        poly2 = actor.polydata()
        df = vtk.vtkDistancePolyDataFilter()
        df.ComputeSecondDistanceOff()
        df.SetInputData(0, poly1)
        df.SetInputData(1, poly2)
        if signed:
            df.SignedDistanceOn()
        else:
            df.SignedDistanceOff()
        if negate:
            df.NegateDistanceOn()
        df.Update()

        scals = df.GetOutput().GetPointData().GetScalars()
        poly1.GetPointData().AddArray(scals)

        poly1.GetPointData().SetActiveScalars(scals.GetName())
        rng = scals.GetRange()
        self._mapper.SetScalarRange(rng[0], rng[1])
        self._mapper.ScalarVisibilityOn()
        return self

    def clone(self, transformed=True):
        """
        Clone a ``Actor(vtkActor)`` and make an exact copy of it.

        :param transformed: if `False` ignore any previous transformation applied to the mesh.

        |mirror| |mirror.py|_
        """
        poly = self.polydata(transformed=transformed)
        polyCopy = vtk.vtkPolyData()
        polyCopy.DeepCopy(poly)

        cloned = Actor()
        cloned._polydata = polyCopy
        cloned._mapper.SetInputData(polyCopy)
        cloned._mapper.SetScalarVisibility(self._mapper.GetScalarVisibility())
        cloned._mapper.SetScalarRange(self._mapper.GetScalarRange())
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.GetProperty())
        cloned.SetProperty(pr)
        if self.trail:
            n = len(self.trailPoints)
            cloned.addTrail(self.trailOffset, self.trailSegmentSize*n, n,
                            None, None, self.trail.GetProperty().GetLineWidth())
        if self.shadow:
            cloned.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                             self.shadow.GetProperty().GetColor(),
                             self.shadow.GetProperty().GetOpacity())
        return cloned

    def transformMesh(self, transformation):
        """
        Apply this transformation to the polygonal `mesh`,
        not to the actor's transformation, which is reset.

        :param transformation: ``vtkTransform`` or ``vtkMatrix4x4`` object.
        """
        if isinstance(transformation, vtk.vtkMatrix4x4):
            tr = vtk.vtkTransform()
            tr.SetMatrix(transformation)
            transformation = tr

        tf = vtk.vtkTransformPolyDataFilter()

        tf.SetTransform(transformation)
        tf.SetInputData(self.polydata())
        tf.Update()
        self.PokeMatrix(vtk.vtkMatrix4x4())  # identity
        return self.updateMesh(tf.GetOutput())

    def normalize(self):
        """
        Shift actor's center of mass at origin and scale its average size to unit.
        """
        cm = self.centerOfMass()
        coords = self.coordinates()
        if not len(coords):
            return
        pts = coords - cm
        xyz2 = np.sum(pts * pts, axis=0)
        scale = 1 / np.sqrt(np.sum(xyz2) / len(pts))
        t = vtk.vtkTransform()
        t.Scale(scale, scale, scale)
        t.Translate(-cm)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(self._polydata)
        tf.SetTransform(t)
        tf.Update()
        return self.updateMesh(tf.GetOutput())

    def mirror(self, axis="x"):
        """
        Mirror the actor polydata along one of the cartesian axes.

        .. note::  ``axis='n'``, will flip only mesh normals.

        |mirror| |mirror.py|_
        """
        poly = self.polydata(transformed=True)
        polyCopy = vtk.vtkPolyData()
        polyCopy.DeepCopy(poly)

        sx, sy, sz = 1, 1, 1
        dx, dy, dz = self.GetPosition()
        if axis.lower() == "x":
            sx = -1
        elif axis.lower() == "y":
            sy = -1
        elif axis.lower() == "z":
            sz = -1
        elif axis.lower() == "n":
            pass
        else:
            colors.printc("~times Error in mirror(): mirror must be set to x, y, z or n.",
                          c=1)
            raise RuntimeError()

        if axis != "n":
            for j in range(polyCopy.GetNumberOfPoints()):
                p = [0, 0, 0]
                polyCopy.GetPoint(j, p)
                polyCopy.GetPoints().SetPoint(
                    j,
                    p[0] * sx - dx * (sx - 1),
                    p[1] * sy - dy * (sy - 1),
                    p[2] * sz - dz * (sz - 1),
                )
        rs = vtk.vtkReverseSense()
        rs.SetInputData(polyCopy)
        rs.ReverseNormalsOn()
        rs.Update()
        polyCopy = rs.GetOutput()

        pdnorm = vtk.vtkPolyDataNormals()
        pdnorm.SetInputData(polyCopy)
        pdnorm.ComputePointNormalsOn()
        pdnorm.ComputeCellNormalsOn()
        pdnorm.FlipNormalsOff()
        pdnorm.ConsistencyOn()
        pdnorm.Update()
        return self.updateMesh(pdnorm.GetOutput())

    def flipNormals(self):
        """
        Flip all mesh normals. Same as `actor.mirror('n')`.
        """
        return self.mirror("n")

    def shrink(self, fraction=0.85):
        """Shrink the triangle polydata in the representation of the input mesh.

        Example:
            .. code-block:: python

                from vtkplotter import *
                pot = load(datadir + 'shapes/teapot.vtk').shrink(0.75)
                s = Sphere(r=0.2).pos(0,0,-0.5)
                show(pot, s)

            |shrink| |shrink.py|_
        """
        poly = self.polydata(True)
        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputData(poly)
        shrink.SetShrinkFactor(fraction)
        shrink.Update()
        return self.updateMesh(shrink.GetOutput())

    def stretch(self, q1, q2):
        """Stretch actor between points `q1` and `q2`. Mesh is not affected.

        |aspring| |aspring.py|_

        .. note:: for ``Actors`` like helices, Line, cylinders, cones etc.,
            two attributes ``actor.base``, and ``actor.top`` are already defined.
        """
        if self.base is None:
            colors.printc('~times Error in stretch(): Please define vectors \
                          actor.base and actor.top at creation.', c='r')
            raise RuntimeError()

        p1, p2 = self.base, self.top
        q1, q2, z = np.array(q1), np.array(q2), np.array([0, 0, 1])
        plength = np.linalg.norm(p2 - p1)
        qlength = np.linalg.norm(q2 - q1)
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.Translate(-p1)
        cosa = np.dot(p2 - p1, z) / plength
        n = np.cross(p2 - p1, z)
        T.RotateWXYZ(np.rad2deg(np.arccos(cosa)), n)

        T.Scale(1, 1, qlength / plength)

        cosa = np.dot(q2 - q1, z) / qlength
        n = np.cross(q2 - q1, z)
        T.RotateWXYZ(-np.rad2deg(np.arccos(cosa)), n)
        T.Translate(q1)

        self.SetUserMatrix(T.GetMatrix())
        if self.trail:
            self.updateTrail()
        if self.shadow:
            self.addShadow(self.shadowX, self.shadowY, self.shadowZ,
                           self.shadow.GetProperty().GetColor(),
                           self.shadow.GetProperty().GetOpacity())
        return self

    def crop(self, top=None, bottom=None, right=None, left=None, front=None, back=None):
        """Crop an ``Actor`` object. Input object is modified.

        :param float top:    fraction to crop from the top plane (positive z)
        :param float bottom: fraction to crop from the bottom plane (negative z)
        :param float front:  fraction to crop from the front plane (positive y)
        :param float back:   fraction to crop from the back plane (negative y)
        :param float right:  fraction to crop from the right plane (positive x)
        :param float left:   fraction to crop from the left plane (negative x)

        Example:
            .. code-block:: python

                from vtkplotter import Sphere
                Sphere().crop(right=0.3, left=0.1).show()

            |cropped|
        """
        x0, x1, y0, y1, z0, z1 = self.GetBounds()

        cu = vtk.vtkBox()
        dx, dy, dz = x1-x0, y1-y0, z1-z0
        if top:    z1 = z1 - top*dz
        if bottom: z0 = z0 + bottom*dz
        if front:  y1 = y1 - front*dy
        if back:   y0 = y0 + back*dy
        if right:  x1 = x1 - right*dx
        if left:   x0 = x0 + left*dx
        cu.SetBounds(x0,x1, y0,y1, z0,z1)

        poly = self.polydata()
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(poly)
        clipper.SetClipFunction(cu)
        clipper.InsideOutOn()
        clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()

        self.updateMesh(clipper.GetOutput())
        return self

    def cutWithPlane(self, origin=(0, 0, 0), normal=(1, 0, 0), showcut=False):
        """
        Takes a ``vtkActor`` and cuts it with the plane defined by a point and a normal.

        :param origin: the cutting plane goes through this point
        :param normal: normal of the cutting plane
        :param showcut: if `True` show the cut off part of the mesh as thin wireframe.

        :Example:
            .. code-block:: python

                from vtkplotter import Cube

                cube = Cube().cutWithPlane(normal=(1,1,1))
                cube.bc('pink').show()

            |cutcube|

        |trail| |trail.py|_
        """
        if str(normal) == "x":
            normal = (1, 0, 0)
        elif str(normal) == "y":
            normal = (0, 1, 0)
        elif str(normal) == "z":
            normal = (0, 0, 1)
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        self.computeNormals()
        poly = self.polydata()
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(poly)
        clipper.SetClipFunction(plane)
        if showcut:
            clipper.GenerateClippedOutputOn()
        else:
            clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()

        self.updateMesh(clipper.GetOutput()).computeNormals()

        if showcut:
            c = self.GetProperty().GetColor()
            cpoly = clipper.GetClippedOutput()
            restActor = Actor(cpoly, c, 0.05).wireframe(True)
            restActor.SetUserMatrix(self.GetMatrix())
            asse = Assembly([self, restActor])
            self = asse
            return asse
        else:
            return self

    def cutWithMesh(self, mesh, invert=False):
        """
        Cut an ``Actor`` mesh with another ``vtkPolyData`` or ``Actor``.

        :param bool invert: if True return cut off part of actor.

        .. hint:: |cutWithMesh.py|_ |cutAndCap.py|_

            |cutWithMesh| |cutAndCap|
        """
        if isinstance(mesh, vtk.vtkPolyData):
            polymesh = mesh
        elif isinstance(mesh, Actor):
            polymesh = mesh.polydata()
        else:
            polymesh = mesh.GetMapper().GetInput()
        poly = self.polydata()

        # Create an array to hold distance information
        signedDistances = vtk.vtkFloatArray()
        signedDistances.SetNumberOfComponents(1)
        signedDistances.SetName("SignedDistances")

        # implicit function that will be used to slice the mesh
        ippd = vtk.vtkImplicitPolyDataDistance()
        ippd.SetInput(polymesh)

        # Evaluate the signed distance function at all of the grid points
        for pointId in range(poly.GetNumberOfPoints()):
            p = poly.GetPoint(pointId)
            signedDistance = ippd.EvaluateFunction(p)
            signedDistances.InsertNextValue(signedDistance)

        # add the SignedDistances to the grid
        poly.GetPointData().SetScalars(signedDistances)

        # use vtkClipDataSet to slice the grid with the polydata
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(poly)
        clipper.SetInsideOut(not invert)
        clipper.SetValue(0.0)
        clipper.Update()
        return self.updateMesh(clipper.GetOutput())

    def cutWithPointLoop(self, points, invert=False):
        """
        Cut an ``Actor`` mesh with a set of points forming a closed loop.
        """
        if isinstance(points, Actor):
            vpts = points.polydata().GetPoints()
            points = points.coordinates()
        else:
            vpts = vtk.vtkPoints()
            for p in points:
                vpts.InsertNextPoint(p)

        spol = vtk.vtkSelectPolyData()
        spol.SetLoop(vpts)
        spol.GenerateSelectionScalarsOn()
        spol.GenerateUnselectedOutputOff()
        spol.SetInputData(self.polydata())
        spol.Update()

        # use vtkClipDataSet to slice the grid with the polydata
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(spol.GetOutput())
        clipper.SetInsideOut(not invert)
        clipper.SetValue(0.0)
        clipper.Update()
        return self.updateMesh(clipper.GetOutput())


    def cap(self, returnCap=False):
        """
        Generate a "cap" on a clipped actor, or caps sharp edges.

        |cutAndCap| |cutAndCap.py|_
        """
        poly = self.polydata(True)

        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(poly)
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ManifoldEdgesOff()
        fe.Update()

        stripper = vtk.vtkStripper()
        stripper.SetInputData(fe.GetOutput())
        stripper.Update()

        boundaryPoly = vtk.vtkPolyData()
        boundaryPoly.SetPoints(stripper.GetOutput().GetPoints())
        boundaryPoly.SetPolys(stripper.GetOutput().GetLines())

        tf = vtk.vtkTriangleFilter()
        tf.SetInputData(boundaryPoly)
        tf.Update()

        if returnCap:
            return Actor(tf.GetOutput())
        else:
            polyapp = vtk.vtkAppendPolyData()
            polyapp.AddInputData(poly)
            polyapp.AddInputData(tf.GetOutput())
            polyapp.Update()
            return self.updateMesh(polyapp.GetOutput()).clean()

    def threshold(self, scalars, vmin=None, vmax=None, useCells=False):
        """
        Extracts cells where scalar value satisfies threshold criterion.

        :param scalars: name of the scalars array.
        :type scalars: str, list
        :param float vmin: minimum value of the scalar
        :param float vmax: maximum value of the scalar
        :param bool useCells: if `True`, assume array scalars refers to cells.

        |mesh_threshold| |mesh_threshold.py|_
        """
        if utils.isSequence(scalars):
            self.addPointScalars(scalars, "threshold")
            scalars = "threshold"
        elif self.scalars(scalars) is None:
            colors.printc("~times No scalars found with name/nr:", scalars, c=1)
            raise RuntimeError()

        thres = vtk.vtkThreshold()
        thres.SetInputData(self._polydata)

        if useCells:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        thres.SetInputArrayToProcess(0, 0, 0, asso, scalars)

        if vmin is None and vmax is not None:
            thres.ThresholdByLower(vmax)
        elif vmax is None and vmin is not None:
            thres.ThresholdByUpper(vmin)
        else:
            thres.ThresholdBetween(vmin, vmax)
        thres.Update()

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(thres.GetOutput())
        gf.Update()
        return self.updateMesh(gf.GetOutput())

    def triangle(self, verts=True, lines=True):
        """
        Converts actor polygons and strips to triangles.

        :param bool verts: if True, break input vertex cells into individual vertex cells
            (one point per cell). If False, the input vertex cells will be ignored.
        :param bool lines: if True, break input polylines into line segments.
            If False, input lines will be ignored and the output will have no lines.
        """
        tf = vtk.vtkTriangleFilter()
        tf.SetPassLines(lines)
        tf.SetPassVerts(verts)
        tf.SetInputData(self._polydata)
        tf.Update()
        return self.updateMesh(tf.GetOutput())

    def pointColors(self, scalars_or_colors, cmap="jet", alpha=1,
                    mode='scalars',
                    bands=None, vmin=None, vmax=None):
        """
        Set individual point colors by providing a list of scalar values and a color map.
        `scalars` can be a string name of the ``vtkArray``.

        if ``mode='colors'``, colorize vertices of a mesh one by one,
        passing a 1-to-1 list of colors.

        :param list alphas: single value or list of transparencies for each vertex

        :param cmap: color map scheme to transform a real number into a color.
        :type cmap: str, list, vtkLookupTable, matplotlib.colors.LinearSegmentedColormap
        :param alpha: mesh transparency. Can be a ``list`` of values one for each vertex.
        :type alpha: float, list
        :param int bands: group scalars in this number of bins, typically to form bands or stripes.
        :param float vmin: clip scalars to this minimum value
        :param float vmax: clip scalars to this maximum value

        .. hint::|mesh_coloring.py|_ |mesh_alphas.py|_ |mesh_bands.py|_ |mesh_custom.py|_

             |mesh_coloring| |mesh_alphas| |mesh_bands| |mesh_custom|
        """
        ####################################################################
        if 'color' in mode:
            return self._pointColors1By1(scalars_or_colors, alpha)
        ####################################################################

        poly = self.polydata(False)

        if isinstance(scalars_or_colors, str):  # if a name is passed
            scalars_or_colors = vtk_to_numpy(poly.GetPointData().GetArray(scalars_or_colors))

        try:
            n = len(scalars_or_colors)
        except TypeError:  # invalid type
            return self

        useAlpha = False
        if n != poly.GetNumberOfPoints():
            colors.printc('~times pointColors Error: nr. of scalars != nr. of points',
                          n, poly.GetNumberOfPoints(), c=1)
        if utils.isSequence(alpha):
            useAlpha = True
            if len(alpha) > n:
                colors.printc('~times pointColors Error: nr. of scalars < nr. of alpha values',
                              n, len(alpha), c=1)
                raise RuntimeError()
        if bands:
            scalars_or_colors = utils.makeBands(scalars_or_colors, bands)

        if vmin is None:
            vmin = np.min(scalars_or_colors)
        if vmax is None:
            vmax = np.max(scalars_or_colors)

        lut = vtk.vtkLookupTable()  # build the look-up table

        if utils.isSequence(cmap):
            sname = "pointColors_custom"
            lut.SetNumberOfTableValues(len(cmap))
            lut.Build()
            for i, c in enumerate(cmap):
                col = colors.getColor(c)
                r, g, b = col
                if useAlpha:
                    lut.SetTableValue(i, r, g, b, alpha[i])
                else:
                    lut.SetTableValue(i, r, g, b, alpha)

        elif isinstance(cmap, vtk.vtkLookupTable):
            sname = "pointColors_lut"
            lut.DeepCopy(cmap)

        else:
            if isinstance(cmap, str):
                self.cmap = cmap
                sname = "pointColors_" + cmap
            else:
                sname = "pointColors"
            lut.SetNumberOfTableValues(256)
            lut.Build()
            for i in range(256):
                r, g, b = colors.colorMap(i, cmap, 0, 256)
                if useAlpha:
                    idx = int(i / 256 * len(alpha))
                    lut.SetTableValue(i, r, g, b, alpha[idx])
                else:
                    lut.SetTableValue(i, r, g, b, alpha)

        arr = numpy_to_vtk(np.ascontiguousarray(scalars_or_colors), deep=True)
        arr.SetName(sname)
        self._mapper.SetArrayName(sname)
        if settings.autoResetScalarRange:
            self._mapper.SetScalarRange(vmin, vmax)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetScalarModeToUsePointData()
        self._mapper.ScalarVisibilityOn()
        poly.GetPointData().SetScalars(arr)
        poly.GetPointData().SetActiveScalars(sname)
        return self

    def _pointColors1By1(self, acolors, alphas=1):
        ptData = vtk.vtkUnsignedIntArray()
        ptData.SetName("VertexColors")
        lut = vtk.vtkLookupTable()
        n = self._polydata.GetNumberOfPoints()
        if len(acolors) != n or (utils.isSequence(alphas) and len(alphas) != n):
            colors.printc("~times _pointColors1By1(): mismatch in input list sizes.", c=1)
            return self
        lut.SetNumberOfTableValues(n)
        lut.Build()
        cols = colors.getColor(acolors)
        if not utils.isSequence(alphas):
            alphas = [alphas] * n
        for i in range(n):
            ptData.InsertNextValue(i)
            c = cols[i]
            lut.SetTableValue(i, c[0], c[1], c[2], alphas[i])
        self._polydata.GetPointData().SetScalars(ptData)
        self._polydata.GetPointData().Modified()
        self._mapper.SetScalarRange(0, n-1)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetArrayName("VertexColors")
        self._mapper.SetScalarModeToUsePointData()
        self._mapper.ScalarVisibilityOn()
        return self


    def cellColors(self, scalars_or_colors, cmap="jet", alpha=1, alphaPerCell=False,
                   mode='scalars',
                   bands=None, vmin=None, vmax=None):
        """
        Set individual cell colors by setting a list of scalars.

        If ``mode='scalars'`` (default), set individual cell colors by the
        provided list of scalars.

        If ``mode='colors'``, colorize the faces of a mesh one by one,
        passing a 1-to-1 list of colors and optionally a list of transparencies.

        :param alpha: mesh transparency. Can be a ``list`` of values one for each cell.
        :type alpha: float, list

        Only relevant with ``mode='scalars'`` (default):

        :param cmap: color map scheme to transform a real number into a color.
        :type cmap: str, list, vtkLookupTable, matplotlib.colors.LinearSegmentedColormap
        :param int bands: group scalars in this number of bins, typically to form bands of stripes.
        :param float vmin: clip scalars to this minimum value
        :param float vmax: clip scalars to this maximum value

        Only relevant with ``mode='colors'``:

        :param bool alphaPerCell: Only matters if `alpha` is a sequence. If so:
            if `True` assume that the list of opacities is independent
            on the colors (same color cells can have different opacity),
            this can be very slow for large meshes,

            if `False` [default] assume that the alpha matches the color list
            (same color has the same opacity).
            This is very fast even for large meshes.

        |mesh_coloring| |mesh_coloring.py|_
        """
        ####################################################################
        if 'color' in mode:
            return self._cellColors1By1(scalars_or_colors, alpha, alphaPerCell)
        ####################################################################

        poly = self.polydata(False)

        if isinstance(scalars_or_colors, str):  # if a name is passed
            scalars_or_colors = vtk_to_numpy(poly.GetCellData().GetArray(scalars_or_colors))

        n = len(scalars_or_colors)
        useAlpha = False
        if n != poly.GetNumberOfCells():
            colors.printc('~times cellColors(): nr. of scalars != nr. of cells',
                          n, poly.GetNumberOfCells(), c=1)
        if utils.isSequence(alpha):
            useAlpha = True
            if len(alpha) > n:
                colors.printc('~times cellColors(): nr. of scalars != nr. of alpha values',
                              n, len(alpha), c=1)
                raise RuntimeError()
        if bands:
            scalars_or_colors = utils.makeBands(scalars_or_colors, bands)

        if vmin is None:
            vmin = np.min(scalars_or_colors)
        if vmax is None:
            vmax = np.max(scalars_or_colors)

        lut = vtk.vtkLookupTable()  # build the look-up table

        if utils.isSequence(cmap):
            sname = "cellColors_custom"
            lut.SetNumberOfTableValues(len(cmap))
            lut.Build()
            for i, c in enumerate(cmap):
                col = colors.getColor(c)
                r, g, b = col
                if useAlpha:
                    lut.SetTableValue(i, r, g, b, alpha[i])
                else:
                    lut.SetTableValue(i, r, g, b, alpha)

        elif isinstance(cmap, vtk.vtkLookupTable):
            sname = "cellColors_lut"
            lut.DeepCopy(cmap)

        else:
            if isinstance(cmap, str):
                self.cmap = cmap
                sname = "cellColors_" + cmap
            else:
                sname = "cellColors"
            lut.SetNumberOfTableValues(256)
            lut.Build()
            for i in range(256):
                r, g, b = colors.colorMap(i, cmap, 0, 256)
                if useAlpha:
                    idx = int(i / 256 * len(alpha))
                    lut.SetTableValue(i, r, g, b, alpha[idx])
                else:
                    lut.SetTableValue(i, r, g, b, alpha)

        arr = numpy_to_vtk(np.ascontiguousarray(scalars_or_colors), deep=True)
        arr.SetName(sname)
        self._mapper.SetArrayName(sname)
        if settings.autoResetScalarRange:
            self._mapper.SetScalarRange(vmin, vmax)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetScalarModeToUseCellData()
        self._mapper.ScalarVisibilityOn()
        poly.GetCellData().SetScalars(arr)
        poly.GetCellData().SetActiveScalars(sname)
        return self

    def _cellColors1By1(self, acolors, alphas, alphaPerCell):
        cellData = vtk.vtkUnsignedIntArray()
        cellData.SetName("CellColors")

        n = self._polydata.GetNumberOfCells()
        if len(acolors) != n or (utils.isSequence(alphas) and len(alphas) != n):
            colors.printc("~times cellColors(): mismatch in input list sizes.",
                          len(acolors), n, c=1)
            return self

        lut = vtk.vtkLookupTable()

        if alphaPerCell:
            lut.SetNumberOfTableValues(n)
            lut.Build()
            cols = colors.getColor(acolors)
            if not utils.isSequence(alphas):
                alphas = [alphas] * n
            for i in range(n):
                cellData.InsertNextValue(i)
                c = cols[i]
                lut.SetTableValue(i, c[0], c[1], c[2], alphas[i])
        else:
            ucolors, uids, inds = np.unique(acolors, axis=0,
                                            return_index=True, return_inverse=True)
            nc = len(ucolors)

            if nc == 1:
                self.color(colors.getColor(ucolors[0]))
                if utils.isSequence(alphas):
                    self.alpha(alphas[0])
                else:
                    self.alpha(alphas)
                return self

            for i in range(n):
                cellData.InsertNextValue(int(inds[i]))

            lut.SetNumberOfTableValues(nc)
            lut.Build()

            cols = colors.getColor(ucolors)

            if not utils.isSequence(alphas):
                alphas = np.ones(n)

            for i in range(nc):
                c = cols[i]
                lut.SetTableValue(i, c[0], c[1], c[2], alphas[uids[i]])

        self._polydata.GetCellData().SetScalars(cellData)
        self._polydata.GetCellData().Modified()
        self._mapper.SetScalarRange(0, lut.GetNumberOfTableValues()-1)
        self._mapper.SetLookupTable(lut)
        self._mapper.SetArrayName("CellColors")
        self._mapper.SetScalarModeToUseCellData()
        self._mapper.ScalarVisibilityOn()
        return self

    def addPointVectors(self, vectors, name):
        """
        Add a point vector field to the actor's polydata assigning it a name.
        """
        poly = self.polydata(False)
        if len(vectors) != poly.GetNumberOfPoints():
            colors.printc('~times addPointVectors Error: Number of vectors != nr. of points',
                          len(vectors), poly.GetNumberOfPoints(), c=1)
            raise RuntimeError()
        arr = vtk.vtkFloatArray()
        arr.SetNumberOfComponents(3)
        arr.SetName(name)
        for v in vectors:
            arr.InsertNextTuple(v)
        poly.GetPointData().AddArray(arr)
        poly.GetPointData().SetActiveVectors(name)
        return self

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
        return self.updateMesh(ids.GetOutput())

    def addCurvatureScalars(self, method=0, lut=None):
        """
        Add scalars to ``Actor`` that contains the
        curvature calculated in three different ways.

        :param int method: 0-gaussian, 1-mean, 2-max, 3-min curvature.
        :param lut: optional vtkLookUpTable up table.

        :Example:
            .. code-block:: python

                from vtkplotter import Torus
                Torus().addCurvatureScalars().show()

            |curvature|
        """
        curve = vtk.vtkCurvatures()
        curve.SetInputData(self._polydata)
        curve.SetCurvatureType(method)
        curve.Update()
        self._polydata = curve.GetOutput()

        self._mapper.SetInputData(self._polydata)
        if lut:
            self._mapper.SetLookupTable(lut)
            self._mapper.SetUseLookupTableScalarRange(1)
        self._mapper.Update()
        self.Modified()
        self._mapper.ScalarVisibilityOn()
        return self

    def addElevationScalars(self, lowPoint=(), highPoint=(), vrange=(), lut=None):
        """
        Add to ``Actor`` a scalar array that contains distance along a specified direction.

        :param list low: one end of the line (small scalar values). Default (0,0,0).
        :param list high: other end of the line (large scalar values). Default (0,0,1).
        :param list vrange: set the range of the scalar. Default is (0, 1).
        :param lut: optional vtkLookUpTable up table (see makeLUT method).

        :Example:
            .. code-block:: python

                from vtkplotter import Sphere

                s = Sphere().addElevationScalars(lowPoint=(0,0,0), highPoint=(1,1,1))
                s.addScalarBar().show(axes=1)

                |elevation|
        """
        ef = vtk.vtkElevationFilter()
        ef.SetInputData(self.polydata())
        if len(lowPoint) == 3:
            ef.SetLowPoint(lowPoint)
        if len(highPoint) == 3:
            ef.SetHighPoint(highPoint)
        if len(vrange) == 2:
            ef.SetScalarRange(vrange)

        ef.Update()
        self._polydata = ef.GetOutput()

        self._mapper.SetInputData(self._polydata)
        if lut:
            self._mapper.SetLookupTable(lut)
            self._mapper.SetUseLookupTableScalarRange(1)
        self._mapper.Update()
        self.Modified()
        self._mapper.ScalarVisibilityOn()
        return self

    def scalars(self, name_or_idx=None, datatype="point"):
        """
        Retrieve point or cell scalars using array name or index number,
        and set it as the active one.

        If no input is given return the list of names of existing arrays.

        :param str datatype: search given name in point-data or cell-data

        .. hint:: |mesh_coloring.py|_
        """
        poly = self.polydata(False)

        # no argument: return list of available arrays
        if name_or_idx is None:
            ncd = poly.GetCellData().GetNumberOfArrays()
            npd = poly.GetPointData().GetNumberOfArrays()
            arrs = []
            for i in range(npd):
                #print(i, "PointData", poly.GetPointData().GetArrayName(i))
                arrs.append(["PointData", poly.GetPointData().GetArrayName(i)])
            for i in range(ncd):
                #print(i, "CellData", poly.GetCellData().GetArrayName(i))
                arrs.append(["CellData", poly.GetCellData().GetArrayName(i)])
            return arrs

        else:  # return a specific array (and set it as active one)

            pdata = poly.GetPointData()
            arr = None

            if 'point' in datatype.lower():
                if isinstance(name_or_idx, int):
                    name = pdata.GetArrayName(name_or_idx)
                else:
                    name = name_or_idx
                if name:
                    arr = pdata.GetArray(name)
                    data = pdata
                    self._mapper.SetScalarModeToUsePointData()


            if not arr or 'cell' in datatype.lower():
                cdata = poly.GetCellData()
                if isinstance(name_or_idx, int):
                    name = cdata.GetArrayName(name_or_idx)
                else:
                    name = name_or_idx
                if name:
                    arr = cdata.GetArray(name)
                    data = cdata
                    self._mapper.SetScalarModeToUseCellData()

            if arr:
                data.SetActiveScalars(name)
                self._mapper.ScalarVisibilityOn()
                if settings.autoResetScalarRange:
                    self._mapper.SetScalarRange(arr.GetRange())
                return vtk_to_numpy(arr)

            return None


    def subdivide(self, N=1, method=0):
        """Increase the number of vertices of a surface mesh.

        :param int N: number of subdivisions.
        :param int method: Loop(0), Linear(1), Adaptive(2), Butterfly(3)

        .. hint:: |tutorial_subdivide| |tutorial.py|_
        """
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputData(self.polydata())
        triangles.Update()
        originalMesh = triangles.GetOutput()
        if method == 0:
            sdf = vtk.vtkLoopSubdivisionFilter()
        elif method == 1:
            sdf = vtk.vtkLinearSubdivisionFilter()
        elif method == 2:
            sdf = vtk.vtkAdaptiveSubdivisionFilter()
        elif method == 3:
            sdf = vtk.vtkButterflySubdivisionFilter()
        else:
            colors.printc("~times Error in subdivide: unknown method.", c="r")
            raise RuntimeError()
        if method != 2:
            sdf.SetNumberOfSubdivisions(N)
        sdf.SetInputData(originalMesh)
        sdf.Update()
        return self.updateMesh(sdf.GetOutput())

    def decimate(self, fraction=0.5, N=None, method='quadric', boundaries=False):
        """
        Downsample the number of vertices in a mesh to `fraction`.

        :param float fraction: the desired target of reduction.
        :param int N: the desired number of final points
            (**fraction** is recalculated based on it).
        :param str method: can be either 'quadric' or 'pro'. In the first case triagulation
            will look like more regular, irrespective of the mesh origianl curvature.
            In the second case triangles are more irregular but mesh is more precise on more
            curved regions.
        :param bool boundaries: (True), in `pro` mode decide whether
            to leave boundaries untouched or not.

        .. note:: Setting ``fraction=0.1`` leaves 10% of the original nr of vertices.

        |skeletonize| |skeletonize.py|_
        """
        poly = self.polydata(True)
        if N:  # N = desired number of points
            Np = poly.GetNumberOfPoints()
            fraction = float(N) / Np
            if fraction >= 1:
                return self

        if 'quad' in method:
            decimate = vtk.vtkQuadricDecimation()
            decimate.SetAttributeErrorMetric(True)
            if self.GetTexture():
                decimate.TCoordsAttributeOn()
            else:
                decimate.SetVolumePreservation(True)
        else:
            decimate = vtk.vtkDecimatePro()
            decimate.PreserveTopologyOn()
            if boundaries:
                decimate.BoundaryVertexDeletionOff()
            else:
                decimate.BoundaryVertexDeletionOn()
        decimate.SetInputData(poly)
        decimate.SetTargetReduction(1 - fraction)
        decimate.Update()
        return self.updateMesh(decimate.GetOutput())

    def addGaussNoise(self, sigma):
        """
        Add gaussian noise.

        :param float sigma: sigma is expressed in percent of the diagonal size of actor.

        :Example:
            .. code-block:: python

                from vtkplotter import Sphere

                Sphere().addGaussNoise(1.0).show()
        """
        sz = self.diagonalSize()
        pts = self.coordinates()
        n = len(pts)
        ns = np.random.randn(n, 3) * sigma * sz / 100
        vpts = vtk.vtkPoints()
        vpts.SetNumberOfPoints(n)
        vpts.SetData(numpy_to_vtk(pts + ns, deep=True))
        self._polydata.SetPoints(vpts)
        self._polydata.GetPoints().Modified()
        self.addPointVectors(-ns, 'GaussNoise')
        return self

    def smoothLaplacian(self, niter=15, relaxfact=0.1, edgeAngle=15, featureAngle=60):
        """
        Adjust mesh point positions using `Laplacian` smoothing.

        :param int niter: number of iterations.
        :param float relaxfact: relaxation factor.
            Small `relaxfact` and large `niter` are more stable.
        :param float edgeAngle: edge angle to control smoothing along edges
            (either interior or boundary).
        :param float featureAngle: specifies the feature angle for sharp edge identification.

        .. hint:: |mesh_smoothers.py|_
        """
        poly = self._polydata
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(poly)
        cl.Update()
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(cl.GetOutput())
        smoothFilter.SetNumberOfIterations(niter)
        smoothFilter.SetRelaxationFactor(relaxfact)
        smoothFilter.SetEdgeAngle(edgeAngle)
        smoothFilter.SetFeatureAngle(featureAngle)
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.FeatureEdgeSmoothingOn()
        smoothFilter.GenerateErrorScalarsOn()
        smoothFilter.Update()
        return self.updateMesh(smoothFilter.GetOutput())

    def smoothWSinc(self, niter=15, passBand=0.1, edgeAngle=15, featureAngle=60):
        """
        Adjust mesh point positions using the `Windowed Sinc` function interpolation kernel.

        :param int niter: number of iterations.
        :param float passBand: set the passband value for the windowed sinc filter.
        :param float edgeAngle: edge angle to control smoothing along edges
             (either interior or boundary).
        :param float featureAngle: specifies the feature angle for sharp edge identification.

        |mesh_smoothers| |mesh_smoothers.py|_
        """
        poly = self._polydata
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(poly)
        cl.Update()
        smoothFilter = vtk.vtkWindowedSincPolyDataFilter()
        smoothFilter.SetInputData(cl.GetOutput())
        smoothFilter.SetNumberOfIterations(niter)
        smoothFilter.SetEdgeAngle(edgeAngle)
        smoothFilter.SetFeatureAngle(featureAngle)
        smoothFilter.SetPassBand(passBand)
        smoothFilter.NormalizeCoordinatesOn()
        smoothFilter.NonManifoldSmoothingOn()
        smoothFilter.FeatureEdgeSmoothingOn()
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.Update()
        return self.updateMesh(smoothFilter.GetOutput())

    def fillHoles(self, size=None):
        """Identifies and fills holes in input mesh.
        Holes are identified by locating boundary edges, linking them together into loops,
        and then triangulating the resulting loops.

        :param float size: approximate limit to the size of the hole that can be filled.

        Example: |fillholes.py|_
        """
        fh = vtk.vtkFillHolesFilter()
        if not size:
            mb = self.maxBoundSize()
            size = mb / 10
        fh.SetHoleSize(size)
        fh.SetInputData(self._polydata)
        fh.Update()
        return self.updateMesh(fh.GetOutput())

    def write(self, filename="mesh.vtk", binary=True):
        """
        Write actor's polydata in its current transformation to file.


        """
        import vtkplotter.vtkio as vtkio

        return vtkio.write(self, filename, binary)

    ########################################################################
    ### stuff that is not returning the input (sometimes modified) actor ###

    def normalAt(self, i):
        """Return the normal vector at vertex point `i`."""
        normals = self.polydata(True).GetPointData().GetNormals()
        return np.array(normals.GetTuple(i))

    def normals(self, cells=False):
        """Retrieve vertex normals as a numpy array.

        :params bool cells: if `True` return cell normals.
        """
        if cells:
            vtknormals = self.polydata().GetCellData().GetNormals()
        else:
            vtknormals = self.polydata().GetPointData().GetNormals()
        return vtk_to_numpy(vtknormals)

    def polydata(self, transformed=True):
        """
        Returns the ``vtkPolyData`` object of an ``Actor``.

        .. note:: If ``transformed=True`` returns a copy of polydata that corresponds
            to the current actor's position in space.
        """
        if not transformed:
            if not self._polydata:
                self._polydata = self._mapper.GetInput()
            return self._polydata
        else:
            M = self.GetMatrix()
            if utils.isIdentity(M):
                # if identity return the original polydata
                if not self._polydata:
                    self._polydata = self._mapper.GetInput()
                return self._polydata
            else:
                # otherwise make a copy that corresponds to
                # the actual position in space of the actor
                transform = vtk.vtkTransform()
                transform.SetMatrix(M)
                tp = vtk.vtkTransformPolyDataFilter()
                tp.SetTransform(transform)
                tp.SetInputData(self._polydata)
                tp.Update()
                return tp.GetOutput()

    def isInside(self, point, tol=0.0001):
        """
        Return True if point is inside a polydata closed surface.
        """
        poly = self.polydata(True)
        points = vtk.vtkPoints()
        points.InsertNextPoint(point)
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(points)
        sep = vtk.vtkSelectEnclosedPoints()
        sep.SetTolerance(tol)
        sep.CheckSurfaceOff()
        sep.SetInputData(pointsPolydata)
        sep.SetSurfaceData(poly)
        sep.Update()
        return sep.IsInside(0)

    def insidePoints(self, points, invert=False, tol=1e-05):
        """
        Return the sublist of points that are inside a polydata closed surface.

        |pca| |pca.py|_
        """
        poly = self.polydata(True)
        # check if the stl file is closed

        #featureEdge = vtk.vtkFeatureEdges()
        # featureEdge.FeatureEdgesOff()
        # featureEdge.BoundaryEdgesOn()
        # featureEdge.NonManifoldEdgesOn()
        # featureEdge.SetInputData(poly)
        # featureEdge.Update()
        #openEdges = featureEdge.GetOutput().GetNumberOfCells()
        # if openEdges != 0:
        #    colors.printc("~lightning Warning: polydata is not a closed surface", c=5)

        vpoints = vtk.vtkPoints()
        vpoints.SetData(numpy_to_vtk(points, deep=True))
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(vpoints)
        sep = vtk.vtkSelectEnclosedPoints()
        sep.SetTolerance(tol)
        sep.SetInputData(pointsPolydata)
        sep.SetSurfaceData(poly)
        sep.Update()

        mask1, mask2 = [], []
        for i, p in enumerate(points):
            if sep.IsInside(i):
                mask1.append(p)
            else:
                mask2.append(p)
        if invert:
            return mask2
        else:
            return mask1

    def cellCenters(self):
        """Get the list of cell centers of the mesh surface.

        |delaunay2d| |delaunay2d.py|_
        """
        vcen = vtk.vtkCellCenters()
        vcen.SetInputData(self.polydata(True))
        vcen.Update()
        return vtk_to_numpy(vcen.GetOutput().GetPoints().GetData())

    def boundaries(self, boundaryEdges=True, featureAngle=65, nonManifoldEdges=True):
        """
        Return an ``Actor`` that shows the boundary lines of an input mesh.

        :param bool boundaryEdges: Turn on/off the extraction of boundary edges.
        :param float featureAngle: Specify the feature angle for extracting feature edges.
        :param bool nonManifoldEdges: Turn on/off the extraction of non-manifold edges.
        """
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(self.polydata())
        fe.SetBoundaryEdges(boundaryEdges)

        if featureAngle:
            fe.FeatureEdgesOn()
            fe.SetFeatureAngle(featureAngle)
        else:
            fe.FeatureEdgesOff()

        fe.SetNonManifoldEdges(nonManifoldEdges)
        fe.ColoringOff()
        fe.Update()
        return Actor(fe.GetOutput(), c="p").lw(5)

    def connectedVertices(self, index, returnIds=False):
        """Find all vertices connected to an input vertex specified by its index.

        :param bool returnIds: return vertex IDs instead of vertex coordinates.

        |connVtx| |connVtx.py|_
        """
        mesh = self.polydata()

        cellIdList = vtk.vtkIdList()
        mesh.GetPointCells(index, cellIdList)

        idxs = []
        for i in range(cellIdList.GetNumberOfIds()):
            pointIdList = vtk.vtkIdList()
            mesh.GetCellPoints(cellIdList.GetId(i), pointIdList)
            for j in range(pointIdList.GetNumberOfIds()):
                idj = pointIdList.GetId(j)
                if idj == index:
                    continue
                if idj in idxs:
                    continue
                idxs.append(idj)

        if returnIds:
            return idxs
        else:
            trgp = []
            for i in idxs:
                p = [0, 0, 0]
                mesh.GetPoints().GetPoint(i, p)
                trgp.append(p)
            return np.array(trgp)

    def connectedCells(self, index, returnIds=False):
        """Find all cellls connected to an input vertex specified by its index."""

        # Find all cells connected to point index
        dpoly = self.polydata()
        cellPointIds = vtk.vtkIdList()
        dpoly.GetPointCells(index, cellPointIds)

        ids = vtk.vtkIdTypeArray()
        ids.SetNumberOfComponents(1)
        rids = []
        for k in range(cellPointIds.GetNumberOfIds()):
            cid = cellPointIds.GetId(k)
            ids.InsertNextValue(cid)
            rids.append(int(cid))
        if returnIds:
            return rids

        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(ids)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, dpoly)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()
        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(extractSelection.GetOutput())
        gf.Update()
        return Actor(gf.GetOutput()).lw(1)

    def intersectWithLine(self, p0, p1):
        """Return the list of points intersecting the actor
        along the segment defined by two points `p0` and `p1`.

        :Example:
            .. code-block:: python

                from vtkplotter import *
                s = Spring(alpha=0.2)
                pts = s.intersectWithLine([0,0,0], [1,0.1,0])
                ln = Line([0,0,0], [1,0.1,0], c='blue')
                ps = Points(pts, r=10, c='r')
                show(s, ln, ps, bg='white')

            |intline|
        """
        if not self.line_locator:
            self.line_locator = vtk.vtkOBBTree()
            self.line_locator.SetDataSet(self.polydata())
            self.line_locator.BuildLocator()

        intersectPoints = vtk.vtkPoints()
        self.line_locator.IntersectWithLine(p0, p1, intersectPoints, None)
        pts = []
        for i in range(intersectPoints.GetNumberOfPoints()):
            intersection = [0, 0, 0]
            intersectPoints.GetPoint(i, intersection)
            pts.append(intersection)
        return pts

    def projectOnPlane(self, direction='z'):
        """
        Project the mesh on one of the Cartesian planes.
        """
        coords = self.coordinates(transformed=True)
        if 'z' == direction:
            coords[:, 2] = self.GetOrigin()[2]
            self.z(self.zbounds()[0])
        elif 'x' == direction:
            coords[:, 0] = self.GetOrigin()[0]
            self.x(self.xbounds()[0])
        elif 'y' == direction:
            coords[:, 1] = self.GetOrigin()[1]
            self.y(self.ybounds()[0])
        else:
            colors.printc("~times Error in projectOnPlane(): unknown direction", direction, c=1)
            raise RuntimeError()
        self.alpha(0.1).polydata(False).GetPoints().Modified()
        return self

    def silhouette(self, direction=None, borderEdges=True, featureAngle=None):
        """
        Return a new line ``Actor`` which corresponds to the outer `silhouette`
        of the input as seen along a specified `direction`, this can also be
        a ``vtkCamera`` object.

        :param list direction: viewpoint direction vector.
            If *None* this is guessed by looking at the minimum
            of the sides of the bounding box.
        :param bool borderEdges: enable or disable generation of border edges
        :param float borderEdges: minimal angle for sharp edges detection.
            If set to `False` the functionality is disabled.

        |silhouette| |silhouette.py|_
        """
        sil = vtk.vtkPolyDataSilhouette()
        sil.SetInputData(self.polydata())
        if direction is None:
            b = self.GetBounds()
            i = np.argmin([b[1]-b[0], b[3]-b[2], b[5]-b[4]])
            d = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            sil.SetVector(d[i])
            sil.SetDirectionToSpecifiedVector()
        elif isinstance(direction, vtk.vtkCamera):
            sil.SetCamera(direction)
        else:
            sil.SetVector(direction)
            sil.SetDirectionToSpecifiedVector()
        if featureAngle is not None:
            sil.SetEnableFeatureAngle(1)
            sil.SetFeatureAngle(featureAngle)
            if featureAngle == False:
                sil.SetEnableFeatureAngle(0)

        sil.SetBorderEdges(borderEdges)
        sil.Update()
        return Actor(sil.GetOutput()).lw(2).c('k')


    def followCamera(self, cam=None):
        """
        Actor object will follow camera movements and stay locked to it.

        :param vtkCamera cam: if `None` the text will auto-orient itself to the active camera.
            A ``vtkCamera`` object can also be passed.
        """
        if cam is False:
            self.SetCamera(None)
            return self
        if isinstance(cam, vtk.vtkCamera):
            self.SetCamera(cam)
        else:
            if not settings.plotter_instance or not settings.plotter_instance.camera:
                colors.printc("Error in followCamera(): needs an already rendered scene,", c=1)
                colors.printc("                         or passing a vtkCamera object.", c=1)
                return self
            self.SetCamera(settings.plotter_instance.camera)
        return self


    def isolines(self, n=10, vmin=None, vmax=None):
        """
        Return a new ``Actor`` representing the isolines of the active scalars.

        :param int n: number of isolines in the range
        :param float vmin: minimum of the range
        :param float vmax: maximum of the range

        |isolines| |isolines.py|_
        """
        bcf = vtk.vtkBandedPolyDataContourFilter()
        bcf.SetInputData(self.polydata())
        bcf.SetScalarModeToValue()
        bcf.GenerateContourEdgesOn()
        r0, r1 = self._polydata.GetScalarRange()
        if vmin is None:
            vmin = r0
        if vmax is None:
            vmax = r1
        bcf.GenerateValues(n, vmin, vmax)
        bcf.Update()
        zpoly = bcf.GetContourEdgesOutput()
        zbandsact = Actor(zpoly, c="k")
        zbandsact.GetProperty().SetLineWidth(1.5)
        return zbandsact


#################################################
class Assembly(vtk.vtkAssembly, Prop):
    """Group many actors as a single new actor as a ``vtkAssembly``.

    |gyroscope1| |gyroscope1.py|_
    """

    def __init__(self, actors):

        vtk.vtkAssembly.__init__(self)
        Prop.__init__(self)

        self.actors = actors

        if len(actors) and hasattr(actors[0], "base"):
            self.base = actors[0].base
            self.top = actors[0].top
        else:
            self.base = None
            self.top = None

        for a in actors:
            if a:
                self.AddPart(a)

    def __add__(self, actors):
        if isinstance(actors, list):
            for a in actors:
                self.AddPart(self)
        elif isinstance(actors, vtk.vtkAssembly):
            acts = actors.getActors()
            for a in acts:
                self.AddPart(a)
        else:  # actors=one actor
            self.AddPart(actors)
        return self

    def getActors(self):
        """Unpack a list of ``vtkActor`` objects from a ``vtkAssembly``."""
        return self.actors

    def getActor(self, i):
        """Get `i-th` ``vtkActor`` object from a ``vtkAssembly``."""
        return self.actors[i]

    def diagonalSize(self):
        """Return the maximum diagonal size of the ``Actors`` of the ``Assembly``."""
        szs = [a.diagonalSize() for a in self.actors]
        return np.max(szs)

    def lighting(self, style='', ambient=None, diffuse=None,
                 specular=None, specularPower=None, specularColor=None, enabled=True):
        """Set the lighting type to all ``Actor`` in the ``Assembly`` object.

        :param str style: preset style, can be `[metallic, plastic, shiny, glossy]`
        :param float ambient: ambient fraction of emission [0-1]
        :param float diffuse: emission of diffused light in fraction [0-1]
        :param float specular: fraction of reflected light [0-1]
        :param float specularPower: precision of reflection [1-100]
        :param color specularColor: color that is being reflected by the surface
        :param bool enabled: enable/disable all surface light emission
        """
        for a in self.actors:
            a.lighting(style, ambient, diffuse,
                       specular, specularPower, specularColor, enabled)
        return self

# this would need a callback to work..
# https://vtk.org/Wiki/VTK/Examples/Cxx/Visualization/MovableAxes
#    def followCamera(self, cam=None):
#        for a in self.actors:
#            a.followCamera(cam)
#        return self

#################################################
class Picture(vtk.vtkImageActor, Prop):
    """
    Derived class of ``vtkImageActor``. Used to represent 2D pictures.
    Can be instantiated with a path file name or with a numpy array.

    |rotateImage| |rotateImage.py|_
    """

    def __init__(self, obj=None):
        vtk.vtkImageActor.__init__(self)
        Prop.__init__(self)

        if utils.isSequence(obj) and len(obj):
            iac = vtk.vtkImageAppendComponents()
            for i in range(3):
                #arr = np.flip(np.flip(array[:,:,i], 0), 0).ravel()
                arr = np.flip(obj[:,:,i], 0).ravel()
                varb = numpy_to_vtk(arr, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
                imgb = vtk.vtkImageData()
                imgb.SetDimensions(obj.shape[1], obj.shape[0], 1)
                imgb.GetPointData().SetScalars(varb)
                iac.AddInputData(0, imgb)
            iac.Update()
            self.SetInputData(iac.GetOutput())
            #self.mirror()

        elif isinstance(obj, str):
            if ".png" in obj:
                picr = vtk.vtkPNGReader()
            elif ".jpg" in obj or ".jpeg" in obj:
                picr = vtk.vtkJPEGReader()
            elif ".bmp" in obj:
                picr = vtk.vtkBMPReader()
            picr.SetFileName(obj)
            picr.Update()
            self.SetInputData(picr.GetOutput())


    def alpha(self, a=None):
        """Set/get picture's transparency."""
        if a is not None:
            self.GetProperty().SetOpacity(a)
            return self
        else:
            return self.GetProperty().GetOpacity()

    def crop(self, top=None, bottom=None, right=None, left=None):
        """Crop picture.

        :param float top: fraction to crop from the top margin
        :param float bottom: fraction to crop from the bottom margin
        :param float left: fraction to crop from the left margin
        :param float right: fraction to crop from the right margin
        """
        extractVOI = vtk.vtkExtractVOI()
        extractVOI.SetInputData(self.GetInput())
        extractVOI.IncludeBoundaryOn()

        d = self.GetInput().GetDimensions()
        bx0, bx1, by0, by1 = 0, d[0]-1, 0, d[1]-1
        if left is not None:   bx0 = int((d[0]-1)*left)
        if right is not None:  bx1 = int((d[0]-1)*(1-right))
        if bottom is not None: by0 = int((d[1]-1)*bottom)
        if top is not None:    by1 = int((d[1]-1)*(1-top))
        extractVOI.SetVOI(bx0, bx1, by0, by1, 0, 0)
        extractVOI.Update()
        img = extractVOI.GetOutput()
        self.GetMapper().SetInputData(img)
        self.GetMapper().Modified()
        return self

    def mirror(self, axis="x"):
        """Mirror picture along x or y axis."""
        ff = vtk.vtkImageFlip()
        ff.SetInputData(self.inputdata())
        if axis.lower() == "x":
            ff.SetFilteredAxis(0)
        elif axis.lower() == "y":
            ff.SetFilteredAxis(1)
        else:
            colors.printc("~times Error in mirror(): mirror must be set to x or y.", c=1)
            raise RuntimeError()
        ff.Update()
        self.GetMapper().SetInputData(ff.GetOutput())
        self.GetMapper().Modified()
        return self


##########################################################################
class Volume(vtk.vtkVolume, Prop):
    """Derived class of ``vtkVolume``.
    Can be initialized with a numpy object, see e.g.: |numpy2volume.py|_

    :param c: sets colors along the scalar range, or a matplotlib color map name
    :type c: list, str
    :param alphas: sets transparencies along the scalar range
    :type c: float, list
    :param list origin: set volume origin coordinates
    :param list spacing: voxel dimensions in x, y and z.
    :param list shape: specify the shape.
    :param str mapperType: either 'gpu', 'opengl_gpu', 'fixed' or 'smart'

    :param int mode: define the volumetric rendering style:

        - 0, Composite rendering
        - 1, maximum projection rendering
        - 2, minimum projection
        - 3, average projection
        - 4, additive mode

    .. hint:: if a `list` of values is used for `alphas` this is interpreted
        as a transfer function along the range of the scalar.

        |read_vti| |read_vti.py|_
    """

    def __init__(self, inputobj,
                 c=('b','lb','lg','y','r'),
                 alpha=(0.0, 0.0, 0.2, 0.4, 0.8, 1),
                 alphaGradient=None,
                 mode=0,
                 origin=None,
                 spacing=None,
                 shape=None,
                 mapperType='gpu',
                 ):

        vtk.vtkVolume.__init__(self)
        Prop.__init__(self)

        inputtype = str(type(inputobj))
        #colors.printc('Volume inputtype', inputtype)

        if inputobj is None:
            img = vtk.vtkImageData()
        elif utils.isSequence(inputobj):
            if "ndarray" not in inputtype:
                inputobj = np.array(inputobj)

            varr = numpy_to_vtk(inputobj.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
            varr.SetName('input_scalars')

            img = vtk.vtkImageData()
            if shape is not None:
                img.SetDimensions(shape)
            else:
                img.SetDimensions(inputobj.shape)
            img.GetPointData().SetScalars(varr)

        elif "ImageData" in inputtype:
            img = inputobj
        elif "UniformGrid" in inputtype:
            img = inputobj
        elif "UnstructuredGrid" in inputtype:
            img = inputobj
            mapperType = 'tetra'
        elif hasattr(inputobj, "GetOutput"): # passing vtk object, try extract imagdedata
            if hasattr(inputobj, "Update"):
                inputobj.Update()
            img = inputobj.GetOutput()
        else:
            colors.printc("Volume(): cannot understand input type:\n", inputtype, c=1)
            return

        if 'gpu' in mapperType:
            self._mapper = vtk.vtkGPUVolumeRayCastMapper()
        elif 'opengl_gpu' in mapperType:
            self._mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
        elif 'smart' in mapperType:
            self._mapper = vtk.vtkSmartVolumeMapper()
        elif 'fixed' in mapperType:
            self._mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        elif 'tetra' in mapperType:
            self._mapper = vtk.vtkProjectedTetrahedraMapper()
        elif 'unstr' in mapperType:
            self._mapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()

        if origin is not None:
            img.SetOrigin(origin)
        if spacing is not None:
            img.SetSpacing(spacing)
        if shape is not None:
            img.SetDimensions(shape)

        self._image = img
        self._mapper.SetInputData(img)
        self.SetMapper(self._mapper)
        self.mode(mode).color(c).alpha(alpha).alphaGradient(alphaGradient)
        # remember stuff:
        self._mode = mode
        self._color = c
        self._alpha = alpha
        self._alphaGrad = alphaGradient

    def updateVolume(self, img):
        """
        Overwrite the polygonal mesh of the actor with a new one.
        """
        self._image = img
        self._mapper.SetInputData(img)
        self._mapper.Modified()
        return self

    def mode(self, mode=None):
        """Define the volumetric rendering style.

            - 0, Composite rendering
            - 1, maximum projection rendering
            - 2, minimum projection
            - 3, average projection
            - 4, additive mode
        """
        if mode is None:
            return self._mapper.GetBlendMode()

        volumeProperty = self.GetProperty()
        self._mapper.SetBlendMode(mode)
        self._mode = mode
        if mode == 0:
            volumeProperty.ShadeOn()
            self.lighting('shiny')
            self.jittering(True)
        elif mode == 1:
            volumeProperty.ShadeOff()
            self.jittering(True)
        return self

    def jittering(self, status=None):
        """If `jittering` is `True`, each ray traversal direction will be perturbed slightly
        using a noise-texture to get rid of wood-grain effects.
        """
        if hasattr(self._mapper, 'SetUseJittering'):
            if status is None:
                return self._mapper.GetUseJittering()
            self._mapper.SetUseJittering(status)
            return self
        return None

    def imagedata(self):
        """Return the underlying ``vtkImagaData`` object."""
        return self._image

    def dimensions(self):
        """Return the nr. of voxels in the 3 dimensions."""
        return self._image.GetDimensions()

    def spacing(self, s=None):
        """Set/get the voxels size in the 3 dimensions."""
        if s is not None:
            self._image.SetSpacing(s)
            self._mapper.Modified()
            return self
        else:
            return np.array(self._image.GetSpacing())

    def permuteAxes(self, x, y ,z):
        """Reorder the axes of the Volume by specifying
        the input axes which are supposed to become the new X, Y, and Z."""
        imp = vtk.vtkImagePermute()
        imp.SetFilteredAxes(x,y,z)
        imp. SetInputData(self.imagedata())
        imp.Update()
        return self.updateVolume(imp.GetOutput())

    def resample(self, newSpacing, interpolation=1):
        """
        Resamples a ``Volume`` to be larger or smaller.

        This method modifies the spacing of the input.
        Linear interpolation is used to resample the data.

        :param list newSpacing: a list of 3 new spacings for the 3 axes.
        :param int interpolation: 0=nearest_neighbor, 1=linear, 2=cubic
        """
        rsp = vtk.vtkImageResample()
        oldsp = self.GetSpacing()
        for i in range(3):
            if oldsp[i] != newSpacing[i]:
                rsp.SetAxisOutputSpacing(i, newSpacing[i])
        rsp.InterpolateOn()
        rsp.SetInterpolationMode(interpolation)
        rsp.OptimizationOn()
        rsp.Update()
        return self.updateVolume(rsp.GetOutput())


    def color(self, col):
        """Assign a color or a set of colors to a volume along the range of the scalar value.
        A single constant color can also be assigned.
        Any matplotlib color map name is also accepted, e.g. ``volume.color('jet')``.

        E.g.: say that your voxel scalar runs from -3 to 6,
        and you want -3 to show red and 1.5 violet and 6 green, then just set:

        ``volume.color(['red', 'violet', 'green'])``
        """
        smin, smax = self._image.GetScalarRange()
        volumeProperty = self.GetProperty()
        ctf = vtk.vtkColorTransferFunction()
        self._color = col

        if utils.isSequence(col):
            for i, ci in enumerate(col):
                r, g, b = colors.getColor(ci)
                xalpha = smin + (smax - smin) * i / (len(col) - 1)
                ctf.AddRGBPoint(xalpha, r, g, b)
                #colors.printc('\tcolor at', round(xalpha, 1),
                #              '\tset to', colors.getColorName((r, g, b)), c='b', bold=0)
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
            colors.printc("volume.color(): unknown input type:", col, c=1)

        volumeProperty.SetColor(ctf)
        volumeProperty.SetInterpolationTypeToLinear()
        #volumeProperty.SetInterpolationTypeToNearest()
        return self

    def alpha(self, alpha):
        """Assign a set of tranparencies to a volume along the range of the scalar value.
        A single constant value can also be assigned.

        E.g.: say alpha=(0.0, 0.3, 0.9, 1) and the scalar range goes from -10 to 150.
        Then all voxels with a value close to -10 will be completely transparent, voxels at 1/4
        of the range will get an alpha equal to 0.3 and voxels with value close to 150
        will be completely opaque.
        """
        volumeProperty = self.GetProperty()
        smin, smax = self._image.GetScalarRange()
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        self._alpha = alpha

        if utils.isSequence(alpha):
            for i, al in enumerate(alpha):
                xalpha = smin + (smax - smin) * i / (len(alpha) - 1)
                # Create transfer mapping scalar value to opacity
                opacityTransferFunction.AddPoint(xalpha, al)
                #colors.printc("alpha at", round(xalpha, 1), "\tset to", al, c="b", bold=0)
        else:
            opacityTransferFunction.AddPoint(smin, alpha) # constant alpha
            opacityTransferFunction.AddPoint(smax, alpha)

        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.SetInterpolationTypeToLinear()
        return self

    def alphaGradient(self, alphaGrad):
        """
        Assign a set of tranparencies to a volume's gradient
        along the range of the scalar value.
        A single constant value can also be assigned.
        The gradient function is used to decrease the opacity
        in the "flat" regions of the volume while maintaining the opacity
        at the boundaries between material types.  The gradient is measured
        as the amount by which the intensity changes over unit distance.

        |read_vti| |read_vti.py|_
        """
        self._alphaGrad = alphaGrad
        volumeProperty = self.GetProperty()
        if alphaGrad is None:
            volumeProperty.DisableGradientOpacityOn()
            return self
        else:
            volumeProperty.DisableGradientOpacityOff()

        #smin, smax = self._image.GetScalarRange()
        smin, smax = 0, 255
        gotf = vtk.vtkPiecewiseFunction()
        if utils.isSequence(alphaGrad):
            for i, al in enumerate(alphaGrad):
                xalpha = smin + (smax - smin) * i / (len(alphaGrad) - 1)
                # Create transfer mapping scalar value to gradient opacity
                gotf.AddPoint(xalpha, al)
                #colors.printc("alphaGrad at", round(xalpha, 1), "\tset to", al, c="b", bold=0)
        else:
            gotf.AddPoint(smin, alphaGrad) # constant alphaGrad
            gotf.AddPoint(smax, alphaGrad)

        volumeProperty.SetGradientOpacity(gotf)
        volumeProperty.SetInterpolationTypeToLinear()
        return self

    def threshold(self, vmin=None, vmax=None, replaceWith=0):
        """
        Binary or continuous volume thresholding.
        Find the voxels that contain the value below/above or inbetween
        [vmin, vmax] and replaces it with the provided value (default is 0).
        """
        th = vtk.vtkImageThreshold()
        th.SetInputData(self.imagedata())

        if vmin is not None and vmax is not None:
            th.ThresholdBetween(vmin, vmax)
        elif vmin is not None:
            th.ThresholdByLower(vmin)
        elif vmax is not None:
            th.ThresholdByUpper(vmax)

        th.SetInValue(replaceWith)
        th.Update()
        return self.updateVolume(th.GetOutput())

    def crop(self,
             top=None, bottom=None,
             right=None, left=None,
             front=None, back=None, VOI=()):
        """Crop a ``Volume`` object.

        :param float top:    fraction to crop from the top plane (positive z)
        :param float bottom: fraction to crop from the bottom plane (negative z)
        :param float front:  fraction to crop from the front plane (positive y)
        :param float back:   fraction to crop from the back plane (negative y)
        :param float right:  fraction to crop from the right plane (positive x)
        :param float left:   fraction to crop from the left plane (negative x)
        :param list VOI:     extract Volume Of Interest expressed in voxel numbers

            Eg.: vol.crop(VOI=(xmin, xmax, ymin, ymax, zmin, zmax)) # all integers nrs
        """
        extractVOI = vtk.vtkExtractVOI()
        extractVOI.SetInputData(self.imagedata())

        if len(VOI):
            extractVOI.SetVOI(VOI)
        else:
            d = self.imagedata().GetDimensions()
            bx0, bx1, by0, by1, bz0, bz1 = 0, d[0]-1, 0, d[1]-1, 0, d[2]-1
            if left is not None:   bx0 = int((d[0]-1)*left)
            if right is not None:  bx1 = int((d[0]-1)*(1-right))
            if back is not None:   by0 = int((d[1]-1)*back)
            if front is not None:  by1 = int((d[1]-1)*(1-front))
            if bottom is not None: bz0 = int((d[2]-1)*bottom)
            if top is not None:    bz1 = int((d[2]-1)*(1-top))
            extractVOI.SetVOI(bx0, bx1, by0, by1, bz0, bz1)
        extractVOI.Update()
        return self.updateVolume(extractVOI.GetOutput())

    def cutWithPlane(self, origin=(0,0,0), normal=(1,0,0)):
        """
        Cuts ``Volume`` with the plane defined by a point and a normal
        creating a tetrahedral mesh object.
        Makes sense only if the plane is not along any of the cartesian planes,
        otherwise use ``crop()`` which is way faster.

        :param origin: the cutting plane goes through this point
        :param normal: normal of the cutting plane
        """
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        clipper = vtk.vtkClipVolume()
        clipper.SetInputData(self._image)
        clipper.SetClipFunction(plane)
        clipper.GenerateClipScalarsOff()
        clipper.GenerateClippedOutputOff()
        clipper.Mixed3DCellGenerationOff() # generate only tets
        clipper.SetValue(0)
        clipper.Update()

        vol = Volume(clipper.GetOutput()).color(self._color)
        return vol #self.updateVolume(clipper.GetOutput())


    def resize(self, *newdims):
        """Increase or reduce the number of voxels of a Volume with interpolation."""
        old_dims = np.array(self.imagedata().GetDimensions())
        old_spac = np.array(self.imagedata().GetSpacing())
        rsz = vtk.vtkImageResize()
        rsz.SetResizeMethodToOutputDimensions()
        rsz.SetInputData(self.imagedata())
        rsz.SetOutputDimensions(newdims)
        rsz.Update()
        self._image = rsz.GetOutput()
        new_spac = old_spac * old_dims/newdims  # keep aspect ratio
        self._image.SetSpacing(new_spac)
        return self.updateVolume(self._image)

    def normalize(self):
        """Normalize that scalar components for each point."""
        norm = vtk.vtkImageNormalize()
        norm.SetInputData(self.imagedata())
        norm.Update()
        return self.updateVolume(norm.GetOutput())

    def scaleVoxels(self, scale=1):
        """Scale the voxel content by factor `scale`."""
        rsl = vtk.vtkImageReslice()
        rsl.SetInputData(self.imagedata())
        rsl.SetScalarScale(scale)
        rsl.Update()
        return self.updateVolume(rsl.GetOutput())

    def mirror(self, axis="x"):
        """
        Mirror flip along one of the cartesian axes.

        .. note::  ``axis='n'``, will flip only mesh normals.

        |mirror| |mirror.py|_
        """
        img = self.imagedata()

        ff = vtk.vtkImageFlip()
        ff.SetInputData(img)
        if axis.lower() == "x":
            ff.SetFilteredAxis(0)
        elif axis.lower() == "y":
            ff.SetFilteredAxis(1)
        elif axis.lower() == "z":
            ff.SetFilteredAxis(2)
        else:
            colors.printc("~times Error in mirror(): mirror must be set to x, y, z or n.", c=1)
            raise RuntimeError()
        ff.Update()
        return self.updateVolume(ff.GetOutput())

    def xSlice(self, i):
        """Extract the slice at index `i` of volume along x-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.imagedata())
        nx, ny, nz = self.imagedata().GetDimensions()
        if i>nx-1:
            i=nx-1
        vslice.SetExtent(i,i, 0,ny, 0,nz)
        vslice.Update()
        return Actor(vslice.GetOutput())

    def ySlice(self, j):
        """Extract the slice at index `j` of volume along y-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.imagedata())
        nx, ny, nz = self.imagedata().GetDimensions()
        if j>ny-1:
            j=ny-1
        vslice.SetExtent(0,nx, j,j, 0,nz)
        vslice.Update()
        return Actor(vslice.GetOutput())

    def zSlice(self, k):
        """Extract the slice at index `i` of volume along z-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.imagedata())
        nx, ny, nz = self.imagedata().GetDimensions()
        if k>nz-1:
            k=nz-1
        vslice.SetExtent(0,nx, 0,ny, k,k)
        vslice.Update()
        return Actor(vslice.GetOutput())


    def isosurface(self, threshold=True, connectivity=False):
        """Return an ``Actor`` isosurface extracted from the ``Volume`` object.

        :param threshold: value or list of values to draw the isosurface(s)
        :type threshold: float, list
        :param bool connectivity: if True only keeps the largest portion of the polydata

        |isosurfaces| |isosurfaces.py|_
        """
        scrange = self._image.GetScalarRange()
        cf = vtk.vtkContourFilter()
        cf.SetInputData(self._image)
        cf.UseScalarTreeOn()
        cf.ComputeScalarsOn()
        cf.ComputeNormalsOn()

        if utils.isSequence(threshold):
            cf.SetNumberOfContours(len(threshold))
            for i, t in enumerate(threshold):
                cf.SetValue(i, t)
            cf.Update()
        else:
            if threshold is True:
                threshold = (2 * scrange[0] + scrange[1]) / 3.0
                print('automatic threshold set to ' + utils.precision(threshold, 3), end=' ')
                print('in [' + utils.precision(scrange[0], 3) + ', ' + utils.precision(scrange[1], 3)+']')
            cf.SetValue(0, threshold)
            cf.Update()

        clp = vtk.vtkCleanPolyData()
        clp.SetInputConnection(cf.GetOutputPort())
        clp.Update()
        poly = clp.GetOutput()

        if connectivity:
            conn = vtk.vtkPolyDataConnectivityFilter()
            conn.SetExtractionModeToLargestRegion()
            conn.SetInputData(poly)
            conn.Update()
            poly = conn.GetOutput()

        a = Actor(poly, c=None).phong()
        a._mapper.SetScalarRange(scrange[0], scrange[1])
        return a


    def legosurface(self, vmin=None, vmax=None, cmap='afmhot_r'):
        """
        Represent a ``Volume`` as lego blocks (voxels).
        By default colors correspond to the volume's scalar.
        Returns an ``Actor``.

        :param float vmin: the lower threshold, voxels below this value are not shown.
        :param float vmax: the upper threshold, voxels above this value are not shown.
        :param str cmap: color mapping of the scalar associated to the voxels.

        |legosurface| |legosurface.py|_
        """
        dataset = vtk.vtkImplicitDataSet()
        dataset.SetDataSet(self._image)
        window = vtk.vtkImplicitWindowFunction()
        window.SetImplicitFunction(dataset)

        srng = list(self._image.GetScalarRange())
        if vmin is not None:
            srng[0] = vmin
        if vmax is not None:
            srng[1] = vmax
        window.SetWindowRange(srng)

        extract = vtk.vtkExtractGeometry()
        extract.SetInputData(self._image)
        extract.SetImplicitFunction(window)
        extract.ExtractInsideOff()
        extract.ExtractBoundaryCellsOff()
        extract.Update()

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(extract.GetOutput())
        gf.Update()

        a = Actor(gf.GetOutput()).lw(0.1).flat()

        scalars = np.array(a.scalars(0), dtype=np.float)

        if cmap:
            a.pointColors(scalars, vmin=self._image.GetScalarRange()[0], cmap=cmap)
            a.mapPointsToCells()
        return a




