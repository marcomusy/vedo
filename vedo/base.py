import numpy as np
import vtk
import vedo
import vedo.colors as colors
import vedo.utils as utils
from deprecated import deprecated

__doc__ = (
    """Base classes. Do not instantiate these."""
    + vedo.docs._defs
)

__all__ = [
           'Base3DProp',
           'BaseActor',
           'BaseGrid',
           "probePoints",
           "probeLine",
           "probePlane",
           "streamLines",
]


###############################################################################
class _DataArrayHelper(object):
    """Helper class to manage data associated to either
    points (or vertices) and cells (or faces). Internal use only."""
    def __init__(self, actor, association):
        self.actor = actor
        self.association = association

    def __getitem__(self, key):
        if self.association == 0:
            data = self.actor._data.GetPointData()
        else:
            data = self.actor._data.GetCellData()

        if isinstance(key, int):
            key = data.GetArrayName(key)
        arr = data.GetArray(key)
        if not arr:
            return None
        return utils.vtk2numpy(arr)

    def __setitem__(self, key, input_array):
        if self.association == 0:
            data = self.actor._data.GetPointData()
            n = self.actor._data.GetNumberOfPoints()
            self.actor._mapper.SetScalarModeToUsePointData()
        else:
            data = self.actor._data.GetCellData()
            n = self.actor._data.GetNumberOfCells()
            self.actor._mapper.SetScalarModeToUseCellData()

        if len(input_array) != n:
            colors.printc(f'Error in point/cell data: length of input {len(input_array)}'
                          f' !=  {n} nr. of elements', c='r')
            raise RuntimeError()

        input_array = np.ascontiguousarray(input_array)
        varr = utils.numpy2vtk(input_array, name=key)
        data.AddArray(varr)

        if len(input_array.shape)==1: # scalars
            data.SetActiveScalars(key)
        elif len(input_array.shape)==2 and input_array.shape[1] == 3: # vectors
            if key == "Normals":
                data.SetActiveNormals(key)
            else:
                data.SetActiveVectors(key)

        return #####################

    def keys(self):
        if self.association == 0:
            data = self.actor._data.GetPointData()
        else:
            data = self.actor._data.GetCellData()
        arrnames = []
        for i in range(data.GetNumberOfArrays()):
            arrnames.append(data.GetArray(i).GetName())
        return arrnames

    def remove(self, key):
        if self.association == 0:
            self.actor._data.GetPointData().RemoveArray(key)
        else:
            self.actor._data.GetCellData().RemoveArray(key)

    def rename(self, oldname, newname):
        if self.association == 0:
            varr = self.actor._data.GetPointData().GetArray(oldname)
        else:
            varr = self.actor._data.GetCellData().GetArray(oldname)
        if varr:
            varr.SetName(newname)
        else:
            colors.printc("Cannot rename non existing array:", oldname, "to:", newname, c='r')

    def select(self, key):
        if self.association == 0:
            data = self.actor._data.GetPointData()
            self.actor._mapper.SetScalarModeToUsePointData()
        else:
            data = self.actor._data.GetCellData()
            self.actor._mapper.SetScalarModeToUseCellData()

        if isinstance(key, int):
            key = data.GetArrayName(key)
        data.SetActiveScalars(key)

        if hasattr(self.actor._mapper, 'SetArrayName'):
            self.actor._mapper.SetArrayName(key)

        if hasattr(self.actor._mapper, 'ScalarVisibilityOn'): # could be volume mapper
            self.actor._mapper.ScalarVisibilityOn()

    def print(self, **kwargs):
        colors.printc(self.keys(), **kwargs)


###############################################################################
class Base3DProp(object):

    def __init__(self):
        self.filename = ""
        self.name = ""
        self.fileSize = ''
        self.created = ''
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
        self.renderedAt = set()
        self.transform = None
        self._set2actcam = False # used by mesh.followCamera()


    def address(self):
        """
        Return a unique memory address integer which can serve as the ID of the
        object, or passed to c++ code.
        """
        # https://www.linkedin.com/pulse/speedup-your-code-accessing-python-vtk-objects-from-c-pletzer/
        # https://github.com/tfmoraes/polydata_connectivity
        return int(self.inputdata().GetAddressAsString('')[5:], 16)

    def pickable(self, value=None):
        """Set/get pickable property of an object."""
        if value is None:
            return self.GetPickable()
        else:
            self.SetPickable(value)
            return self

    def draggable(self, value=None): # NOT FUNCTIONAL?
        """Set/get draggable property of an object."""
        if value is None:
            return self.GetDragable()
        else:
            self.SetDragable(value)
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
        if x is None: # get functionality
            return np.array(self.GetPosition())

        if z is None and y is None: # assume x is of the form (x,y,z)
            if len(x)==3:
                x, y, z = x
            else:
                x, y = x
                z=0
        elif z is None:             # assume x,y is of the form x, y
            z=0
        self.SetPosition(x, y, z)

        if self.trail:
            self.updateTrail()
        if self.shadow:
            self._updateShadow()
        return self  # return itself to concatenate methods

    def addPos(self, dx=0, dy=0, dz=0):
        """Add vector to current object position. Same as `shift()`."""
        p = np.array(self.GetPosition())

        if utils.isSequence(dx):
            if len(dx) == 2:
                self.SetPosition(p + [dx[0], dx[1], 0])
            else:
                self.SetPosition(p + dx)
        else:
            self.SetPosition(p + [dx,dy,dz])

        if self.trail:
            self.updateTrail()
        if self.shadow:
            self._updateShadow()
        return self

    def shift(self, dx=0, dy=0, dz=0):
        """Add vector to current object position. Same as `addPos()`."""
        return self.addPos(dx, dy, dz)

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

    def rotate(self, angle, axis=(1, 0, 0), point=(0, 0, 0), rad=False):
        """Rotate around an arbitrary `axis` passing through `point`.

        :Example:

            .. code-block:: python

                from vedo import *
                c1 = Cube()
                c2 = c1.clone().c('violet').alpha(0.5) # copy of c1
                v = vector(0.2,1,0)
                p = vector(1,0,0)  # axis passes through this point
                c2.rotate(90, axis=v, point=p)
                l = Line(-v+p, v+p).lw(3).c('red')
                show(c1, l, c2, axes=1)
        """
        if rad:
            anglerad = angle
        else:
            anglerad = np.deg2rad(angle)
        axis = utils.versor(axis)
        a = np.cos(anglerad / 2)
        b, c, d = -axis * np.sin(anglerad / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = np.array([
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )
        rv = np.dot(R, self.GetPosition() - np.asarray(point)) + point

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

    def rotateX(self, angle, rad=False, locally=False):
        """Rotate around x-axis. If angle is in radians set ``rad=True``.

        :param bool locally: ignore previous transformations. Rotate around object local origin.
        """
        if rad:
            angle *= 180 / np.pi
        if not locally:
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
        else:
            self.RotateX(angle)
        return self

    def rotateY(self, angle, rad=False, locally=False):
        """Rotate around y-axis. If angle is in radians set ``rad=True``.

        :param bool locally: ignore previous transformations. Rotate around object local origin.
        """
        if rad:
            angle *= 180 / np.pi
        if not locally:
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
        else:
            self.RotateY(angle)
        return self

    def rotateZ(self, angle, rad=False, locally=False):
        """Rotate around z-axis. If angle is in radians set ``rad=True``.

        :param bool locally: ignore previous transformations. Rotate around object local origin.
        """
        if rad:
            angle *= 180 / np.pi
        if not locally:
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
        else:
            self.RotateZ(angle)
        return self


    def orientation(self, newaxis=None, rotation=0, rad=False):
        """
        Set/Get object orientation.

        :param rotation: If != 0 rotate object around newaxis.
        :param rad: set to True if angle is in radians.

        :Example:

            .. code-block:: python

                from vedo import *
                objs = []
                for i in range(-5, 5):
                    p = [i/3, i/2, i]
                    v = vector(i/10, i/20, 1)
                    c = Circle(r=i/5+1.2).pos(p).orientation(v).lw(3)
                    objs += [c, Arrow(p,p+v)]
                show(objs, axes=1)

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


    def getTransform(self, invert=False):
        """
        Check if ``object.transform`` exists and returns a ``vtkTransform``.
        Otherwise return current user transformation (where the object is currently placed).

        :param bool invert: return the inverse of the current transformation

        :Example:

            .. code-block:: python

                from vedo import *

                c1 = Cube()
                c2 = c1.clone().c('violet').alpha(0.5) # copy of c1
                v = vector(0.2,1,0)
                p = vector(1,0,0)  # axis passes through this point
                c2.rotate(90, axis=v, point=p)

                # get the inverse of the current transformation
                T = c2.getTransform(invert=True)
                c2.applyTransform(T)  # put back c2 in place

                l = Line(p-v, p+v).lw(3).c('red')
                show(c1.wireframe().lw(3), l, c2, axes=1)
        """
        if self.transform:
            tr = self.transform
        else:
            T = self.GetMatrix()
            tr = vtk.vtkTransform()
            tr.SetMatrix(T)
        if invert:
            tr = tr.GetInverse()
        return tr

    def applyTransform(self, T):
        """
        Transform object position and orientation.
        """
        if isinstance(T, vtk.vtkMatrix4x4):
            self.SetUserMatrix(T)
        elif isinstance(T, (list,tuple)):
            vm = vtk.vtkMatrix4x4()
            for i in [0, 1, 2, 3]:
               for j in [0, 1, 2, 3]:
                   vm.SetElement(i, j, T[i][j])
            self.SetUserMatrix(vm)
        else:
            self.SetUserTransform(T)
        self.transform = T
        return self

    def alignToBoundingBox(self, msh, rigid=False):
        """
        Align the current object's bounding box to the bounding box
        of the input object.

        :param bool rigid: do not allow scaling if set to True

        :Example:

            .. code-block:: python

                from vedo import *
                eli = Ellipsoid().alpha(0.4)
                cube= Cube().pos(3,2,1).rotateX(10).rotateZ(10).alpha(0.4)
                eli.alignToBoundingBox(cube, rigid=False)
                axes1 = Axes(eli, c='db')
                axes2 = Axes(cube, c='dg')
                show(eli, cube, axes1, axes2)
        """
        lmt = vtk.vtkLandmarkTransform()
        ss = vtk.vtkPoints()
        xss0,xss1,yss0,yss1,zss0,zss1 = self.bounds()
        for p in [[xss0, yss0, zss0],
                  [xss1, yss0, zss0],
                  [xss1, yss1, zss0],
                  [xss0, yss1, zss0],
                  [xss0, yss0, zss1],
                  [xss1, yss0, zss1],
                  [xss1, yss1, zss1],
                  [xss0, yss1, zss1],
                 ]:
            ss.InsertNextPoint(p)
        st = vtk.vtkPoints()
        xst0,xst1,yst0,yst1,zst0,zst1 = msh.bounds()
        for p in [[xst0, yst0, zst0],
                  [xst1, yst0, zst0],
                  [xst1, yst1, zst0],
                  [xst0, yst1, zst0],
                  [xst0, yst0, zst1],
                  [xst1, yst0, zst1],
                  [xst1, yst1, zst1],
                  [xst0, yst1, zst1],
                 ]:
            st.InsertNextPoint(p)

        lmt.SetSourceLandmarks(ss)
        lmt.SetTargetLandmarks(st)
        if rigid:
            lmt.SetModeToRigidBody()
        lmt.Update()
        self.applyTransform(lmt)
        self.transform = lmt
        return self


    def on(self):
        """Switch on  object visibility. Object is not removed."""
        self.VisibilityOn()
        return self

    def off(self):
        """Switch off object visibility. Object is not removed."""
        self.VisibilityOff()
        return self

    def box(self, scale=1, pad=0, fill=False):
        """Return the bounding box as a new ``Mesh``.

        :param float scale: box size can be scaled by a factor
        :param float,list pad: a constant pad can be added (can be a list [padx,pady,padz])

        .. hint:: |latex.py|_
        """
        b = self.GetBounds()
        if not utils.isSequence(pad):
            pad=[pad,pad,pad]
        length, width, height = b[1]-b[0], b[3]-b[2], b[5]-b[4]
        tol = (length+width+height)/30000 # useful for boxing 2D text
        pos = [(b[0]+b[1])/2, (b[3]+b[2])/2, (b[5]+b[4])/2 -tol]
        bx = vedo.shapes.Box(pos,
                             length*scale+pad[0],
                             width*scale+pad[1],
                             height*scale+pad[2],
                             c='gray')
        if hasattr(self, 'GetProperty'): # could be Assembly
            if isinstance(self.GetProperty(), vtk.vtkProperty): # could be volume
                pr = vtk.vtkProperty()
                pr.DeepCopy(self.GetProperty())
                bx.SetProperty(pr)
                bx.property = pr
        bx.flat().lighting('off')
        bx.wireframe(not fill)
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

    def xbounds(self, i=None):
        """Get the bounds [xmin,xmax]. Can specify upper or lower with i (0,1)."""
        b = self.GetBounds()
        if i is not None: return b[i]
        return (b[0], b[1])

    def ybounds(self, i=None):
        """Get the bounds [ymin,ymax]. Can specify upper or lower with i (0,1)."""
        b = self.GetBounds()
        if i == 0: return b[2]
        elif i == 1: return b[3]
        return (b[2], b[3])

    def zbounds(self, i=None):
        """Get the bounds [zmin,zmax]. Can specify upper or lower with i (0,1)."""
        b = self.GetBounds()
        if i == 0: return b[4]
        elif i == 1: return b[5]
        return (b[4], b[5])

    def diagonalSize(self):
        """Get the length of the diagonal of mesh bounding box."""
        b = self.GetBounds()
        return np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)
        # return self.GetLength() # ???different???

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
        """Obsolete, use print() instead."""
        return self.print()

    def print(self):
        """Print information about an object."""
        utils.printInfo(self)
        return self

    def buildAxes(self, **kargs):
        """Generate axes for the input object or for a specified range.
        Returns an ``Assembly`` object.

        :Example:

            .. code-block:: python

                from vedo import Box, show
                b = Box(pos=(1,2,3), length=8, width=9, height=7).alpha(0)
                bax = b.buildAxes(c='k')  # returns Assembly object
                show(b, bax)

        |customAxes1| |customAxes1.py|_ |customAxes2.py|_ |customAxes3.py|_

        |customIndividualAxes| |customIndividualAxes.py|_
        """
        a = vedo.addons.Axes(self, **kargs)
        self.axes = a
        return a


    def show(self, **options):
        """
        Create on the fly an instance of class ``Plotter`` or use the last existing one to
        show one single object.

        This method is meant as a shortcut. If more than one object needs to be visualised
        please use the syntax `show(mesh1, mesh2, volume, ..., options)`.

        :return: the current ``Plotter`` class instance.

        .. note:: E.g.:

            .. code-block:: python

                from vedo import *
                s = Sphere()
                s.show(at=1, N=2)
                c = Cube()
                c.show(at=0, interactive=True)
        """
        return vedo.plotter.show(self, **options)


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
        self.property = None


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


    def points(self, pts=None, transformed=True):
        """
        Set/Get the vertex coordinates of a mesh or point cloud.
        Argument can be an index, a set of indices
        or a complete new set of points to update the mesh.

        :param bool transformed: if `False` ignore any previous transformation
            applied to the mesh.
        """
        if pts is None: ### getter

            if isinstance(self, vedo.Points):
                vpts = self.polydata(transformed).GetPoints()
            else:
                vpts = self._data.GetPoints()

            if vpts:
                return utils.vtk2numpy(vpts.GetData())
            else:
                return np.array([])

        elif (utils.isSequence(pts) and not utils.isSequence(pts[0])) or isinstance(pts, (int, np.integer)):
            #passing a list of indices or a single index
            return utils.vtk2numpy(self.polydata(transformed).GetPoints().GetData())[pts]

        else:           ### setter

            if len(pts) == 3 and len(pts[0]) != 3:
                # assume plist is in the format [all_x, all_y, all_z]
                pts = np.stack((pts[0], pts[1], pts[2]), axis=1)
            vpts = self._data.GetPoints()
            vpts.SetData(utils.numpy2vtk(pts, dtype=float))
            vpts.Modified()
            # reset mesh to identity matrix position/rotation:
            self.PokeMatrix(vtk.vtkMatrix4x4())
            return self

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
        return utils.vtk2numpy(vcen.GetOutput().GetPoints().GetData())


    def deleteCells(self, ids):
        """Remove cells from the mesh object by ID. Points (vertices) are not affected."""
        data = self.inputdata()
        for cid in ids:
            data.DeleteCell(cid)
        data.RemoveDeletedCells()
        data.Modified()
        self._mapper.Modified()
        return self


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
            self._data.GetCellData().AddArray(cellData)
            self._data.GetCellData().SetActiveScalars('CellsWithinBoundsColor')
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
                 specular=None, specularPower=None, specularColor=None,
                 metallicity=None, roughness=None,
        ):
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

            if isinstance(pr, vtk.vtkVolumeProperty):
                self.shade(True)
                if style=='off':
                    self.shade(False)
                elif style=='ambient':
                    style='default'
                    self.shade(False)
            else:
                if style!='off':
                    pr.LightingOn()

            if style=='off':
                pr.SetInterpolationToFlat()
                pr.LightingOff()
                return self ##############

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
            elif style=='ambient' : pars = [0.8, 0.1, 0.0,  1, (1,1,1)]
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
        if utils.vtkVersionIsAtLeast(9):
            if metallicity is not None:
                pr.SetInterpolationToPBR()
                pr.SetMetallic(metallicity)
            if roughness is not None:
                pr.SetInterpolationToPBR()
                pr.SetRoughness(roughness)

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


    def c(self, color=False, alpha=None):
        """
        Shortcut for `color()`.
        If None is passed as input, will use colors from current active scalars.
        """
        return self.color(color, alpha)

    @property
    def pointdata(self):
        """
        Create and/or return a ``numpy.array`` associated to points (vertices).
        A data array can be indexed either as a string or by an integer number.
        E.g.:  ``myobj.pointdata["arrayname"]``

        Use:

            ``myobj.pointdata.keys()`` to return the available data array names

            ``myobj.pointdata.select(name)`` to make this array the active one

            ``myobj.pointdata.remove(name)`` to remove this array
        """
        return _DataArrayHelper(self, 0)

    @property
    def celldata(self):
        """
        Create and/or return a ``numpy.array`` associated to cells (faces).
        A data array can be indexed either as a string or by an integer number.
        E.g.:  ``myobj.celldata["arrayname"]``

        Use:

            ``myobj.celldata.keys()`` to return the available data array names

            ``myobj.celldata.select(name)`` to make this array the active one

            ``myobj.celldata.remove(name)`` to remove this array
        """
        return _DataArrayHelper(self, 1)

    @deprecated(reason=colors.red+"Please use myobj.pointdata[name] instead."+colors.reset)
    def getPointArray(self, name=0):
        """Deprecated. Use myobj.pointdata[name]` instead."""
        return self.pointdata[name]

    @deprecated(reason=colors.red+"Please use myobj.celldata[name] instead."+colors.reset)
    def getCellArray(self, name=0):
        """Deprecated. Use myobj.celldata[name]` instead."""
        return self.celldata[name]

    @deprecated(reason=colors.red+"Please use myobj.pointdata[name] = myarr instead."+colors.reset)
    def addPointArray(self, input_array, name):
        """Deprecated. Use myobj.pointdata[name] = input_array` instead."""
        self.pointdata[name] = input_array
        return self

    @deprecated(reason=colors.red+"Please use myobj.celldata[name] = myarr instead."+colors.reset)
    def addCellArray(self, input_array, name):
        """Deprecated. Use myobj.celldata[name] = input_array` instead."""
        self.celldata[name] = input_array
        return self

    def mapCellsToPoints(self):
        """
        Interpolate cell data (i.e., data specified per cell or face)
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
        Interpolate point data (i.e., data specified per point or vertex)
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

    def addIDs(self, asfield=False):
        """Generate point and cell ids."""
        ids = vtk.vtkIdFilter()
        ids.SetInputData(self._data)
        ids.PointIdsOn()
        ids.CellIdsOn()
        ids.FieldDataOff()
        ids.Update()
        return self._update(ids.GetOutput())


    def gradient(self, arrname=None, on='points', fast=False):
        """
        Compute and return the gradiend of a scalar field as a numpy array.

        :param str arrname: name of an existing field
        :param str on: defined either on 'points' or 'cells'
        :param bool fast: if True, will use a less accurate algorithm
            that performs fewer derivative calculations (and is therefore faster).

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
        gra.SetResultArrayName('Gradient')
        gra.SetFasterApproximation(fast)
        gra.ComputeDivergenceOff()
        gra.ComputeVorticityOff()
        gra.ComputeGradientOn()
        gra.Update()
        if on.startswith('p'):
            gvecs = utils.vtk2numpy(gra.GetOutput().GetPointData().GetArray('Gradient'))
        else:
            gvecs = utils.vtk2numpy(gra.GetOutput().GetCellData().GetArray('Gradient'))
        return gvecs

    def divergence(self, arrname=None, on='points', fast=False):
        """
        Compute and return the divergence of a vector field as a numpy array.

        :param str arrname: name of an existing field
        :param str on: defined either on 'points' or 'cells'
        :param bool fast: if True, will use a less accurate algorithm
            that performs fewer derivative calculations (and is therefore faster).
        """
        div = vtk.vtkGradientFilter()
        if on.startswith('p'):
            varr = self.inputdata().GetPointData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        else:
            varr = self.inputdata().GetCellData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        if not arrname:
            if self.GetVectors():
                arrname = varr.GetVectors().GetName()
            else:
                colors.printc('Error in divergence: no scalars found for', on, c='r')
                raise RuntimeError
        div.SetInputData(self.inputdata())
        div.SetInputScalars(tp, arrname)
        div.ComputeDivergenceOn()
        div.ComputeGradientOff()
        div.ComputeVorticityOff()
        div.SetDivergenceArrayName('Divergence')
        div.SetFasterApproximation(fast)
        div.Update()
        if on.startswith('p'):
            dvecs = utils.vtk2numpy(div.GetOutput().GetPointData().GetArray('Divergence'))
        else:
            dvecs = utils.vtk2numpy(div.GetOutput().GetCellData().GetArray('Divergence'))
        return dvecs

    def vorticity(self, arrname=None, on='points', fast=False):
        """
        Compute and return the vorticity of a vector field as a numpy array.

        :param str arrname: name of an existing field
        :param str on: defined either on 'points' or 'cells'
        :param bool fast: if True, will use a less accurate algorithm
            that performs fewer derivative calculations (and is therefore faster).
        """
        vort = vtk.vtkGradientFilter()
        if on.startswith('p'):
            varr = self.inputdata().GetPointData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        else:
            varr = self.inputdata().GetCellData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        if not arrname:
            if self.GetVectors():
                arrname = varr.GetVectors().GetName()
            else:
                colors.printc('Error in vortergence: no scalars found for', on, c='r')
                raise RuntimeError
        vort.SetInputData(self.inputdata())
        vort.SetInputScalars(tp, arrname)
        vort.ComputeDivergenceOff()
        vort.ComputeGradientOff()
        vort.ComputeVorticityOn()
        vort.SetVorticityArrayName('Vorticity')
        vort.SetFasterApproximation(fast)
        vort.Update()
        if on.startswith('p'):
            vvecs = utils.vtk2numpy(vort.GetOutput().GetPointData().GetArray('Vorticity'))
        else:
            vvecs = utils.vtk2numpy(vort.GetOutput().GetCellData().GetArray('Vorticity'))
        return vvecs

    def addScalarBar(self,
                     title="",
                     pos=(0.8,0.05),
                     titleYOffset=15,
                     titleFontSize=12,
                     size=(None,None),
                     nlabels=None,
                     c=None,
                     horizontal=False,
                     useAlpha=True,
    ):
        """
        Add a 2D scalar bar for the specified obj.

        .. hint:: |mesh_coloring| |mesh_coloring.py|_ |scalarbars.py|_
        """
        self.scalarbar = vedo.addons.addScalarBar(self,
                                                  title,
                                                  pos,
                                                  titleYOffset,
                                                  titleFontSize,
                                                  size,
                                                  nlabels,
                                                  c,
                                                  horizontal,
                                                  useAlpha,
                                                 )
        return self


    def addScalarBar3D( self,
                        title='',
                        pos=None,
                        sx=None,
                        sy=None,
                        titleFont="",
                        titleXOffset = -1.5,
                        titleYOffset = 0.0,
                        titleSize =  1.5,
                        titleRotation = 0.0,
                        nlabels=9,
                        labelFont="",
                        labelSize=1,
                        labelOffset = 0.375,
                        labelRotation=0,
                        italic=0,
                        c=None,
                        useAlpha=True,
                        drawBox=True,
                        aboveText=None,
                        belowText=None,
                        nanText='NaN',
                        categories=None,
        ):
        """
        Draw a 3D scalar bar.

        ``obj`` input can be:
            - a list of numbers,
            - a list of two numbers in the form `(min, max)`,
            - a ``Mesh`` already containing a set of scalars associated to vertices or cells,
            - if ``None`` the last object in the list of actors will be used.

        Return an ``Assembly`` object.

        :param str title: scalar bar title
        :param float sx: thickness of scalarbar
        :param float sy: length of scalarbar
        :param float titleXOffset: horizontal space btw title and color scalarbar
        :param float titleYOffset: vertical space offset
        :param float titleSize: size of title wrt numeric labels
        :param float titleRotation: title rotation in degrees
        :param int nlabels: number of numeric labels
        :param str labelFont: font type for labels
        :param float labelSize: label scale factor
        :param float labelOffset: space btw numeric labels and scale
        :param float labelRotation: label rotation in degrees
        :param bool,float italic: use italic font for title and labels
        :param bool useAlpha: render transparency of the color bar, otherwise ignore
        :param bool drawBox: draw a box around the colorbar (useful with useAlpha=True)
        :param str aboveText: text to show for above scale values
        :param str belowText: text to show for below scale values
        :param str nanText: text to show for nan values

        |mesh_coloring| |mesh_coloring.py|_
        """
        self.scalarbar = vedo.addons.addScalarBar3D(self,
                                                    title,
                                                    pos,
                                                    sx,
                                                    sy,
                                                    titleFont,
                                                    titleXOffset,
                                                    titleYOffset,
                                                    titleSize,
                                                    titleRotation,
                                                    nlabels,
                                                    labelFont,
                                                    labelSize,
                                                    labelOffset,
                                                    labelRotation,
                                                    italic,
                                                    c,
                                                    useAlpha,
                                                    drawBox,
                                                    aboveText,
                                                    belowText,
                                                    nanText,
                                                    categories,
                                                    )
        return self


    def write(self, filename, binary=True):
        """Write object to file."""
        return vedo.io.write(self, filename, binary)


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
        gf = vtk.vtkGeometryFilter()
        if fill:
            sf = vtk.vtkShrinkFilter()
            sf.SetInputData(self._data)
            sf.SetShrinkFactor(shrink)
            sf.Update()
            gf.SetInputData(sf.GetOutput())
            gf.Update()
            poly = gf.GetOutput()
            if shrink==1.0:
                cleanPolyData = vtk.vtkCleanPolyData()
                cleanPolyData.PointMergingOn()
                cleanPolyData.ConvertLinesToPointsOn()
                cleanPolyData.ConvertPolysToLinesOn()
                cleanPolyData.ConvertStripsToPolysOn()
                cleanPolyData.SetInputData(poly)
                cleanPolyData.Update()
                poly = cleanPolyData.GetOutput()
        else:
            gf.SetInputData(self._data)
            gf.Update()
            poly = gf.GetOutput()

        msh = vedo.mesh.Mesh(poly).flat()
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

    def cells(self):
        """
        Get the cells connectivity ids as a numpy array.
        The output format is: [[id0 ... idn], [id0 ... idm],  etc].
        """
        arr1d = utils.vtk2numpy(self._data.GetCells().GetData())
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


    def color(self, col, alpha=None, vmin=None, vmax=None):
        """
        Assign a color or a set of colors along the range of the scalar value.
        A single constant color can also be assigned.
        Any matplotlib color map name is also accepted, e.g. ``volume.color('jet')``.

        E.g.: say that your cells scalar runs from -3 to 6,
        and you want -3 to show red and 1.5 violet and 6 green, then just set:

        ``volume.color(['red', 'violet', 'green'])``

        :param list alpha: use a list to specify transparencies along the scalar range
        :param float vmin: force the min of the scalar range to be this value
        :param float vmax: force the max of the scalar range to be this value
        """
        # superseeds method in Points, Mesh
        if vmin is None:
            vmin, _ = self._data.GetScalarRange()
        if vmax is None:
            _, vmax = self._data.GetScalarRange()
        ctf = self.GetProperty().GetRGBTransferFunction()
        ctf.RemoveAllPoints()
        self._color = col

        if utils.isSequence(col):
            if utils.isSequence(col[0]) and len(col[0])==2:
                # user passing [(value1, color1), ...]
                for x, ci in col:
                    r, g, b = colors.getColor(ci)
                    ctf.AddRGBPoint(x, r, g, b)
                    # colors.printc('color at', round(x, 1),
                    #               'set to', colors.getColorName((r, g, b)),
                    #               c='w', bold=0)
            else:
                # user passing [color1, color2, ..]
                for i, ci in enumerate(col):
                    r, g, b = colors.getColor(ci)
                    x = vmin + (vmax - vmin) * i / (len(col) - 1)
                    ctf.AddRGBPoint(x, r, g, b)
        elif isinstance(col, str):
            if col in colors.colors.keys() or col in colors.color_nicks.keys():
                r, g, b = colors.getColor(col)
                ctf.AddRGBPoint(vmin, r,g,b) # constant color
                ctf.AddRGBPoint(vmax, r,g,b)
            else: # assume it's a colormap
                for x in np.linspace(vmin, vmax, num=64, endpoint=True):
                    r,g,b = colors.colorMap(x, name=col, vmin=vmin, vmax=vmax)
                    ctf.AddRGBPoint(x, r, g, b)
        elif isinstance(col, int):
            r, g, b = colors.getColor(col)
            ctf.AddRGBPoint(vmin, r,g,b) # constant color
            ctf.AddRGBPoint(vmax, r,g,b)
        else:
            colors.printc("color(): unknown input type:", col, c='r')

        if alpha is not None:
            self.alpha(alpha, vmin=vmin, vmax=vmax)
        return self

    def alpha(self, alpha, vmin=None, vmax=None):
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
        if vmin is None:
            vmin, _ = self._data.GetScalarRange()
        if vmax is None:
            _, vmax = self._data.GetScalarRange()
        otf = self.GetProperty().GetScalarOpacity()
        otf.RemoveAllPoints()
        self._alpha = alpha

        if utils.isSequence(alpha):
            alpha = np.array(alpha)
            if len(alpha.shape)==1: # user passing a flat list e.g. (0.0, 0.3, 0.9, 1)
                for i, al in enumerate(alpha):
                    xalpha = vmin + (vmax - vmin) * i / (len(alpha) - 1)
                    # Create transfer mapping scalar value to opacity
                    otf.AddPoint(xalpha, al)
                    #colors.printc("alpha at", round(xalpha, 1), "\tset to", al)
            elif len(alpha.shape)==2: # user passing [(x0,alpha0), ...]
                otf.AddPoint(vmin, alpha[0][1])
                for xalpha, al in alpha:
                    # Create transfer mapping scalar value to opacity
                    otf.AddPoint(xalpha, al)
                otf.AddPoint(vmax, alpha[-1][1])

        else:

            otf.AddPoint(vmin, alpha) # constant alpha
            otf.AddPoint(vmax, alpha)

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

        a = vedo.mesh.Mesh(poly, c=None).phong()
        a._mapper.SetScalarRange(scrange[0], scrange[1])
        return a


    def legosurface(self, vmin=None, vmax=None, invert=False, cmap='afmhot_r'):
        """
        Represent an object - typically a Volume - as lego blocks (voxels).
        By default colors correspond to the volume's scalar.
        Returns an ``Mesh``.

        :param float vmin: the lower threshold, voxels below this value are not shown.
        :param float vmax: the upper threshold, voxels above this value are not shown.
        :param str cmap: color mapping of the scalar associated to the voxels.

        |legosurface| |legosurface.py|_
        """
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

        a = vedo.mesh.Mesh(gf.GetOutput()).lw(0.1).flat().lighting('ambient')
        scalars = a.pointdata[0]
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


    def cutWithBox(self, box):
        """
        Cut the grid with the specified bounding box.

        Parameter box has format [xmin, xmax, ymin, ymax, zmin, zmax].
        If a Mesh is passed, its bounding box is used.

        Example:

            .. code-block:: python

                from vedo import *
                tetmesh = TetMesh(dataurl+'limb_ugrid.vtk')
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
            ug.GetPointData().AddArray(signedDistances)
            ug.GetPointData().SetActiveScalars("SignedDistances")
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
                    self.celldata.select(scalname)
                else:
                    self.pointdata.select(scalname)

        self._update(cug)
        return self


    def tetralize(self, tetsOnly=True):
        """Tetralize the grid.
        If tetsOnly=True will cull all 1D and 2D cells from the output.

        Return a TetMesh.
        """
        return vedo.tetmesh.tetralize(self._data, tetsOnly)


    def extractCellsByID(self, idlist, usePointIDs=False):
        """Return a new UGrid composed of the specified subset of indices."""
        selectionNode = vtk.vtkSelectionNode()
        if usePointIDs:
            selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
            contcells = vtk.vtkSelectionNode.CONTAINING_CELLS()
            selectionNode.GetProperties().Set(contcells, 1)
        else:
            selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        vidlist = utils.numpy2vtk(idlist, dtype='id')
        selectionNode.SetSelectionList(vidlist)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        es = vtk.vtkExtractSelection()
        es.SetInputData(0, self._data)
        es.SetInputData(1, selection)
        es.Update()
        tm_sel = vedo.ugrid.UGrid(es.GetOutput())
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.GetProperty())
        tm_sel.SetProperty(pr)
        tm_sel.property = pr

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
    with `mesh.pointdata['vtkValidPointMask']`.
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
    with `mesh.pointdata['vtkValidPointMask']`.

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

    locator = vtk.vtkStaticPointLocator()
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

    if lw is not None and len(tubes) == 0 and not ribbons:
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




