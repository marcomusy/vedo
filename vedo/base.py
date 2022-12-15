#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from deprecated import deprecated

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

import vedo
from vedo import colors
from vedo import utils

__doc__ = "Base classes. Do not instantiate."

__all__ = [
    "Base3DProp",
    "BaseActor",
    "BaseGrid",
    "probe_points",
    "probe_line",
    "probe_plane",
]


###############################################################################
class _DataArrayHelper:
    # Helper class to manage data associated to either
    # points (or vertices) and cells (or faces).
    # Internal use only.
    def __init__(self, actor, association):
        self.actor = actor
        self.association = association

    def __getitem__(self, key):

        if self.association == 0:
            data = self.actor.inputdata().GetPointData()

        elif self.association == 1:
            data = self.actor.inputdata().GetCellData()

        elif self.association == 2:
            data = self.actor.inputdata().GetFieldData()
            if isinstance(key, int):
                key = data.GetArrayName(key)
            varr = data.GetAbstractArray(key)
            n = varr.GetNumberOfValues()
            narr = np.array([varr.GetValue(i) for i in range(n)], dtype='|U')
            return narr
            ###########

        else:
            raise RuntimeError()

        if isinstance(key, int):
            key = data.GetArrayName(key)

        arr = data.GetArray(key)
        if not arr:
            return None
        return utils.vtk2numpy(arr)

    def __setitem__(self, key, input_array):

        if self.association == 0:
            data = self.actor.inputdata().GetPointData()
            n = self.actor.inputdata().GetNumberOfPoints()
            self.actor._mapper.SetScalarModeToUsePointData()

        elif self.association == 1:
            data = self.actor.inputdata().GetCellData()
            n = self.actor.inputdata().GetNumberOfCells()
            self.actor._mapper.SetScalarModeToUseCellData()

        elif self.association == 2:
            data = self.actor.inputdata().GetFieldData()

            if isinstance(input_array[0], str):
                varr = vtk.vtkStringArray()
                varr.SetName(key)
                varr.SetNumberOfComponents(1)
                for val in input_array:
                    try:
                        varr.InsertNextValue(val)
                    except TypeError:
                        vedo.printc("in metadata cannot add type", type(val), c='r')
            else:
                varr = utils.numpy2vtk(input_array, name=key)

            data.AddArray(varr)
            return ############

        else:
            raise RuntimeError()

        if len(input_array) != n:
            vedo.logger.error(
                f"Error in point/cell data: length of input {len(input_array)}"
                f" !=  {n} nr. of elements"
            )
            raise RuntimeError()

        input_array = np.asarray(input_array)
        varr = utils.numpy2vtk(input_array, name=key)
        data.AddArray(varr)

        if len(input_array.shape) == 1:  # scalars
            data.SetActiveScalars(key)
        elif len(input_array.shape) == 2 and input_array.shape[1] == 3:  # vectors
            if key.lower() == "normals":
                data.SetActiveNormals(key)
            else:
                data.SetActiveVectors(key)

    def keys(self):
        """Return the list of available data array names"""
        if self.association == 0:
            data = self.actor.inputdata().GetPointData()
        elif self.association == 1:
            data = self.actor.inputdata().GetCellData()
        elif  self.association == 2:
            data = self.actor.inputdata().GetFieldData()
        arrnames = []
        for i in range(data.GetNumberOfArrays()):
            arrnames.append(data.GetArray(i).GetName())
        return arrnames

    def remove(self, key):
        """Remove a data array by name or number"""
        if self.association == 0:
            self.actor.inputdata().GetPointData().RemoveArray(key)
        elif self.association == 1:
            self.actor.inputdata().GetCellData().RemoveArray(key)
        elif  self.association == 2:
            self.actor.inputdata().GetFieldData().RemoveArray(key)

    def rename(self, oldname, newname):
        """Rename an array"""
        if self.association == 0:
            varr = self.actor.inputdata().GetPointData().GetArray(oldname)
        elif self.association == 1:
            varr = self.actor.inputdata().GetCellData().GetArray(oldname)
        elif self.association == 2:
            varr = self.actor.inputdata().GetFieldData().GetArray(oldname)
        if varr:
            varr.SetName(newname)
        else:
            vedo.logger.warning(
                f"Cannot rename non existing array {oldname} to {newname}"
            )

    def select(self, key):
        """Select one specific array by its name to make it the `active` one."""
        if self.association == 0:
            data = self.actor.inputdata().GetPointData()
            self.actor.mapper().SetScalarModeToUsePointData()
        else:
            data = self.actor.inputdata().GetCellData()
            self.actor.mapper().SetScalarModeToUseCellData()

        if isinstance(key, int):
            key = data.GetArrayName(key)
        data.SetActiveScalars(key)

        if hasattr(self.actor.mapper(), "SetArrayName"):
            self.actor.mapper().SetArrayName(key)

        if hasattr(self.actor.mapper(), "ScalarVisibilityOn"):  # could be volume mapper
            self.actor.mapper().ScalarVisibilityOn()

    def print(self, **kwargs):
        """Print the array names available to terminal"""
        colors.printc(self.keys(), **kwargs)


###############################################################################
class Base3DProp:
    """Base class to manage positioning and size of the objects in space and other properties"""

    def __init__(self):
        self.filename = ""
        self.name = ""
        self.fileSize = ""
        self.created = ""
        self.trail = None
        self.trail_points = []
        self.trail_segment_size = 0
        self.trail_offset = None
        self.shadows = []
        self.axes = None
        self.picked3d = None
        self.units = None
        self.top = np.array([0, 0, 1])
        self.base = np.array([0, 0, 0])
        self.info = {}
        self._time = 0
        self.rendered_at = set()
        self.transform = None
        self._set2actcam = False  # used by mesh.followCamera()

        self.point_locator = None
        self.cell_locator = None

        self.scalarbar = None


    def address(self):
        """
        Return a unique memory address integer which may serve as the ID of the
        object, or passed to c++ code.
        """
        # https://www.linkedin.com/pulse/speedup-your-code-accessing-python-vtk-objects-from-c-pletzer/
        # https://github.com/tfmoraes/polydata_connectivity
        return int(self.inputdata().GetAddressAsString("")[5:], 16)

    def pickable(self, value=None):
        """Set/get the pickability property of an object."""
        if value is None:
            return self.GetPickable()
        self.SetPickable(value)
        return self

    def draggable(self, value=None):  # NOT FUNCTIONAL?
        """Set/get the draggability property of an object."""
        if value is None:
            return self.GetDragable()
        self.SetDragable(value)
        return self

    def time(self, t=None):
        """Set/get object's absolute time of creation."""
        if t is None:
            return self._time
        self._time = t
        return self  # return itself to concatenate methods

    def origin(self, x=None, y=None, z=None):
        """
        Set/get object's origin.

        Relevant to control the scaling with `scale()` and rotations.
        Has no effect on position.
        """
        if x is None:
            return np.array(self.GetOrigin()) + self.GetPosition()

        if z is None and y is None:  # assume x is of the form (x,y,z)
            if len(x) == 3:
                x, y, z = x
            else:
                x, y = x
                z = 0
        elif z is None:  # assume x,y is of the form x, y
            z = 0
        self.SetOrigin([x, y, z] - np.array(self.GetPosition()))
        return self

    def pos(self, x=None, y=None, z=None):
        """Set/Get object position."""
        if x is None:  # get functionality
            return np.array(self.GetPosition())

        if z is None and y is None:  # assume x is of the form (x,y,z)
            if len(x) == 3:
                x, y, z = x
            else:
                x, y = x
                z = 0
        elif z is None:  # assume x,y is of the form x, y
            z = 0
        self.SetPosition(x, y, z)

        self.point_locator = None
        self.cell_locator = None
        return self  # return itself to concatenate methods

    def shift(self, dx=0, dy=0, dz=0):
        """Add a vector to the current object position."""
        p = np.array(self.GetPosition())

        if utils.is_sequence(dx):
            if len(dx) == 2:
                self.SetPosition(p + [dx[0], dx[1], 0])
            else:
                self.SetPosition(p + dx)
        else:
            self.SetPosition(p + [dx, dy, dz])

        self.point_locator = None
        self.cell_locator = None
        return self

    def x(self, position=None):
        """Set/Get object position along x axis."""
        p = self.GetPosition()
        if position is None:
            return p[0]
        self.pos(position, p[1], p[2])
        return self

    def y(self, position=None):
        """Set/Get object position along y axis."""
        p = self.GetPosition()
        if position is None:
            return p[1]
        self.pos(p[0], position, p[2])
        return self

    def z(self, position=None):
        """Set/Get object position along z axis."""
        p = self.GetPosition()
        if position is None:
            return p[2]
        self.pos(p[0], p[1], position)
        return self

    def rotate(self, angle, axis=(1, 0, 0), point=(0, 0, 0), rad=False):
        """
        Rotate around an arbitrary `axis` passing through `point`.

        Example:
            .. code-block:: python

                from vedo import *
                c1 = Cube()
                c2 = c1.clone().c('violet').alpha(0.5) # copy of c1
                v = vector(0.2,1,0)
                p = vector(1,0,0)  # axis passes through this point
                c2.rotate(90, axis=v, point=p)
                l = Line(-v+p, v+p).lw(3).c('red')
                show(c1, l, c2, axes=1).close()
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
        R = np.array(
            [
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
        self.pos(rv)
        return self

    def _rotatexyz(self, a, angle, rad, around):
        if rad:
            angle *= 180 / np.pi

        T = vtk.vtkTransform()
        T.SetMatrix(self.GetMatrix())
        T.PostMultiply()

        rot = dict(x=T.RotateX, y=T.RotateY, z=T.RotateZ)

        if around is None:
            # rotate around its origin
            rot[a](angle)
        else:
            if around == "itself":
                around = self.GetPosition()
            # displacement needed to bring it back to the origin
            # and disregard origin
            disp = around - np.array(self.GetOrigin())
            T.Translate(-disp)
            rot[a](angle)
            T.Translate(disp)

        self.SetOrientation(T.GetOrientation())
        self.SetPosition(T.GetPosition())

        self.point_locator = None
        self.cell_locator = None
        return self

    @deprecated(reason=vedo.colors.red + "Please use rotate_x()" + vedo.colors.reset)
    def rotateX(self, *a, **b):
        """Deprecated. Please use rotate_x()."""
        return self.rotate_x(*a, **b)
    @deprecated(reason=vedo.colors.red + "Please use rotate_y()" + vedo.colors.reset)
    def rotateY(self, *a, **b):
        """Deprecated. Please use rotate_y()."""
        return self.rotate_y(*a, **b)
    @deprecated(reason=vedo.colors.red + "Please use rotate_z()" + vedo.colors.reset)
    def rotateZ(self, *a, **b):
        """Deprecated. Please use rotate_z()."""
        return self.rotate_z(*a, **b)

    def rotate_x(self, angle, rad=False, around=None):
        """
        Rotate around x-axis. If angle is in radians set ``rad=True``.

        Use `around` to define a pivoting point.
        """
        return self._rotatexyz("x", angle, rad, around)

    def rotate_y(self, angle, rad=False, around=None):
        """
        Rotate around y-axis. If angle is in radians set ``rad=True``.

        Use `around` to define a pivoting point.
        """
        return self._rotatexyz("y", angle, rad, around)

    def rotate_z(self, angle, rad=False, around=None):
        """
        Rotate around z-axis. If angle is in radians set ``rad=True``.

        Use `around` to define a pivoting point.
        """
        return self._rotatexyz("z", angle, rad, around)

    def orientation(self, newaxis=None, rotation=0, concatenate=False, xyplane=False, rad=False):
        """
        Set/Get object orientation.

        Parameters
        ----------
        rotation : float
            rotate object around newaxis.

        concatenate : bool
            concatenate the orietation operation with the previous existing transform (if any)

        xyplane : bool
            make an extra rotation to keep the object aligned to the xy-plane

        rad : bool
            set to True if angle is expressed in radians.

        Example:
            .. code-block:: python

                from vedo import *
                center = np.array([581/2,723/2,0])
                objs = []
                for a in np.linspace(0, 6.28, 7):
                    v = vector(cos(a), sin(a), 0)*1000
                    pic = Picture(dataurl+"images/dog.jpg").rotate_z(10)
                    pic.orientation(v, xyplane=True)
                    pic.origin(center)
                    pic.pos(v - center)
                    objs += [pic, Arrow(v, v+v)]
                show(objs, Point(), axes=1).close()

        .. hint:: examples/simulations/gyroscope2.py
            .. image:: https://vedo.embl.es/images/simulations/50738942-687b5780-11d9-11e9-97f0-72bbd63f7d6e.gif
        """

        if self.top is None or self.base is None:
            initaxis = (0, 0, 1)
        else:
            initaxis = utils.versor(self.top - self.base)

        newaxis = utils.versor(newaxis)
        p = np.array(self.GetPosition())
        crossvec = np.cross(initaxis, newaxis)

        angleth = np.arccos(np.dot(initaxis, newaxis))

        T = vtk.vtkTransform()
        if concatenate:
            try:
                M = self.GetMatrix()
                T.SetMatrix(M)
            except:
                pass
        T.PostMultiply()
        T.Translate(-p)
        if rotation:
            if rad:
                rotation *= 180.0 / np.pi
            T.RotateWXYZ(rotation, initaxis)
        if xyplane:
            angleph = np.arctan2(newaxis[1], newaxis[0])
            T.RotateWXYZ(np.rad2deg(angleph + angleth), initaxis) # compensation
        T.RotateWXYZ(np.rad2deg(angleth), crossvec)
        T.Translate(p)

        self.SetOrientation(T.GetOrientation())

        self.point_locator = None
        self.cell_locator = None
        return self

        # newaxis = utils.versor(newaxis)
        # pos = np.array(self.GetPosition())
        # crossvec = np.cross(initaxis, newaxis)
        # angle = np.arccos(np.dot(initaxis, newaxis))
        # T = vtk.vtkTransform()
        # T.PostMultiply()
        # T.Translate(-pos)
        # if rotation:
        #     T.RotateWXYZ(rotation, initaxis)
        # T.RotateWXYZ(np.rad2deg(angle), crossvec)
        # T.Translate(pos)
        # self.SetUserTransform(T)
        # self.transform = T


    def scale(self, s=None, reset=False):
        """
        Set/get object's scaling factor.

        Parameters
        ----------
        s : list, float
            scaling factor(s).

        reset : bool
            if True previous scaling factors are ignored.

        .. note:: if `s=(sx,sy,sz)` scale differently in the three coordinates.
        """
        if s is None:
            return np.array(self.GetScale())

        # assert s[0] != 0
        # assert s[1] != 0
        # assert s[2] != 0

        if reset:
            self.SetScale(s)
        else:
            self.SetScale(np.multiply(self.GetScale(), s))

        self.point_locator = None
        self.cell_locator = None
        return self

    def get_transform(self, invert=False):
        """
        Check if ``object.transform`` exists and returns a ``vtkTransform``.
        Otherwise return current user transformation (where the object is currently placed).

        Use ``invert`` to return the inverse of the current transformation

        Example:
            .. code-block:: python

                from vedo import *

                c1 = Cube()
                c2 = c1.clone().c('violet').alpha(0.5) # copy of c1
                v = vector(0.2,1,0)
                p = vector(1,0,0)  # axis passes through this point
                c2.rotate(90, axis=v, point=p)

                # get the inverse of the current transformation
                T = c2.get_transform(invert=True)
                c2.apply_transform(T)  # put back c2 in place

                l = Line(p-v, p+v).lw(3).c('red')
                show(c1.wireframe().lw(3), l, c2, axes=1)
        """
        # if self.transform:
        #     tr = self.transform
        # else:
        #     T = self.GetMatrix()
        #     tr = vtk.vtkTransform()
        #     tr.SetMatrix(T)
        # if invert:
        #     tr = tr.GetInverse()
        # return tr if self.transform:
        T = self.GetMatrix()
        tr = vtk.vtkTransform()
        tr.SetMatrix(T)
        if invert:
            tr = tr.GetInverse()
        return tr


    @deprecated(reason=vedo.colors.red + "Please use apply_transform()" + vedo.colors.reset)
    def applyTransform(self, T, reset=False, concatenate=False):
        """Deprecated. Please use apply_transform()"""
        return self.apply_transform(T,reset,concatenate)

    def apply_transform(self, T, reset=False, concatenate=False):
        """
        Transform object position and orientation.

        Parameters
        ----------
        reset : bool
            no effect, this is superseded by pointcloud.apply_transform()

        concatenate : bool
            no effect, this is superseded by pointcloud.apply_transform()
        """
        if isinstance(T, vtk.vtkMatrix4x4):
            self.SetUserMatrix(T)
        elif utils.is_sequence(T):
            vm = vtk.vtkMatrix4x4()
            for i in [0, 1, 2, 3]:
                for j in [0, 1, 2, 3]:
                    vm.SetElement(i, j, T[i][j])
            self.SetUserMatrix(vm)
        else:
            self.SetUserTransform(T)
        self.transform = T

        self.point_locator = None
        self.cell_locator = None
        return self

    def align_to_bounding_box(self, msh, rigid=False):
        """
        Align the current object's bounding box to the bounding box
        of the input object.

        Use ``rigid`` to disable scaling.

        .. hint:: examples/basic/align6.py
        """
        lmt = vtk.vtkLandmarkTransform()
        ss = vtk.vtkPoints()
        xss0, xss1, yss0, yss1, zss0, zss1 = self.bounds()
        for p in [
            [xss0, yss0, zss0],
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
        xst0, xst1, yst0, yst1, zst0, zst1 = msh.bounds()
        for p in [
            [xst0, yst0, zst0],
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
        lmt.SetModeToAffine()
        if rigid:
            lmt.SetModeToRigidBody()
        lmt.Update()
        self.apply_transform(lmt)
        self.transform = lmt

        self.point_locator = None
        self.cell_locator = None
        return self

    def on(self):
        """Switch on  object visibility. Object is not removed."""
        self.VisibilityOn()
        return self

    def off(self):
        """Switch off object visibility. Object is not removed."""
        self.VisibilityOff()
        return self

    def box(self, scale=1, padding=0, fill=False):
        """
        Return the bounding box as a new ``Mesh``.

        Parameters
        ----------
        scale : float
            box size can be scaled by a factor

        padding : float, list
            a constant padding can be added (can be a list [padx,pady,padz])

        .. hint:: examples/pyplot/latex.py
        """
        b = self.bounds()
        if not utils.is_sequence(padding):
            padding = [padding, padding, padding]
        length, width, height = b[1] - b[0], b[3] - b[2], b[5] - b[4]
        tol = (length + width + height) / 30000  # useful for boxing 2D text
        pos = [(b[0] + b[1]) / 2, (b[3] + b[2]) / 2, (b[5] + b[4]) / 2 - tol]
        bx = vedo.shapes.Box(
            pos,
            length * scale + padding[0],
            width  * scale + padding[1],
            height * scale + padding[2],
            c="gray",
        )
        if hasattr(self, "GetProperty"):  # could be Assembly
            if isinstance(self.GetProperty(), vtk.vtkProperty):  # could be volume
                pr = vtk.vtkProperty()
                pr.DeepCopy(self.GetProperty())
                bx.SetProperty(pr)
                bx.property = pr
        bx.wireframe(not fill)
        bx.flat().lighting("off")
        return bx

    def use_bounds(self, ub=True):
        """
        Instruct the current camera to either take into account or ignore
        the object bounds when resetting.
        """
        self.SetUseBounds(ub)
        return self

    def bounds(self):
        """
        Get the object bounds.
        Returns a list in format [xmin,xmax, ymin,ymax, zmin,zmax].
        """
        try:
            pts = self.points()
            xmin, ymin, zmin = np.min(pts, axis=0)
            xmax, ymax, zmax = np.max(pts, axis=0)
            return [xmin,xmax, ymin,ymax, zmin,zmax]
        except (AttributeError, ValueError):
            return self.GetBounds()

    def xbounds(self, i=None):
        """Get the bounds [xmin,xmax]. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i is not None:
            return b[i]
        return (b[0], b[1])

    def ybounds(self, i=None):
        """Get the bounds [ymin,ymax]. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i == 0:
            return b[2]
        if i == 1:
            return b[3]
        return (b[2], b[3])

    def zbounds(self, i=None):
        """Get the bounds [zmin,zmax]. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i == 0: return b[4]
        if i == 1: return b[5]
        return (b[4], b[5])

    @deprecated(reason=vedo.colors.red + "Please use diagonal_size()" + vedo.colors.reset)
    def diagonalSize(self):
        """Deprecated. Please use diagonal_size()."""
        return self.diagonal_size()

    def diagonal_size(self):
        """Get the length of the diagonal of mesh bounding box."""
        b = self.bounds()
        return np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)
        # return self.GetLength() # ???different???


    def copy_data_from(self, obj):
        """Copy all data (point and cell data) from this input object"""
        self._data.GetPointData().PassData(obj._data.GetPointData())
        self._data.GetCellData().PassData(obj._data.GetCellData())
        return self

    def print(self):
        """Print information about an object."""
        utils.print_info(self)
        return self

    def show(self, **options):
        """
        Create on the fly an instance of class ``Plotter`` or use the last existing one to
        show one single object.

        This method is meant as a shortcut. If more than one object needs to be visualised
        please use the syntax `show(mesh1, mesh2, volume, ..., options)`.

        Returns the ``Plotter`` class instance.
        """
        return vedo.plotter.show(self, **options)


########################################################################################
class BaseActor(Base3DProp):
    """
    Base class to add operative and data
    functionality to ``Mesh``, ``Assembly``, ``Volume`` and ``Picture`` objects.

    .. warning:: Do not use this class to instance objects, use one the above instead.
    """

    def __init__(self):
        Base3DProp.__init__(self)

        self._mapper = None
        self._caption = None
        self.property = None


    def mapper(self, new_mapper=None):
        """Return the ``vtkMapper`` data object, or update it with a new one."""
        if new_mapper:
            self.SetMapper(new_mapper)
            if self._mapper:
                iptdata = self._mapper.GetInput()
                if iptdata:
                    new_mapper.SetInputData(self._mapper.GetInput())
            self._mapper = new_mapper
            self._mapper.Modified()
        return self._mapper

    def inputdata(self):
        """Return the VTK input data object."""
        if self._mapper:
            return self._mapper.GetInput()
        return self.GetMapper().GetInput()

    def modified(self):
        """Use in conjunction with ``tonumpy()``
        to update any modifications to the volume array"""
        sc = self.inputdata().GetPointData().GetScalars()
        if sc:
            sc.Modified()
        self.inputdata().GetPointData().Modified()
        return self

    @deprecated(reason=vedo.colors.red + "Please use property object.npoints" + vedo.colors.reset)
    def N(self):
        """Deprecated. Please use property object.npoints"""
        return self.inputdata().GetNumberOfPoints()

    @deprecated(reason=vedo.colors.red + "Please use property object.npoints" + vedo.colors.reset)
    def NPoints(self):
        """Deprecated. Please use property object.npoints"""
        return self.inputdata().GetNumberOfPoints()

    @deprecated(reason=vedo.colors.red + "Please use property object.ncells" + vedo.colors.reset)
    def NCells(self):
        """Deprecated. Please use property object.ncells"""
        return self.inputdata().GetNumberOfCells()

    @property
    def npoints(self):
        """Retrieve the number of points."""
        return self.inputdata().GetNumberOfPoints()

    @property
    def ncells(self):
        """Retrieve the number of cells."""
        return self.inputdata().GetNumberOfCells()


    def points(self, pts=None, transformed=True):
        """
        Set/Get the vertex coordinates of a mesh or point cloud.
        Argument can be an index, a set of indices
        or a complete new set of points to update the mesh.

        Set ``transformed=False`` to ignore any previous transformation applied to the mesh.
        """
        if pts is None:  ### getter

            if isinstance(self, vedo.Points):
                vpts = self.polydata(transformed).GetPoints()
            elif isinstance(self, vedo.BaseVolume):
                v2p = vtk.vtkImageToPoints()
                v2p.SetInputData(self.imagedata())
                v2p.Update()
                vpts = v2p.GetOutput().GetPoints()
            else:  # tetmesh et al
                vpts = self.inputdata().GetPoints()

            if vpts:
                return utils.vtk2numpy(vpts.GetData())
            return np.array([])

        elif (utils.is_sequence(pts) and not utils.is_sequence(pts[0])) or isinstance(
            pts, (int, np.integer)
        ):
            # passing a list of indices or a single index
            return utils.vtk2numpy(self.polydata(transformed).GetPoints().GetData())[pts]

        else:  ### setter

            if len(pts) == 3 and len(pts[0]) != 3:
                # assume plist is in the format [all_x, all_y, all_z]
                pts = np.stack((pts[0], pts[1], pts[2]), axis=1)
            vpts = self.inputdata().GetPoints()
            vpts.SetData(utils.numpy2vtk(pts, dtype=float))
            vpts.Modified()
            # reset mesh to identity matrix position/rotation:
            self.PokeMatrix(vtk.vtkMatrix4x4())
            self.point_locator = None
            self.cell_locator = None
            return self

    @deprecated(reason=vedo.colors.red + "Please use cell_centers()" + vedo.colors.reset)
    def cellCenters(self):
        """Deprecated. Please use cell_centers()"""
        return self.cell_centers()

    def cell_centers(self):
        """
        Get the coordinates of the cell centers.

        .. hint:: examples/basic/delaunay2d.py
        """
        vcen = vtk.vtkCellCenters()
        if hasattr(self, "polydata"):
            vcen.SetInputData(self.polydata())
        else:
            vcen.SetInputData(self.inputdata())
        vcen.Update()
        return utils.vtk2numpy(vcen.GetOutput().GetPoints().GetData())

    def delete_cells(self, ids):
        """
        Remove cells from the mesh object by their ID.
        Points (vertices) are not removed
        (you may use `.clean()` to remove those).
        """
        data = self.inputdata()
        data.BuildLinks()
        for cid in ids:
            data.DeleteCell(cid)
        data.RemoveDeletedCells()
        data.Modified()
        self._mapper.Modified()
        return self


    # @deprecated(reason=vedo.colors.red + "Please use property cellcolors" + vedo.colors.reset)
    # def get_rgba(self, on="points"):
    #     """Deprecated."""
    #     if "point" in on:
    #         return self.pointcolors
    #     elif "cell" in on:
    #         return self.cellcolors
    #     else:
    #         raise ValueError("in get_rgba() input must be 'points' or 'cells'")


    def mark_boundaries(self):
        """Mark cells and vertices of the mesh if they lie on a boundary."""
        mb = vtk.vtkMarkBoundaryFilter()
        mb.SetInputData(self.polydata())
        mb.Update()
        return self._update(mb.GetOutput())

    def find_cells_in(self, xbounds=(), ybounds=(), zbounds=()):
        """
        Find cells that are within the specified bounds.
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

        cids = []
        for i in range(cellIds.GetNumberOfIds()):
            cid = cellIds.GetId(i)
            cids.append(cid)

        return np.array(cids)

    def count_vertices(self):
        """Count the number of vertices each cell has and return it as a numpy array"""
        vc = vtk.vtkCountVertices()
        vc.SetInputData(self._data)
        vc.SetOutputArrayName("VertexCount")
        vc.Update()
        varr = vc.GetOutput().GetCellData().GetArray("VertexCount")
        return utils.vtk2numpy(varr)

    def lighting(
        self,
        style="",
        ambient=None,
        diffuse=None,
        specular=None,
        specular_power=None,
        specular_color=None,
        metallicity=None,
        roughness=None,
    ):
        """
        Set the ambient, diffuse, specular and specular_power lighting constants.

        Parameters
        ----------
        style: str
            preset style, options are `[metallic, plastic, shiny, glossy, ambient, off]`

        ambient : float
            ambient fraction of emission [0-1]

        diffuse : float
            emission of diffused light in fraction [0-1]

        specular : float
            fraction of reflected light [0-1]

        specular_power : float
            precision of reflection [1-100]

        specular_color : color
            color that is being reflected by the surface

        .. image:: https://upload.wikimedia.org/wikipedia/commons/6/6b/Phong_components_version_4.png

        .. hint:: examples/basic/specular.py
        """
        pr = self.GetProperty()

        if style:

            if isinstance(pr, vtk.vtkVolumeProperty):
                self.shade(True)
                if style == "off":
                    self.shade(False)
                elif style == "ambient":
                    style = "default"
                    self.shade(False)
            else:
                if style != "off":
                    pr.LightingOn()

            if style == "off":
                pr.SetInterpolationToFlat()
                pr.LightingOff()
                return self  ##############

            if hasattr(pr, "GetColor"):  # could be Volume
                c = pr.GetColor()
            else:
                c = (1, 1, 0.99)
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
                vedo.logger.error("in lighting(): Available styles are")
                vedo.logger.error(
                    "[default, metallic, plastic, shiny, glossy, ambient, off]"
                )
                raise RuntimeError()
            pr.SetAmbient(pars[0])
            pr.SetDiffuse(pars[1])
            pr.SetSpecular(pars[2])
            pr.SetSpecularPower(pars[3])
            if hasattr(pr, "GetColor"):
                pr.SetSpecularColor(pars[4])

        if ambient is not None: pr.SetAmbient(ambient)
        if diffuse is not None: pr.SetDiffuse(diffuse)
        if specular is not None: pr.SetSpecular(specular)
        if specular_power is not None: pr.SetSpecularPower(specular_power)
        if specular_color is not None: pr.SetSpecularColor(colors.get_color(specular_color))
        if utils.vtk_version_at_least(9):
            if metallicity is not None:
                pr.SetInterpolationToPBR()
                pr.SetMetallic(metallicity)
            if roughness is not None:
                pr.SetInterpolationToPBR()
                pr.SetRoughness(roughness)

        return self

    def print_histogram(
        self,
        bins=10,
        height=10,
        logscale=False,
        minbin=0,
        horizontal=False,
        char="\U00002589",
        c=None,
        bold=True,
        title="Histogram",
    ):
        """
        Ascii histogram printing on terminal.
        Input can be ``Volume`` or ``Mesh`` (will grab the active point array).

        Parameters
        ----------
        bins : int
            number of histogram bins

        height : int
            height of the histogram in character units

        logscale : bool
            use logscale for frequencies

        minbin : int
            ignore bins before minbin

        horizontal : bool
            show histogram horizontally

        char : str
            character to be used as marker

        c : color
            ascii color

        bold : bool
            use boldface

        title : str
            histogram title
        """
        utils.print_histogram(
            self, bins, height, logscale, minbin, horizontal, char, c, bold, title
        )
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

        Usage:

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

        Usage:

            ``myobj.celldata.keys()`` to return the available data array names

            ``myobj.celldata.select(name)`` to make this array the active one

            ``myobj.celldata.remove(name)`` to remove this array
        """
        return _DataArrayHelper(self, 1)

    @property
    def metadata(self):
        """
        Create and/or return a ``numpy.array`` associated to neither cells nor faces.
        A data array can be indexed either as a string or by an integer number.
        E.g.:  ``myobj.metadata["arrayname"]``

        Usage:

            ``myobj.metadata.keys()`` to return the available data array names

            ``myobj.metadata.select(name)`` to make this array the active one

            ``myobj.metadata.remove(name)`` to remove this array
        """
        return _DataArrayHelper(self, 2)


    def map_cells_to_points(self):
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

    def map_points_to_cells(self):
        """
        Interpolate point data (i.e., data specified per point or vertex)
        into cell data (i.e., data specified per cell).
        The method of transformation is based on averaging the data values
        of all points defining a particular cell.

        .. hint:: examples/basic/mesh_map2cell.py
        """
        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(self.inputdata())
        p2c.Update()
        self._mapper.SetScalarModeToUseCellData()
        return self._update(p2c.GetOutput())

    def add_ids(self):
        """Generate point and cell ids arrays."""
        ids = vtk.vtkIdFilter()
        ids.SetInputData(self.inputdata())
        ids.PointIdsOn()
        ids.CellIdsOn()
        ids.FieldDataOff()
        ids.Update()
        return self._update(ids.GetOutput())

    def gradient(self, on="points", fast=False):
        """
        Compute and return the gradiend of the active scalar field as a numpy array.

        Parameters
        ----------
        on : str
            compute either on 'points' or 'cells' data

        fast : bool
            if True, will use a less accurate algorithm
            that performs fewer derivative calculations (and is therefore faster).

        .. hint::  examples/advanced/isolines.py
            .. image:: https://user-images.githubusercontent.com/32848391/72433087-f00a8780-3798-11ea-9778-991f0abeca70.png
        """
        gra = vtk.vtkGradientFilter()
        if on.startswith("p"):
            varr = self.inputdata().GetPointData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        else:
            varr = self.inputdata().GetCellData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS

        if varr.GetScalars():
            arrname = varr.GetScalars().GetName()
        else:
            vedo.logger.error(f"in gradient: no scalars found for {on}")
            raise RuntimeError

        gra.SetInputData(self.inputdata())
        gra.SetInputScalars(tp, arrname)
        gra.SetResultArrayName("Gradient")
        gra.SetFasterApproximation(fast)
        gra.ComputeDivergenceOff()
        gra.ComputeVorticityOff()
        gra.ComputeGradientOn()
        gra.Update()
        if on.startswith("p"):
            gvecs = utils.vtk2numpy(gra.GetOutput().GetPointData().GetArray("Gradient"))
        else:
            gvecs = utils.vtk2numpy(gra.GetOutput().GetCellData().GetArray("Gradient"))
        return gvecs

    def divergence(self, on="points", fast=False):
        """
        Compute and return the divergence of a vector field as a numpy array.

        Parameters
        ----------
        on : str
            compute either on 'points' or 'cells' data

        fast : bool
            if True, will use a less accurate algorithm
            that performs fewer derivative calculations (and is therefore faster).
        """
        div = vtk.vtkGradientFilter()
        if on.startswith("p"):
            varr = self.inputdata().GetPointData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        else:
            varr = self.inputdata().GetCellData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS

        if varr.GetVectors():
            arrname = varr.GetVectors().GetName()
        else:
            vedo.logger.error(f"in divergence(): no vectors found for {on}")
            raise RuntimeError

        div.SetInputData(self.inputdata())
        div.SetInputScalars(tp, arrname)
        div.ComputeDivergenceOn()
        div.ComputeGradientOff()
        div.ComputeVorticityOff()
        div.SetDivergenceArrayName("Divergence")
        div.SetFasterApproximation(fast)
        div.Update()
        if on.startswith('p'):
            dvecs = utils.vtk2numpy(div.GetOutput().GetPointData().GetArray('Divergence'))
        else:
            dvecs = utils.vtk2numpy(div.GetOutput().GetCellData().GetArray('Divergence'))
        return dvecs

    def vorticity(self, on="points", fast=False):
        """
        Compute and return the vorticity of a vector field as a numpy array.

        Parameters
        ----------
        on : str
            compute either on 'points' or 'cells' data

        fast : bool
            if True, will use a less accurate algorithm
            that performs fewer derivative calculations (and is therefore faster).
        """
        vort = vtk.vtkGradientFilter()
        if on.startswith("p"):
            varr = self.inputdata().GetPointData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        else:
            varr = self.inputdata().GetCellData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS

        if varr.GetVectors():
            arrname = varr.GetVectors().GetName()
        else:
            vedo.logger.error(f"in vorticity(): no vectors found for {on}")
            raise RuntimeError

        vort.SetInputData(self.inputdata())
        vort.SetInputScalars(tp, arrname)
        vort.ComputeDivergenceOff()
        vort.ComputeGradientOff()
        vort.ComputeVorticityOn()
        vort.SetVorticityArrayName("Vorticity")
        vort.SetFasterApproximation(fast)
        vort.Update()
        if on.startswith('p'):
            vvecs = utils.vtk2numpy(vort.GetOutput().GetPointData().GetArray('Vorticity'))
        else:
            vvecs = utils.vtk2numpy(vort.GetOutput().GetCellData().GetArray('Vorticity'))
        return vvecs


    @deprecated(reason=vedo.colors.red + "Please use method add_scalarbar()" + vedo.colors.reset)
    def addScalarBar(self, *a, **b):
        """Deprecated. Please use method add_scalarbar()"""
        return self.add_scalarbar(*a, **b)

    def add_scalarbar(
            self,
            title="",
            pos=(0.8,0.05),
            title_yoffset=15,
            font_size=12,
            size=(None,None),
            nlabels=None,
            c=None,
            horizontal=False,
            use_alpha=True,
            label_format=':6.3g',
        ):
        """
        Add a 2D scalar bar for the specified obj.

        .. hint:: examples/basic/mesh_coloring.py, scalarbars.py
        """
        plt = vedo.plotter_instance

        if plt and plt.renderer:
            c = (0.9, 0.9, 0.9)
            if np.sum(plt.renderer.GetBackground()) > 1.5:
                c = (0.1, 0.1, 0.1)
            if isinstance(self.scalarbar, vtk.vtkActor):
                plt.renderer.RemoveActor(self.scalarbar)
            elif isinstance(self.scalarbar, vedo.Assembly):
                for a in self.scalarbar.get_meshes():
                    plt.renderer.RemoveActor(a)
        if c is None:
            c = "gray"

        sb = vedo.addons.ScalarBar(
            self,
            title,
            pos,
            title_yoffset,
            font_size,
            size,
            nlabels,
            c,
            horizontal,
            use_alpha,
            label_format,
        )
        self.scalarbar = sb
        return self

    @deprecated(reason=vedo.colors.red + "Please use method add_scalarbar3d()" + vedo.colors.reset)
    def addScalarBar3D(self, *a, **b):
        """Deprecated. Please use method add_scalarbar3d()"""
        return self.add_scalarbar3d(*a, **b)

    def add_scalarbar3d(
        self,
        title="",
        pos=None,
        s=(None, None),
        title_font="",
        title_xoffset=-1.5,
        title_yoffset=0.0,
        title_size=1.5,
        title_rotation=0.0,
        nlabels=9,
        label_font="",
        label_size=1,
        label_offset=0.375,
        label_rotation=0,
        label_format="",
        italic=0,
        c=None,
        draw_box=True,
        above_text=None,
        below_text=None,
        nan_text="NaN",
        categories=None,
    ):
        """
        Associate a 3D scalar bar to the object and add it to the scene.
        The new scalarbar object (Assembly) will be accessible as obj.scalarbar

        Parameters
        ----------
        s : list
            (thickness, length) of scalarbar

        title : str
            scalar bar title

        title_xoffset : float
            horizontal space btw title and color scalarbar

        title_yoffset : float
            vertical space offset

        title_size : float
            size of title wrt numeric labels

        title_rotation : float
            title rotation in degrees

        nlabels : int
            number of numeric labels

        label_font : str
            font type for labels

        label_size : float
            label scale factor

        label_offset : float
            space btw numeric labels and scale

        label_rotation : float
            label rotation in degrees

        label_format : str
            label format for floats and integers (e.g. ':.2f')

        draw_box : bool
            draw a box around the colorbar

        categories : list
            make a categorical scalarbar,
            the input list will have the format [value, color, alpha, textlabel]

        .. hint:: examples/basic/scalarbars.py
        """
        plt = vedo.plotter_instance
        if plt and c is None:  # automatic black or white
            c = (0.9, 0.9, 0.9)
            if np.sum(vedo.get_color(plt.backgrcol)) > 1.5:
                c = (0.1, 0.1, 0.1)
        if c is None:
            c = (0, 0, 0)
        c = vedo.get_color(c)

        self.scalarbar = vedo.addons.ScalarBar3D(
            self,
            title,
            pos,
            s,
            title_font,
            title_xoffset,
            title_yoffset,
            title_size,
            title_rotation,
            nlabels,
            label_font,
            label_size,
            label_offset,
            label_rotation,
            label_format,
            italic,
            c,
            draw_box,
            above_text,
            below_text,
            nan_text,
            categories,
        )
        return self

    ###################################################################################
    def write(self, filename, binary=True):
        """Write object to file."""
        return vedo.io.write(self, filename, binary)


########################################################################################
class BaseGrid(BaseActor):
    """Base class for grid datasets"""
    def __init__(self):

        BaseActor.__init__(self)

        self._data = None
        self.useCells = True
        self._color = None
        self._alpha = [0,1]

        # -----------------------------------------------------------

    def _update(self, data):
        self._data = data
        self._mapper.SetInputData(self.tomesh().polydata())
        self._mapper.Modified()
        return self

    def tomesh(self, fill=True, shrink=1.0):
        """
        Build a polygonal Mesh from the current Grid object.

        If ``fill=True``, the interior faces of all the cells are created.
        (setting a ``shrink`` value slightly smaller than the default 1.0
        can avoid flickering due to internal adjacent faces).

        If ``fill=False``, only the boundary faces will be generated.
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
            if shrink == 1.0:
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
            msh.mapper().SetLookupTable(lut)
        if self.useCells:
            msh.mapper().SetScalarModeToUseCellData()
        else:
            msh.mapper().SetScalarModeToUsePointData()
        # msh.mapper().SetScalarRange(msh.mapper().GetScalarRange())
        # print(msh.mapper().GetScalarRange(), lut.GetRange())
        # msh.mapper().SetScalarRange()
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

        # Get cell connettivity ids as a 1D array. vtk format is:
        # [nids1, id0 ... idn, niids2, id0 ... idm,  etc].
        i = 0
        conn = []
        n = len(arr1d)
        if n:
            while True:
                cell = [arr1d[i + k] for k in range(1, arr1d[i] + 1)]
                conn.append(cell)
                i += arr1d[i] + 1
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

        You can also assign a specific color to a aspecific value with eg.:

        ``volume.color([(0,'red', (0.5,'violet'), (1,'green')])``

        Parameters
        ----------
        alpha : list
            use a list to specify transparencies along the scalar range

        vmin : float
            force the min of the scalar range to be this value

        vmax : float
            force the max of the scalar range to be this value
        """
        # superseeds method in Points, Mesh
        if vmin is None:
            vmin, _ = self._data.GetScalarRange()
        if vmax is None:
            _, vmax = self._data.GetScalarRange()
        ctf = self.GetProperty().GetRGBTransferFunction()
        ctf.RemoveAllPoints()
        self._color = col

        if utils.is_sequence(col):
            if utils.is_sequence(col[0]) and len(col[0]) == 2:
                # user passing [(value1, color1), ...]
                for x, ci in col:
                    r, g, b = colors.get_color(ci)
                    ctf.AddRGBPoint(x, r, g, b)
                    # colors.printc('color at', round(x, 1),
                    #               'set to', colors.get_color_name((r, g, b)),
                    #               c='w', bold=0)
            else:
                # user passing [color1, color2, ..]
                for i, ci in enumerate(col):
                    r, g, b = colors.get_color(ci)
                    x = vmin + (vmax - vmin) * i / (len(col) - 1)
                    ctf.AddRGBPoint(x, r, g, b)
        elif isinstance(col, str):
            if col in colors.colors.keys() or col in colors.color_nicks.keys():
                r, g, b = colors.get_color(col)
                ctf.AddRGBPoint(vmin, r, g, b)  # constant color
                ctf.AddRGBPoint(vmax, r, g, b)
            else:  # assume it's a colormap
                for x in np.linspace(vmin, vmax, num=64, endpoint=True):
                    r, g, b = colors.color_map(x, name=col, vmin=vmin, vmax=vmax)
                    ctf.AddRGBPoint(x, r, g, b)
        elif isinstance(col, int):
            r, g, b = colors.get_color(col)
            ctf.AddRGBPoint(vmin, r, g, b)  # constant color
            ctf.AddRGBPoint(vmax, r, g, b)
        else:
            vedo.logger.warning(f"in color() unknown input type {type(col)}")

        if alpha is not None:
            self.alpha(alpha, vmin=vmin, vmax=vmax)
        return self

    def alpha(self, alpha, vmin=None, vmax=None):
        """
        Assign a set of tranparencies along the range of the scalar value.
        A single constant value can also be assigned.

        E.g.: say `alpha=(0.0, 0.3, 0.9, 1)` and the scalar range goes from -10 to 150.
        Then all cells with a value close to -10 will be completely transparent, cells at 1/4
        of the range will get an alpha equal to 0.3 and voxels with value close to 150
        will be completely opaque.

        As a second option one can set explicit (x, alpha_x) pairs to define the transfer function.

        E.g.: say `alpha=[(-5, 0), (35, 0.4) (123,0.9)]` and the scalar range goes from -10 to 150.
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

        if utils.is_sequence(alpha):
            alpha = np.array(alpha)
            if len(alpha.shape)==1: # user passing a flat list e.g. (0.0, 0.3, 0.9, 1)
                for i, al in enumerate(alpha):
                    xalpha = vmin + (vmax - vmin) * i / (len(alpha) - 1)
                    # Create transfer mapping scalar value to opacity
                    otf.AddPoint(xalpha, al)
                    # colors.printc("alpha at", round(xalpha, 1), "\tset to", al)
            elif len(alpha.shape) == 2:  # user passing [(x0,alpha0), ...]
                otf.AddPoint(vmin, alpha[0][1])
                for xalpha, al in alpha:
                    # Create transfer mapping scalar value to opacity
                    otf.AddPoint(xalpha, al)
                otf.AddPoint(vmax, alpha[-1][1])

        else:

            otf.AddPoint(vmin, alpha)  # constant alpha
            otf.AddPoint(vmax, alpha)

        return self

    def alpha_unit(self, u=None):
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
        self.GetProperty().SetScalarOpacityUnitDistance(u)
        return self

    def shrink(self, fraction=0.8):
        """Shrink the individual cells to improve visibility."""
        sf = vtk.vtkShrinkFilter()
        sf.SetInputData(self.inputdata())
        sf.SetShrinkFactor(fraction)
        sf.Update()
        return self._update(sf.GetOutput())

    def isosurface(self, value=None, flying_edges=True):
        """
        Return an ``Mesh`` isosurface extracted from the ``Volume`` object.

        Set ``value`` as single float or list of values to draw the isosurface(s).
        Use flying_edges for faster results (but sometimes can interfere with `smooth()`).

        .. hint:: examples/volumetric/isosurfaces.py
        """
        scrange = self._data.GetScalarRange()

        if flying_edges:
            cf = vtk.vtkFlyingEdges3D()
            cf.InterpolateAttributesOn()
        else:
            cf = vtk.vtkContourFilter()
            cf.UseScalarTreeOn()

        cf.SetInputData(self._data)
        cf.ComputeNormalsOn()

        if utils.is_sequence(value):
            cf.SetNumberOfContours(len(value))
            for i, t in enumerate(value):
                cf.SetValue(i, t)
        else:
            if value is None:
                value = (2 * scrange[0] + scrange[1]) / 3.0
            cf.SetValue(0, value)

        cf.Update()
        poly = cf.GetOutput()

        # pdnorm = vtk.vtkPolyDataNormals() # no effect
        # pdnorm.SetInputData(poly)
        # pdnorm.SetComputePointNormals(True)
        # pdnorm.SetComputeCellNormals(True)
        # pdnorm.SetConsistency(True)
        # pdnorm.FlipNormalsOff()
        # pdnorm.SetSplitting(False)
        # pdnorm.Update()
        # poly = pdnorm.GetOutput()

        a = vedo.mesh.Mesh(poly, c=None).phong()
        a.mapper().SetScalarRange(scrange[0], scrange[1])
        return a

    def legosurface(self, vmin=None, vmax=None, invert=False, boundary=False):
        """
        Represent an object - typically a ``Volume`` - as lego blocks (voxels).
        By default colors correspond to the volume's scalar.
        Returns an ``Mesh`` object.

        Parameters
        ----------
        vmin : float
            the lower threshold, voxels below this value are not shown.

        vmax : float
            the upper threshold, voxels above this value are not shown.

        boundary : bool
            controls whether to include cells that are partially inside

        .. hint:: examples/volumetric/legosurface.py
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
        tol = 0.00001 * (srng[1] - srng[0])
        srng[0] -= tol
        srng[1] += tol
        window.SetWindowRange(srng)

        extract = vtk.vtkExtractGeometry()
        extract.SetInputData(self._data)
        extract.SetImplicitFunction(window)
        extract.SetExtractInside(invert)
        extract.SetExtractBoundaryCells(boundary)
        extract.Update()

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(extract.GetOutput())
        gf.Update()

        a = vedo.mesh.Mesh(gf.GetOutput()).lw(0.1).flat()  # .lighting('ambient')
        scalars = a.pointdata[0]
        if scalars is None:
            # print("Error in legosurface(): no scalars found!")
            return a
        a.map_points_to_cells()
        return a

    @deprecated(reason=vedo.colors.red + "Please use cut_with_plane()" + vedo.colors.reset)
    def cutWithPlane(self, origin=(0, 0, 0), normal=(1, 0, 0)):
        """Deprecated. Please use cut_with_plane()"""
        return self.cut_with_plane(origin, normal)

    def cut_with_plane(self, origin=(0, 0, 0), normal="x"):
        """
        Cut the object with the plane defined by a point and a normal.

        Parameters
        ----------
        origin : list
            the cutting plane goes through this point

        normal : list, str
            normal vector to the cutting plane
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

    def cut_with_box(self, box):
        """
        Cut the grid with the specified bounding box.

        Parameter box has format [xmin, xmax, ymin, ymax, zmin, zmax].
        If an object is passed, its bounding box are used.

        Example:
            .. code-block:: python

                from vedo import *
                tetmesh = TetMesh(dataurl+'limb_ugrid.vtk')
                tetmesh.color('rainbow')
                cu = Cube(side=500).x(500) # any Mesh works
                tetmesh.cut_with_box(cu).show(axes=1)
        """
        bc = vtk.vtkBoxClipDataSet()
        bc.SetInputData(self._data)
        if isinstance(box, vtk.vtkProp):
            box = box.bounds()
        bc.SetBoxClip(*box)
        bc.Update()
        cout = bc.GetOutput()
        return self._update(cout)

    def cut_with_mesh(self, mesh, invert=False, whole_cells=False, only_boundary=False):
        """
        Cut a UGrid, TetMesh or Volume with a Mesh.

        Use ``invert`` to return cut off part of the input object.
        """
        polymesh = mesh.polydata()
        ug = self._data

        ippd = vtk.vtkImplicitPolyDataDistance()
        ippd.SetInput(polymesh)

        if whole_cells or only_boundary:
            clipper = vtk.vtkExtractGeometry()
            clipper.SetInputData(ug)
            clipper.SetImplicitFunction(ippd)
            clipper.SetExtractInside(not invert)
            clipper.SetExtractBoundaryCells(False)
            if only_boundary:
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

        if ug.GetCellData().GetScalars():  # not working
            scalname = ug.GetCellData().GetScalars().GetName()
            if scalname:  # not working
                if self.useCells:
                    self.celldata.select(scalname)
                else:
                    self.pointdata.select(scalname)

        self._update(cug)
        return self

    def extract_cells_on_plane(self, origin, normal):
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtk.vtk3DLinearGridCrinkleExtractor()
        bf.SetInputData(self._data)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        bf.SetImplicitFunction(plane)

        bf.Update()
        return self._update(bf.GetOutput())

    def extract_cells_on_sphere(self, center, radius):
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtk.vtk3DLinearGridCrinkleExtractor()
        bf.SetInputData(self._data)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        sph = vtk.vtkSphere()
        sph.SetRadius(radius)
        sph.SetCenter(center)
        bf.SetImplicitFunction(sph)

        bf.Update()
        return self._update(bf.GetOutput())

    def extract_cells_on_cylinder(self, center, axis, radius):
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtk.vtk3DLinearGridCrinkleExtractor()
        bf.SetInputData(self._data)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        cyl = vtk.vtkCylinder()
        cyl.SetRadius(radius)
        cyl.SetCenter(center)
        cyl.SetAxis(axis)
        bf.SetImplicitFunction(cyl)

        bf.Update()
        return self._update(bf.GetOutput())

    def clean(self):
        """
        Cleanup unused points and empty cells
        """
        cl = vtk.vtkStaticCleanUnstructuredGrid()
        cl.SetInputData(self._data)
        cl.RemoveUnusedPointsOn()
        cl.ProduceMergeMapOff()
        cl.AveragePointDataOff()
        cl.Update()
        return self._update(cl.GetOutput())


    def find_cell(self, p):
        """Locate the cell that contains a point and return the cell ID."""
        cell = vtk.vtkTetra()
        cellId = vtk.mutable(0)
        tol2 = vtk.mutable(0)
        subId = vtk.mutable(0)
        pcoords = [0,0,0]
        weights = [0,0,0]
        cid = self._data.FindCell(p, cell, cellId, tol2, subId, pcoords, weights)
        return cid


    def extract_cells_by_id(self, idlist, use_point_ids=False):
        """Return a new UGrid composed of the specified subset of indices."""
        selectionNode = vtk.vtkSelectionNode()
        if use_point_ids:
            selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
            contcells = vtk.vtkSelectionNode.CONTAINING_CELLS()
            selectionNode.GetProperties().Set(contcells, 1)
        else:
            selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        vidlist = utils.numpy2vtk(idlist, dtype="id")
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

        # assign the same transformation to the copy
        tm_sel.SetOrigin(self.GetOrigin())
        tm_sel.SetScale(self.GetScale())
        tm_sel.SetOrientation(self.GetOrientation())
        tm_sel.SetPosition(self.GetPosition())
        tm_sel.mapper().SetLookupTable(utils.ctf2lut(self))
        return tm_sel


############################################################################### funcs
def _getinput(obj):
    if isinstance(obj, (vtk.vtkVolume, vtk.vtkActor)):
        return obj.GetMapper().GetInput()
    return obj


def probe_points(dataset, pts):
    """
    Takes a ``Volume`` (or any other vtk data set)
    and probes its scalars at the specified points in space.

    Note that a mask is also output with valid/invalid points which can be accessed
    with `mesh.pointdata['vtkValidPointMask']`.

    .. hint:: examples/volumetric/probePoints.py
    """
    if isinstance(pts, vedo.pointcloud.Points):
        pts = pts.points()

    def _readpoints():
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
    src.SetExecuteMethod(_readpoints)
    src.Update()
    img = _getinput(dataset)
    probeFilter = vtk.vtkProbeFilter()
    probeFilter.SetSourceData(img)
    probeFilter.SetInputConnection(src.GetOutputPort())
    probeFilter.Update()
    poly = probeFilter.GetOutput()
    pm = vedo.mesh.Mesh(poly)
    pm.name = "ProbePoints"
    return pm


def probe_line(dataset, p1, p2, res=100):
    """
    Takes a ``Volume``  (or any other vtk data set)
    and probes its scalars along a line defined by 2 points `p1` and `p2`.

    Note that a mask is also output with valid/invalid points which can be accessed
    with `mesh.pointdata['vtkValidPointMask']`.

    Use ``res`` to set the nr of points along the line

    .. hint:: examples/volumetric/probeLine1.py, examples/volumetric/probeLine2.py
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
    lnn.name = "ProbeLine"
    return lnn


def probe_plane(dataset, origin=(0, 0, 0), normal=(1, 0, 0)):
    """
    Takes a ``Volume`` (or any other vtk data set)
    and probes its scalars on a plane defined by a point and a normal.

    .. hint:: examples/volumetric/slicePlane1.py, examples/volumetric/slicePlane2.py
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
    cutmesh.name = "ProbePlane"
    return cutmesh


###################################################################################
# def extract_cells_by_type(obj, types=(7,)):    ### VTK9 only
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
