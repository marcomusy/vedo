#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

import vedo
from vedo import colors
from vedo import utils
from vedo.transformations import LinearTransform


__docformat__ = "google"

__doc__ = "Base classes. Do not instantiate."

__all__ = [
    "Base3DProp",
    "BaseActor",
    "BaseActor2D",
    "BaseGrid",
    "probe_points",
    "probe_line",
    "probe_plane",
]


###############################################################################
class _DataArrayHelper:
    # Internal use only.
    # Helper class to manage data associated to either
    # points (or vertices) and cells (or faces).
    def __init__(self, obj, association):
        self.obj = obj
        self.association = association

    def __getitem__(self, key):

        if self.association == 0:
            data = self.obj.GetPointData()

        elif self.association == 1:
            data = self.obj.GetCellData()

        elif self.association == 2:
            data = self.obj.GetFieldData()

            varr = data.GetAbstractArray(key)
            if isinstance(varr, vtk.vtkStringArray):
                if isinstance(key, int):
                    key = data.GetArrayName(key)
                n = varr.GetNumberOfValues()
                narr = [varr.GetValue(i) for i in range(n)]
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
            data = self.obj.GetPointData()
            n = self.obj.GetNumberOfPoints()
            self.obj.mapper.SetScalarModeToUsePointData()

        elif self.association == 1:
            data = self.obj.GetCellData()
            n = self.obj.GetNumberOfCells()
            self.obj.mapper.SetScalarModeToUseCellData()

        elif self.association == 2:
            data = self.obj.GetFieldData()
            if not utils.is_sequence(input_array):
                input_array = [input_array]

            if isinstance(input_array[0], str):
                varr = vtk.vtkStringArray()
                varr.SetName(key)
                varr.SetNumberOfComponents(1)
                varr.SetNumberOfTuples(len(input_array))
                for i, iarr in enumerate(input_array):
                    if isinstance(iarr, np.ndarray):
                        iarr = iarr.tolist()  # better format
                        # Note: a string k can be converted to numpy with
                        # import json; k = np.array(json.loads(k))
                    varr.InsertValue(i, str(iarr))
            else:
                try:
                    varr = utils.numpy2vtk(input_array, name=key)
                except TypeError as e:
                    vedo.logger.error(
                        f"cannot create metadata with input object:\n"
                        f"{input_array}"
                        f"\n\nAllowed content examples are:\n"
                        f"- flat list of strings ['a','b', 1, [1,2,3], ...]"
                        f" (first item must be a string in this case)\n"
                        f"  hint: use k = np.array(json.loads(k)) to convert strings\n"
                        f"- numpy arrays of any shape"
                    )
                    raise e

            data.AddArray(varr)
            return  ############

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
            data = self.obj.GetPointData()
        elif self.association == 1:
            data = self.obj.GetCellData()
        elif self.association == 2:
            data = self.obj.GetFieldData()
        arrnames = []
        for i in range(data.GetNumberOfArrays()):
            name = data.GetArray(i).GetName()
            if name:
                arrnames.append(name)
        return arrnames

    def remove(self, key):
        """Remove a data array by name or number"""
        if self.association == 0:
            self.obj.GetPointData().RemoveArray(key)
        elif self.association == 1:
            self.obj.GetCellData().RemoveArray(key)
        elif self.association == 2:
            self.obj.GetFieldData().RemoveArray(key)

    def clear(self):
        """Remove all data associated to this object"""
        if self.association == 0:
            data = self.obj.GetPointData()
        elif self.association == 1:
            data = self.obj.GetCellData()
        elif self.association == 2:
            data = self.obj.GetFieldData()
        for i in range(data.GetNumberOfArrays()):
            name = data.GetArray(i).GetName()
            data.RemoveArray(name)

    def rename(self, oldname, newname):
        """Rename an array"""
        if self.association == 0:
            varr = self.obj.GetPointData().GetArray(oldname)
        elif self.association == 1:
            varr = self.obj.GetCellData().GetArray(oldname)
        elif self.association == 2:
            varr = self.obj.GetFieldData().GetArray(oldname)
        if varr:
            varr.SetName(newname)
        else:
            vedo.logger.warning(f"Cannot rename non existing array {oldname} to {newname}")

    def select(self, key):
        """Select one specific array by its name to make it the `active` one."""
        if self.association == 0:
            data = self.obj.GetPointData()
            self.obj.mapper.SetScalarModeToUsePointData()
        else:
            data = self.obj.GetCellData()
            self.obj.mapper.SetScalarModeToUseCellData()

        if isinstance(key, int):
            key = data.GetArrayName(key)

        arr = data.GetArray(key)
        if not arr:
            return

        nc = arr.GetNumberOfComponents()
        if nc == 1:
            data.SetActiveScalars(key)
        elif nc >= 2:
            if "rgb" in key.lower():
                data.SetActiveScalars(key)
                # try:
                #     self.mapper.SetColorModeToDirectScalars()
                # except AttributeError:
                #     pass
            else:
                data.SetActiveVectors(key)
        elif nc >= 4:
            data.SetActiveTensors(key)

        try:
            self.obj.mapper.SetArrayName(key)
            self.obj.mapper.ScalarVisibilityOn()
            # .. could be a volume mapper
        except AttributeError:
            pass

    def select_scalars(self, key):
        """Select one specific scalar array by its name to make it the `active` one."""
        if self.association == 0:
            data = self.obj.GetPointData()
            self.obj.mapper.SetScalarModeToUsePointData()
        else:
            data = self.obj.GetCellData()
            self.obj.mapper.SetScalarModeToUseCellData()

        if isinstance(key, int):
            key = data.GetArrayName(key)

        data.SetActiveScalars(key)

        try:
            self.obj.mapper.SetArrayName(key)
            self.obj.mapper.ScalarVisibilityOn()
        except AttributeError:
            pass

    def select_vectors(self, key):
        """Select one specific vector array by its name to make it the `active` one."""
        if self.association == 0:
            data = self.obj.GetPointData()
            self.obj.mapper.SetScalarModeToUsePointData()
        else:
            data = self.obj.GetCellData()
            self.obj.mapper.SetScalarModeToUseCellData()

        if isinstance(key, int):
            key = data.GetArrayName(key)

        data.SetActiveVectors(key)

        try:
            self.obj.mapper.SetArrayName(key)
            self.obj.mapper.ScalarVisibilityOn()
        except AttributeError:
            pass

    def print(self, **kwargs):
        """Print the array names available to terminal"""
        colors.printc(self.keys(), **kwargs)

    def __repr__(self) -> str:
        """Representation"""

        def _get_str(pd, header):
            if pd.GetNumberOfArrays():
                out = f"\x1b[2m\x1b[1m\x1b[7m{header}"
                if self.obj.name:
                    out += f" in {self.obj.name}"
                out += f" contains {pd.GetNumberOfArrays()} array(s)\x1b[0m"
                for i in range(pd.GetNumberOfArrays()):
                    varr = pd.GetArray(i)
                    out += f"\n\x1b[1m\x1b[4mArray name    : {varr.GetName()}\x1b[0m"
                    out += "\nindex".ljust(15) + f": {i}"
                    t = varr.GetDataType()
                    if t in vedo.utils.array_types:
                        out += f"\ntype".ljust(15)
                        out += f": {vedo.utils.array_types[t][1]} ({vedo.utils.array_types[t][0]})"
                    shape = (varr.GetNumberOfTuples(), varr.GetNumberOfComponents())
                    out += "\nshape".ljust(15) + f": {shape}"
                    out += "\nrange".ljust(15) + f": {np.array(varr.GetRange())}"
                    out += "\nmax id".ljust(15) + f": {varr.GetMaxId()}"
                    out += "\nlook up table".ljust(15) + f": {bool(varr.GetLookupTable())}"
                    out += "\nin-memory size".ljust(15) + f": {varr.GetActualMemorySize()} KB"
            else:
                out += " has no associated data."
            return out

        if self.association == 0:
            out = _get_str(self.GetPointData(), "Point Data")
        elif self.association == 1:
            out = _get_str(self.GetCellData(), "Cell Data")
        elif self.association == 2:
            pd = self.GetFieldData()
            if pd.GetNumberOfArrays():
                out = f"\x1b[2m\x1b[1m\x1b[7mMeta Data"
                if self.actor.name:
                    out += f" in {self.actor.name}"
                out += f" contains {pd.GetNumberOfArrays()} entries\x1b[0m"
                for i in range(pd.GetNumberOfArrays()):
                    varr = pd.GetAbstractArray(i)
                    out += f"\n\x1b[1m\x1b[4mEntry name    : {varr.GetName()}\x1b[0m"
                    out += "\nindex".ljust(15) + f": {i}"
                    shape = (varr.GetNumberOfTuples(), varr.GetNumberOfComponents())
                    out += "\nshape".ljust(15) + f": {shape}"

        return out


###############################################################################
class Base3DProp:
    """
    Base class to manage positioning and size of the objects in space and other properties.

    .. warning:: Do not use this class to instantiate objects
    """

    def __init__(self):
        """
        Base class to manage positioning and size of the objects in space and other properties.
        """
        self.filename = ""
        self.name = ""
        self.file_size = ""
        self.trail = None
        self.trail_points = []
        self.trail_segment_size = 0
        self.trail_offset = None
        self.shadows = []
        self.axes = None
        self.picked3d = None

        self.top  = np.array([0, 0, 1])
        self.base = np.array([0, 0, 0])
        self.info = {}
        self.time = time.time()
        self.rendered_at = set()
        self.transform = LinearTransform()
        self._isfollower = False  # set by mesh.follow_camera()

        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None

        self.scalarbar = None
        # self.scalarbars = dict() #TODO
        self.pipeline = None

    def memory_address(self):
        """
        Return a unique memory address integer which may serve as the ID of the
        object, or passed to c++ code.
        """
        # https://www.linkedin.com/pulse/speedup-your-code-accessing-python-vtk-objects-from-c-pletzer/
        # https://github.com/tfmoraes/polydata_connectivity
        return int(self.GetAddressAsString("")[5:], 16)

    def pickable(self, value=None):
        """Set/get the pickability property of an object."""
        if value is None:
            return self.actor.GetPickable()
        self.actor.SetPickable(value)
        return self

    def draggable(self, value=None):  # NOT FUNCTIONAL?
        """Set/get the draggability property of an object."""
        if value is None:
            return self.GetDragable()
        self.actor.SetDragable(value)
        return self


    def apply_transform(self, LT, concatenate=True, deep_copy=True):
        """
        Apply a linear or non-linear transformation to the mesh polygonal data.
            ```python
            from vedo import Cube, show
            c1 = Cube().rotate_z(5).x(2).y(1)
            print("cube1 position", c1.pos())
            T = c1.get_transform()  # rotate by 5 degrees, sum 2 to x and 1 to y
            c2 = Cube().c('r4')
            c2.apply_transform(T)   # ignore previous movements
            c2.apply_transform(T, concatenate=True)
            c2.apply_transform(T, concatenate=True)
            print("cube2 position", c2.pos())
            show(c1, c2, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/apply_transform.png)
        """
        if isinstance(LT, LinearTransform):
            tr = LT.T
            if concatenate:
                self.transform.concatenate(LT)
        elif isinstance(LT, (vtk.vtkMatrix4x4, vtk.vtkTransform, np.ndarray)):
            LT = LinearTransform(LT)
            if LT.is_identity():
                return self
            tr = LT.T
            if concatenate:
                self.transform.concatenate(LT)
        elif isinstance(LT, vtk.vtkThinPlateSplineTransform):
            tr = LT

        tp = vtk.vtkTransformPolyDataFilter()
        tp.SetTransform(tr)
        tp.SetInputData(self)
        tp.Update()
        out = tp.GetOutput()

        if deep_copy:
            self.DeepCopy(out)
        else:
            self.ShallowCopy(out)

        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None
        return self
    

    def pos(self, x=None, y=None, z=None):
        """Set/Get object position."""
        if x is None:  # get functionality
            return self.transform.position

        if z is None and y is None:  # assume x is of the form (x,y,z)
            if len(x) == 3:
                x, y, z = x
            else:
                x, y = x
                z = 0
        elif z is None:  # assume x,y is of the form x, y
            z = 0

        q = self.transform.position
        LT = LinearTransform()
        LT.translate([x,y,z]-q) 
        return self.apply_transform(LT)

    def shift(self, dx=0, dy=0, dz=0):
        """Add a vector to the current object position."""
        if utils.is_sequence(dx):
            utils.make3d(dx)
            dx, dy, dz = dx
        LT = LinearTransform().translate([dx, dy, dz]) 
        return self.apply_transform(LT)

    def x(self, val=None):
        """Set/Get object position along x axis."""
        p = self.transform.position
        if val is None:
            return p[0]
        self.pos(val, p[1], p[2])
        return self

    def y(self, val=None):
        """Set/Get object position along y axis."""
        p = self.transform.position
        if val is None:
            return p[1]
        self.pos(p[0], val, p[2])
        return self

    def z(self, val=None):
        """Set/Get object position along z axis."""
        p = self.transform.position
        if val is None:
            return p[2]
        self.pos(p[0], p[1], val)
        return self

    def rotate(self, angle, axis=(1, 0, 0), point=(0, 0, 0), rad=False):
        """
        Rotate around an arbitrary `axis` passing through `point`.

        Example:
            ```python
            from vedo import *
            c1 = Cube()
            c2 = c1.clone().c('violet').alpha(0.5) # copy of c1
            v = vector(0.2,1,0)
            p = vector(1,0,0)  # axis passes through this point
            c2.rotate(90, axis=v, point=p)
            l = Line(-v+p, v+p).lw(3).c('red')
            show(c1, l, c2, axes=1).close()
            ```
            ![](https://vedo.embl.es/images/feats/rotate_axis.png)
        """
        # self.rotate(angle, axis, point, rad)
        LT = LinearTransform()
        LT.rotate(angle, axis, point, rad)
        return self.apply_transform(LT)

    def rotate_x(self, angle, rad=False, around=None):
        """
        Rotate around x-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        LT = LinearTransform().rotate_x(angle, rad, around)
        return self.apply_transform(LT)

    def rotate_y(self, angle, rad=False, around=None):
        """
        Rotate around y-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        LT = LinearTransform().rotate_y(angle, rad, around)
        return self.apply_transform(LT)

    def rotate_z(self, angle, rad=False, around=None):
        """
        Rotate around z-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        LT = LinearTransform().rotate_z(angle, rad, around)
        return self.apply_transform(LT)

    #TODO
    def orientation(self, newaxis=None, rotation=0, concatenate=False, xyplane=False, rad=False):
        return self


    def scale(self, s=None, reset=False, origin=True):
        """
        Set/get object's scaling factor.

        Arguments:
            s : (list, float)
                scaling factor(s).
            reset : (bool)
                if True previous scaling factors are ignored.
            origin : (bool)
                if True scaling is applied with respect to object's position,
                otherwise is applied respect to (0,0,0).

        Note:
            use `s=(sx,sy,sz)` to scale differently in the three coordinates.
        """
        if s is None:
            return np.array(self.transform.T.GetScale())
        
        LT = LinearTransform()
        if reset:
            LT.set_scale(s)
        else:
            if origin:
                LT.scale(s, origin=self.transform.position)
            else:
                LT.scale(s, origin=False)
        return self.apply_transform(LT)


    def align_to_bounding_box(self, msh, rigid=False):
        """
        Align the current object's bounding box to the bounding box
        of the input object.

        Use `rigid` to disable scaling.

        Examples:
            - [align6.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/align6.py)
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

        LT = LinearTransform(lmt)
        self.apply_transform(LT)
        return self

    def on(self):
        """Switch on  object visibility. Object is not removed."""
        self.VisibilityOn()
        try:
            self.scalarbar.VisibilityOn()
        except AttributeError:
            pass
        try:
            self.trail.VisibilityOn()
        except AttributeError:
            pass
        try:
            for sh in self.shadows:
                sh.VisibilityOn()
        except AttributeError:
            pass
        return self

    def off(self):
        """Switch off object visibility. Object is not removed."""
        self.VisibilityOff()
        try:
            self.scalarbar.VisibilityOff()
        except AttributeError:
            pass
        try:
            self.trail.VisibilityOff()
        except AttributeError:
            pass
        try:
            for sh in self.shadows:
                sh.VisibilityOff()
        except AttributeError:
            pass
        return self

    def toggle(self):
        """Toggle object visibility on/off."""
        v = self.GetVisibility()
        if v:
            self.off()
        else:
            self.on()
        return self

    def box(self, scale=1, padding=0, fill=False):
        """
        Return the bounding box as a new `Mesh`.

        Arguments:
            scale : (float)
                box size can be scaled by a factor
            padding : (float, list)
                a constant padding can be added (can be a list [padx,pady,padz])

        Examples:
            - [latex.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/latex.py)
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
            width * scale + padding[1],
            height * scale + padding[2],
            c="gray",
        )
        try:
            pr = vtk.vtkProperty()
            pr.DeepCopy(self.GetProperty())
            bx.SetProperty(pr)
            bx.property = pr
        except (AttributeError, TypeError):
            pass
        bx.wireframe(not fill)
        bx.flat().lighting("off")
        return bx

    def use_bounds(self, value=True):
        """
        Instruct the current camera to either take into account or ignore
        the object bounds when resetting.
        """
        self.actor.SetUseBounds(value)
        return self

    def bounds(self):
        """
        Get the object bounds.
        Returns a list in format `[xmin,xmax, ymin,ymax, zmin,zmax]`.
        """
        try:
            pts = self.points()
            xmin, ymin, zmin = np.min(pts, axis=0)
            xmax, ymax, zmax = np.max(pts, axis=0)
            return [xmin, xmax, ymin, ymax, zmin, zmax]
        except (AttributeError, ValueError):
            return self.GetBounds()

    def xbounds(self, i=None):
        """Get the bounds `[xmin,xmax]`. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i is not None:
            return b[i]
        return (b[0], b[1])

    def ybounds(self, i=None):
        """Get the bounds `[ymin,ymax]`. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i == 0:
            return b[2]
        if i == 1:
            return b[3]
        return (b[2], b[3])

    def zbounds(self, i=None):
        """Get the bounds `[zmin,zmax]`. Can specify upper or lower with i (0,1)."""
        b = self.bounds()
        if i == 0:
            return b[4]
        if i == 1:
            return b[5]
        return (b[4], b[5])


    def diagonal_size(self):
        """Get the length of the diagonal of mesh bounding box."""
        b = self.bounds()
        return np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)
        # return self.GetLength() # ???different???


    def copy_data_from(self, obj):
        """Copy all data (point and cell data) from this input object"""
        self.GetPointData().PassData(obj.GetPointData())
        self.GetCellData().PassData(obj.GetCellData())
        self.pipeline = utils.OperationNode(
            f"copy_data_from\n{obj.__class__.__name__}",
            parents=[self, obj],
            shape="note",
            c="#ccc5b9",
        )
        return self

    def print(self):
        """Print information about an object."""
        utils.print_info(self)
        return self

    def show(self, **options):
        """
        Create on the fly an instance of class `Plotter` or use the last existing one to
        show one single object.

        This method is meant as a shortcut. If more than one object needs to be visualised
        please use the syntax `show(mesh1, mesh2, volume, ..., options)`.

        Returns the `Plotter` class instance.
        """
        return vedo.plotter.show(self, **options)

    def add_observer(self, event_name, func, priority=0):
        """Add a callback function that will be called when an event occurs."""
        event_name = utils.get_vtk_name_event(event_name)
        idd = self.AddObserver(event_name, func, priority)
        return idd

    def thumbnail(self, zoom=1.25, size=(200, 200), bg="white", azimuth=0, elevation=0, axes=False):
        """Build a thumbnail of the object and return it as an array."""
        # speed is about 20Hz for size=[200,200]
        ren = vtk.vtkRenderer()
        ren.AddActor(self)
        if axes:
            axes = vedo.addons.Axes(self)
            ren.AddActor(axes)
        ren.ResetCamera()
        cam = ren.GetActiveCamera()
        cam.Zoom(zoom)
        cam.Elevation(elevation)
        cam.Azimuth(azimuth)

        ren_win = vtk.vtkRenderWindow()
        ren_win.SetOffScreenRendering(True)
        ren_win.SetSize(size)
        ren.SetBackground(colors.get_color(bg))
        ren_win.AddRenderer(ren)
        ren_win.Render()

        nx, ny = ren_win.GetSize()
        arr = vtk.vtkUnsignedCharArray()
        ren_win.GetRGBACharPixelData(0, 0, nx - 1, ny - 1, 0, arr)
        narr = utils.vtk2numpy(arr).T[:3].T.reshape([ny, nx, 3])
        narr = np.ascontiguousarray(np.flip(narr, axis=0))

        ren.RemoveActor(self)
        if axes:
            ren.RemoveActor(axes)
        ren_win.Finalize()
        del ren_win
        return narr


########################################################################################
class BaseActor(Base3DProp):
    """
    Base class.

    .. warning:: Do not use this class to instantiate objects, use one the above instead.
    """

    def __init__(self):
        """
        Base class to add operative and data
        functionality to `Mesh`, `Assembly`, `Volume` and `Picture` objects.
        """

        super().__init__()

        self.mapper = None
        self._caption = None
        self.property = None
        self.mapper = None


    def inputdata(self):
        """Obsolete, use `self` instead."""
        # """Return the VTK input data object."""
        # if self.mapper:
        #     return self.mapper.GetInput()
        # return self.GetMapper().GetInput()
        return self

    # def modified(self):
    #     """Use in conjunction with `tonumpy()`
    #     to update any modifications to the volume array"""
    #     sc = self.GetPointData().GetScalars()
    #     if sc:
    #         sc.Modified()
    #     self.GetPointData().Modified()
    #     return self

    @property
    def npoints(self):
        """Retrieve the number of points."""
        return self.GetNumberOfPoints()

    @property
    def ncells(self):
        """Retrieve the number of cells."""
        return self.GetNumberOfCells()

    def points(self, pts=None, transformed=True):
        """
        Set/Get the vertex coordinates of a mesh or point cloud.
        Keyword `pts` can also be a list of indices to be retrieved.

        Set `transformed=False` to ignore any previous transformation applied to the mesh.
        """
        if pts is None:  ### getter

            if isinstance(self, vedo.Points):
                vpts = self.GetPoints()
            elif isinstance(self, vedo.BaseVolume):
                v2p = vtk.vtkImageToPoints()
                v2p.SetInputData(self.imagedata())
                v2p.Update()
                vpts = v2p.GetOutput().GetPoints()
            else:  # tetmesh et al
                vpts = self.GetPoints()

            if vpts:
                return utils.vtk2numpy(vpts.GetData())
            return np.array([], dtype=np.float32)

        else:

            pts = np.asarray(pts, dtype=np.float32)

            if pts.ndim == 1:
                ### getter by point index ###################
                indices = pts.astype(int)
                vpts = self.GetPoints()
                arr = utils.vtk2numpy(vpts.GetData())
                return arr[indices] ###########

            ### setter ####################################
            if pts.shape[1] == 2:
                pts = np.c_[pts, np.zeros(pts.shape[0], dtype=np.float32)]
            arr = utils.numpy2vtk(pts, dtype=np.float32)
            
            vpts = self.GetPoints()
            vpts.SetData(arr)
            vpts.Modified()
            # reset mesh to identity matrix position/rotation:
            self.actor.PokeMatrix(vtk.vtkMatrix4x4())
            self.point_locator = None
            self.cell_locator = None
            self.transform = LinearTransform()
            return self


    def cell_centers(self):
        """
        Get the coordinates of the cell centers.

        Examples:
            - [delaunay2d.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/delaunay2d.py)
        """
        vcen = vtk.vtkCellCenters()
        if hasattr(self, "polydata"):
            vcen.SetInputData(self)
        else:
            vcen.SetInputData(self)
        vcen.Update()
        return utils.vtk2numpy(vcen.GetOutput().GetPoints().GetData())

    def delete_cells(self, ids):
        """
        Remove cells from the mesh object by their ID.
        Points (vertices) are not removed (you may use `.clean()` to remove those).
        """
        data = self
        data.BuildLinks()
        for cid in ids:
            data.DeleteCell(cid)
        data.RemoveDeletedCells()
        data.Modified()
        self._mapper.Modified()
        self.pipeline = utils.OperationNode(
            "delete_cells", parents=[self], comment=f"#cells {self.GetNumberOfCells()}"
        )
        return self

    def mark_boundaries(self):
        """
        Mark cells and vertices of the mesh if they lie on a boundary.
        A new array called `BoundaryCells` is added to the mesh.
        """
        mb = vtk.vtkMarkBoundaryFilter()
        mb.SetInputData(self)
        mb.Update()
        self.DeepCopy(mb.GetOutput())
        self.pipeline = utils.OperationNode("mark_boundaries", parents=[self])
        return self

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
        self.cell_locator.SetDataSet(self)
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
        vc.SetInputData(self)
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

        Arguments:
            style : (str)
                preset style, options are `[metallic, plastic, shiny, glossy, ambient, off]`
            ambient : (float)
                ambient fraction of emission [0-1]
            diffuse : (float)
                emission of diffused light in fraction [0-1]
            specular : (float)
                fraction of reflected light [0-1]
            specular_power : (float)
                precision of reflection [1-100]
            specular_color : (color)
                color that is being reflected by the surface

        <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Phong_components_version_4.png" alt="", width=700px>

        Examples:
            - [specular.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/specular.py)
        """
        pr = self.property

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
            mpr = self.mapper
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
                vedo.logger.error("[default, metallic, plastic, shiny, glossy, ambient, off]")
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
        Input can be `Volume` or `Mesh` (will grab the active point array).

        Arguments:
            bins : (int)
                number of histogram bins
            height : (int)
                height of the histogram in character units
            logscale : (bool)
                use logscale for frequencies
            minbin : (int)
                ignore bins before minbin
            horizontal : (bool)
                show histogram horizontally
            char : (str)
                character to be used as marker
            c : (color)
                ascii color
            bold : (bool)
                use boldface
            title : (str)
                histogram title

        ![](https://vedo.embl.es/images/feats/histoprint.png)
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
        Create and/or return a `numpy.array` associated to points (vertices).
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.pointdata["arrayname"]`

        Usage:

            `myobj.pointdata.keys()` to return the available data array names

            `myobj.pointdata.select(name)` to make this array the active one

            `myobj.pointdata.remove(name)` to remove this array
        """
        return _DataArrayHelper(self, 0)

    @property
    def celldata(self):
        """
        Create and/or return a `numpy.array` associated to cells (faces).
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.celldata["arrayname"]`

        Usage:

            `myobj.celldata.keys()` to return the available data array names

            `myobj.celldata.select(name)` to make this array the active one

            `myobj.celldata.remove(name)` to remove this array
        """
        return _DataArrayHelper(self, 1)

    @property
    def metadata(self):
        """
        Create and/or return a `numpy.array` associated to neither cells nor faces.
        A data array can be indexed either as a string or by an integer number.
        E.g.:  `myobj.metadata["arrayname"]`

        Usage:

            `myobj.metadata.keys()` to return the available data array names

            `myobj.metadata.select(name)` to make this array the active one

            `myobj.metadata.remove(name)` to remove this array
        """
        return _DataArrayHelper(self, 2)

    def map_cells_to_points(self, arrays=(), move=False):
        """
        Interpolate cell data (i.e., data specified per cell or face)
        into point data (i.e., data specified at each vertex).
        The method of transformation is based on averaging the data values
        of all cells using a particular point.

        A custom list of arrays to be mapped can be passed in input.

        Set `move=True` to delete the original `celldata` array.
        """
        c2p = vtk.vtkCellDataToPointData()
        c2p.SetInputData(self)
        if not move:
            c2p.PassCellDataOn()
        if arrays:
            c2p.ClearCellDataArrays()
            c2p.ProcessAllArraysOff()
            for arr in arrays:
                c2p.AddCellDataArray(arr)
        else:
            c2p.ProcessAllArraysOn()
        c2p.Update()
        self.mapper.SetScalarModeToUsePointData()
        self.DeepCopy(c2p.GetOutput())
        self.pipeline = utils.OperationNode("map cell\nto point data", parents=[self])
        return self

    def map_points_to_cells(self, arrays=(), move=False):
        """
        Interpolate point data (i.e., data specified per point or vertex)
        into cell data (i.e., data specified per cell).
        The method of transformation is based on averaging the data values
        of all points defining a particular cell.

        A custom list of arrays to be mapped can be passed in input.

        Set `move=True` to delete the original `pointdata` array.

        Examples:
            - [mesh_map2cell.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_map2cell.py)
        """
        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(self)
        if not move:
            p2c.PassPointDataOn()
        if arrays:
            p2c.ClearPointDataArrays()
            p2c.ProcessAllArraysOff()
            for arr in arrays:
                p2c.AddPointDataArray(arr)
        else:
            p2c.ProcessAllArraysOn()
        p2c.Update()
        self.mapper.SetScalarModeToUseCellData()
        self.DeepCopy(p2c.GetOutput())
        self.pipeline = utils.OperationNode("map point\nto cell data", parents=[self])
        return self

    def resample_data_from(self, source, tol=None, categorical=False):
        """
        Resample point and cell data from another dataset.
        The output has the same structure but its point data have
        the resampled values from target.

        Use `tol` to set the tolerance used to compute whether
        a point in the source is in a cell of the current object.
        Points without resampled values, and their cells, are marked as blank.
        If the data is categorical, then the resulting data will be determined
        by a nearest neighbor interpolation scheme.

        Example:
        ```python
        from vedo import *
        m1 = Mesh(dataurl+'bunny.obj')#.add_gaussian_noise(0.1)
        pts = m1.points()
        ces = m1.cell_centers()
        m1.pointdata["xvalues"] = np.power(pts[:,0], 3)
        m1.celldata["yvalues"]  = np.power(ces[:,1], 3)
        m2 = Mesh(dataurl+'bunny.obj')
        m2.resample_arrays_from(m1)
        # print(m2.pointdata["xvalues"])
        show(m1, m2 , N=2, axes=1)
        ```
        """
        rs = vtk.vtkResampleWithDataSet()
        rs.SetInputData(self)
        rs.SetSourceData(source)

        rs.SetPassPointArrays(True)
        rs.SetPassCellArrays(True)
        rs.SetPassFieldArrays(True)
        rs.SetCategoricalData(categorical)

        rs.SetComputeTolerance(True)
        if tol:
            rs.SetComputeTolerance(False)
            rs.SetTolerance(tol)
        rs.Update()
        self.DeepCopy(rs.GetOutput())
        self.pipeline = utils.OperationNode(
            f"resample_data_from\n{source.__class__.__name__}", parents=[self, source]
        )
        return self

    def add_ids(self):
        """Generate point and cell ids arrays."""
        ids = vtk.vtkIdFilter()
        ids.SetInputData(self)
        ids.PointIdsOn()
        ids.CellIdsOn()
        ids.FieldDataOff()
        ids.SetPointIdsArrayName("PointID")
        ids.SetCellIdsArrayName("CellID")
        ids.Update()
        self.DeepCopy(ids.GetOutput())
        self.pipeline = utils.OperationNode("add_ids", parents=[self])
        return self

    def gradient(self, input_array=None, on="points", fast=False):
        """
        Compute and return the gradiend of the active scalar field as a numpy array.

        Arguments:
            input_array : (str)
                array of the scalars to compute the gradient,
                if None the current active array is selected
            on : (str)
                compute either on 'points' or 'cells' data
            fast : (bool)
                if True, will use a less accurate algorithm
                that performs fewer derivative calculations (and is therefore faster).

        Examples:
            - [isolines.py](https://github.com/marcomusy/vedo/tree/master/examples/advanced/isolines.py)

            ![](https://user-images.githubusercontent.com/32848391/72433087-f00a8780-3798-11ea-9778-991f0abeca70.png)
        """
        gra = vtk.vtkGradientFilter()
        if on.startswith("p"):
            varr = self.GetPointData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        else:
            varr = self.GetCellData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS

        if input_array is None:
            if varr.GetScalars():
                input_array = varr.GetScalars().GetName()
            else:
                vedo.logger.error(f"in gradient: no scalars found for {on}")
                raise RuntimeError

        gra.SetInputData(self)
        gra.SetInputScalars(tp, input_array)
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

    def divergence(self, array_name=None, on="points", fast=False):
        """
        Compute and return the divergence of a vector field as a numpy array.

        Arguments:
            array_name : (str)
                name of the array of vectors to compute the divergence,
                if None the current active array is selected
            on : (str)
                compute either on 'points' or 'cells' data
            fast : (bool)
                if True, will use a less accurate algorithm
                that performs fewer derivative calculations (and is therefore faster).
        """
        div = vtk.vtkGradientFilter()
        if on.startswith("p"):
            varr = self.GetPointData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        else:
            varr = self.GetCellData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS

        if array_name is None:
            if varr.GetVectors():
                array_name = varr.GetVectors().GetName()
            else:
                vedo.logger.error(f"in divergence(): no vectors found for {on}")
                raise RuntimeError

        div.SetInputData(self)
        div.SetInputScalars(tp, array_name)
        div.ComputeDivergenceOn()
        div.ComputeGradientOff()
        div.ComputeVorticityOff()
        div.SetDivergenceArrayName("Divergence")
        div.SetFasterApproximation(fast)
        div.Update()
        if on.startswith("p"):
            dvecs = utils.vtk2numpy(div.GetOutput().GetPointData().GetArray("Divergence"))
        else:
            dvecs = utils.vtk2numpy(div.GetOutput().GetCellData().GetArray("Divergence"))
        return dvecs

    def vorticity(self, array_name=None, on="points", fast=False):
        """
        Compute and return the vorticity of a vector field as a numpy array.

        Arguments:
            array_name : (str)
                name of the array to compute the vorticity,
                if None the current active array is selected
            on : (str)
                compute either on 'points' or 'cells' data
            fast : (bool)
                if True, will use a less accurate algorithm
                that performs fewer derivative calculations (and is therefore faster).
        """
        vort = vtk.vtkGradientFilter()
        if on.startswith("p"):
            varr = self.GetPointData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        else:
            varr = self.GetCellData()
            tp = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS

        if array_name is None:
            if varr.GetVectors():
                array_name = varr.GetVectors().GetName()
            else:
                vedo.logger.error(f"in vorticity(): no vectors found for {on}")
                raise RuntimeError

        vort.SetInputData(self)
        vort.SetInputScalars(tp, array_name)
        vort.ComputeDivergenceOff()
        vort.ComputeGradientOff()
        vort.ComputeVorticityOn()
        vort.SetVorticityArrayName("Vorticity")
        vort.SetFasterApproximation(fast)
        vort.Update()
        if on.startswith("p"):
            vvecs = utils.vtk2numpy(vort.GetOutput().GetPointData().GetArray("Vorticity"))
        else:
            vvecs = utils.vtk2numpy(vort.GetOutput().GetCellData().GetArray("Vorticity"))
        return vvecs

    def add_scalarbar(
        self,
        title="",
        pos=(0.8, 0.05),
        title_yoffset=15,
        font_size=12,
        size=(None, None),
        nlabels=None,
        c=None,
        horizontal=False,
        use_alpha=True,
        label_format=":6.3g",
    ):
        """
        Add a 2D scalar bar for the specified obj.

        Examples:
            - [mesh_coloring.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_coloring.py)
            - [scalarbars.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/scalarbars.py)
        """
        plt = vedo.plotter_instance

        if plt and plt.renderer:
            c = (0.9, 0.9, 0.9)
            if np.sum(plt.renderer.GetBackground()) > 1.5:
                c = (0.1, 0.1, 0.1)
            if isinstance(self.scalarbar, vtk.vtkActor):
                plt.renderer.RemoveActor(self.scalarbar)
            elif isinstance(self.scalarbar, vedo.Assembly):
                for a in self.scalarbar.unpack():
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

    def add_scalarbar3d(
        self,
        title="",
        pos=None,
        size=(None, None),
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

        Arguments:
            size : (list)
                (thickness, length) of scalarbar
            title : (str)
                scalar bar title
            title_xoffset : (float)
                horizontal space btw title and color scalarbar
            title_yoffset : (float)
                vertical space offset
            title_size : (float)
                size of title wrt numeric labels
            title_rotation : (float)
                title rotation in degrees
            nlabels : (int)
                number of numeric labels
            label_font : (str)
                font type for labels
            label_size : (float)
                label scale factor
            label_offset : (float)
                space btw numeric labels and scale
            label_rotation : (float)
                label rotation in degrees
            label_format : (str)
                label format for floats and integers (e.g. `':.2f'`)
            draw_box : (bool)
                draw a box around the colorbar
            categories : (list)
                make a categorical scalarbar,
                the input list will have the format `[value, color, alpha, textlabel]`

        Examples:
            - [scalarbars.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/scalarbars.py)
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
            size,
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
        out = vedo.file_io.write(self, filename, binary)
        out.pipeline = utils.OperationNode(
            "write", parents=[self], comment=filename[:15], shape="folder", c="#8a817c"
        )
        return out


########################################################################################
class BaseGrid(BaseActor):
    """
    Base class for grid datasets.

    .. warning:: Do not use this class to instantiate objects.
    """

    def __init__(self):
        """Base class for grid datasets."""

        super().__init__()

        self._data = None
        self.useCells = True
        self._color = None
        self._alpha = [0, 1]

        # -----------------------------------------------------------

    # def _update(self, data):
    #     self.mapper.SetInputData(self.tomesh())
    #     self.mapper.Modified()
    #     return self

    def tomesh(self, fill=True, shrink=1.0):
        """
        Build a polygonal Mesh from the current Grid object.

        If `fill=True`, the interior faces of all the cells are created.
        (setting a `shrink` value slightly smaller than the default 1.0
        can avoid flickering due to internal adjacent faces).

        If `fill=False`, only the boundary faces will be generated.
        """
        gf = vtk.vtkGeometryFilter()
        if fill:
            sf = vtk.vtkShrinkFilter()
            sf.SetInputData(self)
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
            gf.SetInputData(self)
            gf.Update()
            poly = gf.GetOutput()

        msh = vedo.mesh.Mesh(poly).flat()
        msh.scalarbar = self.scalarbar
        lut = utils.ctf2lut(self)
        if lut:
            msh.mapper.SetLookupTable(lut)
        if self.useCells:
            msh.mapper.SetScalarModeToUseCellData()
        else:
            msh.mapper.SetScalarModeToUsePointData()

        msh.pipeline = utils.OperationNode(
            "tomesh", parents=[self], comment=f"fill={fill}", c="#9e2a2b:#e9c46a"
        )
        return msh

    def cells(self):
        """
        Get the cells connectivity ids as a numpy array.

        The output format is: `[[id0 ... idn], [id0 ... idm],  etc]`.
        """
        arr1d = utils.vtk2numpy(self.GetCells().GetData())
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
        Any matplotlib color map name is also accepted, e.g. `volume.color('jet')`.

        E.g.: say that your cells scalar runs from -3 to 6,
        and you want -3 to show red and 1.5 violet and 6 green, then just set:

        `volume.color(['red', 'violet', 'green'])`

        You can also assign a specific color to a aspecific value with eg.:

        `volume.color([(0,'red', (0.5,'violet'), (1,'green')])`

        Arguments:
            alpha : (list)
                use a list to specify transparencies along the scalar range
            vmin : (float)
                force the min of the scalar range to be this value
            vmax : (float)
                force the max of the scalar range to be this value
        """
        # supersedes method in Points, Mesh

        if col is None:
            return self
        
        if vmin is None:
            vmin, _ = self.GetScalarRange()
        if vmax is None:
            _, vmax = self.GetScalarRange()
        ctf = self.property.GetRGBTransferFunction()
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
            vmin, _ = self.GetScalarRange()
        if vmax is None:
            _, vmax = self.GetScalarRange()
        otf = self.property.GetScalarOpacity()
        otf.RemoveAllPoints()
        self._alpha = alpha

        if utils.is_sequence(alpha):
            alpha = np.array(alpha)
            if len(alpha.shape) == 1:  # user passing a flat list e.g. (0.0, 0.3, 0.9, 1)
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
            return self.property.GetScalarOpacityUnitDistance()
        self.property.SetScalarOpacityUnitDistance(u)
        return self

    def shrink(self, fraction=0.8):
        """
        Shrink the individual cells to improve visibility.

        ![](https://vedo.embl.es/images/feats/shrink_hex.png)
        """
        sf = vtk.vtkShrinkFilter()
        sf.SetInputData(self)
        sf.SetShrinkFactor(fraction)
        sf.Update()
        self.DeepCopy(sf.GetOutput())
        self.pipeline = utils.OperationNode(
            "shrink", comment=f"by {fraction}", parents=[self], c="#9e2a2b"
        )
        return self

    def isosurface(self, value=None, flying_edges=True):
        """
        Return an `Mesh` isosurface extracted from the `Volume` object.

        Set `value` as single float or list of values to draw the isosurface(s).
        Use flying_edges for faster results (but sometimes can interfere with `smooth()`).

        Examples:
            - [isosurfaces.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/isosurfaces.py)

                ![](https://vedo.embl.es/images/volumetric/isosurfaces.png)
        """
        scrange = self.GetScalarRange()

        if flying_edges:
            cf = vtk.vtkFlyingEdges3D()
            cf.InterpolateAttributesOn()
        else:
            cf = vtk.vtkContourFilter()
            cf.UseScalarTreeOn()

        cf.SetInputData(self)
        cf.ComputeNormalsOn()

        if utils.is_sequence(value):
            cf.SetNumberOfContours(len(value))
            for i, t in enumerate(value):
                cf.SetValue(i, t)
        else:
            if value is None:
                value = (2 * scrange[0] + scrange[1]) / 3.0
                # print("automatic isosurface value =", value)
            cf.SetValue(0, value)

        cf.Update()
        poly = cf.GetOutput()

        out = vedo.mesh.Mesh(poly, c=None).phong()
        out.mapper.SetScalarRange(scrange[0], scrange[1])

        out.pipeline = utils.OperationNode(
            "isosurface",
            parents=[self],
            comment=f"#pts {out.GetNumberOfPoints()}",
            c="#4cc9f0:#e9c46a",
        )
        return out

    def legosurface(
        self, vmin=None, vmax=None, invert=False, boundary=False, array_name="input_scalars"
    ):
        """
        Represent an object - typically a `Volume` - as lego blocks (voxels).
        By default colors correspond to the volume's scalar.
        Returns an `Mesh` object.

        Arguments:
            vmin : (float)
                the lower threshold, voxels below this value are not shown.
            vmax : (float)
                the upper threshold, voxels above this value are not shown.
            boundary : (bool)
                controls whether to include cells that are partially inside
            array_name : (int, str)
                name or index of the scalar array to be considered

        Examples:
            - [legosurface.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/legosurface.py)

                ![](https://vedo.embl.es/images/volumetric/56820682-da40e500-684c-11e9-8ea3-91cbcba24b3a.png)
        """
        dataset = vtk.vtkImplicitDataSet()
        dataset.SetDataSet(self)
        window = vtk.vtkImplicitWindowFunction()
        window.SetImplicitFunction(dataset)

        srng = list(self.GetScalarRange())
        if vmin is not None:
            srng[0] = vmin
        if vmax is not None:
            srng[1] = vmax
        tol = 0.00001 * (srng[1] - srng[0])
        srng[0] -= tol
        srng[1] += tol
        window.SetWindowRange(srng)

        extract = vtk.vtkExtractGeometry()
        extract.SetInputData(self)
        extract.SetImplicitFunction(window)
        extract.SetExtractInside(invert)
        extract.SetExtractBoundaryCells(boundary)
        extract.Update()

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(extract.GetOutput())
        gf.Update()

        m = vedo.mesh.Mesh(gf.GetOutput()).lw(0.1).flat()
        m.map_points_to_cells()
        m.celldata.select(array_name)

        m.pipeline = utils.OperationNode(
            "legosurface", parents=[self], comment=f"array: {array_name}", c="#4cc9f0:#e9c46a"
        )
        return m

    def cut_with_plane(self, origin=(0, 0, 0), normal="x"):
        """
        Cut the object with the plane defined by a point and a normal.

        Arguments:
            origin : (list)
                the cutting plane goes through this point
            normal : (list, str)
                normal vector to the cutting plane
        """
        # if isinstance(self, vedo.Volume):
        #     raise RuntimeError("cut_with_plane() is not applicable to Volume objects.")    

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
        clipper.SetInputData(self)
        clipper.SetClipFunction(plane)
        clipper.GenerateClipScalarsOff()
        clipper.GenerateClippedOutputOff()
        clipper.SetValue(0)
        clipper.Update()
        cout = clipper.GetOutput()

        if isinstance(cout, vtk.vtkUnstructuredGrid):
            ug = vedo.UGrid(cout)
            if isinstance(self, vedo.UGrid):
                self.DeepCopy(cout)
                self.pipeline = utils.OperationNode("cut_with_plane", parents=[self], c="#9e2a2b")
                return self
            ug.pipeline = utils.OperationNode("cut_with_plane", parents=[self], c="#9e2a2b")
            return ug

        else:
            self.DeepCopy(cout)
            self.pipeline = utils.OperationNode("cut_with_plane", parents=[self], c="#9e2a2b")
            return self

    def cut_with_box(self, box):
        """
        Cut the grid with the specified bounding box.

        Parameter box has format [xmin, xmax, ymin, ymax, zmin, zmax].
        If an object is passed, its bounding box are used.

        Example:
            ```python
            from vedo import *
            tetmesh = TetMesh(dataurl+'limb_ugrid.vtk')
            tetmesh.color('rainbow')
            cu = Cube(side=500).x(500) # any Mesh works
            tetmesh.cut_with_box(cu).show(axes=1)
            ```
            ![](https://vedo.embl.es/images/feats/tet_cut_box.png)
        """
        # if isinstance(self, vedo.Volume):
        #     raise RuntimeError("cut_with_box() is not applicable to Volume objects.")    

        bc = vtk.vtkBoxClipDataSet()
        bc.SetInputData(self)
        if isinstance(box, vtk.vtkProp):
            boxb = box.GetBounds()
        else:
            boxb = box
        bc.SetBoxClip(*boxb)
        bc.Update()
        cout = bc.GetOutput()

        if isinstance(cout, vtk.vtkUnstructuredGrid):
            ug = vedo.UGrid(cout)
            if isinstance(self, vedo.UGrid):
                self.DeepCopy(cout)
                self.pipeline = utils.OperationNode("cut_with_box", parents=[self], c="#9e2a2b")
                return self
            ug.pipeline = utils.OperationNode("cut_with_box", parents=[self], c="#9e2a2b")
            return ug

        else:
            self.DeepCopy(cout)
            self.pipeline = utils.OperationNode("cut_with_box", parents=[self], c="#9e2a2b")
            return self


    def cut_with_mesh(self, mesh, invert=False, whole_cells=False, only_boundary=False):
        """
        Cut a UGrid or TetMesh with a Mesh.

        Use `invert` to return cut off part of the input object.
        """
        # if isinstance(self, vedo.Volume):
        #     raise RuntimeError("cut_with_mesh() is not applicable to Volume objects.")    

        ug = self

        ippd = vtk.vtkImplicitPolyDataDistance()
        ippd.SetInput(mesh)

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
        cout = clipper.GetOutput()

        # if ug.GetCellData().GetScalars():  # not working
        #     scalname = ug.GetCellData().GetScalars().GetName()
        #     if scalname:  # not working
        #         if self.useCells:
        #             self.celldata.select(scalname)
        #         else:
        #             self.pointdata.select(scalname)
        # self._update(cout)
        # self.pipeline = utils.OperationNode("cut_with_mesh", parents=[self, mesh], c="#9e2a2b")
        # return self

        if isinstance(cout, vtk.vtkUnstructuredGrid):
            ug = vedo.UGrid(cout)
            if isinstance(self, vedo.UGrid):
                self.DeepCopy(cout)
                self.pipeline = utils.OperationNode("cut_with_mesh", parents=[self], c="#9e2a2b")
                return self
            ug.pipeline = utils.OperationNode("cut_with_mesh", parents=[self], c="#9e2a2b")
            return ug

        else:
            self.DeepCopy(cout)
            self.pipeline = utils.OperationNode("cut_with_mesh", parents=[self], c="#9e2a2b")
            return self

    def extract_cells_on_plane(self, origin, normal):
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtk.vtk3DLinearGridCrinkleExtractor()
        bf.SetInputData(self)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        bf.SetImplicitFunction(plane)
        bf.Update()

        self.DeepCopy(bf.GetOutput())
        self.pipeline = utils.OperationNode(
            "extract_cells_on_plane",
            parents=[self],
            comment=f"#cells {self.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        return self

    def extract_cells_on_sphere(self, center, radius):
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtk.vtk3DLinearGridCrinkleExtractor()
        bf.SetInputData(self)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        sph = vtk.vtkSphere()
        sph.SetRadius(radius)
        sph.SetCenter(center)
        bf.SetImplicitFunction(sph)
        bf.Update()

        self.DeepCopy(bf.GetOutput())
        self.pipeline = utils.OperationNode(
            "extract_cells_on_sphere",
            parents=[self],
            comment=f"#cells {self.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        return self

    def extract_cells_on_cylinder(self, center, axis, radius):
        """
        Extract cells that are lying of the specified surface.
        """
        bf = vtk.vtk3DLinearGridCrinkleExtractor()
        bf.SetInputData(self)
        bf.CopyPointDataOn()
        bf.CopyCellDataOn()
        bf.RemoveUnusedPointsOff()

        cyl = vtk.vtkCylinder()
        cyl.SetRadius(radius)
        cyl.SetCenter(center)
        cyl.SetAxis(axis)
        bf.SetImplicitFunction(cyl)
        bf.Update()

        self.pipeline = utils.OperationNode(
            "extract_cells_on_cylinder",
            parents=[self],
            comment=f"#cells {self.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        self.DeepCopy(bf.GetOutput())
        return self

    def clean(self):
        """
        Cleanup unused points and empty cells
        """
        cl = vtk.vtkStaticCleanUnstructuredGrid()
        cl.SetInputData(self)
        cl.RemoveUnusedPointsOn()
        cl.ProduceMergeMapOff()
        cl.AveragePointDataOff()
        cl.Update()

        self.DeepCopy(cl.GetOutput())
        self.pipeline = utils.OperationNode(
            "clean", parents=[self], comment=f"#cells {self.GetNumberOfCells()}", c="#9e2a2b"
        )
        return self

    def find_cell(self, p):
        """Locate the cell that contains a point and return the cell ID."""
        cell = vtk.vtkTetra()
        cellId = vtk.mutable(0)
        tol2 = vtk.mutable(0)
        subId = vtk.mutable(0)
        pcoords = [0, 0, 0]
        weights = [0, 0, 0]
        cid = self.FindCell(p, cell, cellId, tol2, subId, pcoords, weights)
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
        es.SetInputData(0, self)
        es.SetInputData(1, selection)
        es.Update()

        ug = vedo.ugrid.UGrid(es.GetOutput())
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.property)
        ug.SetProperty(pr)
        ug.property = pr

        # assign the same transformation to the copy
        ug.SetOrigin(self.GetOrigin())
        ug.SetScale(self.GetScale())
        ug.SetOrientation(self.GetOrientation())
        ug.SetPosition(self.GetPosition())
        ug.mapper.SetLookupTable(utils.ctf2lut(self))
        ug.pipeline = utils.OperationNode(
            "extract_cells_by_id",
            parents=[self],
            comment=f"#cells {self.GetNumberOfCells()}",
            c="#9e2a2b",
        )
        return ug


########################################################################################
class BaseActor2D(vtk.vtkActor2D):
    """
    Base class.

    .. warning:: Do not use this class to instantiate objects.
    """

    def __init__(self):
        """Manage 2D objects."""
        super().__init__()
        self.mapper = None
        self.property = self.GetProperty()
        self.filename = ""

    def layer(self, value=None):
        """Set/Get the layer number in the overlay planes into which to render."""
        if value is None:
            return self.GetLayerNumber()
        self.SetLayerNumber(value)
        return self

    def pos(self, px=None, py=None):
        """Set/Get the screen-coordinate position."""
        if isinstance(px, str):
            vedo.logger.error("Use string descriptors only inside the constructor")
            return self
        if px is None:
            return np.array(self.GetPosition(), dtype=int)
        if py is not None:
            p = [px, py]
        else:
            p = px
        assert len(p) == 2, "Error: len(pos) must be 2 for BaseActor2D"
        self.SetPosition(p)
        return self

    def coordinate_system(self, value=None):
        """
        Set/get the coordinate system which this coordinate is defined in.

        The options are:
            0. Display
            1. Normalized Display
            2. Viewport
            3. Normalized Viewport
            4. View
            5. Pose
            6. World
        """
        coor = self.GetPositionCoordinate()
        if value is None:
            return coor.GetCoordinateSystem()
        coor.SetCoordinateSystem(value)
        return self

    def on(self):
        """Set object visibility."""
        self.VisibilityOn()
        return self

    def off(self):
        """Set object visibility."""
        self.VisibilityOn()
        return self

    def toggle(self):
        """Toggle object visibility."""
        self.SetVisibility(not self.GetVisibility())
        return self

    def pickable(self, value=True):
        """Set object pickability."""
        self.SetPickable(value)
        return self

    def alpha(self, value=None):
        """Set/Get the object opacity."""
        if value is None:
            return self.property.GetOpacity()
        self.property.SetOpacity(value)
        return self

    def ps(self, point_size=None):
        if point_size is None:
            return self.property.GetPointSize()
        self.property.SetPointSize(point_size)
        return self

    def ontop(self, value=True):
        """Keep the object always on top of everything else."""
        if value:
            self.property.SetDisplayLocationToForeground()
        else:
            self.property.SetDisplayLocationToBackground()
        return self
    
    def add_observer(self, event_name, func, priority=0):
        """Add a callback function that will be called when an event occurs."""
        event_name = utils.get_vtk_name_event(event_name)
        idd = self.AddObserver(event_name, func, priority)
        return idd


############################################################################### funcs
def _getinput(obj):
    if isinstance(obj, (vtk.vtkVolume, vtk.vtkActor)):
        return obj.GetMapper().GetInput()
    return obj


def probe_points(dataset, pts):
    """
    Takes a `Volume` (or any other vtk data set)
    and probes its scalars at the specified points in space.

    Note that a mask is also output with valid/invalid points which can be accessed
    with `mesh.pointdata['vtkValidPointMask']`.

    Examples:
        - [probe_points.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/probe_points.py)

            ![](https://vedo.embl.es/images/volumetric/probePoints.png)
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
    pm.pipeline = utils.OperationNode("probe_points", parents=[dataset])
    return pm


def probe_line(dataset, p1, p2, res=100):
    """
    Takes a `Volume`  (or any other vtk data set)
    and probes its scalars along a line defined by 2 points `p1` and `p2`.

    Note that a mask is also output with valid/invalid points which can be accessed
    with `mesh.pointdata['vtkValidPointMask']`.

    Use `res` to set the nr of points along the line

    Examples:
        - [probe_line1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/probe_line1.py)
        - [probe_line2.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/probe_line2.py)

            ![](https://vedo.embl.es/images/volumetric/probeLine2.png)
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
    lnn.pipeline = utils.OperationNode("probe_line", parents=[dataset])
    return lnn


def probe_plane(dataset, origin=(0, 0, 0), normal=(1, 0, 0)):
    """
    Takes a `Volume` (or any other vtk data set)
    and probes its scalars on a plane defined by a point and a normal.

    Examples:
        - [slice_plane1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slice_plane1.py)
        - [slice_plane2.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slice_plane2.py)

            ![](https://vedo.embl.es/images/volumetric/slicePlane2.png)
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
    cutmesh.pipeline = utils.OperationNode("probe_plane", parents=[dataset])
    return cutmesh
