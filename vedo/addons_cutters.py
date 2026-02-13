#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cutter widgets and frame/progress helpers extracted from vedo.addons."""

from typing import Union
from typing_extensions import Self
import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import settings
from vedo import utils
from vedo import shapes
from vedo.transformations import LinearTransform
from vedo.assembly import Assembly, Group
from vedo.colors import get_color, build_lut, color_map
from vedo.mesh import Mesh
from vedo.pointcloud import Points, Point, merge
from vedo.grids import TetMesh
from vedo.volume import Volume
from vedo.visual import Actor2D
class BaseCutter:
    """
    Base class for Cutter widgets.
    """

    def __init__(self):
        self.__implicit_func = None
        self.widget = None
        self.clipper = None
        self.cutter = None
        self.mesh = None
        self.remnant = None
        self._alpha = 0.5

        self.can_translate = True
        self.can_scale = True
        self.can_rotate = True
        self.is_inverted = False


    @property
    def transform(self) -> LinearTransform:
        """Get the transformation matrix."""
        t = vtki.vtkTransform()
        self.widget.GetTransform(t)
        return LinearTransform(t)

    def get_cut_mesh(self, invert=False) -> Mesh:
        """
        Get the mesh resulting from the cut operation.
        Returns the original mesh and the remnant mesh.
        """
        self.clipper.Update()
        if invert:
            poly = self.clipper.GetClippedOutput()
        else:
            poly = self.clipper.GetOutput()
        out = Mesh(poly)
        out.copy_properties_from(self.mesh)
        return out

    def invert(self) -> Self:
        """Invert selection."""
        self.clipper.SetInsideOut(not self.clipper.GetInsideOut())
        self.is_inverted = not self.clipper.GetInsideOut()
        return self
    
    def on(self) -> Self:
        """Switch the widget on or off."""
        self.widget.On()
        return self

    def off(self) -> Self:
        """Switch the widget on or off."""
        self.widget.Off()
        return self

    def toggle(self) -> Self:
        """Toggle the widget on or off."""
        if self.widget.GetEnabled():
            self.off()
        else:
            self.on()
        return self

    def add_to(self, plt) -> Self:
        """Assign the widget to the provided `Plotter` instance."""
        self.widget.SetInteractor(plt.interactor)
        self.widget.SetCurrentRenderer(plt.renderer)
        if self.widget not in plt.widgets:
            plt.widgets.append(self.widget)

        cpoly = self.clipper.GetOutput()
        self.mesh._update(cpoly)

        out = self.clipper.GetClippedOutputPort()
        if self._alpha:
            self.remnant.mapper.SetInputConnection(out)
            self.remnant.alpha(self._alpha).color((0.5, 0.5, 0.5))
            self.remnant.lighting("off").wireframe()
            plt.add(self.mesh, self.remnant)
        else:
            plt.add(self.mesh)

        if plt.interactor and plt.interactor.GetInitialized():
            self.widget.On()
            self._select_polygons(self.widget, "InteractionEvent")
            plt.interactor.Render()
        return self

    def remove_from(self, plt) -> Self:
        """Remove the widget to the provided `Plotter` instance."""
        self.widget.Off()
        plt.remove(self.remnant)
        if self.widget in plt.widgets:
            plt.widgets.remove(self.widget)
        return self

    def add_observer(self, event, func, priority=1) -> int:
        """Add an observer to the widget."""
        event = utils.get_vtk_name_event(event)
        cid = self.widget.AddObserver(event, func, priority)
        return cid

    def remove_observers(self, event="") -> Self:
        """Remove all observers from the widget."""
        if not event:
            self.widget.RemoveAllObservers()
        else:
            event = utils.get_vtk_name_event(event)
            self.widget.RemoveObservers(event)
        return self

    def keypress_activation(self, value=True) -> Self:
        """Enable or disable keypress activation of the widget."""
        self.widget.SetKeyPressActivation(value)
        return self
    
    def render(self) -> Self:
        """Render the current state of the widget."""
        if self.widget.GetInteractor() and self.widget.GetInteractor().GetInitialized():
            self.widget.GetInteractor().Render()
        return self


class PlaneCutter(BaseCutter, vtki.vtkPlaneWidget):
    """
    Create a box widget to cut away parts of a Mesh.
    """

    def __init__(
        self,
        mesh,
        invert=False,
        origin=(),
        normal=(),
        padding=0.05,
        delayed=False,
        c=(0.25, 0.25, 0.25),
        alpha=0.05,
    ):
        """
        Create a box widget to cut away parts of a `Mesh`.

        Arguments:
            mesh : (Mesh)
                the input mesh
            invert : (bool)
                invert the clipping plane
            origin : (list)
                origin of the plane
            normal : (list)
                normal to the plane
            padding : (float)
                padding around the input mesh
            delayed : (bool)
                if True the callback is delayed until
                when the mouse button is released (useful for large meshes)
            c : (color)
                color of the box cutter widget
            alpha : (float)
                transparency of the cut-off part of the input mesh

        Examples:
            - [slice_plane3.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slice_plane3.py)
        """
        super().__init__()
        self.name = "PlaneCutter"

        self.mesh = mesh
        self.remnant = Mesh()
        self.remnant.name = mesh.name + "Remnant"
        self.remnant.pickable(False)

        self._alpha = alpha

        self.__implicit_func = vtki.new("Plane")

        poly = mesh.dataset
        self.clipper = vtki.new("ClipPolyData")
        self.clipper.GenerateClipScalarsOff()
        self.clipper.SetInputData(poly)
        self.clipper.SetClipFunction(self.__implicit_func)
        self.clipper.SetInsideOut(invert)
        self.clipper.GenerateClippedOutputOn()
        self.clipper.Update()

        self.widget = vtki.new("ImplicitPlaneWidget")

        self.widget.GetOutlineProperty().SetColor(get_color(c))
        self.widget.GetOutlineProperty().SetOpacity(0.25)
        self.widget.GetOutlineProperty().SetLineWidth(1)
        self.widget.GetOutlineProperty().LightingOff()

        self.widget.GetSelectedOutlineProperty().SetColor(get_color("red3"))

        self.widget.SetTubing(0)
        self.widget.SetDrawPlane(bool(alpha))
        self.widget.GetPlaneProperty().LightingOff()
        self.widget.GetPlaneProperty().SetOpacity(alpha)
        self.widget.GetSelectedPlaneProperty().SetColor(get_color("red5"))
        self.widget.GetSelectedPlaneProperty().LightingOff()

        self.widget.SetPlaceFactor(1.0 + padding)
        self.widget.SetInputData(poly)
        self.widget.PlaceWidget()
        if delayed:
            self.widget.AddObserver("EndInteractionEvent", self._select_polygons)
        else:
            self.widget.AddObserver("InteractionEvent", self._select_polygons)

        if len(origin) == 3:
            self.widget.SetOrigin(origin)
        else:
            self.widget.SetOrigin(mesh.center_of_mass())

        if len(normal) == 3:
            self.widget.SetNormal(normal)
        else:
            self.widget.SetNormal((1, 0, 0))

    @property
    def origin(self):
        """Get the origin of the plane."""
        return np.array(self.widget.GetOrigin())

    @origin.setter
    def origin(self, value):
        """Set the origin of the plane."""
        self.widget.SetOrigin(value)

    @property
    def normal(self):
        """Get the normal of the plane."""
        return np.array(self.widget.GetNormal())

    @normal.setter
    def normal(self, value):
        """Set the normal of the plane."""
        self.widget.SetNormal(value)

    def _select_polygons(self, vobj, _event) -> None:
        vobj.GetPlane(self.__implicit_func)

    def enable_translation(self, value=True) -> Self:
        """Enable or disable translation of the widget."""
        self.widget.SetOutlineTranslation(value)
        self.widget.SetOriginTranslation(value)
        self.can_translate = bool(value)
        return self
    
    def enable_origin_translation(self, value=True) -> Self:
        """Enable or disable rotation of the widget."""
        self.widget.SetOriginTranslation(value)
        return self
    
    def enable_scaling(self, value=True) -> Self:
        """Enable or disable scaling of the widget."""
        self.widget.SetScaleEnabled(value)
        self.can_scale = bool(value)
        return self
    
    def enable_rotation(self, value=True) -> Self:
        """Dummy."""
        self.can_rotate = bool(value)
        return self


class BoxCutter(BaseCutter, vtki.vtkBoxWidget):
    """
    Create a box widget to cut away parts of a Mesh.
    """

    def __init__(
        self,
        mesh,
        invert=False,
        initial_bounds=(),
        padding=0.025,
        delayed=False,
        c=(0.25, 0.25, 0.25),
        alpha=0.05,
    ):
        """
        Create a box widget to cut away parts of a Mesh.

        Arguments:
            mesh : (Mesh)
                the input mesh
            invert : (bool)
                invert the clipping plane
            initial_bounds : (list)
                initial bounds of the box widget
            padding : (float)
                padding space around the input mesh
            delayed : (bool)
                if True the callback is delayed until
                when the mouse button is released (useful for large meshes)
            c : (color)
                color of the box cutter widget
            alpha : (float)
                transparency of the cut-off part of the input mesh
        """
        super().__init__()
        self.name = "BoxCutter"

        self.mesh = mesh
        self.remnant = Mesh()
        self.remnant.name = mesh.name + "Remnant"
        self.remnant.pickable(False)

        self._alpha = alpha

        self._init_bounds = initial_bounds
        if len(self._init_bounds) == 0:
            self._init_bounds = mesh.bounds()
        else:
            self._init_bounds = initial_bounds

        self.__implicit_func = vtki.new("Planes")
        self.__implicit_func.SetBounds(self._init_bounds)

        poly = mesh.dataset
        self.clipper = vtki.new("ClipPolyData")
        self.clipper.GenerateClipScalarsOff()
        self.clipper.SetInputData(poly)
        self.clipper.SetClipFunction(self.__implicit_func)
        self.clipper.SetInsideOut(not invert)
        self.clipper.GenerateClippedOutputOn()
        self.clipper.Update()

        self.widget = vtki.vtkBoxWidget()

        self.widget.OutlineCursorWiresOn()
        self.widget.GetSelectedOutlineProperty().SetColor(get_color("red3"))
        self.widget.GetSelectedHandleProperty().SetColor(get_color("red5"))

        self.widget.GetOutlineProperty().SetColor(c)
        self.widget.GetOutlineProperty().SetOpacity(1)
        self.widget.GetOutlineProperty().SetLineWidth(1)
        self.widget.GetOutlineProperty().LightingOff()

        self.widget.GetSelectedFaceProperty().LightingOff()
        self.widget.GetSelectedFaceProperty().SetOpacity(0.1)

        self.widget.SetPlaceFactor(1.0 + padding)
        self.widget.SetInputData(poly)
        self.widget.PlaceWidget()
        if delayed:
            self.widget.AddObserver("EndInteractionEvent", self._select_polygons)
        else:
            self.widget.AddObserver("InteractionEvent", self._select_polygons)

    def _select_polygons(self, vobj, _event):
        vobj.GetPlanes(self.__implicit_func)

    def set_bounds(self, bb) -> Self:
        """Set the bounding box as a list of 6 values."""
        self.__implicit_func.SetBounds(bb)
        return self

    def enable_translation(self, value=True) -> Self:
        """Enable or disable translation of the widget."""
        self.widget.SetTranslationEnabled(value)
        self.can_translate = bool(value)
        return self
    
    def enable_scaling(self, value=True) -> Self:
        """Enable or disable scaling of the widget."""
        self.widget.SetScalingEnabled(value)
        self.can_scale = bool(value)
        return self
    
    def enable_rotation(self, value=True) -> Self:
        """Enable or disable rotation of the widget."""
        self.widget.SetRotationEnabled(value)
        self.can_rotate = bool(value)
        return self

class SphereCutter(BaseCutter, vtki.vtkSphereWidget):
    """
    Create a box widget to cut away parts of a Mesh.
    """

    def __init__(
        self,
        mesh,
        invert=False,
        origin=(),
        radius=0,
        res=60,
        delayed=False,
        c="white",
        alpha=0.05,
    ):
        """
        Create a box widget to cut away parts of a Mesh.

        Arguments:
            mesh : Mesh
                the input mesh
            invert : bool
                invert the clipping
            origin : list
                initial position of the sphere widget
            radius : float
                initial radius of the sphere widget
            res : int
                resolution of the sphere widget
            delayed : bool
                if True the cutting callback is delayed until
                when the mouse button is released (useful for large meshes)
            c : color
                color of the box cutter widget
            alpha : float
                transparency of the cut-off part of the input mesh
        """
        super().__init__()
        self.name = "SphereCutter"

        self.mesh = mesh
        self.remnant = Mesh()
        self.remnant.name = mesh.name + "Remnant"
        self.remnant.pickable(False)

        self._alpha = alpha

        self.__implicit_func = vtki.new("Sphere")

        if len(origin) == 3:
            self.__implicit_func.SetCenter(origin)
        else:
            origin = mesh.center_of_mass()
            self.__implicit_func.SetCenter(origin)

        if radius > 0:
            self.__implicit_func.SetRadius(radius)
        else:
            radius = mesh.average_size() * 2
            self.__implicit_func.SetRadius(radius)

        poly = mesh.dataset
        self.clipper = vtki.new("ClipPolyData")
        self.clipper.GenerateClipScalarsOff()
        self.clipper.SetInputData(poly)
        self.clipper.SetClipFunction(self.__implicit_func)
        self.clipper.SetInsideOut(not invert)
        self.clipper.GenerateClippedOutputOn()
        self.clipper.Update()

        self.widget = vtki.vtkSphereWidget()

        self.widget.SetThetaResolution(res * 2)
        self.widget.SetPhiResolution(res)
        self.widget.SetRadius(radius)
        self.widget.SetCenter(origin)
        self.widget.SetRepresentation(2)
        self.widget.HandleVisibilityOff()

        self.widget.HandleVisibilityOff()
        self.widget.GetSphereProperty().SetColor(get_color(c))
        self.widget.GetSphereProperty().SetOpacity(0.2)
        self.widget.GetSelectedSphereProperty().SetColor(get_color("red5"))
        self.widget.GetSelectedSphereProperty().SetOpacity(0.2)

        self.widget.SetPlaceFactor(1.0)
        self.widget.SetInputData(poly)
        self.widget.PlaceWidget()
        if delayed:
            self.widget.AddObserver("EndInteractionEvent", self._select_polygons)
        else:
            self.widget.AddObserver("InteractionEvent", self._select_polygons)

    def _select_polygons(self, vobj, _event):
        vobj.GetSphere(self.__implicit_func)


    @property
    def center(self):
        """Get the center of the sphere."""
        return np.array(self.widget.GetCenter())

    @center.setter
    def center(self, value):
        """Set the center of the sphere."""
        self.widget.SetCenter(value)

    @property
    def radius(self):
        """Get the radius of the sphere."""
        return self.widget.GetRadius()

    @radius.setter
    def radius(self, value):
        """Set the radius of the sphere."""
        self.widget.SetRadius(value)

    def enable_translation(self, value=True) -> Self:
        """Enable or disable translation of the widget."""
        self.widget.SetTranslation(value)
        self.can_translate = bool(value)
        return self
    
    def enable_scaling(self, value=True) -> Self:
        """Enable or disable scaling of the widget."""
        self.widget.SetScale(value)
        self.can_scale = bool(value)
        return self
    
    def enable_rotation(self, value=True) -> Self:
        """Enable or disable rotation of the widget."""
        # This is dummy anyway
        self.can_rotate = bool(value)
        return self


#####################################################################
class RendererFrame(Actor2D):
    """
    Add a line around the renderer subwindow.
    """

    def __init__(self, c="k", alpha=None, lw=None, padding=None, pattern="brtl"):
        """
        Add a line around the renderer subwindow.

        Arguments:
            c : (color)
                color of the line.
            alpha : (float)
                opacity.
            lw : (int)
                line width in pixels.
            padding : (int)
                padding in pixel units.
            pattern : (str)
                combination of characters `b` for bottom, `r` for right,
                `t` for top, `l` for left.
        """
        if lw is None:
            lw = settings.renderer_frame_width

        if alpha is None:
            alpha = settings.renderer_frame_alpha

        if padding is None:
            padding = settings.renderer_frame_padding

        if lw == 0 or alpha == 0:
            return
        c = get_color(c)

        a = padding
        b = 1 - padding
        p0 = [a, a]
        p1 = [b, a]
        p2 = [b, b]
        p3 = [a, b]
        disconnected = False
        if "b" in pattern and "r" in pattern and "t" in pattern and "l" in pattern:
            psqr = [p0, p1, p2, p3, p0]
        elif "b" in pattern and "r" in pattern and "t" in pattern:
            psqr = [p0, p1, p2, p3]
        elif "b" in pattern and "r" in pattern and "l" in pattern:
            psqr = [p3, p0, p1, p2]
        elif "b" in pattern and "t" in pattern and "l" in pattern:
            psqr = [p2, p3, p0, p1]
        elif "b" in pattern and "r" in pattern:
            psqr = [p0, p1, p2]
        elif "b" in pattern and "l" in pattern:
            psqr = [p3, p0, p1]
        elif "r" in pattern and "t" in pattern:
            psqr = [p1, p2, p3]
        elif "t" in pattern and "l" in pattern:
            psqr = [p3, p2, p1]
        elif "b" in pattern and "t" in pattern:
            psqr = [p0, p1, p3, p2]
            disconnected = True
        elif "r" in pattern and "l" in pattern:
            psqr = [p0, p3, p1, p2]
            disconnected = True
        elif "b" in pattern:
            psqr = [p0, p1]
        elif "r" in pattern:
            psqr = [p1, p2]
        elif "t" in pattern:
            psqr = [p3, p2]
        elif "l" in pattern:
            psqr = [p0, p3]
        else:
            vedo.printc("Error in RendererFrame: pattern not recognized", pattern, c='r')
       
        ppoints = vtki.vtkPoints()  # Generate the polyline
        for i, pt in enumerate(psqr):
            ppoints.InsertPoint(i, pt[0], pt[1], 0)

        lines = vtki.vtkCellArray()
        if disconnected:
            lines.InsertNextCell(2)
            lines.InsertCellPoint(0)
            lines.InsertCellPoint(1)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(2)
            lines.InsertCellPoint(3)
        else:
            n = len(psqr)
            lines.InsertNextCell(n)
            for i in range(n):
                lines.InsertCellPoint(i)

        polydata = vtki.vtkPolyData()
        polydata.SetPoints(ppoints)
        polydata.SetLines(lines)

        super().__init__(polydata)
        self.name = "RendererFrame"
        
        self.coordinate = vtki.vtkCoordinate()
        self.coordinate.SetCoordinateSystemToNormalizedViewport()
        self.mapper.SetTransformCoordinate(self.coordinate)

        self.set_position_coordinates([0, 1], [1, 1])
        self.color(c)
        self.alpha(alpha)
        self.lw(lw)



#####################################################################
class ProgressBarWidget(Actor2D):
    """
    Add a progress bar in the rendering window.
    """

    def __init__(self, n=None, c="blue5", alpha=0.8, lw=10, autohide=True):
        """
        Add a progress bar window.

        Arguments:
            n : (int)
                number of iterations.
                If None, you need to call `update(fraction)` manually.
            c : (color)
                color of the line.
            alpha : (float)
                opacity of the line.
            lw : (int)
                line width in pixels.
            autohide : (bool)
                if True, hide the progress bar when completed.
        """
        self.n = 0
        self.iterations = n
        self.autohide = autohide

        ppoints = vtki.vtkPoints()  # Generate the line
        psqr = [[0, 0, 0], [1, 0, 0]]
        for i, pt in enumerate(psqr):
            ppoints.InsertPoint(i, *pt)
        lines = vtki.vtkCellArray()
        lines.InsertNextCell(len(psqr))
        for i in range(len(psqr)):
            lines.InsertCellPoint(i)

        pd = vtki.vtkPolyData()
        pd.SetPoints(ppoints)
        pd.SetLines(lines)

        super().__init__(pd)
        self.name = "ProgressBarWidget"

        self.coordinate = vtki.vtkCoordinate()
        self.coordinate.SetCoordinateSystemToNormalizedViewport()
        self.mapper.SetTransformCoordinate(self.coordinate)

        self.alpha(alpha)
        self.color(get_color(c))
        self.lw(lw * 2)

    def update(self, fraction=None) -> Self:
        """Update progress bar to fraction of the window width."""
        if fraction is None:
            if self.iterations is None:
                vedo.printc("Error in ProgressBarWindow: must specify iterations", c='r')
                return self
            self.n += 1
            fraction = self.n / self.iterations

        if fraction >= 1 and self.autohide:
            fraction = 0

        psqr = [[0, 0, 0], [fraction, 0, 0]]
        vpts = utils.numpy2vtk(psqr, dtype=np.float32)
        self.dataset.GetPoints().SetData(vpts)
        return self

    def reset(self):
        """Reset progress bar."""
        self.n = 0
        self.update(0)
        return self


#####################################################################
