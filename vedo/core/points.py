#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Point-cloud and mesh geometric transform algorithms."""

from typing import Any
from typing_extensions import Self
import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import utils
from vedo.core.transformations import LinearTransform, NonLinearTransform
from vedo.core.common import CommonAlgorithms

__all__ = ["PointAlgorithms"]


class PointAlgorithms(CommonAlgorithms):
    """Methods for point clouds."""

    def apply_transform(self, LT: Any, deep_copy=True) -> Self:
        """
        Apply a linear or non-linear transformation to the mesh polygonal data.

        Examples:
        ```python
        from vedo import Cube, show, settings
        settings.use_parallel_projection = True
        c1 = Cube().rotate_z(25).pos(2,1).mirror().alpha(0.5)
        T = c1.transform  # rotate by 5 degrees, place at (2,1)
        c2 = Cube().c('red4').wireframe().lw(10).lighting('off')
        c2.apply_transform(T)
        show(c1, c2, "The 2 cubes should overlap!", axes=1).close()
        ```

        ![](https://vedo.embl.es/images/feats/apply_transform.png)
        """
        if self.dataset.GetNumberOfPoints() == 0 or LT is None:
            return self

        if isinstance(LT, LinearTransform):
            LT_is_linear = True
            tr = LT.T
            if LT.is_identity():
                return self

        elif isinstance(
            LT, (vtki.vtkMatrix4x4, vtki.vtkLinearTransform)
        ) or utils.is_sequence(LT):
            LT_is_linear = True
            LT = LinearTransform(LT)
            tr = LT.T
            if LT.is_identity():
                return self

        elif isinstance(LT, NonLinearTransform):
            LT_is_linear = False
            tr = LT.T
            self.transform = LT  # reset

        elif isinstance(LT, vtki.vtkThinPlateSplineTransform):
            LT_is_linear = False
            tr = LT
            self.transform = NonLinearTransform(LT)  # reset

        else:
            vedo.logger.error(f"apply_transform(), unknown input type:\n{LT}")
            return self

        ################
        if LT_is_linear:
            try:
                # self.transform might still not be linear
                self.transform.concatenate(LT)
            except AttributeError:
                # in that case reset it
                self.transform = LT

        ################
        if isinstance(self.dataset, vtki.vtkPolyData):
            tp = vtki.new("TransformPolyDataFilter")
        elif isinstance(
            self.dataset, (vtki.vtkStructuredGrid, vtki.vtkUnstructuredGrid)
        ):
            tp = vtki.new("TransformFilter")
            tp.TransformAllInputVectorsOn()
        else:
            vedo.logger.error(f"apply_transform(), unknown input type: {type(self.dataset)}")
            return self

        tp.SetTransform(tr)
        tp.SetInputData(self.dataset)
        tp.Update()
        out = tp.GetOutput()

        if deep_copy:
            self.dataset.DeepCopy(out)
        else:
            self.dataset.ShallowCopy(out)

        # reset the locators
        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None
        return self

    def apply_transform_from_actor(self) -> LinearTransform:
        """
        Apply the current transformation of the actor to the data.
        Useful when manually moving an actor (eg. when pressing "a").
        Returns the `LinearTransform` object.

        Note that this method is automatically called when the window is closed,
        or the interactor style is changed.
        """
        M = self.actor.GetMatrix()
        self.apply_transform(M)
        LT = LinearTransform(M)
        iden = vtki.vtkMatrix4x4()
        self.actor.PokeMatrix(iden)
        return LT

    def get_transform_from_actor(self) -> LinearTransform:
        """
        Get the current transformation of the actor as a `LinearTransform` object.
        This is useful to retrieve the transformation matrix without applying it to the data.
        """
        M = self.actor.GetMatrix()
        if M is None:
            return LinearTransform()
        return LinearTransform(M)

    def pos(self, x=None, y=None, z=None) -> Self:
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
        delta = np.array([x, y, z]) - q
        if delta[0] == delta[1] == delta[2] == 0:
            return self
        LT = LinearTransform().translate(delta)
        return self.apply_transform(LT)

    def shift(self, dx=0, dy=0, dz=0) -> Self:
        """Add a vector to the current object position."""
        if utils.is_sequence(dx):
            dx, dy, dz = utils.make3d(dx)
        if dx == dy == dz == 0:
            return self
        LT = LinearTransform().translate([dx, dy, dz])
        return self.apply_transform(LT)

    def x(self, val=None) -> Self:
        """Set/Get object position along x axis."""
        p = self.transform.position
        if val is None:
            return p[0]
        self.pos(val, p[1], p[2])
        return self

    def y(self, val=None) -> Self:
        """Set/Get object position along y axis."""
        p = self.transform.position
        if val is None:
            return p[1]
        self.pos(p[0], val, p[2])
        return self

    def z(self, val=None) -> Self:
        """Set/Get object position along z axis."""
        p = self.transform.position
        if val is None:
            return p[2]
        self.pos(p[0], p[1], val)
        return self

    def rotate(self, angle: float, axis=(1, 0, 0), point=(0, 0, 0), rad=False) -> Self:
        """
        Rotate around an arbitrary `axis` passing through `point`.

        Examples:
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
        LT = LinearTransform()
        LT.rotate(angle, axis, point, rad)
        return self.apply_transform(LT)

    def rotate_x(self, angle: float, rad=False, around=None) -> Self:
        """
        Rotate around x-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return self
        LT = LinearTransform().rotate_x(angle, rad, around)
        return self.apply_transform(LT)

    def rotate_y(self, angle: float, rad=False, around=None) -> Self:
        """
        Rotate around y-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return self
        LT = LinearTransform().rotate_y(angle, rad, around)
        return self.apply_transform(LT)

    def rotate_z(self, angle: float, rad=False, around=None) -> Self:
        """
        Rotate around z-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return self
        LT = LinearTransform().rotate_z(angle, rad, around)
        return self.apply_transform(LT)

    def reorient(self, initaxis, newaxis, rotation=0, rad=False, xyplane=False) -> Self:
        """
        Reorient the object to point to a new direction from an initial one.
        If `initaxis` is None, the object will be assumed in its "default" orientation.
        If `xyplane` is True, the object will be rotated to lie on the xy plane.

        Use `rotation` to first rotate the object around its `initaxis`.
        """
        q = self.transform.position
        LT = LinearTransform()
        LT.reorient(initaxis, newaxis, q, rotation, rad, xyplane)
        return self.apply_transform(LT)

    def scale(self, s=None, reset=False, origin=True) -> Self | np.ndarray:
        """
        Set/get object's scaling factor.

        Args:
            s (list, float):
                scaling factor(s).
            reset (bool):
                if True previous scaling factors are ignored.
            origin (bool):
                if True scaling is applied with respect to object's position,
                otherwise is applied respect to (0,0,0).

        Note:
            use `s=(sx,sy,sz)` to scale differently in the three coordinates.
        """
        if s is None:
            return np.array(self.transform.T.GetScale())

        if not utils.is_sequence(s):
            s = [s, s, s]

        LT = LinearTransform()
        if reset:
            old_s = np.array(self.transform.T.GetScale())
            LT.scale(np.array(s) / old_s)
        else:
            if origin is True:
                LT.scale(s, origin=self.transform.position)
            elif origin is False:
                LT.scale(s, origin=False)
            else:
                LT.scale(s, origin=origin)

        return self.apply_transform(LT)


###############################################################################
