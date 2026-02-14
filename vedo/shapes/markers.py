#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Marker and special symbol shapes extracted from vedo.shapes."""

from typing import Union, Any
import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import settings, utils
from vedo.transformations import LinearTransform
from vedo.colors import get_color
from vedo.mesh import Mesh
from vedo.pointcloud import Points, merge
from vedo.shapes.curves import Line, Arc, Arrow, Arrow2D, Ribbon
from vedo.shapes.primitives import Polygon, Star, Circle, Disc, Sphere, Cylinder, Cone, Rectangle, Grid
from vedo.shapes.text3d import Text3D
def Marker(symbol, pos=(0, 0, 0), c="k", alpha=1.0, s=0.1, filled=True) -> Any:
    """
    Generate a marker shape. Typically used in association with `Glyph`.
    """
    if isinstance(symbol, Mesh):
        return symbol.c(c).alpha(alpha).lighting("off")

    if isinstance(symbol, int):
        symbs = [".", "o", "O", "0", "p", "*", "h", "D", "d", "v", "^", ">", "<", "s", "x", "a"]
        symbol = symbol % len(symbs)
        symbol = symbs[symbol]

    if symbol == ".":
        mesh = Polygon(nsides=24, r=s * 0.6)
    elif symbol == "o":
        mesh = Polygon(nsides=24, r=s * 0.75)
    elif symbol == "O":
        mesh = Disc(r1=s * 0.6, r2=s * 0.75, res=(1, 24))
    elif symbol == "0":
        m1 = Disc(r1=s * 0.6, r2=s * 0.75, res=(1, 24))
        m2 = Circle(r=s * 0.36).reverse()
        mesh = merge(m1, m2)
    elif symbol == "p":
        mesh = Polygon(nsides=5, r=s)
    elif symbol == "*":
        mesh = Star(r1=0.65 * s * 1.1, r2=s * 1.1, line=not filled)
    elif symbol == "h":
        mesh = Polygon(nsides=6, r=s)
    elif symbol == "D":
        mesh = Polygon(nsides=4, r=s)
    elif symbol == "d":
        mesh = Polygon(nsides=4, r=s * 1.1).scale([0.5, 1, 1])
    elif symbol == "v":
        mesh = Polygon(nsides=3, r=s).rotate_z(180)
    elif symbol == "^":
        mesh = Polygon(nsides=3, r=s)
    elif symbol == ">":
        mesh = Polygon(nsides=3, r=s).rotate_z(-90)
    elif symbol == "<":
        mesh = Polygon(nsides=3, r=s).rotate_z(90)
    elif symbol == "s":
        mesh = Mesh(
            [[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2, 3]]]
        ).scale(s / 1.4)
    elif symbol == "x":
        mesh = Text3D("+", pos=(0, 0, 0), s=s * 2.6, justify="center", depth=0)
        # mesh.rotate_z(45)
    elif symbol == "a":
        mesh = Text3D("*", pos=(0, 0, 0), s=s * 2.6, justify="center", depth=0)
    else:
        mesh = Text3D(symbol, pos=(0, 0, 0), s=s * 2, justify="center", depth=0)
    mesh.flat().lighting("off").wireframe(not filled).c(c).alpha(alpha)
    if len(pos) == 2:
        pos = (pos[0], pos[1], 0)
    mesh.pos(pos)
    mesh.name = "Marker"
    return mesh


class Brace(Mesh):
    """
    Create a brace (bracket) shape.
    """

    def __init__(
        self,
        q1,
        q2,
        style="}",
        padding1=0.0,
        font="Theemim",
        comment="",
        justify=None,
        angle=0.0,
        padding2=0.2,
        s=1.0,
        italic=0,
        c="k1",
        alpha=1.0,
    ) -> None:
        """
        Create a brace (bracket) shape which spans from point q1 to point q2.

        Arguments:
            q1 : (list)
                point 1.
            q2 : (list)
                point 2.
            style : (str)
                style of the bracket, eg. `{}, [], (), <>`.
            padding1 : (float)
                padding space in percent form the input points.
            font : (str)
                font type
            comment : (str)
                additional text to appear next to the brace symbol.
            justify : (str)
                specify the anchor point to justify text comment, e.g. "top-left".
            italic : float
                italicness of the text comment (can be a positive or negative number)
            angle : (float)
                rotation angle of text. Use `None` to keep it horizontal.
            padding2 : (float)
                padding space in percent form brace to text comment.
            s : (float)
                scale factor for the comment

        Examples:
            - [scatter3.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/scatter3.py)

                ![](https://vedo.embl.es/images/pyplot/scatter3.png)
        """
        if isinstance(q1, vtki.vtkActor):
            q1 = q1.GetPosition()
        if isinstance(q2, vtki.vtkActor):
            q2 = q2.GetPosition()
        if len(q1) == 2:
            q1 = [q1[0], q1[1], 0.0]
        if len(q2) == 2:
            q2 = [q2[0], q2[1], 0.0]
        q1 = np.array(q1, dtype=float)
        q2 = np.array(q2, dtype=float)
        mq = (q1 + q2) / 2
        q1 = q1 - mq
        q2 = q2 - mq
        d = np.linalg.norm(q2 - q1)
        q2[2] = q1[2]

        if style not in "{}[]()<>|I":
            vedo.logger.error(f"unknown style {style}." + "Use {}[]()<>|I")
            style = "}"

        flip = False
        if style in ["{", "[", "(", "<"]:
            flip = True
            i = ["{", "[", "(", "<"].index(style)
            style = ["}", "]", ")", ">"][i]

        br = Text3D(style, font="Theemim", justify="center-left")
        br.scale([0.4, 1, 1])

        angler = np.arctan2(q2[1], q2[0]) * 180 / np.pi - 90
        if flip:
            angler += 180

        _, x1, y0, y1, _, _ = br.bounds()
        if comment:
            just = "center-top"
            if angle is None:
                angle = -angler + 90
                if not flip:
                    angle += 180

            if flip:
                angle += 180
                just = "center-bottom"
            if justify is not None:
                just = justify
            cmt = Text3D(comment, font=font, justify=just, italic=italic)
            cx0, cx1 = cmt.xbounds()
            cmt.rotate_z(90 + angle)
            cmt.scale(1 / (cx1 - cx0) * s * len(comment) / 5)
            cmt.shift(x1 * (1 + padding2), 0, 0)
            poly = merge(br, cmt).dataset

        else:
            poly = br.dataset

        tr = vtki.vtkTransform()
        tr.Translate(mq)
        tr.RotateZ(angler)
        tr.Translate(padding1 * d, 0, 0)
        pscale = 1
        tr.Scale(pscale / (y1 - y0) * d, pscale / (y1 - y0) * d, 1)

        tf = vtki.new("TransformPolyDataFilter")
        tf.SetInputData(poly)
        tf.SetTransform(tr)
        tf.Update()
        poly = tf.GetOutput()

        super().__init__(poly, c, alpha)

        self.base = q1
        self.top  = q2
        self.name = "Brace"


class Star3D(Mesh):
    """
    Build a 3D starred shape.
    """

    def __init__(self, pos=(0, 0, 0), r=1.0, thickness=0.1, c="blue4", alpha=1.0) -> None:
        """
        Build a 3D star shape of 5 cusps, mainly useful as a 3D marker.
        """
        pts = ((1.34, 0., -0.37), (5.75e-3, -0.588, thickness/10), (0.377, 0.,-0.38),
               (0.0116, 0., -1.35), (-0.366, 0., -0.384), (-1.33, 0., -0.385),
               (-0.600, 0., 0.321), (-0.829, 0., 1.19), (-1.17e-3, 0., 0.761),
               (0.824, 0., 1.20), (0.602, 0., 0.328), (6.07e-3, 0.588, thickness/10))
        fcs = [[0, 1, 2], [0, 11,10], [2, 1, 3], [2, 11, 0], [3, 1, 4], [3, 11, 2],
               [4, 1, 5], [4, 11, 3], [5, 1, 6], [5, 11, 4], [6, 1, 7], [6, 11, 5],
               [7, 1, 8], [7, 11, 6], [8, 1, 9], [8, 11, 7], [9, 1,10], [9, 11, 8],
               [10,1, 0],[10,11, 9]]

        super().__init__([pts, fcs], c, alpha)
        self.rotate_x(90)
        self.scale(r).lighting("shiny")

        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)
        self.pos(pos)
        self.name = "Star3D"


class Cross3D(Mesh):
    """
    Build a 3D cross shape.
    """

    def __init__(self, pos=(0, 0, 0), s=1.0, thickness=0.3, c="b", alpha=1.0) -> None:
        """
        Build a 3D cross shape, mainly useful as a 3D marker.
        """
        if len(pos) == 2:
            pos = (pos[0], pos[1], 0)

        c1 = Cylinder(r=thickness * s, height=2 * s)
        c2 = Cylinder(r=thickness * s, height=2 * s).rotate_x(90)
        c3 = Cylinder(r=thickness * s, height=2 * s).rotate_y(90)
        poly = merge(c1, c2, c3).color(c).alpha(alpha).pos(pos).dataset
        super().__init__(poly, c, alpha)
        self.name = "Cross3D"


class ParametricShape(Mesh):
    """
    A set of built-in shapes mainly for illustration purposes.
    """

    def __init__(self, name, res=51, n=25, seed=1):
        """
        A set of built-in shapes mainly for illustration purposes.

        Name can be an integer or a string in this list:
            `['Boy', 'ConicSpiral', 'CrossCap', 'Dini', 'Enneper',
            'Figure8Klein', 'Klein', 'Mobius', 'RandomHills', 'Roman',
            'SuperEllipsoid', 'BohemianDome', 'Bour', 'CatalanMinimal',
            'Henneberg', 'Kuen', 'PluckerConoid', 'Pseudosphere']`.

        Example:
            ```python
            from vedo import *
            settings.immediate_rendering = False
            plt = Plotter(N=18)
            for i in range(18):
                ps = ParametricShape(i).color(i)
                plt.at(i).show(ps, ps.name)
            plt.interactive().close()
            ```
            <img src="https://user-images.githubusercontent.com/32848391/69181075-bb6aae80-0b0e-11ea-92f7-d0cd3b9087bf.png" width="700">
        """

        shapes = [
            "Boy",
            "ConicSpiral",
            "CrossCap",
            "Enneper",
            "Figure8Klein",
            "Klein",
            "Dini",
            "Mobius",
            "RandomHills",
            "Roman",
            "SuperEllipsoid",
            "BohemianDome",
            "Bour",
            "CatalanMinimal",
            "Henneberg",
            "Kuen",
            "PluckerConoid",
            "Pseudosphere",
        ]

        if isinstance(name, int):
            name = name % len(shapes)
            name = shapes[name]

        if name == "Boy":
            ps = vtki.new("ParametricBoy")
        elif name == "ConicSpiral":
            ps = vtki.new("ParametricConicSpiral")
        elif name == "CrossCap":
            ps = vtki.new("ParametricCrossCap")
        elif name == "Dini":
            ps = vtki.new("ParametricDini")
        elif name == "Enneper":
            ps = vtki.new("ParametricEnneper")
        elif name == "Figure8Klein":
            ps = vtki.new("ParametricFigure8Klein")
        elif name == "Klein":
            ps = vtki.new("ParametricKlein")
        elif name == "Mobius":
            ps = vtki.new("ParametricMobius")
            ps.SetRadius(2.0)
            ps.SetMinimumV(-0.5)
            ps.SetMaximumV(0.5)
        elif name == "RandomHills":
            ps = vtki.new("ParametricRandomHills")
            ps.AllowRandomGenerationOn()
            ps.SetRandomSeed(seed)
            ps.SetNumberOfHills(n)
        elif name == "Roman":
            ps = vtki.new("ParametricRoman")
        elif name == "SuperEllipsoid":
            ps = vtki.new("ParametricSuperEllipsoid")
            ps.SetN1(0.5)
            ps.SetN2(0.4)
        elif name == "BohemianDome":
            ps = vtki.new("ParametricBohemianDome")
            ps.SetA(5.0)
            ps.SetB(1.0)
            ps.SetC(2.0)
        elif name == "Bour":
            ps = vtki.new("ParametricBour")
        elif name == "CatalanMinimal":
            ps = vtki.new("ParametricCatalanMinimal")
        elif name == "Henneberg":
            ps = vtki.new("ParametricHenneberg")
        elif name == "Kuen":
            ps = vtki.new("ParametricKuen")
            ps.SetDeltaV0(0.001)
        elif name == "PluckerConoid":
            ps = vtki.new("ParametricPluckerConoid")
        elif name == "Pseudosphere":
            ps = vtki.new("ParametricPseudosphere")
        else:
            vedo.logger.error(f"unknown ParametricShape {name}")
            return

        pfs = vtki.new("ParametricFunctionSource")
        pfs.SetParametricFunction(ps)
        pfs.SetUResolution(res)
        pfs.SetVResolution(res)
        pfs.SetWResolution(res)
        pfs.SetScalarModeToZ()
        pfs.Update()

        super().__init__(pfs.GetOutput())

        if name == "RandomHills": self.shift([0,-10,-2.25])
        if name != 'Kuen': self.normalize()
        if name == 'Dini': self.scale(0.4)
        if name == 'Enneper': self.scale(0.4)
        if name == 'ConicSpiral': self.bc('tomato')
        self.name = name
