#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Axes construction helpers extracted from vedo.addons."""

import os
from typing import Union
import numpy as np
from typing_extensions import Self

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
from vedo.addons_measure import compute_visible_bounds, Ruler3D, RulerAxes
from vedo.addons_cutters import BoxCutter, PlaneCutter, SphereCutter
def Axes(
        obj=None,
        xtitle='x', ytitle='y', ztitle='z',
        xrange=None, yrange=None, zrange=None,
        c=None,
        number_of_divisions=None,
        digits=None,
        limit_ratio=0.04,
        title_depth=0,
        title_font="", # grab settings.default_font
        text_scale=1.0,
        x_values_and_labels=None, y_values_and_labels=None, z_values_and_labels=None,
        htitle="",
        htitle_size=0.03,
        htitle_font=None,
        htitle_italic=False,
        htitle_color=None, htitle_backface_color=None,
        htitle_justify='bottom-left',
        htitle_rotation=0,
        htitle_offset=(0, 0.01, 0),
        xtitle_position=0.95, ytitle_position=0.95, ztitle_position=0.95,
        # xtitle_offset can be a list (dx,dy,dz)
        xtitle_offset=0.025,  ytitle_offset=0.0275, ztitle_offset=0.02,
        xtitle_justify=None,  ytitle_justify=None,  ztitle_justify=None,
        # xtitle_rotation can be a list (rx,ry,rz)
        xtitle_rotation=0, ytitle_rotation=0, ztitle_rotation=0,
        xtitle_box=False,  ytitle_box=False,
        xtitle_size=0.025, ytitle_size=0.025, ztitle_size=0.025,
        xtitle_color=None, ytitle_color=None, ztitle_color=None,
        xtitle_backface_color=None, ytitle_backface_color=None, ztitle_backface_color=None,
        xtitle_italic=0, ytitle_italic=0, ztitle_italic=0,
        grid_linewidth=1,
        xygrid=True,   yzgrid=False,  zxgrid=False,
        xygrid2=False, yzgrid2=False, zxgrid2=False,
        xygrid_transparent=False,  yzgrid_transparent=False,  zxgrid_transparent=False,
        xygrid2_transparent=False, yzgrid2_transparent=False, zxgrid2_transparent=False,
        xyplane_color=None, yzplane_color=None, zxplane_color=None,
        xygrid_color=None, yzgrid_color=None, zxgrid_color=None,
        xyalpha=0.075, yzalpha=0.075, zxalpha=0.075,
        xyframe_line=None, yzframe_line=None, zxframe_line=None,
        xyframe_color=None, yzframe_color=None, zxframe_color=None,
        axes_linewidth=1,
        xline_color=None, yline_color=None, zline_color=None,
        xhighlight_zero=False, yhighlight_zero=False, zhighlight_zero=False,
        xhighlight_zero_color='red4', yhighlight_zero_color='green4', zhighlight_zero_color='blue4',
        show_ticks=True,
        xtick_length=0.015, ytick_length=0.015, ztick_length=0.015,
        xtick_thickness=0.0025, ytick_thickness=0.0025, ztick_thickness=0.0025,
        xminor_ticks=1, yminor_ticks=1, zminor_ticks=1,
        tip_size=None,
        label_font="", # grab settings.default_font
        xlabel_color=None, ylabel_color=None, zlabel_color=None,
        xlabel_backface_color=None, ylabel_backface_color=None, zlabel_backface_color=None,
        xlabel_size=0.016, ylabel_size=0.016, zlabel_size=0.016,
        xlabel_offset=0.8, ylabel_offset=0.8, zlabel_offset=0.8, # each can be a list (dx,dy,dz)
        xlabel_justify=None, ylabel_justify=None, zlabel_justify=None,
        xlabel_rotation=0, ylabel_rotation=0, zlabel_rotation=0, # each can be a list (rx,ry,rz)
        xaxis_rotation=0, yaxis_rotation=0, zaxis_rotation=0,    # rotate all elements around axis
        xyshift=0, yzshift=0, zxshift=0,
        xshift_along_y=0, xshift_along_z=0,
        yshift_along_x=0, yshift_along_z=0,
        zshift_along_x=0, zshift_along_y=0,
        x_use_bounds=True, y_use_bounds=True, z_use_bounds=False,
        x_inverted=False, y_inverted=False, z_inverted=False,
        use_global=False,
        tol=0.001,
    ) -> Union[Assembly, None]:
    """
    Draw axes for the input object.
    Check [available fonts here](https://vedo.embl.es/fonts).

    Returns an `vedo.Assembly` object.

    Parameters
    ----------

    - `xtitle`,                 ['x'], x-axis title text
    - `xrange`,                [None], x-axis range in format (xmin, ymin), default is automatic.
    - `number_of_divisions`,   [None], approximate number of divisions on the longest axis
    - `axes_linewidth`,           [1], width of the axes lines
    - `grid_linewidth`,           [1], width of the grid lines
    - `title_depth`,              [0], extrusion fractional depth of title text
    - `x_values_and_labels`        [], assign custom tick positions and labels [(pos1, label1), ...]
    - `xygrid`,                [True], show a gridded wall on plane xy
    - `yzgrid`,                [True], show a gridded wall on plane yz
    - `zxgrid`,                [True], show a gridded wall on plane zx
    - `yzgrid2`,              [False], show yz plane on opposite side of the bounding box
    - `zxgrid2`,              [False], show zx plane on opposite side of the bounding box
    - `xygrid_transparent`    [False], make grid plane completely transparent
    - `xygrid2_transparent`   [False], make grid plane completely transparent on opposite side box
    - `xyplane_color`,       ['None'], color of the plane
    - `xygrid_color`,        ['None'], grid line color
    - `xyalpha`,               [0.15], grid plane opacity
    - `xyframe_line`,             [0], add a frame for the plane, use value as the thickness
    - `xyframe_color`,         [None], color for the frame of the plane
    - `show_ticks`,            [True], show major ticks
    - `digits`,                [None], use this number of significant digits in scientific notation
    - `title_font`,              [''], font for axes titles
    - `label_font`,              [''], font for numeric labels
    - `text_scale`,             [1.0], global scaling factor for all text elements (titles, labels)
    - `htitle`,                  [''], header title
    - `htitle_size`,           [0.03], header title size
    - `htitle_font`,           [None], header font (defaults to `title_font`)
    - `htitle_italic`,         [True], header font is italic
    - `htitle_color`,          [None], header title color (defaults to `xtitle_color`)
    - `htitle_backface_color`, [None], header title color on its backface
    - `htitle_justify`, ['bottom-center'], origin of the title justification
    - `htitle_offset`,   [(0,0.01,0)], control offsets of header title in x, y and z
    - `xtitle_position`,       [0.32], title fractional positions along axis
    - `xtitle_offset`,         [0.05], title fractional offset distance from axis line, can be a list
    - `xtitle_justify`,        [None], choose the origin of the bounding box of title
    - `xtitle_rotation`,          [0], add a rotation of the axis title, can be a list (rx,ry,rz)
    - `xtitle_box`,           [False], add a box around title text
    - `xline_color`,      [automatic], color of the x-axis
    - `xtitle_color`,     [automatic], color of the axis title
    - `xtitle_backface_color`, [None], color of axis title on its backface
    - `xtitle_size`,          [0.025], size of the axis title
    - `xtitle_italic`,            [0], a bool or float to make the font italic
    - `xhighlight_zero`,       [True], draw a line highlighting zero position if in range
    - `xhighlight_zero_color`, [auto], color of the line highlighting the zero position
    - `xtick_length`,         [0.005], radius of the major ticks
    - `xtick_thickness`,     [0.0025], thickness of the major ticks along their axis
    - `xminor_ticks`,             [1], number of minor ticks between two major ticks
    - `xlabel_color`,     [automatic], color of numeric labels and ticks
    - `xlabel_backface_color`, [auto], back face color of numeric labels and ticks
    - `xlabel_size`,          [0.015], size of the numeric labels along axis
    - `xlabel_rotation`,     [0,list], numeric labels rotation (can be a list of 3 rotations)
    - `xlabel_offset`,     [0.8,list], offset of the numeric labels (can be a list of 3 offsets)
    - `xlabel_justify`,        [None], choose the origin of the bounding box of labels
    - `xaxis_rotation`,           [0], rotate the X axis elements (ticks and labels) around this same axis
    - `xyshift`                 [0.0], slide the xy-plane along z (the range is [0,1])
    - `xshift_along_y`          [0.0], slide x-axis along the y-axis (the range is [0,1])
    - `tip_size`,              [0.01], size of the arrow tip as a fraction of the bounding box diagonal
    - `limit_ratio`,           [0.04], below this ratio don't plot smaller axis
    - `x_use_bounds`,          [True], keep into account space occupied by labels when setting camera
    - `x_inverted`,           [False], invert labels order and direction (only visually!)
    - `use_global`,           [False], try to compute the global bounding box of visible actors

    Example:
        ```python
        from vedo import Axes, Box, show
        box = Box(pos=(1,2,3), size=(8,9,7)).alpha(0.1)
        axs = Axes(box, c='k')  # returns an Assembly object
        for a in axs.unpack():
            print(a.name)
        show(box, axs).close()
        ```
        ![](https://vedo.embl.es/images/feats/axes1.png)

    Examples:
        - [custom_axes1.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/custom_axes1.py)
        - [custom_axes2.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/custom_axes2.py)
        - [custom_axes3.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/custom_axes3.py)
        - [custom_axes4.py](https://github.com/marcomusy/vedo/tree/master/examples/pyplot/custom_axes4.py)

        ![](https://vedo.embl.es/images/pyplot/customAxes3.png)
    """
    if not title_font:
        title_font = vedo.settings.default_font
    if not label_font:
        label_font = vedo.settings.default_font

    if c is None:  # automatic black or white
        c = (0.1, 0.1, 0.1)
        plt = vedo.current_plotter()
        if plt and plt.renderer:
            bgcol = plt.renderer.GetBackground()
        else:
            bgcol = (1, 1, 1)
        if np.sum(bgcol) < 1.5:
            c = (0.9, 0.9, 0.9)
    else:
        c = get_color(c)

    # Check if obj has bounds, if so use those
    if obj is not None:
        try:
            bb = obj.bounds()
        except AttributeError:
            try:
                bb = obj.GetBounds()
                if xrange is None: xrange = (bb[0], bb[1])
                if yrange is None: yrange = (bb[2], bb[3])
                if zrange is None: zrange = (bb[4], bb[5])
                obj = None # dont need it anymore
            except AttributeError:
                pass
        if utils.is_sequence(obj) and len(obj) == 6 and utils.is_number(obj[0]):
            # passing a list of numeric bounds
            if xrange is None: xrange = (obj[0], obj[1])
            if yrange is None: yrange = (obj[2], obj[3])
            if zrange is None: zrange = (obj[4], obj[5])

    if use_global:
        vbb, drange, min_bns, max_bns = compute_visible_bounds()
    else:
        if obj is not None:
            vbb, drange, min_bns, max_bns = compute_visible_bounds(obj)
        else:
            vbb = np.zeros(6)
            drange = np.zeros(3)
            if zrange is None:
                zrange = (0, 0)
            if xrange is None or yrange is None:
                vedo.logger.error("in Axes() must specify axes ranges!")
                return None  ###########################################

    if xrange is not None:
        if xrange[1] < xrange[0]:
            x_inverted = True
            xrange = [xrange[1], xrange[0]]
        vbb[0], vbb[1] = xrange
        drange[0] = vbb[1] - vbb[0]
        min_bns = vbb
        max_bns = vbb
    if yrange is not None:
        if yrange[1] < yrange[0]:
            y_inverted = True
            yrange = [yrange[1], yrange[0]]
        vbb[2], vbb[3] = yrange
        drange[1] = vbb[3] - vbb[2]
        min_bns = vbb
        max_bns = vbb
    if zrange is not None:
        if zrange[1] < zrange[0]:
            z_inverted = True
            zrange = [zrange[1], zrange[0]]
        vbb[4], vbb[5] = zrange
        drange[2] = vbb[5] - vbb[4]
        min_bns = vbb
        max_bns = vbb

    drangemax = max(drange)
    if not drangemax:
        return None

    if drange[0] / drangemax < limit_ratio:
        drange[0] = 0
        xtitle = ""
    if drange[1] / drangemax < limit_ratio:
        drange[1] = 0
        ytitle = ""
    if drange[2] / drangemax < limit_ratio:
        drange[2] = 0
        ztitle = ""

    x0, x1, y0, y1, z0, z1 = vbb
    dx, dy, dz = drange

    gscale = np.sqrt(dx * dx + dy * dy + dz * dz) * 0.75

    if not xyplane_color: xyplane_color = c
    if not yzplane_color: yzplane_color = c
    if not zxplane_color: zxplane_color = c
    if not xygrid_color:  xygrid_color = c
    if not yzgrid_color:  yzgrid_color = c
    if not zxgrid_color:  zxgrid_color = c
    if not xtitle_color:  xtitle_color = c
    if not ytitle_color:  ytitle_color = c
    if not ztitle_color:  ztitle_color = c
    if not xline_color:   xline_color = c
    if not yline_color:   yline_color = c
    if not zline_color:   zline_color = c
    if not xlabel_color:  xlabel_color = xline_color
    if not ylabel_color:  ylabel_color = yline_color
    if not zlabel_color:  zlabel_color = zline_color

    if tip_size is None:
        tip_size = 0.005 * gscale
        if not ztitle:
            tip_size = 0  # switch off in xy 2d

    ndiv = 4
    if not ztitle or not ytitle or not xtitle:  # make more default ticks if 2D
        ndiv = 6
        if not ztitle:
            if xyframe_line is None:
                xyframe_line = True
            if tip_size is None:
                tip_size = False

    if utils.is_sequence(number_of_divisions):
        rx, ry, rz = number_of_divisions
    else:
        if not number_of_divisions:
            number_of_divisions = ndiv
        if not drangemax or np.any(np.isnan(drange)):
            rx, ry, rz = 1, 1, 1
        else:
            rx, ry, rz = np.ceil(drange / drangemax * number_of_divisions).astype(int)

    if xtitle:
        xticks_float, xticks_str = utils.make_ticks(x0, x1, rx, x_values_and_labels, digits)
        xticks_float = xticks_float * dx
        if x_inverted:
            xticks_float = np.flip(-(xticks_float - xticks_float[-1]))
            xticks_str = list(reversed(xticks_str))
            xticks_str[-1] = ""
            xhighlight_zero = False
    if ytitle:
        yticks_float, yticks_str = utils.make_ticks(y0, y1, ry, y_values_and_labels, digits)
        yticks_float = yticks_float * dy
        if y_inverted:
            yticks_float = np.flip(-(yticks_float - yticks_float[-1]))
            yticks_str = list(reversed(yticks_str))
            yticks_str[-1] = ""
            yhighlight_zero = False
    if ztitle:
        zticks_float, zticks_str = utils.make_ticks(z0, z1, rz, z_values_and_labels, digits)
        zticks_float = zticks_float * dz
        if z_inverted:
            zticks_float = np.flip(-(zticks_float - zticks_float[-1]))
            zticks_str = list(reversed(zticks_str))
            zticks_str[-1] = ""
            zhighlight_zero = False

    ################################################ axes lines
    lines = []
    if xtitle:
        axlinex = shapes.Line([0,0,0], [dx,0,0], c=xline_color, lw=axes_linewidth)
        axlinex.shift([0, zxshift*dy + xshift_along_y*dy, xyshift*dz + xshift_along_z*dz])
        axlinex.name = 'xAxis'
        lines.append(axlinex)
    if ytitle:
        axliney = shapes.Line([0,0,0], [0,dy,0], c=yline_color, lw=axes_linewidth)
        axliney.shift([yzshift*dx + yshift_along_x*dx, 0, xyshift*dz + yshift_along_z*dz])
        axliney.name = 'yAxis'
        lines.append(axliney)
    if ztitle:
        axlinez = shapes.Line([0,0,0], [0,0,dz], c=zline_color, lw=axes_linewidth)
        axlinez.shift([yzshift*dx + zshift_along_x*dx, zxshift*dy + zshift_along_y*dy, 0])
        axlinez.name = 'zAxis'
        lines.append(axlinez)

    ################################################ grid planes
    # all shapes have a name to keep track of them in the Assembly
    # if user wants to unpack it
    grids = []
    if xygrid and xtitle and ytitle:
        if not xygrid_transparent:
            gxy = shapes.Grid(s=(xticks_float, yticks_float))
            gxy.alpha(xyalpha).c(xyplane_color).lw(0)
            if xyshift: gxy.shift([0,0,xyshift*dz])
            elif tol:   gxy.shift([0,0,-tol*gscale])
            gxy.name = "xyGrid"
            grids.append(gxy)
        if grid_linewidth:
            gxy_lines = shapes.Grid(s=(xticks_float, yticks_float))
            gxy_lines.c(xyplane_color).lw(grid_linewidth).alpha(xyalpha)
            if xyshift: gxy_lines.shift([0,0,xyshift*dz])
            elif tol:   gxy_lines.shift([0,0,-tol*gscale])
            gxy_lines.name = "xyGridLines"
            grids.append(gxy_lines)

    if yzgrid and ytitle and ztitle:
        if not yzgrid_transparent:
            gyz = shapes.Grid(s=(zticks_float, yticks_float))
            gyz.alpha(yzalpha).c(yzplane_color).lw(0).rotate_y(-90)
            if yzshift: gyz.shift([yzshift*dx,0,0])
            elif tol:   gyz.shift([-tol*gscale,0,0])
            gyz.name = "yzGrid"
            grids.append(gyz)
        if grid_linewidth:
            gyz_lines = shapes.Grid(s=(zticks_float, yticks_float))
            gyz_lines.c(yzplane_color).lw(grid_linewidth).alpha(yzalpha).rotate_y(-90)
            if yzshift: gyz_lines.shift([yzshift*dx,0,0])
            elif tol:   gyz_lines.shift([-tol*gscale,0,0])
            gyz_lines.name = "yzGridLines"
            grids.append(gyz_lines)

    if zxgrid and ztitle and xtitle:
        if not zxgrid_transparent:
            gzx = shapes.Grid(s=(xticks_float, zticks_float))
            gzx.alpha(zxalpha).c(zxplane_color).lw(0).rotate_x(90)
            if zxshift: gzx.shift([0,zxshift*dy,0])
            elif tol:   gzx.shift([0,-tol*gscale,0])
            gzx.name = "zxGrid"
            grids.append(gzx)
        if grid_linewidth:
            gzx_lines = shapes.Grid(s=(xticks_float, zticks_float))
            gzx_lines.c(zxplane_color).lw(grid_linewidth).alpha(zxalpha).rotate_x(90)
            if zxshift: gzx_lines.shift([0,zxshift*dy,0])
            elif tol:   gzx_lines.shift([0,-tol*gscale,0])
            gzx_lines.name = "zxGridLines"
            grids.append(gzx_lines)

    # Grid2
    if xygrid2 and xtitle and ytitle:
        if not xygrid2_transparent:
            gxy2 = shapes.Grid(s=(xticks_float, yticks_float)).z(dz)
            gxy2.alpha(xyalpha).c(xyplane_color).lw(0)
            gxy2.shift([0, tol * gscale, 0])
            gxy2.name = "xyGrid2"
            grids.append(gxy2)
        if grid_linewidth:
            gxy2_lines = shapes.Grid(s=(xticks_float, yticks_float)).z(dz)
            gxy2_lines.c(xyplane_color).lw(grid_linewidth).alpha(xyalpha)
            gxy2_lines.shift([0, tol * gscale, 0])
            gxy2_lines.name = "xygrid2Lines"
            grids.append(gxy2_lines)

    if yzgrid2 and ytitle and ztitle:
        if not yzgrid2_transparent:
            gyz2 = shapes.Grid(s=(zticks_float, yticks_float))
            gyz2.alpha(yzalpha).c(yzplane_color).lw(0)
            gyz2.rotate_y(-90).x(dx).shift([tol * gscale, 0, 0])
            gyz2.name = "yzGrid2"
            grids.append(gyz2)
        if grid_linewidth:
            gyz2_lines = shapes.Grid(s=(zticks_float, yticks_float))
            gyz2_lines.c(yzplane_color).lw(grid_linewidth).alpha(yzalpha)
            gyz2_lines.rotate_y(-90).x(dx).shift([tol * gscale, 0, 0])
            gyz2_lines.name = "yzGrid2Lines"
            grids.append(gyz2_lines)

    if zxgrid2 and ztitle and xtitle:
        if not zxgrid2_transparent:
            gzx2 = shapes.Grid(s=(xticks_float, zticks_float))
            gzx2.alpha(zxalpha).c(zxplane_color).lw(0)
            gzx2.rotate_x(90).y(dy).shift([0, tol * gscale, 0])
            gzx2.name = "zxGrid2"
            grids.append(gzx2)
        if grid_linewidth:
            gzx2_lines = shapes.Grid(s=(xticks_float, zticks_float))
            gzx2_lines.c(zxplane_color).lw(grid_linewidth).alpha(zxalpha)
            gzx2_lines.rotate_x(90).y(dy).shift([0, tol * gscale, 0])
            gzx2_lines.name = "zxGrid2Lines"
            grids.append(gzx2_lines)

    ################################################ frame lines
    framelines = []
    if xyframe_line and xtitle and ytitle:
        if not xyframe_color:
            xyframe_color = xygrid_color
        frxy = shapes.Line(
            [[0, dy, 0], [dx, dy, 0], [dx, 0, 0], [0, 0, 0], [0, dy, 0]],
            c=xyframe_color,
            lw=xyframe_line,
        )
        frxy.shift([0, 0, xyshift * dz])
        frxy.name = "xyFrameLine"
        framelines.append(frxy)
    if yzframe_line and ytitle and ztitle:
        if not yzframe_color:
            yzframe_color = yzgrid_color
        fryz = shapes.Line(
            [[0, 0, dz], [0, dy, dz], [0, dy, 0], [0, 0, 0], [0, 0, dz]],
            c=yzframe_color,
            lw=yzframe_line,
        )
        fryz.shift([yzshift * dx, 0, 0])
        fryz.name = "yzFrameLine"
        framelines.append(fryz)
    if zxframe_line and ztitle and xtitle:
        if not zxframe_color:
            zxframe_color = zxgrid_color
        frzx = shapes.Line(
            [[0, 0, dz], [dx, 0, dz], [dx, 0, 0], [0, 0, 0], [0, 0, dz]],
            c=zxframe_color,
            lw=zxframe_line,
        )
        frzx.shift([0, zxshift * dy, 0])
        frzx.name = "zxFrameLine"
        framelines.append(frzx)

    ################################################ zero lines highlights
    highlights = []
    if xygrid and xtitle and ytitle:
        if xhighlight_zero and min_bns[0] <= 0 and max_bns[1] > 0:
            xhl = -min_bns[0]
            hxy = shapes.Line([xhl, 0, 0], [xhl, dy, 0], c=xhighlight_zero_color)
            hxy.alpha(np.sqrt(xyalpha)).lw(grid_linewidth * 2)
            hxy.shift([0, 0, xyshift * dz])
            hxy.name = "xyHighlightZero"
            highlights.append(hxy)
        if yhighlight_zero and min_bns[2] <= 0 and max_bns[3] > 0:
            yhl = -min_bns[2]
            hyx = shapes.Line([0, yhl, 0], [dx, yhl, 0], c=yhighlight_zero_color)
            hyx.alpha(np.sqrt(yzalpha)).lw(grid_linewidth * 2)
            hyx.shift([0, 0, xyshift * dz])
            hyx.name = "yxHighlightZero"
            highlights.append(hyx)

    if yzgrid and ytitle and ztitle:
        if yhighlight_zero and min_bns[2] <= 0 and max_bns[3] > 0:
            yhl = -min_bns[2]
            hyz = shapes.Line([0, yhl, 0], [0, yhl, dz], c=yhighlight_zero_color)
            hyz.alpha(np.sqrt(yzalpha)).lw(grid_linewidth * 2)
            hyz.shift([yzshift * dx, 0, 0])
            hyz.name = "yzHighlightZero"
            highlights.append(hyz)
        if zhighlight_zero and min_bns[4] <= 0 and max_bns[5] > 0:
            zhl = -min_bns[4]
            hzy = shapes.Line([0, 0, zhl], [0, dy, zhl], c=zhighlight_zero_color)
            hzy.alpha(np.sqrt(yzalpha)).lw(grid_linewidth * 2)
            hzy.shift([yzshift * dx, 0, 0])
            hzy.name = "zyHighlightZero"
            highlights.append(hzy)

    if zxgrid and ztitle and xtitle:
        if zhighlight_zero and min_bns[4] <= 0 and max_bns[5] > 0:
            zhl = -min_bns[4]
            hzx = shapes.Line([0, 0, zhl], [dx, 0, zhl], c=zhighlight_zero_color)
            hzx.alpha(np.sqrt(zxalpha)).lw(grid_linewidth * 2)
            hzx.shift([0, zxshift * dy, 0])
            hzx.name = "zxHighlightZero"
            highlights.append(hzx)
        if xhighlight_zero and min_bns[0] <= 0 and max_bns[1] > 0:
            xhl = -min_bns[0]
            hxz = shapes.Line([xhl, 0, 0], [xhl, 0, dz], c=xhighlight_zero_color)
            hxz.alpha(np.sqrt(zxalpha)).lw(grid_linewidth * 2)
            hxz.shift([0, zxshift * dy, 0])
            hxz.name = "xzHighlightZero"
            highlights.append(hxz)

    ################################################ arrow cone
    cones = []

    if tip_size:

        if xtitle:
            if x_inverted:
                cx = shapes.Cone(
                    r=tip_size,
                    height=tip_size * 2,
                    axis=(-1, 0, 0),
                    c=xline_color,
                    res=12,
                )
            else:
                cx = shapes.Cone(
                    (dx, 0, 0),
                    r=tip_size,
                    height=tip_size * 2,
                    axis=(1, 0, 0),
                    c=xline_color,
                    res=12,
                )
            T = LinearTransform()
            T.translate(
                [
                    0,
                    zxshift * dy + xshift_along_y * dy,
                    xyshift * dz + xshift_along_z * dz,
                ]
            )
            cx.apply_transform(T)
            cx.name = "xTipCone"
            cones.append(cx)

        if ytitle:
            if y_inverted:
                cy = shapes.Cone(
                    r=tip_size,
                    height=tip_size * 2,
                    axis=(0, -1, 0),
                    c=yline_color,
                    res=12,
                )
            else:
                cy = shapes.Cone(
                    (0, dy, 0),
                    r=tip_size,
                    height=tip_size * 2,
                    axis=(0, 1, 0),
                    c=yline_color,
                    res=12,
                )
            T = LinearTransform()
            T.translate(
                [
                    yzshift * dx + yshift_along_x * dx,
                    0,
                    xyshift * dz + yshift_along_z * dz,
                ]
            )
            cy.apply_transform(T)
            cy.name = "yTipCone"
            cones.append(cy)

        if ztitle:
            if z_inverted:
                cz = shapes.Cone(
                    r=tip_size,
                    height=tip_size * 2,
                    axis=(0, 0, -1),
                    c=zline_color,
                    res=12,
                )
            else:
                cz = shapes.Cone(
                    (0, 0, dz),
                    r=tip_size,
                    height=tip_size * 2,
                    axis=(0, 0, 1),
                    c=zline_color,
                    res=12,
                )
            T = LinearTransform()
            T.translate(
                [
                    yzshift * dx + zshift_along_x * dx,
                    zxshift * dy + zshift_along_y * dy,
                    0,
                ]
            )
            cz.apply_transform(T)
            cz.name = "zTipCone"
            cones.append(cz)

    ################################################################# MAJOR ticks
    majorticks, minorticks = [], []
    xticks, yticks, zticks = [], [], []
    if show_ticks:
        if xtitle:
            tick_thickness = xtick_thickness * gscale / 2
            tick_length = xtick_length * gscale / 2
            for i in range(1, len(xticks_float) - 1):
                v1 = (xticks_float[i] - tick_thickness, -tick_length, 0)
                v2 = (xticks_float[i] + tick_thickness, tick_length, 0)
                xticks.append(shapes.Rectangle(v1, v2))
            if len(xticks) > 1:
                xmajticks = merge(xticks).c(xlabel_color)
                T = LinearTransform()
                T.rotate_x(xaxis_rotation)
                T.translate([0, zxshift*dy + xshift_along_y*dy, xyshift*dz + xshift_along_z*dz])
                xmajticks.apply_transform(T)
                xmajticks.name = "xMajorTicks"
                majorticks.append(xmajticks)
        if ytitle:
            tick_thickness = ytick_thickness * gscale / 2
            tick_length = ytick_length * gscale / 2
            for i in range(1, len(yticks_float) - 1):
                v1 = (-tick_length, yticks_float[i] - tick_thickness, 0)
                v2 = (tick_length, yticks_float[i] + tick_thickness, 0)
                yticks.append(shapes.Rectangle(v1, v2))
            if len(yticks) > 1:
                ymajticks = merge(yticks).c(ylabel_color)
                T = LinearTransform()
                T.rotate_y(yaxis_rotation)
                T.translate([yzshift*dx + yshift_along_x*dx, 0, xyshift*dz + yshift_along_z*dz])
                ymajticks.apply_transform(T)
                ymajticks.name = "yMajorTicks"
                majorticks.append(ymajticks)
        if ztitle:
            tick_thickness = ztick_thickness * gscale / 2
            tick_length = ztick_length * gscale / 2.85
            for i in range(1, len(zticks_float) - 1):
                v1 = (zticks_float[i] - tick_thickness, -tick_length, 0)
                v2 = (zticks_float[i] + tick_thickness, tick_length, 0)
                zticks.append(shapes.Rectangle(v1, v2))
            if len(zticks) > 1:
                zmajticks = merge(zticks).c(zlabel_color)
                T = LinearTransform()
                T.rotate_y(-90).rotate_z(-45 + zaxis_rotation)
                T.translate([yzshift*dx + zshift_along_x*dx, zxshift*dy + zshift_along_y*dy, 0])
                zmajticks.apply_transform(T)
                zmajticks.name = "zMajorTicks"
                majorticks.append(zmajticks)

        ############################################################# MINOR ticks
        if xtitle and xminor_ticks and len(xticks) > 1:
            tick_thickness = xtick_thickness * gscale / 4
            tick_length = xtick_length * gscale / 4
            xminor_ticks += 1
            ticks = []
            for i in range(1, len(xticks)):
                t0, t1 = xticks[i - 1].pos(), xticks[i].pos()
                dt = t1 - t0
                for j in range(1, xminor_ticks):
                    mt = dt * (j / xminor_ticks) + t0
                    v1 = (mt[0] - tick_thickness, -tick_length, 0)
                    v2 = (mt[0] + tick_thickness, tick_length, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = xticks[0].pos(), xticks[1].pos()
            dt = t1 - t0
            for j in range(1, xminor_ticks):
                mt = t0 - dt * (j / xminor_ticks)
                if mt[0] < 0:
                    break
                v1 = (mt[0] - tick_thickness, -tick_length, 0)
                v2 = (mt[0] + tick_thickness, tick_length, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = xticks[-2].pos(), xticks[-1].pos()
            dt = t1 - t0
            for j in range(1, xminor_ticks):
                mt = t1 + dt * (j / xminor_ticks)
                if mt[0] > dx:
                    break
                v1 = (mt[0] - tick_thickness, -tick_length, 0)
                v2 = (mt[0] + tick_thickness, tick_length, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if ticks:
                xminticks = merge(ticks).c(xlabel_color)
                T = LinearTransform()
                T.rotate_x(xaxis_rotation)
                T.translate([0, zxshift*dy + xshift_along_y*dy, xyshift*dz + xshift_along_z*dz])
                xminticks.apply_transform(T)
                xminticks.name = "xMinorTicks"
                minorticks.append(xminticks)

        if ytitle and yminor_ticks and len(yticks) > 1:  ##### y
            tick_thickness = ytick_thickness * gscale / 4
            tick_length = ytick_length * gscale / 4
            yminor_ticks += 1
            ticks = []
            for i in range(1, len(yticks)):
                t0, t1 = yticks[i - 1].pos(), yticks[i].pos()
                dt = t1 - t0
                for j in range(1, yminor_ticks):
                    mt = dt * (j / yminor_ticks) + t0
                    v1 = (-tick_length, mt[1] - tick_thickness, 0)
                    v2 = (tick_length, mt[1] + tick_thickness, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = yticks[0].pos(), yticks[1].pos()
            dt = t1 - t0
            for j in range(1, yminor_ticks):
                mt = t0 - dt * (j / yminor_ticks)
                if mt[1] < 0:
                    break
                v1 = (-tick_length, mt[1] - tick_thickness, 0)
                v2 = (tick_length, mt[1] + tick_thickness, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = yticks[-2].pos(), yticks[-1].pos()
            dt = t1 - t0
            for j in range(1, yminor_ticks):
                mt = t1 + dt * (j / yminor_ticks)
                if mt[1] > dy:
                    break
                v1 = (-tick_length, mt[1] - tick_thickness, 0)
                v2 = (tick_length, mt[1] + tick_thickness, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if ticks:
                yminticks = merge(ticks).c(ylabel_color)
                T = LinearTransform()
                T.rotate_y(yaxis_rotation)
                T.translate([yzshift*dx + yshift_along_x*dx, 0, xyshift*dz + yshift_along_z*dz])
                yminticks.apply_transform(T)
                yminticks.name = "yMinorTicks"
                minorticks.append(yminticks)

        if ztitle and zminor_ticks and len(zticks) > 1:  ##### z
            tick_thickness = ztick_thickness * gscale / 4
            tick_length = ztick_length * gscale / 5
            zminor_ticks += 1
            ticks = []
            for i in range(1, len(zticks)):
                t0, t1 = zticks[i - 1].pos(), zticks[i].pos()
                dt = t1 - t0
                for j in range(1, zminor_ticks):
                    mt = dt * (j / zminor_ticks) + t0
                    v1 = (mt[0] - tick_thickness, -tick_length, 0)
                    v2 = (mt[0] + tick_thickness, tick_length, 0)
                    ticks.append(shapes.Rectangle(v1, v2))

            # finish off the fist lower range from start to first tick
            t0, t1 = zticks[0].pos(), zticks[1].pos()
            dt = t1 - t0
            for j in range(1, zminor_ticks):
                mt = t0 - dt * (j / zminor_ticks)
                if mt[0] < 0:
                    break
                v1 = (mt[0] - tick_thickness, -tick_length, 0)
                v2 = (mt[0] + tick_thickness, tick_length, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            # finish off the last upper range from last tick to end
            t0, t1 = zticks[-2].pos(), zticks[-1].pos()
            dt = t1 - t0
            for j in range(1, zminor_ticks):
                mt = t1 + dt * (j / zminor_ticks)
                if mt[0] > dz:
                    break
                v1 = (mt[0] - tick_thickness, -tick_length, 0)
                v2 = (mt[0] + tick_thickness, tick_length, 0)
                ticks.append(shapes.Rectangle(v1, v2))

            if ticks:
                zminticks = merge(ticks).c(zlabel_color)
                T = LinearTransform()
                T.rotate_y(-90).rotate_z(-45 + zaxis_rotation)
                T.translate([yzshift*dx + zshift_along_x*dx, zxshift*dy + zshift_along_y*dy, 0])
                zminticks.apply_transform(T)
                zminticks.name = "zMinorTicks"
                minorticks.append(zminticks)

    ################################################ axes NUMERIC text labels
    labels = []
    xlab, ylab, zlab = None, None, None

    if xlabel_size and xtitle:

        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(xlabel_rotation):  # unpck 3 rotations
            zRot, xRot, yRot = xlabel_rotation
        else:
            zRot = xlabel_rotation
        if zRot < 0:  # deal with negative angles
            zRot += 360

        jus = "center-top"
        if zRot:
            if zRot >  24: jus = "top-right"
            if zRot >  67: jus = "center-right"
            if zRot > 112: jus = "right-bottom"
            if zRot > 157: jus = "center-bottom"
            if zRot > 202: jus = "bottom-left"
            if zRot > 247: jus = "center-left"
            if zRot > 292: jus = "top-left"
            if zRot > 337: jus = "top-center"
        if xlabel_justify is not None:
            jus = xlabel_justify

        for i in range(1, len(xticks_str)):
            t = xticks_str[i]
            if not t:
                continue
            if utils.is_sequence(xlabel_offset):
                xoffs, yoffs, zoffs = xlabel_offset
            else:
                xoffs, yoffs, zoffs = 0, xlabel_offset, 0

            xlab = shapes.Text3D(
                t, s=xlabel_size * text_scale * gscale, font=label_font, justify=jus
            )
            tb = xlab.ybounds()  # must be ybounds: height of char

            v = (xticks_float[i], 0, 0)
            offs = -np.array([xoffs, yoffs, zoffs]) * (tb[1] - tb[0])

            T = LinearTransform()
            T.rotate_x(xaxis_rotation).rotate_y(yRot).rotate_x(xRot).rotate_z(zRot)
            T.translate(v + offs)
            T.translate([0, zxshift*dy + xshift_along_y*dy, xyshift*dz + xshift_along_z*dz])
            xlab.apply_transform(T)

            xlab.use_bounds(x_use_bounds)

            xlab.c(xlabel_color)
            if xlabel_backface_color is None:
                bfc = 1 - np.array(get_color(xlabel_color))
                xlab.backcolor(bfc)
            xlab.name = f"xNumericLabel {i}"
            labels.append(xlab)

    if ylabel_size and ytitle:

        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(ylabel_rotation):  # unpck 3 rotations
            zRot, yRot, xRot = ylabel_rotation
        else:
            zRot = ylabel_rotation
        if zRot < 0:
            zRot += 360  # deal with negative angles

        jus = "center-right"
        if zRot:
            if zRot >  24: jus = "bottom-right"
            if zRot >  67: jus = "center-bottom"
            if zRot > 112: jus = "left-bottom"
            if zRot > 157: jus = "center-left"
            if zRot > 202: jus = "top-left"
            if zRot > 247: jus = "center-top"
            if zRot > 292: jus = "top-right"
            if zRot > 337: jus = "right-center"
        if ylabel_justify is not None:
            jus = ylabel_justify

        for i in range(1, len(yticks_str)):
            t = yticks_str[i]
            if not t:
                continue
            if utils.is_sequence(ylabel_offset):
                xoffs, yoffs, zoffs = ylabel_offset
            else:
                xoffs, yoffs, zoffs = ylabel_offset, 0, 0
            ylab = shapes.Text3D(
                t, s=ylabel_size * text_scale * gscale, font=label_font, justify=jus
            )
            tb = ylab.ybounds()  # must be ybounds: height of char
            v = (0, yticks_float[i], 0)
            offs = -np.array([xoffs, yoffs, zoffs]) * (tb[1] - tb[0])

            T = LinearTransform()
            T.rotate_y(yaxis_rotation).rotate_x(xRot).rotate_y(yRot).rotate_z(zRot)
            T.translate(v + offs)
            T.translate([yzshift*dx + yshift_along_x*dx, 0, xyshift*dz + yshift_along_z*dz])
            ylab.apply_transform(T)

            ylab.use_bounds(y_use_bounds)

            ylab.c(ylabel_color)
            if ylabel_backface_color is None:
                bfc = 1 - np.array(get_color(ylabel_color))
                ylab.backcolor(bfc)
            ylab.name = f"yNumericLabel {i}"
            labels.append(ylab)

    if zlabel_size and ztitle:

        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(zlabel_rotation):  # unpck 3 rotations
            xRot, yRot, zRot = zlabel_rotation
        else:
            xRot = zlabel_rotation
        if xRot < 0: xRot += 360 # deal with negative angles

        jus = "center-right"
        if xRot:
            if xRot >  24: jus = "bottom-right"
            if xRot >  67: jus = "center-bottom"
            if xRot > 112: jus = "left-bottom"
            if xRot > 157: jus = "center-left"
            if xRot > 202: jus = "top-left"
            if xRot > 247: jus = "center-top"
            if xRot > 292: jus = "top-right"
            if xRot > 337: jus = "right-center"
        if zlabel_justify is not None:
            jus = zlabel_justify

        for i in range(1, len(zticks_str)):
            t = zticks_str[i]
            if not t:
                continue
            if utils.is_sequence(zlabel_offset):
                xoffs, yoffs, zoffs = zlabel_offset
            else:
                xoffs, yoffs, zoffs = zlabel_offset, zlabel_offset, 0
            zlab = shapes.Text3D(t, s=zlabel_size*text_scale*gscale, font=label_font, justify=jus)
            tb = zlab.ybounds()  # must be ybounds: height of char

            v = (0, 0, zticks_float[i])
            offs = -np.array([xoffs, yoffs, zoffs]) * (tb[1] - tb[0]) / 1.5
            angle = np.arctan2(dy, dx) * 57.3

            T = LinearTransform()
            T.rotate_x(90 + zRot).rotate_y(-xRot).rotate_z(angle + yRot + zaxis_rotation)
            T.translate(v + offs)
            T.translate([yzshift*dx + zshift_along_x*dx, zxshift*dy + zshift_along_y*dy, 0])
            zlab.apply_transform(T)

            zlab.use_bounds(z_use_bounds)

            zlab.c(zlabel_color)
            if zlabel_backface_color is None:
                bfc = 1 - np.array(get_color(zlabel_color))
                zlab.backcolor(bfc)
            zlab.name = f"zNumericLabel {i}"
            labels.append(zlab)

    ################################################ axes titles
    titles = []

    if xtitle:
        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(xtitle_rotation):  # unpack 3 rotations
            zRot, xRot, yRot = xtitle_rotation
        else:
            zRot = xtitle_rotation
        if zRot < 0:  # deal with negative angles
            zRot += 360

        if utils.is_sequence(xtitle_offset):
            xoffs, yoffs, zoffs = xtitle_offset
        else:
            xoffs, yoffs, zoffs = 0, xtitle_offset, 0

        if xtitle_justify is not None:
            jus = xtitle_justify
        else:
            # find best justfication for given rotation(s)
            jus = "right-top"
            if zRot:
                if zRot >  24: jus = "center-right"
                if zRot >  67: jus = "right-bottom"
                if zRot > 157: jus = "bottom-left"
                if zRot > 202: jus = "center-left"
                if zRot > 247: jus = "top-left"
                if zRot > 337: jus = "top-right"

        xt = shapes.Text3D(
            xtitle,
            s=xtitle_size * text_scale * gscale,
            font=title_font,
            c=xtitle_color,
            justify=jus,
            depth=title_depth,
            italic=xtitle_italic,
        )
        if xtitle_backface_color is None:
            xtitle_backface_color = 1 - np.array(get_color(xtitle_color))
        xt.backcolor(xtitle_backface_color)

        shift = 0
        if xlab:  # xlab is the last created numeric text label..
            lt0, lt1 = xlab.bounds()[2:4]
            shift = lt1 - lt0

        T = LinearTransform()
        T.rotate_x(xRot).rotate_y(yRot).rotate_z(zRot)
        T.set_position(
            [(xoffs + xtitle_position) * dx,
            -(yoffs + xtick_length / 2) * dy - shift,
            zoffs * dz]
        )
        T.rotate_x(xaxis_rotation)
        T.translate([0, xshift_along_y * dy, xyshift * dz + xshift_along_z * dz])
        xt.apply_transform(T)

        xt.use_bounds(x_use_bounds)
        if xtitle == " ":
            xt.use_bounds(False)
        xt.name = "xtitle"
        titles.append(xt)
        if xtitle_box:
            titles.append(xt.box(scale=1.1).use_bounds(x_use_bounds))

    if ytitle:
        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(ytitle_rotation):  # unpck 3 rotations
            zRot, yRot, xRot = ytitle_rotation
        else:
            zRot = ytitle_rotation
            if len(ytitle) > 3:
                zRot += 90
                ytitle_position *= 0.975
        if zRot < 0:
            zRot += 360  # deal with negative angles

        if utils.is_sequence(ytitle_offset):
            xoffs, yoffs, zoffs = ytitle_offset
        else:
            xoffs, yoffs, zoffs = ytitle_offset, 0, 0

        if ytitle_justify is not None:
            jus = ytitle_justify
        else:
            jus = "center-right"
            if zRot:
                if zRot >  24: jus = "bottom-right"
                if zRot > 112: jus = "left-bottom"
                if zRot > 157: jus = "center-left"
                if zRot > 202: jus = "top-left"
                if zRot > 292: jus = "top-right"
                if zRot > 337: jus = "right-center"

        yt = shapes.Text3D(
            ytitle,
            s=ytitle_size * text_scale * gscale,
            font=title_font,
            c=ytitle_color,
            justify=jus,
            depth=title_depth,
            italic=ytitle_italic,
        )
        if ytitle_backface_color is None:
            ytitle_backface_color = 1 - np.array(get_color(ytitle_color))
        yt.backcolor(ytitle_backface_color)

        shift = 0
        if ylab:  # this is the last created num label..
            lt0, lt1 = ylab.bounds()[0:2]
            shift = lt1 - lt0

        T = LinearTransform()
        T.rotate_x(xRot).rotate_y(yRot).rotate_z(zRot)
        T.set_position(
            [-(xoffs + ytick_length / 2) * dx - shift,
            (yoffs + ytitle_position) * dy,
            zoffs * dz]
        )
        T.rotate_y(yaxis_rotation)
        T.translate([yshift_along_x * dx, 0, xyshift * dz + yshift_along_z * dz])
        yt.apply_transform(T)

        yt.use_bounds(y_use_bounds)
        if ytitle == " ":
            yt.use_bounds(False)
        yt.name = "ytitle"
        titles.append(yt)
        if ytitle_box:
            titles.append(yt.box(scale=1.1).use_bounds(y_use_bounds))

    if ztitle:
        xRot, yRot, zRot = 0, 0, 0
        if utils.is_sequence(ztitle_rotation):  # unpck 3 rotations
            xRot, yRot, zRot = ztitle_rotation
        else:
            xRot = ztitle_rotation
            if len(ztitle) > 3:
                xRot += 90
                ztitle_position *= 0.975
        if xRot < 0:
            xRot += 360  # deal with negative angles

        if ztitle_justify is not None:
            jus = ztitle_justify
        else:
            jus = "center-right"
            if xRot:
                if xRot >  24: jus = "bottom-right"
                if xRot > 112: jus = "left-bottom"
                if xRot > 157: jus = "center-left"
                if xRot > 202: jus = "top-left"
                if xRot > 292: jus = "top-right"
                if xRot > 337: jus = "right-center"

        zt = shapes.Text3D(
            ztitle,
            s=ztitle_size * text_scale * gscale,
            font=title_font,
            c=ztitle_color,
            justify=jus,
            depth=title_depth,
            italic=ztitle_italic,
        )

        if ztitle_backface_color is None:
            ztitle_backface_color = 1 - np.array(get_color(ztitle_color))
        zt.backcolor(ztitle_backface_color)

        angle = np.arctan2(dy, dx) * 57.3
        shift = 0
        if zlab:  # this is the last created one..
            lt0, lt1 = zlab.bounds()[0:2]
            shift = lt1 - lt0

        T = LinearTransform()
        T.rotate_x(90 + zRot).rotate_y(-xRot).rotate_z(angle + yRot)
        T.set_position([
            -(ztitle_offset + ztick_length / 5) * dx - shift,
            -(ztitle_offset + ztick_length / 5) * dy - shift,
            ztitle_position * dz]
        )
        T.rotate_z(zaxis_rotation)
        T.translate([zshift_along_x * dx, zxshift * dy + zshift_along_y * dy, 0])
        zt.apply_transform(T)

        zt.use_bounds(z_use_bounds)
        if ztitle == " ":
            zt.use_bounds(False)
        zt.name = "ztitle"
        titles.append(zt)

    ################################################### header title
    if htitle:
        if htitle_font is None:
            htitle_font = title_font
        if htitle_color is None:
            htitle_color = xtitle_color
        htit = shapes.Text3D(
            htitle,
            s=htitle_size * gscale * text_scale,
            font=htitle_font,
            c=htitle_color,
            justify=htitle_justify,
            depth=title_depth,
            italic=htitle_italic,
        )
        if htitle_backface_color is None:
            htitle_backface_color = 1 - np.array(get_color(htitle_color))
            htit.backcolor(htitle_backface_color)
        htit.rotate_x(htitle_rotation)
        wpos = [htitle_offset[0]*dx, (1 + htitle_offset[1])*dy, htitle_offset[2]*dz]
        htit.shift(np.array(wpos) + [0, 0, xyshift*dz])
        htit.name = "htitle"
        titles.append(htit)

    ######
    acts = titles + lines + labels + grids + framelines
    acts += highlights + majorticks + minorticks + cones
    orig = (min_bns[0], min_bns[2], min_bns[4])
    for a in acts:
        a.shift(orig)
        a.actor.PickableOff()
        a.properties.LightingOff()
    asse = Assembly(acts)
    asse.actor.PickableOff()
    asse.name = "Axes"
    return asse


def add_global_axes(axtype=None, c=None, bounds=()) -> None:
    """
    Draw axes on scene. Available axes types are

    Parameters
    ----------
    axtype : (int)
        - 0,  no axes,
        - 1,  draw three gray grid walls
        - 2,  show cartesian axes from (0,0,0)
        - 3,  show positive range of cartesian axes from (0,0,0)
        - 4,  show a triad at bottom left
        - 5,  show a cube at bottom left
        - 6,  mark the corners of the bounding box
        - 7,  draw a 3D ruler at each side of the cartesian axes
        - 8,  show the `vtkCubeAxesActor` object
        - 9,  show the bounding box outLine
        - 10, show three circles representing the maximum bounding box
        - 11, show a large grid on the x-y plane (use with zoom=8)
        - 12, show polar axes
        - 13, draw a simple ruler at the bottom of the window
        - 14, show the vtk default `vtkCameraOrientationWidget` object

    Axis type-1 can be fully customized by passing a dictionary `axes=dict()`,
    see `vedo.Axes` for the complete list of options.

    Example
    -------
        .. code-block:: python

            from vedo import Box, show
            b = Box(pos=(0, 0, 0), size=(80, 90, 70).alpha(0.1)
            show(
                b,
                axes={
                    "xtitle": "Some long variable [a.u.]",
                    "number_of_divisions": 4,
                    # ...
                },
            )
    """
    plt = vedo.current_plotter()
    if plt is None:
        return

    if axtype is not None:
        plt.axes = axtype  # override

    r = plt.renderers.index(plt.renderer)

    if not plt.axes:
        return

    if c is None:  # automatic black or white
        c = (0.9, 0.9, 0.9)
        if np.sum(plt.renderer.GetBackground()) > 1.5:
            c = (0.1, 0.1, 0.1)
    else:
        c = get_color(c)  # for speed

    if not plt.renderer:
        return

    if plt.axes_instances[r]:
        return

    ############################################################
    # custom grid walls
    if plt.axes == 1 or plt.axes is True or isinstance(plt.axes, dict):

        if len(bounds) == 6:
            bnds = bounds
            xrange = (bnds[0], bnds[1])
            yrange = (bnds[2], bnds[3])
            zrange = (bnds[4], bnds[5])
        else:
            xrange = None
            yrange = None
            zrange = None

        if isinstance(plt.axes, dict):
            plt.axes.update({"use_global": True})
            # protect from invalid camelCase options from vedo<=2.3
            for k in plt.axes:
                if k.lower() != k:
                    return
            if "xrange" in plt.axes:
                xrange = plt.axes.pop("xrange")
            if "yrange" in plt.axes:
                yrange = plt.axes.pop("yrange")
            if "zrange" in plt.axes:
                zrange = plt.axes.pop("zrange")
            asse = Axes(**plt.axes, xrange=xrange, yrange=yrange, zrange=zrange)
        else:
            asse = Axes(xrange=xrange, yrange=yrange, zrange=zrange)

        plt.add(asse)
        plt.axes_instances[r] = asse

    elif plt.axes in (2, 3):
        x0, x1, y0, y1, z0, z1 = plt.renderer.ComputeVisiblePropBounds()
        xcol, ycol, zcol = "dr", "dg", "db"
        s = 1
        alpha = 1
        centered = False
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        aves = np.sqrt(dx * dx + dy * dy + dz * dz) / 2
        x0, x1 = min(x0, 0), max(x1, 0)
        y0, y1 = min(y0, 0), max(y1, 0)
        z0, z1 = min(z0, 0), max(z1, 0)

        if plt.axes == 3:
            if x1 > 0:
                x0 = 0
            if y1 > 0:
                y0 = 0
            if z1 > 0:
                z0 = 0

        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        acts = []
        if x0 * x1 <= 0 or y0 * z1 <= 0 or z0 * z1 <= 0:  # some ranges contain origin
            zero = shapes.Sphere(r=aves / 120 * s, c="k", alpha=alpha, res=10)
            acts += [zero]

        if dx > aves / 100:
            xl = shapes.Cylinder([[x0, 0, 0], [x1, 0, 0]], r=aves / 250 * s, c=xcol, alpha=alpha)
            xc = shapes.Cone(
                pos=[x1, 0, 0],
                c=xcol,
                alpha=alpha,
                r=aves / 100 * s,
                height=aves / 25 * s,
                axis=[1, 0, 0],
                res=10,
            )
            wpos = [x1, -aves / 25 * s, 0]  # aligned to arrow tip
            if centered:
                wpos = [(x0 + x1) / 2, -aves / 25 * s, 0]
            xt = shapes.Text3D("x", pos=wpos, s=aves / 40 * s, c=xcol)
            acts += [xl, xc, xt]

        if dy > aves / 100:
            yl = shapes.Cylinder([[0, y0, 0], [0, y1, 0]], r=aves / 250 * s, c=ycol, alpha=alpha)
            yc = shapes.Cone(
                pos=[0, y1, 0],
                c=ycol,
                alpha=alpha,
                r=aves / 100 * s,
                height=aves / 25 * s,
                axis=[0, 1, 0],
                res=10,
            )
            wpos = [-aves / 40 * s, y1, 0]
            if centered:
                wpos = [-aves / 40 * s, (y0 + y1) / 2, 0]
            yt = shapes.Text3D("y", pos=(0, 0, 0), s=aves / 40 * s, c=ycol)
            yt.rotate_z(90)
            yt.pos(wpos)
            acts += [yl, yc, yt]

        if dz > aves / 100:
            zl = shapes.Cylinder([[0, 0, z0], [0, 0, z1]], r=aves / 250 * s, c=zcol, alpha=alpha)
            zc = shapes.Cone(
                pos=[0, 0, z1],
                c=zcol,
                alpha=alpha,
                r=aves / 100 * s,
                height=aves / 25 * s,
                axis=[0, 0, 1],
                res=10,
            )
            wpos = [-aves / 50 * s, -aves / 50 * s, z1]
            if centered:
                wpos = [-aves / 50 * s, -aves / 50 * s, (z0 + z1) / 2]
            zt = shapes.Text3D("z", pos=(0, 0, 0), s=aves / 40 * s, c=zcol)
            zt.rotate_z(45)
            zt.rotate_x(90)
            zt.pos(wpos)
            acts += [zl, zc, zt]
        for a in acts:
            a.actor.PickableOff()
        asse = Assembly(acts)
        asse.actor.PickableOff()
        plt.add(asse)
        plt.axes_instances[r] = asse

    elif plt.axes == 4:
        axact = vtki.vtkAxesActor()
        axact.SetShaftTypeToCylinder()
        axact.SetCylinderRadius(0.03)
        axact.SetXAxisLabelText("x")
        axact.SetYAxisLabelText("y")
        axact.SetZAxisLabelText("z")
        axact.GetXAxisShaftProperty().SetColor(1, 0, 0)
        axact.GetYAxisShaftProperty().SetColor(0, 1, 0)
        axact.GetZAxisShaftProperty().SetColor(0, 0, 1)
        axact.GetXAxisTipProperty().SetColor(1, 0, 0)
        axact.GetYAxisTipProperty().SetColor(0, 1, 0)
        axact.GetZAxisTipProperty().SetColor(0, 0, 1)
        bc = np.array(plt.renderer.GetBackground())
        if np.sum(bc) < 1.5:
            lc = (1, 1, 1)
        else:
            lc = (0, 0, 0)
        axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
        axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
        axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
        axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
        axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
        axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
        axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
        axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
        axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
        axact.PickableOff()
        icn = Icon(axact, size=0.1)
        plt.axes_instances[r] = icn
        icn.SetInteractor(plt.interactor)
        icn.EnabledOn()
        icn.InteractiveOff()
        plt.widgets.append(icn)

    elif plt.axes == 5:
        axact = vtki.new("AnnotatedCubeActor")
        axact.GetCubeProperty().SetColor(get_color(settings.annotated_cube_color))
        axact.SetTextEdgesVisibility(0)
        axact.SetFaceTextScale(settings.annotated_cube_text_scale)
        axact.SetXPlusFaceText(settings.annotated_cube_texts[0])  # XPlus
        axact.SetXMinusFaceText(settings.annotated_cube_texts[1])  # XMinus
        axact.SetYPlusFaceText(settings.annotated_cube_texts[2])  # YPlus
        axact.SetYMinusFaceText(settings.annotated_cube_texts[3])  # YMinus
        axact.SetZPlusFaceText(settings.annotated_cube_texts[4])  # ZPlus
        axact.SetZMinusFaceText(settings.annotated_cube_texts[5])  # ZMinus
        axact.SetXFaceTextRotation(settings.annotated_cube_text_rotations[0])
        axact.SetYFaceTextRotation(settings.annotated_cube_text_rotations[1])
        axact.SetZFaceTextRotation(settings.annotated_cube_text_rotations[2])

        if settings.annotated_cube_text_color is None:  # use default
            axact.GetXPlusFaceProperty().SetColor(get_color("r"))
            axact.GetXMinusFaceProperty().SetColor(get_color("dr"))
            axact.GetYPlusFaceProperty().SetColor(get_color("g"))
            axact.GetYMinusFaceProperty().SetColor(get_color("dg"))
            axact.GetZPlusFaceProperty().SetColor(get_color("b"))
            axact.GetZMinusFaceProperty().SetColor(get_color("db"))
        else:  # use single user color
            ac = get_color(settings.annotated_cube_text_color)
            axact.GetXPlusFaceProperty().SetColor(ac)
            axact.GetXMinusFaceProperty().SetColor(ac)
            axact.GetYPlusFaceProperty().SetColor(ac)
            axact.GetYMinusFaceProperty().SetColor(ac)
            axact.GetZPlusFaceProperty().SetColor(ac)
            axact.GetZMinusFaceProperty().SetColor(ac)

        axact.PickableOff()
        icn = Icon(axact, size=0.06)
        plt.axes_instances[r] = icn
        icn.SetInteractor(plt.interactor)
        icn.EnabledOn()
        icn.InteractiveOff()
        plt.widgets.append(icn)

    elif plt.axes == 6:
        ocf = vtki.new("OutlineCornerFilter")
        ocf.SetCornerFactor(0.1)
        largestact, sz = None, -1
        for a in plt.objects:
            try:
                if a.pickable():
                    b = a.bounds()
                    if b is None:
                        return
                    d = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
                    if sz < d:
                        largestact = a
                        sz = d
            except AttributeError:
                pass

        try:
            ocf.SetInputData(largestact)
        except TypeError:
            try:
                ocf.SetInputData(largestact.dataset)
            except (TypeError, AttributeError):
                return
        ocf.Update()

        oc_mapper = vtki.new("HierarchicalPolyDataMapper")
        oc_mapper.SetInputConnection(0, ocf.GetOutputPort(0))
        oc_actor = vtki.vtkActor()
        oc_actor.SetMapper(oc_mapper)
        bc = np.array(plt.renderer.GetBackground())
        if np.sum(bc) < 1.5:
            lc = (1, 1, 1)
        else:
            lc = (0, 0, 0)
        oc_actor.GetProperty().SetColor(lc)
        oc_actor.PickableOff()
        oc_actor.UseBoundsOn()
        plt.axes_instances[r] = oc_actor
        plt.add(oc_actor)

    elif plt.axes == 7:
        vbb = compute_visible_bounds()[0]
        rulax = RulerAxes(vbb, c=c, xtitle="x - ", ytitle="y - ", ztitle="z - ")
        plt.axes_instances[r] = rulax
        if not rulax:
            return
        rulax.actor.UseBoundsOn()
        rulax.actor.PickableOff()
        plt.add(rulax)

    elif plt.axes == 8:
        vbb = compute_visible_bounds()[0]
        ca = vtki.new("CubeAxesActor")
        ca.SetBounds(vbb)
        ca.SetCamera(plt.renderer.GetActiveCamera())
        ca.GetXAxesLinesProperty().SetColor(c)
        ca.GetYAxesLinesProperty().SetColor(c)
        ca.GetZAxesLinesProperty().SetColor(c)
        for i in range(3):
            ca.GetLabelTextProperty(i).SetColor(c)
            ca.GetTitleTextProperty(i).SetColor(c)
        # ca.SetTitleOffset(5)
        ca.SetFlyMode(3)
        ca.SetXTitle("x")
        ca.SetYTitle("y")
        ca.SetZTitle("z")
        ca.PickableOff()
        ca.UseBoundsOff()
        plt.axes_instances[r] = ca
        plt.renderer.AddActor(ca)

    elif plt.axes == 9:
        vbb = compute_visible_bounds()[0]
        src = vtki.new("CubeSource")
        src.SetXLength(vbb[1] - vbb[0])
        src.SetYLength(vbb[3] - vbb[2])
        src.SetZLength(vbb[5] - vbb[4])
        src.Update()
        ca = Mesh(src.GetOutput(), c, 0.5).wireframe(True)
        ca.pos((vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2)
        ca.actor.PickableOff()
        ca.actor.UseBoundsOff()
        plt.axes_instances[r] = ca
        plt.add(ca)

    elif plt.axes == 10:
        vbb = compute_visible_bounds()[0]
        x0 = (vbb[0] + vbb[1]) / 2, (vbb[3] + vbb[2]) / 2, (vbb[5] + vbb[4]) / 2
        rx, ry, rz = (vbb[1] - vbb[0]) / 2, (vbb[3] - vbb[2]) / 2, (vbb[5] - vbb[4]) / 2
        # compute diagonal length of the bounding box
        rm = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        d = 0.005 * rm
        xc = shapes.Disc(x0, r1=rm, r2=rm+d, c="lr", res=(1, 120))
        yc = shapes.Disc(x0, r1=rm, r2=rm+d, c="lg", res=(1, 120)).rotate_x(90)
        zc = shapes.Disc(x0, r1=rm, r2=rm+d, c="lb", res=(1, 120)).rotate_y(90)
        xc.pickable(0).lighting("off")
        yc.pickable(0).lighting("off")
        zc.pickable(0).lighting("off")
        ca = xc + yc + zc
        ca.pickable(False)
        ca.actor.UseBoundsOff()
        plt.axes_instances[r] = ca
        plt.add(ca)

    elif plt.axes == 11:
        vbb, ss = compute_visible_bounds()[0:2]
        xpos, ypos = (vbb[1] + vbb[0]) / 2, (vbb[3] + vbb[2]) / 2
        gs = sum(ss) * 3
        gr = shapes.Grid((xpos, ypos, vbb[4]), s=(gs, gs), res=(11, 11), c=c, alpha=0.1)
        gr.lighting("off").actor.PickableOff()
        gr.actor.UseBoundsOff()
        plt.axes_instances[r] = gr
        plt.add(gr)

    elif plt.axes == 12:
        polaxes = vtki.new("PolarAxesActor")
        vbb = compute_visible_bounds()[0]

        polaxes.SetPolarAxisTitle("radial distance")
        polaxes.SetPole(0, 0, vbb[4])
        rd = max(abs(vbb[0]), abs(vbb[2]), abs(vbb[1]), abs(vbb[3]))
        polaxes.SetMaximumRadius(rd)
        try: # fails in vtk 9.5
            polaxes.AutoSubdividePolarAxisOff()
            polaxes.SetNumberOfPolarAxisTicks(10)
            polaxes.SetNumberOfPolarAxisTicks(5)
        except Exception as e:
            pass
            # vedo.logger.warning("Failed to set polar axis properties")
        polaxes.SetCamera(plt.renderer.GetActiveCamera())
        polaxes.SetPolarLabelFormat("%6.1f")
        polaxes.PolarLabelVisibilityOff()  # due to bad overlap of labels

        polaxes.GetPolarArcsProperty().SetColor(c)
        polaxes.GetPolarAxisProperty().SetColor(c)
        polaxes.GetPolarAxisTitleTextProperty().SetColor(c)
        polaxes.GetPolarAxisLabelTextProperty().SetColor(c)
        polaxes.GetLastRadialAxisTextProperty().SetColor(c)
        polaxes.GetSecondaryRadialAxesTextProperty().SetColor(c)
        polaxes.GetSecondaryRadialAxesProperty().SetColor(c)
        polaxes.GetSecondaryPolarArcsProperty().SetColor(c)

        polaxes.SetMinimumAngle(0.0)
        polaxes.SetMaximumAngle(315.0)
        polaxes.UseBoundsOn()
        polaxes.PickableOff()
        plt.axes_instances[r] = polaxes
        plt.renderer.AddActor(polaxes)

    elif plt.axes == 13:
        # draws a simple ruler at the bottom of the window
        ls = vtki.new("LegendScaleActor")
        ls.RightAxisVisibilityOff()
        ls.TopAxisVisibilityOff()
        ls.LeftAxisVisibilityOff()
        ls.LegendVisibilityOff()
        ls.SetBottomBorderOffset(50)
        ls.GetBottomAxis().SetNumberOfMinorTicks(1)
        ls.GetBottomAxis().SetFontFactor(1.1)
        ls.GetBottomAxis().GetProperty().SetColor(c)
        ls.GetBottomAxis().GetProperty().SetOpacity(1.0)
        ls.GetBottomAxis().GetProperty().SetLineWidth(2)
        ls.GetBottomAxis().GetLabelTextProperty().SetColor(c)
        ls.GetBottomAxis().GetLabelTextProperty().BoldOff()
        ls.GetBottomAxis().GetLabelTextProperty().ItalicOff()
        pr = ls.GetBottomAxis().GetLabelTextProperty()
        pr.SetFontFamily(vtki.VTK_FONT_FILE)
        pr.SetFontFile(utils.get_font_path(settings.default_font))
        ls.PickableOff()
        # if not plt.renderer.GetActiveCamera().GetParallelProjection():
        #     vedo.logger.warning("Axes type 13 should be used with parallel projection")
        plt.axes_instances[r] = ls
        plt.renderer.AddActor(ls)

    elif plt.axes == 14:
        try:
            cow = vtki.new("CameraOrientationWidget")
            cow.SetParentRenderer(plt.renderer)
            cow.On()
            plt.axes_instances[r] = cow
        except ImportError:
            vedo.logger.warning("axes mode 14 is unavailable in this vtk version")

    else:
        e = "Keyword axes type must be in range [0-13]."
        e += "Available axes types are:\n\n"
        e += "0 = no axes\n"
        e += "1 = draw three customizable gray grid walls\n"
        e += "2 = show cartesian axes from (0,0,0)\n"
        e += "3 = show positive range of cartesian axes from (0,0,0)\n"
        e += "4 = show a triad at bottom left\n"
        e += "5 = show a cube at bottom left\n"
        e += "6 = mark the corners of the bounding box\n"
        e += "7 = draw a 3D ruler at each side of the cartesian axes\n"
        e += "8 = show the vtkCubeAxesActor object\n"
        e += "9 = show the bounding box outline\n"
        e += "10 = show three circles representing the maximum bounding box\n"
        e += "11 = show a large grid on the x-y plane (use with zoom=8)\n"
        e += "12 = show polar axes\n"
        e += "13 = draw a simple ruler at the bottom of the window\n"
        e += "14 = show the CameraOrientationWidget object"
        vedo.logger.warning(e)

    if not plt.axes_instances[r]:
        plt.axes_instances[r] = True
