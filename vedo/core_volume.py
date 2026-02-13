#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Volume-specific algorithm mixins."""

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import utils
from vedo.core_common import CommonAlgorithms

__all__ = ["VolumeAlgorithms"]

class VolumeAlgorithms(CommonAlgorithms):
    """Methods for Volume objects."""

    def bounds(cls) -> np.ndarray:
        """
        Get the object bounds.
        Returns a list in format `[xmin,xmax, ymin,ymax, zmin,zmax]`.
        """
        # OVERRIDE CommonAlgorithms.bounds() which is too slow
        return np.array(cls.dataset.GetBounds())

    def isosurface(cls, value=None, flying_edges=False) -> "vedo.mesh.Mesh":
        """
        Return an `Mesh` isosurface extracted from the `Volume` object.

        Set `value` as single float or list of values to draw the isosurface(s).
        Use flying_edges for faster results (but sometimes can interfere with `smooth()`).

        The isosurface values can be accessed with `mesh.metadata["isovalue"]`.

        Examples:
            - [isosurfaces1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/isosurfaces1.py)

                ![](https://vedo.embl.es/images/volumetric/isosurfaces.png)
        """
        scrange = cls.dataset.GetScalarRange()

        if flying_edges:
            cf = vtki.new("FlyingEdges3D")
            cf.InterpolateAttributesOn()
        else:
            cf = vtki.new("ContourFilter")
            cf.UseScalarTreeOn()

        cf.SetInputData(cls.dataset)
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
        out.metadata["isovalue"] = value

        out.pipeline = utils.OperationNode(
            "isosurface",
            parents=[cls],
            comment=f"#pts {out.dataset.GetNumberOfPoints()}",
            c="#4cc9f0:#e9c46a",
        )
        return out

    def isosurface_discrete(
            cls, values, background_label=None, internal_boundaries=True, use_quads=False, nsmooth=0,
        ) -> "vedo.mesh.Mesh":
        """
        Create boundary/isocontour surfaces from a label map (e.g., a segmented image) using a threaded,
        3D version of the multiple objects/labels Surface Nets algorithm.
        The input is a 3D image (i.e., volume) where each voxel is labeled
        (integer labels are preferred to real values), and the output data is a polygonal mesh separating
        labeled regions / objects.
        (Note that on output each region [corresponding to a different segmented object] will share
        points/edges on a common boundary, i.e., two neighboring objects will share the boundary that separates them).

        Besides output geometry defining the surface net, the filter outputs a two-component celldata array indicating
        the labels on either side of the polygons composing the output Mesh.
        (This can be used for advanced operations like extracting shared/contacting boundaries between two objects.
        The name of this celldata array is "BoundaryLabels").

        The values can be accessed with `mesh.metadata["isovalue"]`.

        Arguments:
            value : (float, list)
                single value or list of values to draw the isosurface(s).
            background_label : (float)
                this value specifies the label value to use when referencing the background
                region outside of any of the specified regions.
            boundaries : (bool, list)
                if True, the output will only contain the boundary surface. Internal surfaces will be removed.
                If a list of integers is provided, only the boundaries between the specified labels will be extracted.
            use_quads : (bool)
                if True, the output polygons will be quads. If False, the output polygons will be triangles.
            nsmooth : (int)
                number of iterations of smoothing (0 means no smoothing).

        Examples:
            - [isosurfaces2.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/isosurfaces2.py)
        """
        logger = vtki.get_class("Logger")
        logger.SetStderrVerbosity(logger.VERBOSITY_ERROR)

        snets = vtki.new("SurfaceNets3D")
        snets.SetInputData(cls.dataset)

        if nsmooth:
            snets.SmoothingOn()
            snets.AutomaticSmoothingConstraintsOn()
            snets.GetSmoother().SetNumberOfIterations(nsmooth)
            # snets.GetSmoother().SetRelaxationFactor(relaxation_factor)
            # snets.GetSmoother().SetConstraintDistance(constraint_distance)
        else:
            snets.SmoothingOff()

        if internal_boundaries is False:
            snets.SetOutputStyleToBoundary()
        elif internal_boundaries is True:
            snets.SetOutputStyleToDefault()
        elif utils.is_sequence(internal_boundaries):
            snets.SetOutputStyleToSelected()
            snets.InitializeSelectedLabelsList()
            for val in internal_boundaries:
                snets.AddSelectedLabel(val)
        else:
            vedo.logger.error("isosurface_discrete(): unknown boundaries option")

        n = len(values)
        snets.SetNumberOfContours(n)
        snets.SetNumberOfLabels(n)

        if background_label is not None:
            snets.SetBackgroundLabel(background_label)

        for i, val in enumerate(values):
            snets.SetValue(i, val)

        if use_quads:
            snets.SetOutputMeshTypeToQuads()
        else:
            snets.SetOutputMeshTypeToTriangles()
        snets.Update()

        out = vedo.mesh.Mesh(snets.GetOutput())
        out.metadata["isovalue"] = values
        out.pipeline = utils.OperationNode(
            "isosurface_discrete",
            parents=[cls],
            comment=f"#pts {out.dataset.GetNumberOfPoints()}",
            c="#4cc9f0:#e9c46a",
        )

        logger.SetStderrVerbosity(logger.VERBOSITY_INFO)
        return out


    def legosurface(
        cls,
        vmin=None,
        vmax=None,
        invert=False,
        boundary=True,
        array_name="input_scalars",
    ) -> "vedo.mesh.Mesh":
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
        imp_dataset = vtki.new("ImplicitDataSet")
        imp_dataset.SetDataSet(cls.dataset)
        window = vtki.new("ImplicitWindowFunction")
        window.SetImplicitFunction(imp_dataset)

        srng = list(cls.dataset.GetScalarRange())
        if vmin is not None:
            srng[0] = vmin
        if vmax is not None:
            srng[1] = vmax
        if not boundary:
            tol = 0.00001 * (srng[1] - srng[0])
            srng[0] -= tol
            srng[1] += tol
        window.SetWindowRange(srng)
        # print("legosurface window range:", srng)

        extract = vtki.new("ExtractGeometry")
        extract.SetInputData(cls.dataset)
        extract.SetImplicitFunction(window)
        extract.SetExtractInside(invert)
        extract.SetExtractBoundaryCells(boundary)
        extract.Update()

        gf = vtki.new("GeometryFilter")
        gf.SetInputData(extract.GetOutput())
        gf.Update()

        m = vedo.mesh.Mesh(gf.GetOutput()).lw(0.1).flat()
        m.map_points_to_cells()
        m.celldata.select(array_name)

        m.pipeline = utils.OperationNode(
            "legosurface",
            parents=[cls],
            comment=f"array: {array_name}",
            c="#4cc9f0:#e9c46a",
        )
        return m

    def tomesh(cls, fill=True, shrink=1.0) -> "vedo.mesh.Mesh":
        """
        Build a polygonal Mesh from the current object.

        If `fill=True`, the interior faces of all the cells are created.
        (setting a `shrink` value slightly smaller than the default 1.0
        can avoid flickering due to internal adjacent faces).

        If `fill=False`, only the boundary faces will be generated.
        """
        gf = vtki.new("GeometryFilter")
        if fill:
            sf = vtki.new("ShrinkFilter")
            sf.SetInputData(cls.dataset)
            sf.SetShrinkFactor(shrink)
            sf.Update()
            gf.SetInputData(sf.GetOutput())
            gf.Update()
            poly = gf.GetOutput()
            if shrink == 1.0:
                clean_poly = vtki.new("CleanPolyData")
                clean_poly.PointMergingOn()
                clean_poly.ConvertLinesToPointsOn()
                clean_poly.ConvertPolysToLinesOn()
                clean_poly.ConvertStripsToPolysOn()
                clean_poly.SetInputData(poly)
                clean_poly.Update()
                poly = clean_poly.GetOutput()
        else:
            gf.SetInputData(cls.dataset)
            gf.Update()
            poly = gf.GetOutput()

        msh = vedo.mesh.Mesh(poly).flat()
        msh.scalarbar = cls.scalarbar
        lut = utils.ctf2lut(cls)
        if lut:
            msh.mapper.SetLookupTable(lut)

        msh.pipeline = utils.OperationNode(
            "tomesh", parents=[cls], comment=f"fill={fill}", c="#9e2a2b:#e9c46a"
        )
        return msh
