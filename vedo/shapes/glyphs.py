#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Glyph-related shapes extracted from vedo.shapes."""

import numpy as np

import vedo
import vedo.vtkclasses as vtki

from vedo import utils
from vedo.colors import cmaps_names, get_color
from vedo.mesh import Mesh
from vedo.pointcloud import Points
class Glyph(Mesh):
    """
    At each vertex of a mesh, another mesh, i.e. a "glyph", is shown with
    various orientation options and coloring.

    The input can also be a simple list of 2D or 3D coordinates.
    Color can be specified as a colormap which maps the size of the orientation
    vectors in `orientation_array`.
    """

    def __init__(
        self,
        mesh,
        glyph,
        orientation_array=None,
        scale_by_scalar=False,
        scale_by_vector_size=False,
        scale_by_vector_components=False,
        color_by_scalar=False,
        color_by_vector_size=False,
        c="k8",
        alpha=1.0,
    ) -> None:
        """
        Arguments:
            orientation_array: (list, str, vtkArray)
                list of vectors, `vtkArray` or name of an already existing pointdata array
            scale_by_scalar : (bool)
                glyph mesh is scaled by the active scalars
            scale_by_vector_size : (bool)
                glyph mesh is scaled by the size of the vectors
            scale_by_vector_components : (bool)
                glyph mesh is scaled by the 3 vectors components
            color_by_scalar : (bool)
                glyph mesh is colored based on the scalar value
            color_by_vector_size : (bool)
                glyph mesh is colored based on the vector size

        Examples:
            - [glyphs1.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/glyphs1.py)
            - [glyphs2.py](https://github.com/marcomusy/vedo/tree/master/examples/basic/glyphs2.py)

            ![](https://vedo.embl.es/images/basic/glyphs.png)
        """
        if utils.is_sequence(mesh):
            # create a cloud of points
            poly = utils.buildPolyData(mesh)
        else:
            poly = mesh.dataset

        cmap = ""
        if isinstance(c, str) and c in cmaps_names:
            cmap = c
            c = None
        elif utils.is_sequence(c):  # user passing an array of point colors
            ucols = vtki.vtkUnsignedCharArray()
            ucols.SetNumberOfComponents(3)
            ucols.SetName("GlyphRGB")
            for col in c:
                cl = get_color(col)
                ucols.InsertNextTuple3(cl[0] * 255, cl[1] * 255, cl[2] * 255)
            poly.GetPointData().AddArray(ucols)
            poly.GetPointData().SetActiveScalars("GlyphRGB")
            c = None

        gly = vtki.vtkGlyph3D()
        gly.GeneratePointIdsOn()
        gly.SetInputData(poly)
        try:
            gly.SetSourceData(glyph)
        except TypeError:
            gly.SetSourceData(glyph.dataset)

        if scale_by_scalar:
            gly.SetScaleModeToScaleByScalar()
        elif scale_by_vector_size:
            gly.SetScaleModeToScaleByVector()
        elif scale_by_vector_components:
            gly.SetScaleModeToScaleByVectorComponents()
        else:
            gly.SetScaleModeToDataScalingOff()

        if color_by_vector_size:
            gly.SetVectorModeToUseVector()
            gly.SetColorModeToColorByVector()
        elif color_by_scalar:
            gly.SetColorModeToColorByScalar()
        else:
            gly.SetColorModeToColorByScale()

        if orientation_array is not None:
            gly.OrientOn()
            if isinstance(orientation_array, str):
                if orientation_array.lower() == "normals":
                    gly.SetVectorModeToUseNormal()
                else:  # passing a name
                    poly.GetPointData().SetActiveVectors(orientation_array)
                    gly.SetInputArrayToProcess(0, 0, 0, 0, orientation_array)
                    gly.SetVectorModeToUseVector()
            elif utils.is_sequence(orientation_array):  # passing a list
                varr = vtki.vtkFloatArray()
                varr.SetNumberOfComponents(3)
                varr.SetName("glyph_vectors")
                for v in orientation_array:
                    varr.InsertNextTuple(v)
                poly.GetPointData().AddArray(varr)
                poly.GetPointData().SetActiveVectors("glyph_vectors")
                gly.SetInputArrayToProcess(0, 0, 0, 0, "glyph_vectors")
                gly.SetVectorModeToUseVector()

        gly.Update()

        super().__init__(gly.GetOutput(), c, alpha)
        self.flat()

        if cmap:
            self.cmap(cmap, "VectorMagnitude")
        elif c is None:
            self.pointdata.select("GlyphRGB")

        self.name = "Glyph"


class Tensors(Mesh):
    """
    Geometric representation of tensors defined on a domain or set of points.
    Tensors can be scaled and/or rotated according to the source at each input point.
    Scaling and rotation is controlled by the eigenvalues/eigenvectors of the
    symmetrical part of the tensor as follows:

    For each tensor, the eigenvalues (and associated eigenvectors) are sorted
    to determine the major, medium, and minor eigenvalues/eigenvectors.
    The eigenvalue decomposition only makes sense for symmetric tensors,
    hence the need to only consider the symmetric part of the tensor,
    which is `1/2*(T+T.transposed())`.
    """

    def __init__(
        self,
        domain,
        source="ellipsoid",
        use_eigenvalues=True,
        is_symmetric=True,
        three_axes=False,
        scale=1.0,
        max_scale=None,
        length=None,
        res=24,
        c=None,
        alpha=1.0,
    ) -> None:
        """
        Arguments:
            source : (str, Mesh)
                preset types of source shapes is "ellipsoid", "cylinder", "cube" or a `Mesh` object.
            use_eigenvalues : (bool)
                color source glyph using the eigenvalues or by scalars
            three_axes : (bool)
                if `False` scale the source in the x-direction,
                the medium in the y-direction, and the minor in the z-direction.
                Then, the source is rotated so that the glyph's local x-axis lies
                along the major eigenvector, y-axis along the medium eigenvector,
                and z-axis along the minor.

                If `True` three sources are produced, each of them oriented along an eigenvector
                and scaled according to the corresponding eigenvector.
            is_symmetric : (bool)
                If `True` each source glyph is mirrored (2 or 6 glyphs will be produced).
                The x-axis of the source glyph will correspond to the eigenvector on output.
            length : (float)
                distance from the origin to the tip of the source glyph along the x-axis
            scale : (float)
                scaling factor of the source glyph.
            max_scale : (float)
                clamp scaling at this factor.

        Examples:
            - [tensors.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/tensors.py)
            - [tensor_grid1.py](https://github.com/marcomusy/vedo/tree/master/examples/other/tensor_grid1.py)

            ![](https://vedo.embl.es/images/volumetric/tensor_grid.png)
        """
        if isinstance(source, Points):
            src = source.dataset
        else: # is string
            if "ellip" in source:
                src = vtki.new("SphereSource")
                src.SetPhiResolution(res)
                src.SetThetaResolution(res*2)
            elif "cyl" in source:
                src = vtki.new("CylinderSource")
                src.SetResolution(res)
                src.CappingOn()
            elif source == "cube":
                src = vtki.new("CubeSource")
            else:
                vedo.logger.error(f"Unknown source type {source}")
                raise ValueError()
            src.Update()
            src = src.GetOutput()

        tg = vtki.new("TensorGlyph")
        if isinstance(domain, vtki.vtkPolyData):
            tg.SetInputData(domain)
        else:
            tg.SetInputData(domain.dataset)
        tg.SetSourceData(src)

        if c is None:
            tg.ColorGlyphsOn()
        else:
            tg.ColorGlyphsOff()

        tg.SetSymmetric(int(is_symmetric))

        if length is not None:
            tg.SetLength(length)
        if use_eigenvalues:
            tg.ExtractEigenvaluesOn()
            tg.SetColorModeToEigenvalues()
        else:
            tg.SetColorModeToScalars()

        tg.SetThreeGlyphs(three_axes)
        tg.ScalingOn()
        tg.SetScaleFactor(scale)
        if max_scale is None:
            tg.ClampScalingOn()
            max_scale = scale * 10
        tg.SetMaxScaleFactor(max_scale)

        tg.Update()
        tgn = vtki.new("PolyDataNormals")
        tgn.ComputeCellNormalsOff()
        tgn.SetInputData(tg.GetOutput())
        tgn.Update()

        super().__init__(tgn.GetOutput(), c, alpha)
        self.name = "Tensors"
