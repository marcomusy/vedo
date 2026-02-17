#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Slicing mixin for Volume."""

import os

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import transformations
from vedo import utils
from vedo.mesh import Mesh


class VolumeSlicingMixin:
    def xslice(self, i: int) -> Mesh:
        """Extract the slice at index `i` of volume along x-axis."""
        vslice = vtki.new("ImageDataGeometryFilter")
        vslice.SetInputData(self.dataset)
        nx, ny, nz = self.dataset.GetDimensions()
        if i > nx - 1:
            i = nx - 1
        vslice.SetExtent(i, i, 0, ny, 0, nz)
        vslice.Update()
        m = Mesh(vslice.GetOutput())
        m.pipeline = utils.OperationNode(f"xslice {i}", parents=[self], c="#4cc9f0:#e9c46a")
        return m


    def yslice(self, j: int) -> Mesh:
        """Extract the slice at index `j` of volume along y-axis."""
        vslice = vtki.new("ImageDataGeometryFilter")
        vslice.SetInputData(self.dataset)
        nx, ny, nz = self.dataset.GetDimensions()
        if j > ny - 1:
            j = ny - 1
        vslice.SetExtent(0, nx, j, j, 0, nz)
        vslice.Update()
        m = Mesh(vslice.GetOutput())
        m.pipeline = utils.OperationNode(f"yslice {j}", parents=[self], c="#4cc9f0:#e9c46a")
        return m


    def zslice(self, k: int) -> Mesh:
        """Extract the slice at index `i` of volume along z-axis."""
        vslice = vtki.new("ImageDataGeometryFilter")
        vslice.SetInputData(self.dataset)
        nx, ny, nz = self.dataset.GetDimensions()
        if k > nz - 1:
            k = nz - 1
        vslice.SetExtent(0, nx, 0, ny, k, k)
        vslice.Update()
        m = Mesh(vslice.GetOutput())
        m.pipeline = utils.OperationNode(f"zslice {k}", parents=[self], c="#4cc9f0:#e9c46a")
        return m


    def slice_plane(self, origin: list[float], normal: list[float], autocrop=False, border=0.5, mode="linear") -> Mesh:
        """
        Extract the slice along a given plane position and normal.

        Two metadata arrays are added to the output Mesh:
            - "shape" : contains the shape of the slice
            - "original_bounds" : contains the original bounds of the slice
        One can access them with e.g. `myslice.metadata["shape"]`.

        Arguments:
            origin : (list)
                position of the plane
            normal : (list)
                normal to the plane
            autocrop : (bool)
                crop the output to the minimal possible size
            border : (float)
                add a border to the output slice
            mode : (str)
                interpolation mode, one of the following: "linear", "nearest", "cubic"

        Example:
            - [slice_plane1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slice_plane1.py)

                ![](https://vedo.embl.es/images/volumetric/slicePlane1.gif)

            - [slice_plane2.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slice_plane2.py)

                ![](https://vedo.embl.es/images/volumetric/slicePlane2.png)

            - [slice_plane3.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slice_plane3.py)

                ![](https://vedo.embl.es/images/volumetric/slicePlane3.jpg)
        """
        newaxis = utils.versor(normal)
        pos = np.array(origin)
        initaxis = (0, 0, 1)
        crossvec = np.cross(initaxis, newaxis)
        angle = np.arccos(np.dot(initaxis, newaxis))
        T = vtki.vtkTransform()
        T.PostMultiply()
        T.RotateWXYZ(np.rad2deg(angle), crossvec.tolist())
        T.Translate(pos.tolist())

        reslice = vtki.new("ImageReslice")
        reslice.SetResliceAxes(T.GetMatrix())
        reslice.SetInputData(self.dataset)
        reslice.SetOutputDimensionality(2)
        reslice.SetTransformInputSampling(True)
        reslice.SetGenerateStencilOutput(False)
        if border:
            reslice.SetBorder(True)
            reslice.SetBorderThickness(border)
        else:
            reslice.SetBorder(False)
        if mode == "linear":
            reslice.SetInterpolationModeToLinear()
        elif mode == "nearest":
            reslice.SetInterpolationModeToNearestNeighbor()
        elif mode == "cubic":
            reslice.SetInterpolationModeToCubic()
        else:
            vedo.logger.error(f"in slice_plane(): unknown interpolation mode {mode}")
            raise ValueError()
        reslice.SetAutoCropOutput(not autocrop)
        reslice.Update()
        img = reslice.GetOutput()

        vslice = vtki.new("ImageDataGeometryFilter")
        vslice.SetInputData(img)
        vslice.Update()

        msh = Mesh(vslice.GetOutput()).apply_transform(T)
        msh.properties.LightingOff()

        d0, d1, _ = img.GetDimensions()
        varr1 = utils.numpy2vtk([d1, d0], name="shape")
        varr2 = utils.numpy2vtk(img.GetBounds(), name="original_bounds")
        msh.dataset.GetFieldData().AddArray(varr1)
        msh.dataset.GetFieldData().AddArray(varr2)
        msh.pipeline = utils.OperationNode("slice_plane", parents=[self], c="#4cc9f0:#e9c46a")
        return msh


    def slab(self, slice_range=(), axis='z', operation="mean") -> Mesh:
        """
        Extract a slab from a `Volume` by combining
        all of the slices of an image to create a single slice.

        Returns a `Mesh` containing metadata which
        can be accessed with e.g. `mesh.metadata["slab_range"]`.

        Metadata:
            slab_range : (list)
                contains the range of slices extracted
            slab_axis : (str)
                contains the axis along which the slab was extracted
            slab_operation : (str)
                contains the operation performed on the slab
            slab_bounding_box : (list)
                contains the bounding box of the slab

        Arguments:
            slice_range : (list)
                range of slices to extract
            axis : (str)
                axis along which to extract the slab
            operation : (str)
                operation to perform on the slab,
                allowed values are: "sum", "min", "max", "mean".

        Example:
            - [slab.py](https://github.com/marcomusy/vedo/blob/master/examples/volumetric/slab_vol.py)

            ![](https://vedo.embl.es/images/volumetric/slab_vol.jpg)
        """
        if len(slice_range) != 2:
            vedo.logger.error("in slab(): slice_range is empty or invalid")
            raise ValueError()
        slab_range = [int(slice_range[0]), int(slice_range[1])]

        islab = vtki.new("ImageSlab")
        islab.SetInputData(self.dataset)

        if operation in ["+", "add", "sum"]:
            islab.SetOperationToSum()
        elif "min" in operation:
            islab.SetOperationToMin()
        elif "max" in operation:
            islab.SetOperationToMax()
        elif "mean" in operation:
            islab.SetOperationToMean()
        else:
            vedo.logger.error(f"in slab(): unknown operation {operation}")
            raise ValueError()

        dims = self.dimensions()
        if axis == 'x':
            islab.SetOrientationToX()
            if slab_range[0] > dims[0] - 1:
                slab_range[0] = int(dims[0] - 1)
            if slab_range[1] > dims[0] - 1:
                slab_range[1] = int(dims[0] - 1)
        elif axis == 'y':
            islab.SetOrientationToY()
            if slab_range[0] > dims[1] - 1:
                slab_range[0] = int(dims[1] - 1)
            if slab_range[1] > dims[1] - 1:
                slab_range[1] = int(dims[1] - 1)
        elif axis == 'z':
            islab.SetOrientationToZ()
            if slab_range[0] > dims[2] - 1:
                slab_range[0] = int(dims[2] - 1)
            if slab_range[1] > dims[2] - 1:
                slab_range[1] = int(dims[2] - 1)
        else:
            vedo.logger.error(f"Error in slab(): unknown axis {axis}")
            raise RuntimeError()

        islab.SetSliceRange(slab_range)
        islab.Update()

        msh = Mesh(islab.GetOutput()).lighting('off')
        msh.mapper.SetLookupTable(utils.ctf2lut(self, msh))
        msh.mapper.SetScalarRange(self.scalar_range())

        msh.metadata["slab_range"] = slab_range
        msh.metadata["slab_axis"]  = axis
        msh.metadata["slab_operation"] = operation

        # compute bounds of slab
        origin = list(self.origin())
        spacing = list(self.spacing())
        if axis == 'x':
            msh.metadata["slab_bounding_box"] = [
                origin[0] + slab_range[0] * spacing[0],
                origin[0] + slab_range[1] * spacing[0],
                origin[1],
                origin[1] + dims[1]*spacing[1],
                origin[2],
                origin[2] + dims[2]*spacing[2],
            ]
        elif axis == 'y':
            msh.metadata["slab_bounding_box"] = [
                origin[0],
                origin[0] + dims[0]*spacing[0],
                origin[1] + slab_range[0] * spacing[1],
                origin[1] + slab_range[1] * spacing[1],
                origin[2],
                origin[2] + dims[2]*spacing[2],
            ]
        elif axis == 'z':
            msh.metadata["slab_bounding_box"] = [
                origin[0],
                origin[0] + dims[0]*spacing[0],
                origin[1],
                origin[1] + dims[1]*spacing[1],
                origin[2] + slab_range[0] * spacing[2],
                origin[2] + slab_range[1] * spacing[2],
            ]

        msh.pipeline = utils.OperationNode(
            f"slab{slab_range}",
            comment=f"axis={axis}, operation={operation}",
            parents=[self],
            c="#4cc9f0:#e9c46a",
        )
        msh.name = "SlabMesh"
        return msh


