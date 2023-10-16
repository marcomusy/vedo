import glob
import os

import numpy as np

try:
    import vedo.vtkclasses as vtk
except ImportError:
    import vtkmodules.all as vtk

import vedo
from vedo import utils
from vedo.mesh import Mesh
from vedo.core import VolumeAlgorithms
from vedo.visual import VolumeVisual


__docformat__ = "google"

__doc__ = """
Work with volumetric datasets (voxel data).

![](https://vedo.embl.es/images/volumetric/slicePlane2.png)
"""

__all__ = ["Volume"]


##########################################################################
class Volume(VolumeVisual, VolumeAlgorithms):
    """
    Class to describe dataset that are defined on "voxels":
    the 3D equivalent of 2D pixels.
    """

    def __init__(
        self,
        inputobj=None,
        c="RdBu_r",
        alpha=(0.0, 0.0, 0.2, 0.4, 0.8, 1.0),
        alpha_gradient=None,
        alpha_unit=1,
        mode=0,
        spacing=None,
        dims=None,
        origin=None,
        mapper="smart",
    ):
        """
        This class can be initialized with a numpy object, a `vtkImageData`
        or a list of 2D bmp files.

        Arguments:
            c : (list, str)
                sets colors along the scalar range, or a matplotlib color map name
            alphas : (float, list)
                sets transparencies along the scalar range
            alpha_unit : (float)
                low values make composite rendering look brighter and denser
            origin : (list)
                set volume origin coordinates
            spacing : (list)
                voxel dimensions in x, y and z.
            dims : (list)
                specify the dimensions of the volume.
            mapper : (str)
                either 'gpu', 'opengl_gpu', 'fixed' or 'smart'
            mode : (int)
                define the volumetric rendering style:
                    - 0, composite rendering
                    - 1, maximum projection
                    - 2, minimum projection
                    - 3, average projection
                    - 4, additive mode

                <br>The default mode is "composite" where the scalar values are sampled through
                the volume and composited in a front-to-back scheme through alpha blending.
                The final color and opacity is determined using the color and opacity transfer
                functions specified in alpha keyword.

                Maximum and minimum intensity blend modes use the maximum and minimum
                scalar values, respectively, along the sampling ray.
                The final color and opacity is determined by passing the resultant value
                through the color and opacity transfer functions.

                Additive blend mode accumulates scalar values by passing each value
                through the opacity transfer function and then adding up the product
                of the value and its opacity. In other words, the scalar values are scaled
                using the opacity transfer function and summed to derive the final color.
                Note that the resulting image is always grayscale i.e. aggregated values
                are not passed through the color transfer function.
                This is because the final value is a derived value and not a real data value
                along the sampling ray.

                Average intensity blend mode works similar to the additive blend mode where
                the scalar values are multiplied by opacity calculated from the opacity
                transfer function and then added.
                The additional step here is to divide the sum by the number of samples
                taken through the volume.
                As is the case with the additive intensity projection, the final image will
                always be grayscale i.e. the aggregated values are not passed through the
                color transfer function.

        Example:
            ```python
            from vedo import Volume
            vol = Volume("path/to/mydata/rec*.bmp")
            vol.show()
            ```

        Examples:
            - [numpy2volume1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/numpy2volume1.py)

                ![](https://vedo.embl.es/images/volumetric/numpy2volume1.png)

            - [read_volume2.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/read_volume2.py)

                ![](https://vedo.embl.es/images/volumetric/read_volume2.png)

        .. note::
            if a `list` of values is used for `alphas` this is interpreted
            as a transfer function along the range of the scalar.
        """
        self.actor = vtk.vtkVolume()
        self.actor.data = self
        self.property = self.actor.GetProperty()
        self.dataset = None
        self.mapper = None
        self.pipeline = None
        self.info = {}
        self.name = "Volume"
        self.filename = ""

        ###################
        if isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = vedo.file_io.download(inputobj, verbose=False)  # fpath
            elif os.path.isfile(inputobj):
                pass
            else:
                inputobj = sorted(glob.glob(inputobj))

        ###################
        if "gpu" in mapper:
            self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        elif "opengl_gpu" in mapper:
            self.mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
        elif "smart" in mapper:
            self.mapper = vtk.vtkSmartVolumeMapper()
        elif "fixed" in mapper:
            self.mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        elif isinstance(mapper, vtk.vtkMapper):
            self.mapper = mapper
        else:
            print("Error unknown mapper type", [mapper])
            raise RuntimeError()

        self.actor.SetMapper(self.mapper)

        ###################
        inputtype = str(type(inputobj))

        # print('Volume inputtype', inputtype, c='b')

        if inputobj is None:
            img = vtk.vtkImageData()

        elif utils.is_sequence(inputobj):

            if isinstance(inputobj[0], str) and ".bmp" in inputobj[0].lower():
                # scan sequence of BMP files
                ima = vtk.vtkImageAppend()
                ima.SetAppendAxis(2)
                pb = utils.ProgressBar(0, len(inputobj))
                for i in pb.range():
                    f = inputobj[i]
                    if "_rec_spr.bmp" in f:
                        continue
                    picr = vtk.vtkBMPReader()
                    picr.SetFileName(f)
                    picr.Update()
                    mgf = vtk.vtkImageMagnitude()
                    mgf.SetInputData(picr.GetOutput())
                    mgf.Update()
                    ima.AddInputData(mgf.GetOutput())
                    pb.print("loading...")
                ima.Update()
                img = ima.GetOutput()

            else:

                if len(inputobj.shape) == 1:
                    varr = utils.numpy2vtk(inputobj)
                else:
                    varr = utils.numpy2vtk(inputobj.ravel(order="F"))
                varr.SetName("input_scalars")

                img = vtk.vtkImageData()
                if dims is not None:
                    img.SetDimensions(dims[2], dims[1], dims[0])
                else:
                    if len(inputobj.shape) == 1:
                        vedo.logger.error("must set dimensions (dims keyword) in Volume")
                        raise RuntimeError()
                    img.SetDimensions(inputobj.shape)
                img.GetPointData().AddArray(varr)
                img.GetPointData().SetActiveScalars(varr.GetName())

        elif isinstance(inputobj, vtk.vtkImageData):
            img = inputobj

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = vedo.file_io.download(inputobj, verbose=False)
            img = vedo.file_io.loadImageData(inputobj)

        else:
            vedo.logger.error(f"cannot understand input type {inputtype}")
            return

        if dims is not None:
            img.SetDimensions(dims)

        if origin is not None:
            img.SetOrigin(origin)  ### DIFFERENT from volume.origin()!

        if spacing is not None:
            img.SetSpacing(spacing)

        self.dataset = img
        self.mapper.SetInputData(img)

        if img.GetPointData().GetScalars():
            if img.GetPointData().GetScalars().GetNumberOfComponents() == 1:
                self.mode(mode).color(c).alpha(alpha).alpha_gradient(alpha_gradient)
                self.property.SetShade(True)
                self.property.SetInterpolationType(1)
                self.property.SetScalarOpacityUnitDistance(alpha_unit)


        self.pipeline = utils.OperationNode(
            "Volume", comment=f"dims={tuple(self.dimensions())}", c="#4cc9f0"
        )
        #######################################################################

    def _update(self, data):
        self.dataset = data
        self.mapper.SetInputData(data)
        self.dataset.GetPointData().Modified()
        self.mapper.Modified()
        self.mapper.Update()
        return self


    def _repr_html_(self):
        """
        HTML representation of the Volume object for Jupyter Notebooks.

        Returns:
            HTML text with the image and some properties.
        """
        import io
        import base64
        from PIL import Image

        library_name = "vedo.volume.Volume"
        help_url = "https://vedo.embl.es/docs/vedo/volume.html"

        arr = self.thumbnail(azimuth=0, elevation=-60, zoom=1.4, axes=True)

        im = Image.fromarray(arr)
        buffered = io.BytesIO()
        im.save(buffered, format="PNG", quality=100)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = "data:image/png;base64," + encoded
        image = f"<img src='{url}'></img>"

        # statisitics
        bounds = "<br/>".join(
            [
                utils.precision(min_x, 4) + " ... " + utils.precision(max_x, 4)
                for min_x, max_x in zip(self.bounds()[::2], self.bounds()[1::2])
            ]
        )

        help_text = ""
        if self.name:
            help_text += f"<b> {self.name}: &nbsp&nbsp</b>"
        help_text += '<b><a href="' + help_url + '" target="_blank">' + library_name + "</a></b>"
        if self.filename:
            dots = ""
            if len(self.filename) > 30:
                dots = "..."
            help_text += f"<br/><code><i>({dots}{self.filename[-30:]})</i></code>"

        pdata = ""
        if self.dataset.GetPointData().GetScalars():
            if self.dataset.GetPointData().GetScalars().GetName():
                name = self.dataset.GetPointData().GetScalars().GetName()
                pdata = "<tr><td><b> point data array </b></td><td>" + name + "</td></tr>"

        cdata = ""
        if self.dataset.GetCellData().GetScalars():
            if self.dataset.GetCellData().GetScalars().GetName():
                name = self.dataset.GetCellData().GetScalars().GetName()
                cdata = "<tr><td><b> voxel data array </b></td><td>" + name + "</td></tr>"

        img = self.mapper.GetInput()

        allt = [
            "<table>",
            "<tr>",
            "<td>",
            image,
            "</td>",
            "<td style='text-align: center; vertical-align: center;'><br/>",
            help_text,
            "<table>",
            "<tr><td><b> bounds </b> <br/> (x/y/z) </td><td>" + str(bounds) + "</td></tr>",
            "<tr><td><b> dimensions </b></td><td>" + str(img.GetDimensions()) + "</td></tr>",
            "<tr><td><b> voxel spacing </b></td><td>"
            + utils.precision(img.GetSpacing(), 3)
            + "</td></tr>",
            "<tr><td><b> in memory size </b></td><td>"
            + str(int(img.GetActualMemorySize() / 1024))
            + "MB</td></tr>",
            pdata,
            cdata,
            "<tr><td><b> scalar range </b></td><td>"
            + utils.precision(img.GetScalarRange(), 4)
            + "</td></tr>",
            "</table>",
            "</table>",
        ]
        return "\n".join(allt)

    def clone(self, deep=True):
        """Return a clone copy of the Volume."""
        if deep:
            newimg = vtk.vtkImageData()
            newimg.CopyStructure(self.dataset)
            newimg.CopyAttributes(self.dataset)
            newvol = Volume(newimg)
        else:
            newvol = Volume(self.dataset)

        prop = vtk.vtkVolumeProperty()
        prop.DeepCopy(self.property)
        newvol.actor.SetProperty(prop)
        newvol.property = prop

        newvol.pipeline = utils.OperationNode("clone", parents=[self], c="#bbd0ff", shape="diamond")
        return newvol
    
    def component_weight(self, i, weight):
        """Set the scalar component weight in range [0,1]."""
        self.property.SetComponentWeight(i, weight)
        return self

    def xslice(self, i):
        """Extract the slice at index `i` of volume along x-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.dataset)
        nx, ny, nz = self.dataset.GetDimensions()
        if i > nx - 1:
            i = nx - 1
        vslice.SetExtent(i, i, 0, ny, 0, nz)
        vslice.Update()
        m = Mesh(vslice.GetOutput())
        m.pipeline = utils.OperationNode(f"xslice {i}", parents=[self], c="#4cc9f0:#e9c46a")
        return m

    def yslice(self, j):
        """Extract the slice at index `j` of volume along y-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.dataset)
        nx, ny, nz = self.dataset.GetDimensions()
        if j > ny - 1:
            j = ny - 1
        vslice.SetExtent(0, nx, j, j, 0, nz)
        vslice.Update()
        m = Mesh(vslice.GetOutput())
        m.pipeline = utils.OperationNode(f"yslice {j}", parents=[self], c="#4cc9f0:#e9c46a")
        return m

    def zslice(self, k):
        """Extract the slice at index `i` of volume along z-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.dataset)
        nx, ny, nz = self.dataset.GetDimensions()
        if k > nz - 1:
            k = nz - 1
        vslice.SetExtent(0, nx, 0, ny, k, k)
        vslice.Update()
        m = Mesh(vslice.GetOutput())
        m.pipeline = utils.OperationNode(f"zslice {k}", parents=[self], c="#4cc9f0:#e9c46a")
        return m

    def slice_plane(self, origin=(0, 0, 0), normal=(1, 1, 1), autocrop=False):
        """
        Extract the slice along a given plane position and normal.

        Example:
            - [slice_plane1.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slice_plane1.py)

                ![](https://vedo.embl.es/images/volumetric/slicePlane1.gif)
        """
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(self.dataset)
        reslice.SetOutputDimensionality(2)
        newaxis = utils.versor(normal)
        pos = np.array(origin)
        initaxis = (0, 0, 1)
        crossvec = np.cross(initaxis, newaxis)
        angle = np.arccos(np.dot(initaxis, newaxis))
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.RotateWXYZ(np.rad2deg(angle), crossvec)
        T.Translate(pos)
        M = T.GetMatrix()
        reslice.SetResliceAxes(M)
        reslice.SetInterpolationModeToLinear()
        reslice.SetAutoCropOutput(not autocrop)
        reslice.Update()
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(reslice.GetOutput())
        vslice.Update()
        msh = Mesh(vslice.GetOutput())
        # msh.SetOrientation(T.GetOrientation())
        # msh.SetPosition(pos)
        msh.apply_transform(T)
        msh.pipeline = utils.OperationNode("slice_plane", parents=[self], c="#4cc9f0:#e9c46a")
        return msh

    def warp(self, source, target, sigma=1, mode="3d", fit=False):
        """
        Warp volume scalars within a Volume by specifying
        source and target sets of points.

        Arguments:
            source : (Points, list)
                the list of source points
            target : (Points, list)
                the list of target points
            fit : (bool)
                fit/adapt the old bounding box to the warped geometry
        """
        if isinstance(source, vedo.Points):
            source = source.vertices
        if isinstance(target, vedo.Points):
            target = target.vertices

        ns = len(source)
        ptsou = vtk.vtkPoints()
        ptsou.SetNumberOfPoints(ns)
        for i in range(ns):
            ptsou.SetPoint(i, source[i])

        nt = len(target)
        if ns != nt:
            vedo.logger.error(f"#source {ns} != {nt} #target points")
            raise RuntimeError()

        pttar = vtk.vtkPoints()
        pttar.SetNumberOfPoints(nt)
        for i in range(ns):
            pttar.SetPoint(i, target[i])

        T = vtk.vtkThinPlateSplineTransform()
        if mode.lower() == "3d":
            T.SetBasisToR()
        elif mode.lower() == "2d":
            T.SetBasisToR2LogR()
        else:
            vedo.logger.error(f"unknown mode {mode}")
            raise RuntimeError()

        T.SetSigma(sigma)
        T.SetSourceLandmarks(ptsou)
        T.SetTargetLandmarks(pttar)
        T.Inverse()
        self.apply_transform(T, fit=fit)
        self.pipeline = utils.OperationNode("warp", parents=[self], c="#4cc9f0")
        return self

    def apply_transform(self, T, fit=False):
        """
        Apply a transform to the scalars in the volume.

        Arguments:
            T : (vtkTransform, matrix)
                The transformation to be applied
            fit : (bool)
                fit/adapt the old bounding box to the warped geometry
        """
        if isinstance(T, vtk.vtkMatrix4x4):
            tr = vtk.vtkTransform()
            tr.SetMatrix(T)
            T = tr

        elif utils.is_sequence(T):
            M = vtk.vtkMatrix4x4()
            n = len(T[0])
            for i in range(n):
                for j in range(n):
                    M.SetElement(i, j, T[i][j])
            tr = vtk.vtkTransform()
            tr.SetMatrix(M)
            T = tr

        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(self.dataset)
        reslice.SetResliceTransform(T)
        reslice.SetOutputDimensionality(3)
        reslice.SetInterpolationModeToLinear()

        spacing = self.dataset.GetSpacing()
        origin = self.dataset.GetOrigin()

        if fit:
            bb = self.box()
            if isinstance(T, vtk.vtkThinPlateSplineTransform):
                TI = vtk.vtkThinPlateSplineTransform()
                TI.DeepCopy(T)
                TI.Inverse()
            else:
                TI = vtk.vtkTransform()
                TI.DeepCopy(T)
            bb.apply_transform(TI)
            bounds = bb.bounds()
            bounds = (
                bounds[0] / spacing[0],
                bounds[1] / spacing[0],
                bounds[2] / spacing[1],
                bounds[3] / spacing[1],
                bounds[4] / spacing[2],
                bounds[5] / spacing[2],
            )
            bounds = np.round(bounds).astype(int)
            reslice.SetOutputExtent(bounds)
            reslice.SetOutputSpacing(spacing[0], spacing[1], spacing[2])
            reslice.SetOutputOrigin(origin[0], origin[1], origin[2])

        reslice.Update()
        self._update(reslice.GetOutput())
        self.pipeline = utils.OperationNode("apply_transform", parents=[self], c="#4cc9f0")
        return self

    def imagedata(self):
        """
        DEPRECATED:
        Use `Volume.dataset` instead.

        Return the underlying `vtkImagaData` object.
        """
        print("Volume.imagedata() is deprecated, use Volume.dataset instead")
        return self.dataset

    def tonumpy(self):
        """
        Get read-write access to voxels of a Volume object as a numpy array.

        When you set values in the output image, you don't want numpy to reallocate the array
        but instead set values in the existing array, so use the [:] operator.

        Example:
            `arr[:] = arr*2 + 15`

        If the array is modified add a call to:
        `volume.dataset.GetPointData().GetScalars().Modified()`
        when all your modifications are completed.
        """
        narray_shape = tuple(reversed(self.dataset.GetDimensions()))

        scals = self.dataset.GetPointData().GetScalars()
        comps = scals.GetNumberOfComponents()
        if comps == 1:
            narray = utils.vtk2numpy(scals).reshape(narray_shape)
            narray = np.transpose(narray, axes=[2, 1, 0])
        else:
            narray = utils.vtk2numpy(scals).reshape(*narray_shape, comps)
            narray = np.transpose(narray, axes=[2, 1, 0, 3])

        # narray = utils.vtk2numpy(self.dataset.GetPointData().GetScalars()).reshape(narray_shape)
        # narray = np.transpose(narray, axes=[2, 1, 0])

        return narray

    def dimensions(self):
        """Return the nr. of voxels in the 3 dimensions."""
        return np.array(self.dataset.GetDimensions())

    def scalar_range(self):
        """Return the range of the scalar values."""
        return np.array(self.dataset.GetScalarRange())

    def spacing(self, s=None):
        """Set/get the voxels size in the 3 dimensions."""
        if s is not None:
            self.dataset.SetSpacing(s)
            return self
        return np.array(self.dataset.GetSpacing())

    def origin(self, s=None):
        """Set/get the origin of the volumetric dataset."""
        ### supersedes base.origin()
        ### DIFFERENT from base.origin()!
        if s is not None:
            self.dataset.SetOrigin(s)
            return self
        return np.array(self.dataset.GetOrigin())

    def center(self, p=None):
        """Set/get the center of the volumetric dataset."""
        if p is not None:
            self.dataset.SetCenter(p)
            return self
        return np.array(self.dataset.GetCenter())

    def permute_axes(self, x, y, z):
        """
        Reorder the axes of the Volume by specifying
        the input axes which are supposed to become the new X, Y, and Z.
        """
        imp = vtk.vtkImagePermute()
        imp.SetFilteredAxes(x, y, z)
        imp.SetInputData(self.dataset)
        imp.Update()
        self._update(imp.GetOutput())
        self.pipeline = utils.OperationNode(
            f"permute_axes\n{(x,y,z)}", parents=[self], c="#4cc9f0"
        )
        return self

    def resample(self, new_spacing, interpolation=1):
        """
        Resamples a `Volume` to be larger or smaller.

        This method modifies the spacing of the input.
        Linear interpolation is used to resample the data.

        Arguments:
            new_spacing : (list)
                a list of 3 new spacings for the 3 axes
            interpolation : (int)
                0=nearest_neighbor, 1=linear, 2=cubic
        """
        rsp = vtk.vtkImageResample()
        oldsp = self.spacing()
        for i in range(3):
            if oldsp[i] != new_spacing[i]:
                rsp.SetAxisOutputSpacing(i, new_spacing[i])
        rsp.InterpolateOn()
        rsp.SetInterpolationMode(interpolation)
        rsp.OptimizationOn()
        rsp.Update()
        self._update(rsp.GetOutput())
        self.pipeline = utils.OperationNode(
            f"resample\n{tuple(new_spacing)}", parents=[self], c="#4cc9f0"
        )
        return self


    def threshold(self, above=None, below=None, replace=None, replace_value=None):
        """
        Binary or continuous volume thresholding.
        Find the voxels that contain a value above/below the input values
        and replace them with a new value (default is 0).
        """
        th = vtk.vtkImageThreshold()
        th.SetInputData(self.dataset)

        # sanity checks
        if above is not None and below is not None:
            if above == below:
                return self
            if above > below:
                vedo.logger.warning("in volume.threshold(), above > below, skip.")
                return self

        ## cases
        if below is not None and above is not None:
            th.ThresholdBetween(above, below)

        elif above is not None:
            th.ThresholdByUpper(above)

        elif below is not None:
            th.ThresholdByLower(below)

        ##
        if replace is not None:
            th.SetReplaceIn(True)
            th.SetInValue(replace)
        else:
            th.SetReplaceIn(False)

        if replace_value is not None:
            th.SetReplaceOut(True)
            th.SetOutValue(replace_value)
        else:
            th.SetReplaceOut(False)

        th.Update()
        self._update(th.GetOutput())
        self.pipeline = utils.OperationNode("threshold", parents=[self], c="#4cc9f0")
        return self

    def crop(self, left=None, right=None, back=None, front=None, bottom=None, top=None, VOI=()):
        """
        Crop a `Volume` object.

        Arguments:
            left : (float)
                fraction to crop from the left plane (negative x)
            right : (float)
                fraction to crop from the right plane (positive x)
            back : (float)
                fraction to crop from the back plane (negative y)
            front : (float)
                fraction to crop from the front plane (positive y)
            bottom : (float)
                fraction to crop from the bottom plane (negative z)
            top : (float)
                fraction to crop from the top plane (positive z)
            VOI : (list)
                extract Volume Of Interest expressed in voxel numbers

        Example:
            `vol.crop(VOI=(xmin, xmax, ymin, ymax, zmin, zmax)) # all integers nrs`
        """
        extractVOI = vtk.vtkExtractVOI()
        extractVOI.SetInputData(self.dataset)

        if VOI:
            extractVOI.SetVOI(VOI)
        else:
            d = self.dataset.GetDimensions()
            bx0, bx1, by0, by1, bz0, bz1 = 0, d[0]-1, 0, d[1]-1, 0, d[2]-1
            if left is not None:   bx0 = int((d[0]-1)*left)
            if right is not None:  bx1 = int((d[0]-1)*(1-right))
            if back is not None:   by0 = int((d[1]-1)*back)
            if front is not None:  by1 = int((d[1]-1)*(1-front))
            if bottom is not None: bz0 = int((d[2]-1)*bottom)
            if top is not None:    bz1 = int((d[2]-1)*(1-top))
            extractVOI.SetVOI(bx0, bx1, by0, by1, bz0, bz1)
        extractVOI.Update()
        self._update(extractVOI.GetOutput())

        self.pipeline = utils.OperationNode(
            "crop", parents=[self], c="#4cc9f0", comment=f"dims={tuple(self.dimensions())}"
        )
        return self

    def append(self, volumes, axis="z", preserve_extents=False):
        """
        Take the components from multiple inputs and merges them into one output.
        Except for the append axis, all inputs must have the same extent.
        All inputs must have the same number of scalar components.
        The output has the same origin and spacing as the first input.
        The origin and spacing of all other inputs are ignored.
        All inputs must have the same scalar type.

        Arguments:
            axis : (int, str)
                axis expanded to hold the multiple images
            preserve_extents : (bool)
                if True, the extent of the inputs is used to place
                the image in the output. The whole extent of the output is the union of the input
                whole extents. Any portion of the output not covered by the inputs is set to zero.
                The origin and spacing is taken from the first input.

        Example:
            ```python
            from vedo import Volume, dataurl
            vol = Volume(dataurl+'embryo.tif')
            vol.append(vol, axis='x').show().close()
            ```
            ![](https://vedo.embl.es/images/feats/volume_append.png)
        """
        ima = vtk.vtkImageAppend()
        ima.SetInputData(self.dataset)
        if not utils.is_sequence(volumes):
            volumes = [volumes]
        for volume in volumes:
            if isinstance(volume, vtk.vtkImageData):
                ima.AddInputData(volume)
            else:
                ima.AddInputData(volume.dataset)
        ima.SetPreserveExtents(preserve_extents)
        if axis == "x":
            axis = 0
        elif axis == "y":
            axis = 1
        elif axis == "z":
            axis = 2
        ima.SetAppendAxis(axis)
        ima.Update()
        self._update(ima.GetOutput())

        self.pipeline = utils.OperationNode(
            "append",
            parents=[self, *volumes],
            c="#4cc9f0",
            comment=f"dims={tuple(self.dimensions())}",
        )
        return self

    def pad(self, voxels=10, value=0):
        """
        Add the specified number of voxels at the `Volume` borders.
        Voxels can be a list formatted as `[nx0, nx1, ny0, ny1, nz0, nz1]`.

        Arguments:
            voxels : (int, list)
                number of voxels to be added (or a list of length 4)
            value : (int)
                intensity value (gray-scale color) of the padding

        Example:
            ```python
            from vedo import Volume, dataurl, show
            iso = Volume(dataurl+'embryo.tif').isosurface()
            vol = iso.binarize(spacing=(100, 100, 100)).pad(10)
            vol.dilate([15,15,15])
            show(iso, vol.isosurface(), N=2, axes=1)
            ```
            ![](https://vedo.embl.es/images/volumetric/volume_pad.png)
        """
        x0, x1, y0, y1, z0, z1 = self.dataset.GetExtent()
        pf = vtk.vtkImageConstantPad()
        pf.SetInputData(self.dataset)
        pf.SetConstant(value)
        if utils.is_sequence(voxels):
            pf.SetOutputWholeExtent(
                x0 - voxels[0], x1 + voxels[1],
                y0 - voxels[2], y1 + voxels[3],
                z0 - voxels[4], z1 + voxels[5],
            )
        else:
            pf.SetOutputWholeExtent(
                x0 - voxels, x1 + voxels,
                y0 - voxels, y1 + voxels,
                z0 - voxels, z1 + voxels,
            )
        pf.Update()
        self._update(pf.GetOutput())
        self.pipeline = utils.OperationNode(
            "pad", comment=f"{voxels} voxels", parents=[self], c="#f28482"
        )
        return self

    def resize(self, *newdims):
        """Increase or reduce the number of voxels of a Volume with interpolation."""
        old_dims = np.array(self.dataset.GetDimensions())
        old_spac = np.array(self.dataset.GetSpacing())
        rsz = vtk.vtkImageResize()
        rsz.SetResizeMethodToOutputDimensions()
        rsz.SetInputData(self.dataset)
        rsz.SetOutputDimensions(newdims)
        rsz.Update()
        self.dataset = rsz.GetOutput()
        new_spac = old_spac * old_dims / newdims  # keep aspect ratio
        self.dataset.SetSpacing(new_spac)
        self._update(self.dataset)
        self.pipeline = utils.OperationNode(
            "resize", parents=[self], c="#4cc9f0", comment=f"dims={tuple(self.dimensions())}"
        )
        return self

    def normalize(self):
        """Normalize that scalar components for each point."""
        norm = vtk.vtkImageNormalize()
        norm.SetInputData(self.dataset)
        norm.Update()
        self._update(norm.GetOutput())
        self.pipeline = utils.OperationNode("normalize", parents=[self], c="#4cc9f0")
        return self

    def mirror(self, axis="x"):
        """
        Mirror flip along one of the cartesian axes.
        """
        img = self.dataset

        ff = vtk.vtkImageFlip()
        ff.SetInputData(img)
        if axis.lower() == "x":
            ff.SetFilteredAxis(0)
        elif axis.lower() == "y":
            ff.SetFilteredAxis(1)
        elif axis.lower() == "z":
            ff.SetFilteredAxis(2)
        else:
            vedo.logger.error("mirror must be set to either x, y, z or n")
            raise RuntimeError()
        ff.Update()
        self._update(ff.GetOutput())
        self.pipeline = utils.OperationNode(f"mirror {axis}", parents=[self], c="#4cc9f0")
        return self

    def operation(self, operation, volume2=None):
        """
        Perform operations with `Volume` objects.
        Keyword `volume2` can be a constant float.

        Possible operations are:
        ```
        +, -, /, 1/x, sin, cos, exp, log,
        abs, **2, sqrt, min, max, atan, atan2, median,
        mag, dot, gradient, divergence, laplacian.
        ```

        Examples:
            - [volume_operations.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/volume_operations.py)
        """
        op = operation.lower()
        image1 = self.dataset

        mf = None
        if op in ["median"]:
            mf = vtk.vtkImageMedian3D()
            mf.SetInputData(image1)
        elif op in ["mag"]:
            mf = vtk.vtkImageMagnitude()
            mf.SetInputData(image1)
        elif op in ["dot", "dotproduct"]:
            mf = vtk.vtkImageDotProduct()
            mf.SetInput1Data(image1)
            mf.SetInput2Data(volume2.dataset)
        elif op in ["grad", "gradient"]:
            mf = vtk.vtkImageGradient()
            mf.SetDimensionality(3)
            mf.SetInputData(image1)
        elif op in ["div", "divergence"]:
            mf = vtk.vtkImageDivergence()
            mf.SetInputData(image1)
        elif op in ["laplacian"]:
            mf = vtk.vtkImageLaplacian()
            mf.SetDimensionality(3)
            mf.SetInputData(image1)

        if mf is not None:
            mf.Update()
            vol = Volume(mf.GetOutput())
            vol.pipeline = utils.OperationNode(
                f"operation\n{op}", parents=[self], c="#4cc9f0", shape="cylinder"
            )
            return vol  ###########################

        mat = vtk.vtkImageMathematics()
        mat.SetInput1Data(image1)

        K = None

        if utils.is_number(volume2):
            K = volume2
            mat.SetConstantK(K)
            mat.SetConstantC(K)

        elif volume2 is not None:  # assume image2 is a constant value
            mat.SetInput2Data(volume2.dataset)

        # ###########################
        if op in ["+", "add", "plus"]:
            if K:
                mat.SetOperationToAddConstant()
            else:
                mat.SetOperationToAdd()

        elif op in ["-", "subtract", "minus"]:
            if K:
                mat.SetConstantC(-float(K))
                mat.SetOperationToAddConstant()
            else:
                mat.SetOperationToSubtract()

        elif op in ["*", "multiply", "times"]:
            if K:
                mat.SetOperationToMultiplyByK()
            else:
                mat.SetOperationToMultiply()

        elif op in ["/", "divide"]:
            if K:
                mat.SetConstantK(1.0 / K)
                mat.SetOperationToMultiplyByK()
            else:
                mat.SetOperationToDivide()

        elif op in ["1/x", "invert"]:
            mat.SetOperationToInvert()
        elif op in ["sin"]:
            mat.SetOperationToSin()
        elif op in ["cos"]:
            mat.SetOperationToCos()
        elif op in ["exp"]:
            mat.SetOperationToExp()
        elif op in ["log"]:
            mat.SetOperationToLog()
        elif op in ["abs"]:
            mat.SetOperationToAbsoluteValue()
        elif op in ["**2", "square"]:
            mat.SetOperationToSquare()
        elif op in ["sqrt", "sqr"]:
            mat.SetOperationToSquareRoot()
        elif op in ["min"]:
            mat.SetOperationToMin()
        elif op in ["max"]:
            mat.SetOperationToMax()
        elif op in ["atan"]:
            mat.SetOperationToATAN()
        elif op in ["atan2"]:
            mat.SetOperationToATAN2()
        else:
            vedo.logger.error(f"unknown operation {operation}")
            raise RuntimeError()
        mat.Update()

        self._update(mat.GetOutput())

        self.pipeline = utils.OperationNode(
            f"operation\n{op}", parents=[self, volume2], shape="cylinder", c="#4cc9f0"
        )
        return self

    def frequency_pass_filter(self, low_cutoff=None, high_cutoff=None, order=1):
        """
        Low-pass and high-pass filtering become trivial in the frequency domain.
        A portion of the pixels/voxels are simply masked or attenuated.
        This function applies a high pass Butterworth filter that attenuates the
        frequency domain image.

        The gradual attenuation of the filter is important.
        A simple high-pass filter would simply mask a set of pixels in the frequency domain,
        but the abrupt transition would cause a ringing effect in the spatial domain.

        Arguments:
            low_cutoff : (list)
                the cutoff frequencies for x, y and z
            high_cutoff : (list)
                the cutoff frequencies for x, y and z
            order : (int)
                order determines sharpness of the cutoff curve
        """
        # https://lorensen.github.io/VTKExamples/site/Cxx/ImageProcessing/IdealHighPass
        fft = vtk.vtkImageFFT()
        fft.SetInputData(self.dataset)
        fft.Update()
        out = fft.GetOutput()

        if high_cutoff:
            blp = vtk.vtkImageButterworthLowPass()
            blp.SetInputData(out)
            blp.SetCutOff(high_cutoff)
            blp.SetOrder(order)
            blp.Update()
            out = blp.GetOutput()

        if low_cutoff:
            bhp = vtk.vtkImageButterworthHighPass()
            bhp.SetInputData(out)
            bhp.SetCutOff(low_cutoff)
            bhp.SetOrder(order)
            bhp.Update()
            out = bhp.GetOutput()

        rfft = vtk.vtkImageRFFT()
        rfft.SetInputData(out)
        rfft.Update()

        ecomp = vtk.vtkImageExtractComponents()
        ecomp.SetInputData(rfft.GetOutput())
        ecomp.SetComponents(0)
        ecomp.Update()
        self._update(ecomp.GetOutput())
        self.pipeline = utils.OperationNode("frequency_pass_filter", parents=[self], c="#4cc9f0")
        return self

    def smooth_gaussian(self, sigma=(2, 2, 2), radius=None):
        """
        Performs a convolution of the input Volume with a gaussian.

        Arguments:
            sigma : (float, list)
                standard deviation(s) in voxel units.
                A list can be given to smooth in the three direction differently.
            radius : (float, list)
                radius factor(s) determine how far out the gaussian
                kernel will go before being clamped to zero. A list can be given too.
        """
        gsf = vtk.vtkImageGaussianSmooth()
        gsf.SetDimensionality(3)
        gsf.SetInputData(self.dataset)
        if utils.is_sequence(sigma):
            gsf.SetStandardDeviations(sigma)
        else:
            gsf.SetStandardDeviation(sigma)
        if radius is not None:
            if utils.is_sequence(radius):
                gsf.SetRadiusFactors(radius)
            else:
                gsf.SetRadiusFactor(radius)
        gsf.Update()
        self._update(gsf.GetOutput())
        self.pipeline = utils.OperationNode("smooth_gaussian", parents=[self], c="#4cc9f0")
        return self

    def smooth_median(self, neighbours=(2, 2, 2)):
        """
        Median filter that replaces each pixel with the median value
        from a rectangular neighborhood around that pixel.
        """
        imgm = vtk.vtkImageMedian3D()
        imgm.SetInputData(self.dataset)
        if utils.is_sequence(neighbours):
            imgm.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        else:
            imgm.SetKernelSize(neighbours, neighbours, neighbours)
        imgm.Update()
        self._update(imgm.GetOutput())
        self.pipeline = utils.OperationNode("smooth_median", parents=[self], c="#4cc9f0")
        return self

    def erode(self, neighbours=(2, 2, 2)):
        """
        Replace a voxel with the minimum over an ellipsoidal neighborhood of voxels.
        If `neighbours` of an axis is 1, no processing is done on that axis.

        Examples:
            - [erode_dilate.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/erode_dilate.py)

                ![](https://vedo.embl.es/images/volumetric/erode_dilate.png)
        """
        ver = vtk.vtkImageContinuousErode3D()
        ver.SetInputData(self.dataset)
        ver.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        ver.Update()
        self._update(ver.GetOutput())
        self.pipeline = utils.OperationNode("erode", parents=[self], c="#4cc9f0")
        return self

    def dilate(self, neighbours=(2, 2, 2)):
        """
        Replace a voxel with the maximum over an ellipsoidal neighborhood of voxels.
        If `neighbours` of an axis is 1, no processing is done on that axis.

        Check also `erode()` and `pad()`.

        Examples:
            - [erode_dilate.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/erode_dilate.py)
        """
        ver = vtk.vtkImageContinuousDilate3D()
        ver.SetInputData(self.dataset)
        ver.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        ver.Update()
        self._update(ver.GetOutput())
        self.pipeline = utils.OperationNode("dilate", parents=[self], c="#4cc9f0")
        return self

    def magnitude(self):
        """Colapses components with magnitude function."""
        imgm = vtk.vtkImageMagnitude()
        imgm.SetInputData(self.dataset)
        imgm.Update()
        self._update(imgm.GetOutput())
        self.pipeline = utils.OperationNode("magnitude", parents=[self], c="#4cc9f0")
        return self

    def topoints(self):
        """
        Extract all image voxels as points.
        This function takes an input `Volume` and creates an `Mesh`
        that contains the points and the point attributes.

        Examples:
            - [vol2points.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/vol2points.py)
        """
        v2p = vtk.vtkImageToPoints()
        v2p.SetInputData(self.dataset)
        v2p.Update()
        mpts = vedo.Points(v2p.GetOutput())
        mpts.pipeline = utils.OperationNode("topoints", parents=[self], c="#4cc9f0:#e9c46a")
        return mpts

    def euclidean_distance(self, anisotropy=False, max_distance=None):
        """
        Implementation of the Euclidean DT (Distance Transform) using Saito's algorithm.
        The distance map produced contains the square of the Euclidean distance values.
        The algorithm has a O(n^(D+1)) complexity over n x n x...x n images in D dimensions.

        Check out also: https://en.wikipedia.org/wiki/Distance_transform

        Arguments:
            anisotropy : bool
                used to define whether Spacing should be used in the
                computation of the distances.
            max_distance : bool
                any distance bigger than max_distance will not be
                computed but set to this specified value instead.

        Examples:
            - [euclidian_dist.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/euclidian_dist.py)
        """
        euv = vtk.vtkImageEuclideanDistance()
        euv.SetInputData(self.dataset)
        euv.SetConsiderAnisotropy(anisotropy)
        if max_distance is not None:
            euv.InitializeOn()
            euv.SetMaximumDistance(max_distance)
        euv.SetAlgorithmToSaito()
        euv.Update()
        vol = Volume(euv.GetOutput())
        vol.pipeline = utils.OperationNode("euclidean_distance", parents=[self], c="#4cc9f0")
        return vol

    def correlation_with(self, vol2, dim=2):
        """
        Find the correlation between two volumetric data sets.
        Keyword `dim` determines whether the correlation will be 3D, 2D or 1D.
        The default is a 2D Correlation.

        The output size will match the size of the first input.
        The second input is considered the correlation kernel.
        """
        imc = vtk.vtkImageCorrelation()
        imc.SetInput1Data(self.dataset)
        imc.SetInput2Data(vol2.dataset)
        imc.SetDimensionality(dim)
        imc.Update()
        vol = Volume(imc.GetOutput())

        vol.pipeline = utils.OperationNode("correlation_with", parents=[self, vol2], c="#4cc9f0")
        return vol

    def scale_voxels(self, scale=1):
        """Scale the voxel content by factor `scale`."""
        rsl = vtk.vtkImageReslice()
        rsl.SetInputData(self.dataset)
        rsl.SetScalarScale(scale)
        rsl.Update()
        self._update(rsl.GetOutput())
        self.pipeline = utils.OperationNode(
            f"scale_voxels\nscale={scale}", parents=[self], c="#4cc9f0"
        )
        return self


