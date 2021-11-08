import glob, os
import numpy as np
import vtk
import vedo.colors as colors
import vedo.docs as docs
import vedo.utils as utils
from vedo.mesh import Mesh
from vedo.base import BaseGrid, Base3DProp
from deprecated import deprecated

__doc__ = ("""Submodule extending the ``vtkVolume`` object functionality."""
    + docs._defs
)

__all__ = [
           "Volume",
           "VolumeSlice",
           "mesh2Volume",
           "volumeFromMesh",
           "interpolateToVolume",
           "signedDistanceFromPointCloud",
          ]


def mesh2Volume(mesh, spacing=(1, 1, 1)):
    """
    Convert a mesh it into a ``Volume``
    where the foreground (exterior) voxels value is 1 and the background
    (interior) voxels value is 0.
    Internally the ``vtkPolyDataToImageStencil`` class is used.

    |mesh2volume| |mesh2volume.py|_
    """
    # https://vtk.org/Wiki/VTK/Examples/Cxx/PolyData/PolyDataToImageData
    pd = mesh.polydata()

    whiteImage = vtk.vtkImageData()
    bounds = pd.GetBounds()

    whiteImage.SetSpacing(spacing)

    # compute dimensions
    dim = [0, 0, 0]
    for i in [0, 1, 2]:
        dim[i] = int(np.ceil((bounds[i * 2 + 1] - bounds[i * 2]) / spacing[i]))
    whiteImage.SetDimensions(dim)
    whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)

    origin = [0, 0, 0]
    origin[0] = bounds[0] + spacing[0] / 2
    origin[1] = bounds[2] + spacing[1] / 2
    origin[2] = bounds[4] + spacing[2] / 2
    whiteImage.SetOrigin(origin)
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # fill the image with foreground voxels:
    inval = 255
    count = whiteImage.GetNumberOfPoints()
    for i in range(count):
        whiteImage.GetPointData().GetScalars().SetTuple1(i, inval)

    # polygonal data --> image stencil:
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(pd)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
    pol2stenc.Update()

    # cut the corresponding white image and set the background:
    outval = 0
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImage)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()
    return Volume(imgstenc.GetOutput())


def volumeFromMesh(mesh, bounds=None, dims=(20,20,20), signed=True, negate=False):
    """
    Compute signed distances over a volume from an input mesh.
    The output is a ``Volume`` object whose voxels contains the signed distance from
    the mesh.

    :param list bounds: bounds of the output volume.
    :param list dims: dimensions (nr. of voxels) of the output volume.

    See example script: |volumeFromMesh.py|_
    """
    if bounds is None:
        bounds = mesh.GetBounds()
    sx = (bounds[1]-bounds[0])/dims[0]
    sy = (bounds[3]-bounds[2])/dims[1]
    sz = (bounds[5]-bounds[4])/dims[2]

    img = vtk.vtkImageData()
    img.SetDimensions(dims)
    img.SetSpacing(sx, sy, sz)
    img.SetOrigin(bounds[0], bounds[2], bounds[4])
    img.AllocateScalars(vtk.VTK_FLOAT, 1)

    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(mesh.polydata())
    b4 = bounds[4]
    r2 = range(dims[2])

    for i in range(dims[0]):
        x = i*sx+bounds[0]
        for j in range(dims[1]):
            y = j*sy+bounds[2]
            for k in r2:
                v = imp.EvaluateFunction((x, y, k*sz+b4))
                if signed:
                    if negate:
                        v = -v
                else:
                    v = abs(v)
                img.SetScalarComponentFromFloat(i,j,k, 0, v)

    vol = Volume(img)
    vol.name = "VolumeFromMesh"
    return vol

def interpolateToVolume(mesh, kernel='shepard',
                        radius=None, N=None,
                        bounds=None, nullValue=None,
                        dims=(25,25,25)):
    """
    Generate a ``Volume`` by interpolating a scalar
    or vector field which is only known on a scattered set of points or mesh.
    Available interpolation kernels are: shepard, gaussian, or linear.

    :param str kernel: interpolation kernel type [shepard]
    :param float radius: radius of the local search
    :param list bounds: bounding box of the output Volume object
    :param list dims: dimensions of the output Volume object
    :param float nullValue: value to be assigned to invalid points

    |interpolateVolume| |interpolateVolume.py|_
    """
    if isinstance(mesh, vtk.vtkPolyData):
        output = mesh
    else:
        output = mesh.polydata()

    if radius is None and not N:
        colors.printc("Error in interpolateToVolume(): please set either radius or N", c='r')
        raise RuntimeError

    # Create a probe volume
    probe = vtk.vtkImageData()
    probe.SetDimensions(dims)
    if bounds is None:
        bounds = output.GetBounds()
    probe.SetOrigin(bounds[0],bounds[2],bounds[4])
    probe.SetSpacing((bounds[1]-bounds[0])/dims[0],
                     (bounds[3]-bounds[2])/dims[1],
                     (bounds[5]-bounds[4])/dims[2])

    # if radius is None:
    #     radius = min(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])/3

    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(output)
    locator.BuildLocator()

    if kernel == 'shepard':
        kern = vtk.vtkShepardKernel()
        kern.SetPowerParameter(2)
    elif kernel == 'gaussian':
        kern = vtk.vtkGaussianKernel()
    elif kernel == 'linear':
        kern = vtk.vtkLinearKernel()
    else:
        print('Error in interpolateToVolume, available kernels are:')
        print(' [shepard, gaussian, linear]')
        raise RuntimeError()

    if radius:
        kern.SetRadius(radius)

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(probe)
    interpolator.SetSourceData(output)
    interpolator.SetKernel(kern)
    interpolator.SetLocator(locator)

    if N:
        kern.SetNumberOfPoints(N)
        kern.SetKernelFootprintToNClosest()
    else:
        kern.SetRadius(radius)

    if nullValue is not None:
        interpolator.SetNullValue(nullValue)
    else:
        interpolator.SetNullPointsStrategyToClosestPoint()
    interpolator.Update()
    return Volume(interpolator.GetOutput())



def signedDistanceFromPointCloud(mesh, maxradius=None, bounds=None, dims=(20,20,20)):
    """
    Compute signed distances over a volume from an input point cloud.
    The output is a ``Volume`` object whose voxels contains the signed distance from
    the cloud.

    :param float maxradius: how far out to propagate distance calculation
    :param list bounds: volume bounds.
    :param list dims: dimensions (nr. of voxels) of the output volume.
    """
    if bounds is None:
        bounds = mesh.GetBounds()
    if maxradius is None:
        maxradius = mesh.diagonalSize()/10.
    dist = vtk.vtkSignedDistance()
    dist.SetInputData(mesh.polydata(True))
    dist.SetRadius(maxradius)
    dist.SetBounds(bounds)
    dist.SetDimensions(dims)
    dist.Update()
    vol = Volume(dist.GetOutput())
    vol.name = "signedDistanceVolume"
    return vol


##########################################################################
class BaseVolume:
    def __init__(self, inputobj=None):
        self._data = None
        self._mapper = None

    def _update(self, img):
        self._data = img
        self._data.GetPointData().Modified()
        self._mapper.SetInputData(img)
        self._mapper.Modified()
        self._mapper.Update()
        return self

    def clone(self):
        """Return a clone copy of the Volume."""
        newimg = vtk.vtkImageData()
        newimg.CopyStructure(self._data)
        newimg.CopyAttributes(self._data)
        newvol = Volume(newimg)
        prop = vtk.vtkVolumeProperty()
        prop.DeepCopy(self.GetProperty())
        newvol.SetProperty(prop)
        newvol.SetOrigin(self.GetOrigin())
        newvol.SetScale(self.GetScale())
        newvol.SetOrientation(self.GetOrientation())
        newvol.SetPosition(self.GetPosition())
        return newvol


    def imagedata(self):
        """Return the underlying vtkImagaData object."""
        return self._data

    @deprecated(reason=colors.red+"Please use tonumpy()"+colors.reset)
    def getDataArray(self):
        return self.tonumpy()

    def tonumpy(self):
        """
        Get read-write access to voxels of a Volume object as a numpy array.

        When you set values in the output image, you don't want numpy to reallocate the array
        but instead set values in the existing array, so use the [:] operator.
        Example: arr[:] = arr*2 + 15

        If the array is modified call:
        ``volume.imagedata().GetPointData().GetScalars().Modified()``
        when all your modifications are completed.
        """
        narray_shape = tuple(reversed(self._data.GetDimensions()))
        narray = utils.vtk2numpy(self._data.GetPointData().GetScalars()).reshape(narray_shape)
        narray = np.transpose(narray, axes=[2, 1, 0])
        return narray

    def modified(self):
        """Use in conjunction with ``tonumpy()`` to update any modifications to the volume array"""
        self._data.GetPointData().GetScalars().Modified()
        return self

    def dimensions(self):
        """Return the nr. of voxels in the 3 dimensions."""
        return np.array(self._data.GetDimensions())

    def scalarRange(self):
        """Return the range of the scalar values."""
        return np.array(self._data.GetScalarRange())

    def spacing(self, s=None):
        """Set/get the voxels size in the 3 dimensions."""
        if s is not None:
            self._data.SetSpacing(s)
            return self
        else:
            return np.array(self._data.GetSpacing())

    def origin(self, s=None):
        """Set/get the origin of the volumetric dataset."""
        ### superseedes base.origin()
        ### DIFFERENT from base.origin()!
        if s is not None:
            self._data.SetOrigin(s)
            return self
        else:
            return np.array(self._data.GetOrigin())

    def center(self, center=None):
        """Set/get the volume coordinates of its center.
        Position is reset to (0,0,0)."""
        if center is not None:
            cn = self._data.GetCenter()
            self._data.SetOrigin(-np.array(cn)/2)
            self._update(self._data)
            self.pos(0,0,0)
            return self
        else:
            return np.array(self._data.GetCenter())

    def permuteAxes(self, x, y ,z):
        """Reorder the axes of the Volume by specifying
        the input axes which are supposed to become the new X, Y, and Z."""
        imp = vtk.vtkImagePermute()
        imp.SetFilteredAxes(x,y,z)
        imp. SetInputData(self.imagedata())
        imp.Update()
        return self._update(imp.GetOutput())

    def resample(self, newSpacing, interpolation=1):
        """
        Resamples a ``Volume`` to be larger or smaller.

        This method modifies the spacing of the input.
        Linear interpolation is used to resample the data.

        :param list newSpacing: a list of 3 new spacings for the 3 axes.
        :param int interpolation: 0=nearest_neighbor, 1=linear, 2=cubic
        """
        rsp = vtk.vtkImageResample()
        oldsp = self.spacing()
        for i in range(3):
            if oldsp[i] != newSpacing[i]:
                rsp.SetAxisOutputSpacing(i, newSpacing[i])
        rsp.InterpolateOn()
        rsp.SetInterpolationMode(interpolation)
        rsp.OptimizationOn()
        rsp.Update()
        return self._update(rsp.GetOutput())


    def interpolation(self, itype):
        """
        Set interpolation type.

        0 = nearest neighbour
        1 = linear
        """
        self.property.SetInterpolationType(itype)
        return self

    def threshold(self, above=None, below=None, replace=None, replaceOut=None):
        """
        Binary or continuous volume thresholding.
        Find the voxels that contain a value above/below the input values
        and replace them with a new value (default is 0).
        """
        th = vtk.vtkImageThreshold()
        th.SetInputData(self.imagedata())

        # sanity checks
        if above is not None and below is not None:
            if above==below:
                return self
            if above > below:
                colors.printc("Warning in volume.threshold(): above > below, skip.", c='y')
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

        if replaceOut is not None:
            th.SetReplaceOut(True)
            th.SetOutValue(replaceOut)
        else:
            th.SetReplaceOut(False)

        th.Update()
        out = th.GetOutput()
        return self._update(out)


    def crop(self,
             left=None, right=None,
             back=None, front=None,
             bottom=None, top=None,
             VOI=()
            ):
        """Crop a ``Volume`` object.

        :param float left:   fraction to crop from the left plane (negative x)
        :param float right:  fraction to crop from the right plane (positive x)
        :param float back:   fraction to crop from the back plane (negative y)
        :param float front:  fraction to crop from the front plane (positive y)
        :param float bottom: fraction to crop from the bottom plane (negative z)
        :param float top:    fraction to crop from the top plane (positive z)
        :param list VOI:     extract Volume Of Interest expressed in voxel numbers

            Eg.: vol.crop(VOI=(xmin, xmax, ymin, ymax, zmin, zmax)) # all integers nrs
        """
        extractVOI = vtk.vtkExtractVOI()
        extractVOI.SetInputData(self.imagedata())

        if len(VOI):
            extractVOI.SetVOI(VOI)
        else:
            d = self.imagedata().GetDimensions()
            bx0, bx1, by0, by1, bz0, bz1 = 0, d[0]-1, 0, d[1]-1, 0, d[2]-1
            if left is not None:   bx0 = int((d[0]-1)*left)
            if right is not None:  bx1 = int((d[0]-1)*(1-right))
            if back is not None:   by0 = int((d[1]-1)*back)
            if front is not None:  by1 = int((d[1]-1)*(1-front))
            if bottom is not None: bz0 = int((d[2]-1)*bottom)
            if top is not None:    bz1 = int((d[2]-1)*(1-top))
            extractVOI.SetVOI(bx0, bx1, by0, by1, bz0, bz1)
        extractVOI.Update()
        return self._update(extractVOI.GetOutput())

    def append(self, volumes, axis='z', preserveExtents=False):
        """
        Take the components from multiple inputs and merges them into one output.
        Except for the append axis, all inputs must have the same extent.
        All inputs must have the same number of scalar components.
        The output has the same origin and spacing as the first input.
        The origin and spacing of all other inputs are ignored.
        All inputs must have the same scalar type.

        :param int,str axis: axis expanded to hold the multiple images.
        :param bool preserveExtents: if True, the extent of the inputs is used to place
            the image in the output. The whole extent of the output is the union of the input
            whole extents. Any portion of the output not covered by the inputs is set to zero.
            The origin and spacing is taken from the first input.

        .. code-block:: python

            from vedo import Volume, dataurl
            vol = Volume(dataurl+'embryo.tif')
            vol.append(vol, axis='x').show()
        """
        ima = vtk.vtkImageAppend()
        ima.SetInputData(self.imagedata())
        if not utils.isSequence(volumes):
            volumes = [volumes]
        for volume in volumes:
            if isinstance(volume, vtk.vtkImageData):
                ima.AddInputData(volume)
            else:
                ima.AddInputData(volume.imagedata())
        ima.SetPreserveExtents(preserveExtents)
        if axis   == "x":
            axis = 0
        elif axis == "y":
            axis = 1
        elif axis == "z":
            axis = 2
        ima.SetAppendAxis(axis)
        ima.Update()
        return self._update(ima.GetOutput())

    def resize(self, *newdims):
        """Increase or reduce the number of voxels of a Volume with interpolation."""
        old_dims = np.array(self.imagedata().GetDimensions())
        old_spac = np.array(self.imagedata().GetSpacing())
        rsz = vtk.vtkImageResize()
        rsz.SetResizeMethodToOutputDimensions()
        rsz.SetInputData(self.imagedata())
        rsz.SetOutputDimensions(newdims)
        rsz.Update()
        self._data = rsz.GetOutput()
        new_spac = old_spac * old_dims/newdims  # keep aspect ratio
        self._data.SetSpacing(new_spac)
        return self._update(self._data)

    def normalize(self):
        """Normalize that scalar components for each point."""
        norm = vtk.vtkImageNormalize()
        norm.SetInputData(self.imagedata())
        norm.Update()
        return self._update(norm.GetOutput())

    def scaleVoxels(self, scale=1):
        """Scale the voxel content by factor `scale`."""
        rsl = vtk.vtkImageReslice()
        rsl.SetInputData(self.imagedata())
        rsl.SetScalarScale(scale)
        rsl.Update()
        return self._update(rsl.GetOutput())

    def mirror(self, axis="x"):
        """
        Mirror flip along one of the cartesian axes.

        .. note::  ``axis='n'``, will flip only mesh normals.

        |mirror| |mirror.py|_
        """
        img = self.imagedata()

        ff = vtk.vtkImageFlip()
        ff.SetInputData(img)
        if axis.lower() == "x":
            ff.SetFilteredAxis(0)
        elif axis.lower() == "y":
            ff.SetFilteredAxis(1)
        elif axis.lower() == "z":
            ff.SetFilteredAxis(2)
        else:
            colors.printc("Error in mirror(): mirror must be set to x, y, z or n.", c='r')
            raise RuntimeError()
        ff.Update()
        return self._update(ff.GetOutput())


    def operation(self, operation, volume2=None):
        """
        Perform operations with ``Volume`` objects.

        `volume2` can be a constant value.

        Possible operations are: ``+``, ``-``, ``/``, ``1/x``, ``sin``, ``cos``, ``exp``, ``log``,
        ``abs``, ``**2``, ``sqrt``, ``min``, ``max``, ``atan``, ``atan2``, ``median``,
        ``mag``, ``dot``, ``gradient``, ``divergence``, ``laplacian``.

        |volumeOperations| |volumeOperations.py|_
        """
        op = operation.lower()
        image1 = self._data

        if op in ["median"]:
            mf = vtk.vtkImageMedian3D()
            mf.SetInputData(image1)
            mf.Update()
            return Volume(mf.GetOutput())
        elif op in ["mag"]:
            mf = vtk.vtkImageMagnitude()
            mf.SetInputData(image1)
            mf.Update()
            return Volume(mf.GetOutput())
        elif op in ["dot", "dotproduct"]:
            mf = vtk.vtkImageDotProduct()
            mf.SetInput1Data(image1)
            mf.SetInput2Data(volume2._data)
            mf.Update()
            return Volume(mf.GetOutput())
        elif op in ["grad", "gradient"]:
            mf = vtk.vtkImageGradient()
            mf.SetDimensionality(3)
            mf.SetInputData(image1)
            mf.Update()
            return Volume(mf.GetOutput())
        elif op in ["div", "divergence"]:
            mf = vtk.vtkImageDivergence()
            mf.SetInputData(image1)
            mf.Update()
            return Volume(mf.GetOutput())
        elif op in ["laplacian"]:
            mf = vtk.vtkImageLaplacian()
            mf.SetDimensionality(3)
            mf.SetInputData(image1)
            mf.Update()
            return Volume(mf.GetOutput())

        mat = vtk.vtkImageMathematics()
        mat.SetInput1Data(image1)

        K = None

        if isinstance(volume2, (int, float)):
            K = volume2
            mat.SetConstantK(K)
            mat.SetConstantC(K)
        elif volume2 is not None:  # assume image2 is a constant value
            mat.SetInput2Data(volume2._data)

        if op in ["+", "add", "plus"]:
            if K:
                mat.SetOperationToAddConstant()
            else:
                mat.SetOperationToAdd()

        elif op in ["-", "subtract", "minus"]:
            if K:
                mat.SetConstantC(-K)
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
            colors.printc("Error in volume.operation(): unknown operation", operation, c='r')
            raise RuntimeError()
        mat.Update()
        return self._update(mat.GetOutput())


    def frequencyPassFilter(self, lowcutoff=None, highcutoff=None, order=1):
        """
        Low-pass and high-pass filtering become trivial in the frequency domain.
        A portion of the pixels/voxels are simply masked or attenuated.
        This function applies a high pass Butterworth filter that attenuates the
        frequency domain image with the function

        |G_Of_Omega|

        The gradual attenuation of the filter is important.
        A simple high-pass filter would simply mask a set of pixels in the frequency domain,
        but the abrupt transition would cause a ringing effect in the spatial domain.

        :param list lowcutoff:  the cutoff frequencies for x, y and z
        :param list highcutoff: the cutoff frequencies for x, y and z
        :param int order: order determines sharpness of the cutoff curve

        Check out also this example:

        |idealpass|
        """
        #https://lorensen.github.io/VTKExamples/site/Cxx/ImageProcessing/IdealHighPass
        fft = vtk.vtkImageFFT()
        fft.SetInputData(self._data)
        fft.Update()
        out = fft.GetOutput()

        if highcutoff:
            butterworthLowPass = vtk.vtkImageButterworthLowPass()
            butterworthLowPass.SetInputData(out)
            butterworthLowPass.SetCutOff(highcutoff)
            butterworthLowPass.SetOrder(order)
            butterworthLowPass.Update()
            out = butterworthLowPass.GetOutput()

        if lowcutoff:
            butterworthHighPass = vtk.vtkImageButterworthHighPass()
            butterworthHighPass.SetInputData(out)
            butterworthHighPass.SetCutOff(lowcutoff)
            butterworthHighPass.SetOrder(order)
            butterworthHighPass.Update()
            out = butterworthHighPass.GetOutput()

        butterworthRfft = vtk.vtkImageRFFT()
        butterworthRfft.SetInputData(out)
        butterworthRfft.Update()

        butterworthReal = vtk.vtkImageExtractComponents()
        butterworthReal.SetInputData(butterworthRfft.GetOutput())
        butterworthReal.SetComponents(0)
        butterworthReal.Update()
        return self._update(butterworthReal.GetOutput())

    def gaussianSmooth(self, sigma=(2,2,2), radius=None):
        """Performs a convolution of the input Volume with a gaussian.

        :param float,list sigma: standard deviation(s) in voxel units.
            A list can be given to smooth in the three direction differently.
        :param float,list radius: radius factor(s) determine how far out the gaussian
            kernel will go before being clamped to zero. A list can be given too.
        """
        gsf = vtk.vtkImageGaussianSmooth()
        gsf.SetDimensionality(3)
        gsf.SetInputData(self.imagedata())
        if utils.isSequence(sigma):
            gsf.SetStandardDeviations(sigma)
        else:
            gsf.SetStandardDeviation(sigma)
        if radius is not None:
            if utils.isSequence(radius):
                gsf.SetRadiusFactors(radius)
            else:
                gsf.SetRadiusFactor(radius)
        gsf.Update()
        return self._update(gsf.GetOutput())

    def medianSmooth(self, neighbours=(2,2,2)):
        """Median filter that replaces each pixel with the median value
        from a rectangular neighborhood around that pixel.
        """
        imgm = vtk.vtkImageMedian3D()
        imgm.SetInputData(self.imagedata())
        if utils.isSequence(neighbours):
            imgm.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        else:
            imgm.SetKernelSize(neighbours, neighbours, neighbours)
        imgm.Update()
        return self._update(imgm.GetOutput())

    def erode(self, neighbours=(2,2,2)):
        """Replace a voxel with the minimum over an ellipsoidal neighborhood of voxels.
        If `neighbours` of an axis is 1, no processing is done on that axis.

        See example script: |erode_dilate.py|_
        """
        ver = vtk.vtkImageContinuousErode3D()
        ver.SetInputData(self._data)
        ver.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        ver.Update()
        return self._update(ver.GetOutput())


    def dilate(self, neighbours=(2,2,2)):
        """Replace a voxel with the maximum over an ellipsoidal neighborhood of voxels.
        If `neighbours` of an axis is 1, no processing is done on that axis.

        See example script: |erode_dilate.py|_
        """
        ver = vtk.vtkImageContinuousDilate3D()
        ver.SetInputData(self._data)
        ver.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        ver.Update()
        return self._update(ver.GetOutput())


    def magnitude(self):
        """Colapses components with magnitude function."""
        imgm = vtk.vtkImageMagnitude()
        imgm.SetInputData(self.imagedata())
        imgm.Update()
        return self._update(imgm.GetOutput())


    def topoints(self):
        """Extract all image voxels as points.
        This function takes an input ``Volume`` and creates an ``Mesh``
        that contains the points and the point attributes.

        See example script: |vol2points.py|_
        """
        v2p = vtk.vtkImageToPoints()
        v2p.SetInputData(self.imagedata())
        v2p.Update()
        mpts = Mesh(v2p.GetOutput())
        return mpts


    def euclideanDistance(self, anisotropy=False, maxDistance=None):
        """Implementation of the Euclidean DT (Distance Transform) using Saito's algorithm.
        The distance map produced contains the square of the Euclidean distance values.
        The algorithm has a O(n^(D+1)) complexity over nxnx...xn images in D dimensions.

        Check out also: https://en.wikipedia.org/wiki/Distance_transform

        :param bool anisotropy: used to define whether Spacing should be used
            in the computation of the distances.
        :param float maxDistance: any distance bigger than maxDistance will not be
            computed but set to this specified value instead.

        See example script: |euclDist.py|_
        """
        euv = vtk.vtkImageEuclideanDistance()
        euv.SetInputData(self._data)
        euv.SetConsiderAnisotropy(anisotropy)
        if maxDistance is not None:
            euv.InitializeOn()
            euv.SetMaximumDistance(maxDistance)
        euv.SetAlgorithmToSaito()
        euv.Update()
        return Volume(euv.GetOutput())


    def correlationWith(self, vol2, dim=2):
        """Find the correlation between two volumetric data sets.
        Keyword `dim` determines whether the correlation will be 3D, 2D or 1D.
        The default is a 2D Correlation.
        The output size will match the size of the first input.
        The second input is considered the correlation kernel.
        """
        imc = vtk.vtkImageCorrelation()
        imc.SetInput1Data(self._data)
        imc.SetInput2Data(vol2._data)
        imc.SetDimensionality(dim)
        imc.Update()
        return Volume(imc.GetOutput())



##########################################################################
class Volume(vtk.vtkVolume, BaseGrid, BaseVolume):
    """Derived class of ``vtkVolume``.
    Can be initialized with a numpy object, a ``vtkImageData``
    or a list of 2D bmp files.

    See e.g.: |numpy2volume1.py|_

    :param list,str c: sets colors along the scalar range, or a matplotlib color map name
    :param float,list alphas: sets transparencies along the scalar range
    :param float alphaUnit: low values make composite rendering look brighter and denser
    :param list origin: set volume origin coordinates
    :param list spacing: voxel dimensions in x, y and z.
    :param list dims: specify the dimensions of the volume.
    :param str mapper: either 'gpu', 'opengl_gpu', 'fixed' or 'smart'

    :param int mode: define the volumetric rendering style:

        - 0, composite rendering
        - 1, maximum projection rendering
        - 2, minimum projection
        - 3, average projection
        - 4, additive mode

        The default mode is "composite" where the scalar values are sampled through
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

        .. code-block:: python

            from vedo import Volume
            vol = Volume("path/to/mydata/rec*.bmp", c='jet', mode=1)
            vol.show(axes=1)

    .. hint:: if a `list` of values is used for `alphas` this is interpreted
        as a transfer function along the range of the scalar.

        |read_volume2| |read_volume2.py|_
    """
    def __init__(self, inputobj=None,
                 c='RdBu_r',
                 alpha=(0.0, 0.0, 0.2, 0.4, 0.8, 1.0),
                 alphaGradient=None,
                 alphaUnit=1,
                 mode=0,
                 shade=False,
                 spacing=None,
                 dims=None,
                 origin=None,
                 mapper='smart',
        ):

        vtk.vtkVolume.__init__(self)
        BaseGrid.__init__(self)
        BaseVolume.__init__(self)

        ###################
        if isinstance(inputobj, str):

            if "https://" in inputobj:
                from vedo.io import download
                inputobj = download(inputobj, verbose=False) # fpath
            elif os.path.isfile(inputobj):
                pass
            else:
                inputobj = sorted(glob.glob(inputobj))

        ###################
        if 'gpu' in mapper:
            self._mapper = vtk.vtkGPUVolumeRayCastMapper()
        elif 'opengl_gpu' in mapper:
            self._mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
        elif 'smart' in mapper:
            self._mapper = vtk.vtkSmartVolumeMapper()
        elif 'fixed' in mapper:
            self._mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        elif isinstance(mapper, vtk.vtkMapper):
            self._mapper = mapper
        else:
            print("Error unknown mapper type", [mapper])
            raise RuntimeError()
        self.SetMapper(self._mapper)

        ###################
        inputtype = str(type(inputobj))

        # colors.printc('Volume inputtype', inputtype, c='b')

        if inputobj is None:
            img = vtk.vtkImageData()

        elif utils.isSequence(inputobj):

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
                    pb.print('loading...')
                ima.Update()
                img = ima.GetOutput()

            else:
                if "ndarray" not in inputtype:
                    inputobj = np.asarray(inputobj)

                if len(inputobj.shape)==1:
                    varr = utils.numpy2vtk(inputobj, dtype=float)
                else:
                    if len(inputobj.shape)>2:
                        inputobj = np.transpose(inputobj, axes=[2, 1, 0])
                    varr = utils.numpy2vtk(inputobj.ravel(order='F'), dtype=float)
                varr.SetName('input_scalars')

                img = vtk.vtkImageData()
                if dims is not None:
                    img.SetDimensions(dims)
                else:
                    if len(inputobj.shape)==1:
                        colors.printc("Error: must set dimensions (dims keyword) in Volume.", c='r')
                        raise RuntimeError()
                    img.SetDimensions(inputobj.shape)
                img.GetPointData().AddArray(varr)
                img.GetPointData().SetActiveScalars(varr.GetName())

                #to convert rgb to numpy
                #        img_scalar = data.GetPointData().GetScalars()
                #        dims = data.GetDimensions()
                #        n_comp = img_scalar.GetNumberOfComponents()
                #        temp = utils.vtk2numpy(img_scalar)
                #        numpy_data = temp.reshape(dims[1],dims[0],n_comp)
                #        numpy_data = numpy_data.transpose(0,1,2)
                #        numpy_data = np.flipud(numpy_data)

        elif "ImageData" in inputtype:
            img = inputobj

        elif isinstance(inputobj, Volume):
            img = inputobj.inputdata()

        elif "UniformGrid" in inputtype:
            img = inputobj

        elif hasattr(inputobj, "GetOutput"): # passing vtk object, try extract imagdedata
            if hasattr(inputobj, "Update"):
                inputobj.Update()
            img = inputobj.GetOutput()

        elif isinstance(inputobj, str):
            from vedo.io import loadImageData, download
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            img = loadImageData(inputobj)

        else:
            colors.printc("Volume(): cannot understand input type:\n", inputtype, c='r')
            return


        if dims is not None:
            img.SetDimensions(dims)

        if origin is not None:
            img.SetOrigin(origin) ### DIFFERENT from volume.origin()!

        if spacing is not None:
            img.SetSpacing(spacing)

        self._data = img
        self._mapper.SetInputData(img)
        self.mode(mode).color(c).alpha(alpha).alphaGradient(alphaGradient)
        self.GetProperty().SetShade(True)
        self.GetProperty().SetInterpolationType(1)
        self.GetProperty().SetScalarOpacityUnitDistance(alphaUnit)

        # remember stuff:
        self._mode = mode
        self._color = c
        self._alpha = alpha
        self._alphaGrad = alphaGradient
        self._alphaUnit = alphaUnit

    def _update(self, img):
        self._data = img
        self._data.GetPointData().Modified()
        self._mapper.SetInputData(img)
        self._mapper.Modified()
        self._mapper.Update()
        return self

    def mode(self, mode=None):
        """Define the volumetric rendering style.

            - 0, composite rendering
            - 1, maximum projection rendering
            - 2, minimum projection rendering
            - 3, average projection rendering
            - 4, additive mode

        The default mode is "composite" where the scalar values are sampled through
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
        """
        if mode is None:
            return self._mapper.GetBlendMode()

        if isinstance(mode, str):
            if 'comp' in mode:
                mode = 0
            elif 'proj' in mode:
                if 'max' in mode:
                    mode = 1
                elif 'min' in mode:
                    mode = 2
                elif 'ave' in mode:
                    mode = 3
                else:
                    colors.printc("Error in volume.mode(): unknown mode", mode, c='r')
                    mode = 0
            elif 'add' in mode:
                mode = 4
            else:
                colors.printc("Error in volume.mode(): unknown mode", mode, c='r')
                mode = 0

        self._mapper.SetBlendMode(mode)
        self._mode = mode
        return self

    def shade(self, status=None):
        """
        Set/Get the shading of a Volume.
        Shading can be further controlled with ``volume.lighting()`` method.

        If shading is turned on, the mapper may perform shading calculations.
        In some cases shading does not apply
        (for example, in maximum intensity projection mode).
        """
        if status is None:
            return self.GetProperty().GetShade()
        self.GetProperty().SetShade(status)
        return self

    def cmap(self, c, alpha=None, vmin=None, vmax=None):
        """Same as color().

        :param list alpha: use a list to specify transparencies along the scalar range
        :param float vmin: force the min of the scalar range to be this value
        :param float vmax: force the max of the scalar range to be this value
        """
        return self.color(c, alpha, vmin, vmax)

    def jittering(self, status=None):
        """If `jittering` is `True`, each ray traversal direction will be perturbed slightly
        using a noise-texture to get rid of wood-grain effects.
        """
        if hasattr(self._mapper, 'SetUseJittering'): # tetmesh doesnt have it
            if status is None:
                return self._mapper.GetUseJittering()
            self._mapper.SetUseJittering(status)
        return self

    def alphaGradient(self, alphaGrad, vmin=None, vmax=None):
        """
        Assign a set of tranparencies to a volume's gradient
        along the range of the scalar value.
        A single constant value can also be assigned.
        The gradient function is used to decrease the opacity
        in the "flat" regions of the volume while maintaining the opacity
        at the boundaries between material types.  The gradient is measured
        as the amount by which the intensity changes over unit distance.

        The format for alphaGrad is the same as for method ``volume.alpha()``.

        |read_volume2| |read_volume2.py|_
        """
        if vmin is None:
            vmin, _ = self._data.GetScalarRange()
        if vmax is None:
            _, vmax = self._data.GetScalarRange()
        self._alphaGrad = alphaGrad
        volumeProperty = self.GetProperty()
        if alphaGrad is None:
            volumeProperty.DisableGradientOpacityOn()
            return self
        else:
            volumeProperty.DisableGradientOpacityOff()

        gotf = volumeProperty.GetGradientOpacity()
        if utils.isSequence(alphaGrad):
            alphaGrad = np.array(alphaGrad)
            if len(alphaGrad.shape)==1: # user passing a flat list e.g. (0.0, 0.3, 0.9, 1)
                for i, al in enumerate(alphaGrad):
                    xalpha = vmin + (vmax - vmin) * i / (len(alphaGrad) - 1)
                    # Create transfer mapping scalar value to gradient opacity
                    gotf.AddPoint(xalpha, al)
            elif len(alphaGrad.shape)==2: # user passing [(x0,alpha0), ...]
                gotf.AddPoint(vmin, alphaGrad[0][1])
                for xalpha, al in alphaGrad:
                    # Create transfer mapping scalar value to opacity
                    gotf.AddPoint(xalpha, al)
                gotf.AddPoint(vmax, alphaGrad[-1][1])
            #colors.printc("alphaGrad at", round(xalpha, 1), "\tset to", al, c="b", bold=0)
        else:
            gotf.AddPoint(vmin, alphaGrad) # constant alphaGrad
            gotf.AddPoint(vmax, alphaGrad)
        return self

    def componentWeight(self, i, weight):
        """
        Set the scalar component weight in range [0,1].
        """
        self.GetProperty().SetComponentWeight(i, weight)
        return self

    def xSlice(self, i):
        """Extract the slice at index `i` of volume along x-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.imagedata())
        nx, ny, nz = self.imagedata().GetDimensions()
        if i>nx-1:
            i=nx-1
        vslice.SetExtent(i,i, 0,ny, 0,nz)
        vslice.Update()
        return Mesh(vslice.GetOutput())

    def ySlice(self, j):
        """Extract the slice at index `j` of volume along y-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.imagedata())
        nx, ny, nz = self.imagedata().GetDimensions()
        if j>ny-1:
            j=ny-1
        vslice.SetExtent(0,nx, j,j, 0,nz)
        vslice.Update()
        return Mesh(vslice.GetOutput())

    def zSlice(self, k):
        """Extract the slice at index `i` of volume along z-axis."""
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(self.imagedata())
        nx, ny, nz = self.imagedata().GetDimensions()
        if k>nz-1:
            k=nz-1
        vslice.SetExtent(0,nx, 0,ny, k,k)
        vslice.Update()
        return Mesh(vslice.GetOutput())

    def slicePlane(self, origin=(0,0,0), normal=(1,1,1)):
        """Extract the slice along a given plane position and normal.

        |slicePlane1| |slicePlane1.py|_
        """
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(self._data)
        reslice.SetOutputDimensionality(2)
        newaxis = utils.versor(normal)
        pos = np.array(origin)
        initaxis = (0,0,1)
        crossvec = np.cross(initaxis, newaxis)
        angle = np.arccos(np.dot(initaxis, newaxis))
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.RotateWXYZ(np.rad2deg(angle), crossvec)
        T.Translate(pos)
        M = T.GetMatrix()
        reslice.SetResliceAxes(M)
        reslice.SetInterpolationModeToLinear()
        reslice.Update()
        vslice = vtk.vtkImageDataGeometryFilter()
        vslice.SetInputData(reslice.GetOutput())
        vslice.Update()
        msh = Mesh(vslice.GetOutput())
        msh.SetOrientation(T.GetOrientation())
        msh.SetPosition(pos)
        return msh



##########################################################################
class VolumeSlice(vtk.vtkImageSlice, Base3DProp, BaseVolume):
    """
    Derived class of ``vtkImageSlice``.
    This class is equivalent to ``Volume`` except for its representation.
    The main purpose of this class is to be used in conjunction with ``Volume``
    for visualization using ``mode="image"``.
    """
    def __init__(self, inputobj=None):

        vtk.vtkImageSlice.__init__(self)
        Base3DProp.__init__(self)
        BaseVolume.__init__(self)

        self._mapper = vtk.vtkImageResliceMapper()
        self._mapper.SliceFacesCameraOn()
        self._mapper.SliceAtFocalPointOn()
        self._mapper.SetAutoAdjustImageQuality(False)
        self._mapper.BorderOff()

        self.lut = None

        self.property = vtk.vtkImageProperty()
        self.property.SetInterpolationTypeToLinear()
        self.SetProperty(self.property)

        ###################
        if isinstance(inputobj, str):
            if "https://" in inputobj:
                from vedo.io import download
                inputobj = download(inputobj, verbose=False) # fpath
            elif os.path.isfile(inputobj):
                pass
            else:
                inputobj = sorted(glob.glob(inputobj))

        ###################
        inputtype = str(type(inputobj))

        if inputobj is None:
            img = vtk.vtkImageData()

        if isinstance(inputobj, Volume):
            img = inputobj.imagedata()
            self.lut = utils.ctf2lut(inputobj)

        elif utils.isSequence(inputobj):

            if isinstance(inputobj[0], str): # scan sequence of BMP files
                ima = vtk.vtkImageAppend()
                ima.SetAppendAxis(2)
                pb = utils.ProgressBar(0, len(inputobj))
                for i in pb.range():
                    f = inputobj[i]
                    picr = vtk.vtkBMPReader()
                    picr.SetFileName(f)
                    picr.Update()
                    mgf = vtk.vtkImageMagnitude()
                    mgf.SetInputData(picr.GetOutput())
                    mgf.Update()
                    ima.AddInputData(mgf.GetOutput())
                    pb.print('loading...')
                ima.Update()
                img = ima.GetOutput()

            else:
                if "ndarray" not in inputtype:
                    inputobj = np.array(inputobj)

                if len(inputobj.shape)==1:
                    varr = utils.numpy2vtk(inputobj, dtype=float)
                else:
                    if len(inputobj.shape)>2:
                        inputobj = np.transpose(inputobj, axes=[2, 1, 0])
                    varr = utils.numpy2vtk(inputobj.ravel(order='F'), dtype=float)
                varr.SetName('input_scalars')

                img = vtk.vtkImageData()
                img.SetDimensions(inputobj.shape)
                img.GetPointData().AddArray(varr)
                img.GetPointData().SetActiveScalars(varr.GetName())

        elif "ImageData" in inputtype:
            img = inputobj

        elif isinstance(inputobj, Volume):
            img = inputobj.inputdata()

        elif "UniformGrid" in inputtype:
            img = inputobj

        elif hasattr(inputobj, "GetOutput"): # passing vtk object, try extract imagdedata
            if hasattr(inputobj, "Update"):
                inputobj.Update()
            img = inputobj.GetOutput()

        elif isinstance(inputobj, str):
            from vedo.io import loadImageData, download
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            img = loadImageData(inputobj)

        else:
            colors.printc("VolumeSlice: cannot understand input type:\n", inputtype, c='r')
            return

        self._data = img
        self._mapper.SetInputData(img)
        self.SetMapper(self._mapper)

    def bounds(self):
        """Return the bounding box as [x0,x1, y0,y1, z0,z1]"""
        bns = [0,0,0,0,0,0]
        self.GetBounds(bns)
        return bns

    def colorize(self, lut=None, fixScalarRange=False):
        """
        Assign a LUT (Look Up Table) to colorize the slice, leave it ``None``
        to reuse an exisiting Volume color map.
        Use "bw" for automatic black and white.
        """
        if lut is None and self.lut:
            self.property.SetLookupTable(self.lut)
        elif isinstance(lut, vtk.vtkLookupTable):
            self.property.SetLookupTable(lut)
        elif lut == "bw":
            self.property.SetLookupTable(None)
        self.property.SetUseLookupTableScalarRange(fixScalarRange)
        return self

    def alpha(self, value):
        """Set opacity to the slice"""
        self.property.SetOpacity(value)
        return self

    def autoAdjustQuality(self, value=True):
        """Automatically reduce the rendering quality for greater speed when interacting"""
        self._mapper.SetAutoAdjustImageQuality(value)
        return self

    def slab(self, thickness=0, mode=0, sampleFactor=2):
        """
        Make a thick slice (slab).

        Parameters
        ----------
        thickness : float, optional
            set the slab thickness, for thick slicing
        mode : int, optional
            The slab type:
                0 = min
                1 = max
                2 = mean
                3 = sum
        sampleFactor : float, optional
            Set the number of slab samples to use as a factor of the number of input slices
            within the slab thickness. The default value is 2, but 1 will increase speed
            with very little loss of quality.
        """
        self._mapper.SetSlabThickness(thickness)
        self._mapper.SetSlabType(mode)
        self._mapper.SetSlabSampleFactor(sampleFactor)
        return self


    def faceCamera(self, value=True):
        """Make the slice always face the camera or not."""
        self._mapper.SetSliceFacesCameraOn(value)
        return self

    def jumpToNearestSlice(self, value=True):
        """This causes the slicing to occur at the closest slice to the focal point,
        instead of the default behavior where a new slice is interpolated between the original slices.
        Nothing happens if the plane is oblique to the original slices."""
        self.SetJumpToNearestSlice(value)
        return self

    def fillBackground(self, value=True):
        """Instead of rendering only to the image border, render out to the viewport boundary with
        the background color. The background color will be the lowest color on the lookup
        table that is being used for the image."""
        self._mapper.SetBackground(value)
        return self

    def lighting(self, window, level, ambient=1.0, diffuse=0.0):
        """Assign the values for window and color level."""
        self.property.SetColorWindow(window)
        self.property.SetColorLevel(level)
        self.property.SetAmbient(ambient)
        self.property.SetDiffuse(diffuse)
        return self



