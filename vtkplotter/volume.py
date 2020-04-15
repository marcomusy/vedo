from __future__ import division, print_function
import numpy as np
import vtk
import vtkplotter.colors as colors
import vtkplotter.docs as docs
import vtkplotter.utils as utils
from vtk.util.numpy_support import numpy_to_vtk
from vtkplotter.base import ActorBase
from vtkplotter.mesh import Mesh

__doc__ = (
    """
Submodule extending the ``vtkVolume`` object functionality.
"""
    + docs._defs
)

__all__ = ["Volume"]


##########################################################################
class Volume(vtk.vtkVolume, ActorBase):
    """Derived class of ``vtkVolume``.
    Can be initialized with a numpy object, a ``vtkImageData``
    or a list of 2D bmp files.

    See e.g.: |numpy2volume.py|_

    :param c: sets colors along the scalar range, or a matplotlib color map name
    :type c: list, str
    :param alphas: sets transparencies along the scalar range
    :type c: float, list
    :param list origin: set volume origin coordinates
    :param list spacing: voxel dimensions in x, y and z.
    :param list shape: specify the shape.
    :param str mapperType: either 'gpu', 'opengl_gpu', 'fixed' or 'smart'

    :param int mode: define the volumetric rendering style:

        - 0, Composite rendering
        - 1, maximum projection rendering
        - 2, minimum projection
        - 3, average projection
        - 4, additive mode

    .. hint:: if a `list` of values is used for `alphas` this is interpreted
        as a transfer function along the range of the scalar.

        |read_vti| |read_vti.py|_
    """

    def __init__(self, inputobj=None,
                 c=('b','lb','lg','y','r'),
                 alpha=(0.0, 0.0, 0.2, 0.4, 0.8, 1),
                 alphaGradient=None,
                 mode=0,
                 origin=None,
                 spacing=None,
                 shape=None,
                 mapperType='smart',
                 ):

        vtk.vtkVolume.__init__(self)
        ActorBase.__init__(self)

        inputtype = str(type(inputobj))
        #colors.printc('Volume inputtype', inputtype)

        if isinstance(inputobj, str):
            import glob
            inputobj = sorted(glob.glob(inputobj))


        if inputobj is None:
            img = vtk.vtkImageData()

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
                    pb.print('loading..')
                ima.Update()
                img = ima.GetOutput()

            else:
                if "ndarray" not in inputtype:
                    inputobj = np.array(inputobj)

                varr = numpy_to_vtk(inputobj.ravel(order='F'),
                                    deep=True, array_type=vtk.VTK_FLOAT)
                varr.SetName('input_scalars')

                img = vtk.vtkImageData()
                if shape is not None:
                    img.SetDimensions(shape)
                else:
                    img.SetDimensions(inputobj.shape)
                img.GetPointData().SetScalars(varr)

                #to convert rgb to numpy
                #        img_scalar = data.GetPointData().GetScalars()
                #        dims = data.GetDimensions()
                #        n_comp = img_scalar.GetNumberOfComponents()
                #        temp = numpy_support.vtk_to_numpy(img_scalar)
                #        numpy_data = temp.reshape(dims[1],dims[0],n_comp)
                #        numpy_data = numpy_data.transpose(0,1,2)
                #        numpy_data = np.flipud(numpy_data)

        elif "ImageData" in inputtype:
            img = inputobj
        elif "UniformGrid" in inputtype:
            img = inputobj
        elif "UnstructuredGrid" in inputtype:
            img = inputobj
            mapperType = 'tetra'
        elif hasattr(inputobj, "GetOutput"): # passing vtk object, try extract imagdedata
            if hasattr(inputobj, "Update"):
                inputobj.Update()
            img = inputobj.GetOutput()
        else:
            colors.printc("Volume(): cannot understand input type:\n", inputtype, c=1)
            return

        if 'gpu' in mapperType:
            self._mapper = vtk.vtkGPUVolumeRayCastMapper()
        elif 'opengl_gpu' in mapperType:
            self._mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
        elif 'smart' in mapperType:
            self._mapper = vtk.vtkSmartVolumeMapper()
        elif 'fixed' in mapperType:
            self._mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        elif 'tetra' in mapperType:
            self._mapper = vtk.vtkProjectedTetrahedraMapper()
        elif 'unstr' in mapperType:
            self._mapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()
        else:
            print("Error unknown mapperType", mapperType)
            raise RuntimeError()

        if origin is not None:
            img.SetOrigin(origin)
        if spacing is not None:
            img.SetSpacing(spacing)
        if shape is not None:
            img.SetDimensions(shape)

        self._imagedata = img
        self._mapper.SetInputData(img)
        self.SetMapper(self._mapper)
        self.mode(mode).color(c).alpha(alpha).alphaGradient(alphaGradient)
        self.GetProperty().SetInterpolationType(1)
        # remember stuff:
        self._mode = mode
        self._color = c
        self._alpha = alpha
        self._alphaGrad = alphaGradient

    def _update(self, img):
        self._imagedata = img
        self._mapper.SetInputData(img)
        self._mapper.Modified()
        return self

    def mode(self, mode=None):
        """Define the volumetric rendering style.

            - 0, Composite rendering
            - 1, maximum projection rendering
            - 2, minimum projection
            - 3, average projection
            - 4, additive mode
        """
        if mode is None:
            return self._mapper.GetBlendMode()

        volumeProperty = self.GetProperty()
        self._mapper.SetBlendMode(mode)
        self._mode = mode
        if mode == 0:
            volumeProperty.ShadeOn()
            self.lighting('shiny')
            self.jittering(True)
        elif mode == 1:
            volumeProperty.ShadeOff()
            self.jittering(True)
        return self

    def clone(self):
        """Return a clone copy of the Volume."""
        newimg = vtk.vtkImageData()
        newimg.CopyStructure(self._imagedata)
        newimg.CopyAttributes(self._imagedata)
        newvol = Volume(newimg)
        prop = vtk.vtkVolumeProperty()
        prop.DeepCopy(self.GetProperty())
        newvol.SetProperty(prop)
        return newvol

    def jittering(self, status=None):
        """If `jittering` is `True`, each ray traversal direction will be perturbed slightly
        using a noise-texture to get rid of wood-grain effects.
        """
        if hasattr(self._mapper, 'SetUseJittering'):
            if status is None:
                return self._mapper.GetUseJittering()
            self._mapper.SetUseJittering(status)
            return self
        return None

    def imagedata(self):
        """Return the underlying ``vtkImagaData`` object."""
        return self._imagedata

    def getDataArray(self):
        """Get read-write access to voxels of a Volume object as a numpy array.

        When you set values in the output image, you don't want numpy to reallocate the array
        but instead set values in the existing array, so use the [:] operator.
        Example: arr[:] = arr*2 + 15

        If the array is modified call:
        ``volume.imagedata().GetPointData().GetScalars().Modified()``
        when all your modifications are completed.
        """
        from vtk.util.numpy_support import vtk_to_numpy
        narray_shape = tuple(reversed(self._imagedata.GetDimensions()))
        narray = vtk_to_numpy(self._imagedata.GetPointData().GetScalars()).reshape(narray_shape)
        narray = np.transpose(narray, axes=[2, 1, 0])
        return narray

    def dimensions(self):
        """Return the nr. of voxels in the 3 dimensions."""
        return self._imagedata.GetDimensions()

    def spacing(self, s=None):
        """Set/get the voxels size in the 3 dimensions."""
        if s is not None:
            self._imagedata.SetSpacing(s)
            self._update(self._imagedata)
            return self
        else:
            return np.array(self._imagedata.GetSpacing())

    def center(self, center=None):
        """Set/get the volume coordinates of its center.
        Position is reset to (0,0,0)."""
        if center is not None:
            cn = self._imagedata.GetCenter()
            self._imagedata.SetOrigin(-np.array(cn)/2)
            self._update(self._imagedata)
            self.pos(0,0,0)
            return self
        else:
            return np.array(self._imagedata.GetCenter())

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
        oldsp = self.GetSpacing()
        for i in range(3):
            if oldsp[i] != newSpacing[i]:
                rsp.SetAxisOutputSpacing(i, newSpacing[i])
        rsp.InterpolateOn()
        rsp.SetInterpolationMode(interpolation)
        rsp.OptimizationOn()
        rsp.Update()
        return self._update(rsp.GetOutput())


    def color(self, col):
        """Assign a color or a set of colors to a volume along the range of the scalar value.
        A single constant color can also be assigned.
        Any matplotlib color map name is also accepted, e.g. ``volume.color('jet')``.

        E.g.: say that your voxel scalar runs from -3 to 6,
        and you want -3 to show red and 1.5 violet and 6 green, then just set:

        ``volume.color(['red', 'violet', 'green'])``
        """
        smin, smax = self._imagedata.GetScalarRange()
        volumeProperty = self.GetProperty()
        ctf = vtk.vtkColorTransferFunction()
        self._color = col

        if utils.isSequence(col):
            for i, ci in enumerate(col):
                r, g, b = colors.getColor(ci)
                xalpha = smin + (smax - smin) * i / (len(col) - 1)
                ctf.AddRGBPoint(xalpha, r, g, b)
                #colors.printc('\tcolor at', round(xalpha, 1),
                #              '\tset to', colors.getColorName((r, g, b)), c='b', bold=0)
        elif isinstance(col, str):
            if col in colors.colors.keys() or col in colors.color_nicks.keys():
                r, g, b = colors.getColor(col)
                ctf.AddRGBPoint(smin, r,g,b) # constant color
                ctf.AddRGBPoint(smax, r,g,b)
            elif colors._mapscales:
                for x in np.linspace(smin, smax, num=64, endpoint=True):
                    r,g,b = colors.colorMap(x, name=col, vmin=smin, vmax=smax)
                    ctf.AddRGBPoint(x, r, g, b)
        elif isinstance(col, int):
            r, g, b = colors.getColor(col)
            ctf.AddRGBPoint(smin, r,g,b) # constant color
            ctf.AddRGBPoint(smax, r,g,b)
        else:
            colors.printc("volume.color(): unknown input type:", col, c=1)

        volumeProperty.SetColor(ctf)
        return self

    def alpha(self, alpha):
        """Assign a set of tranparencies to a volume along the range of the scalar value.
        A single constant value can also be assigned.

        E.g.: say alpha=(0.0, 0.3, 0.9, 1) and the scalar range goes from -10 to 150.
        Then all voxels with a value close to -10 will be completely transparent, voxels at 1/4
        of the range will get an alpha equal to 0.3 and voxels with value close to 150
        will be completely opaque.
        """
        volumeProperty = self.GetProperty()
        smin, smax = self._imagedata.GetScalarRange()
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        self._alpha = alpha

        if utils.isSequence(alpha):
            for i, al in enumerate(alpha):
                xalpha = smin + (smax - smin) * i / (len(alpha) - 1)
                # Create transfer mapping scalar value to opacity
                opacityTransferFunction.AddPoint(xalpha, al)
                #colors.printc("alpha at", round(xalpha, 1), "\tset to", al, c="b", bold=0)
        else:
            opacityTransferFunction.AddPoint(smin, alpha) # constant alpha
            opacityTransferFunction.AddPoint(smax, alpha)

        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        return self

    def alphaGradient(self, alphaGrad):
        """
        Assign a set of tranparencies to a volume's gradient
        along the range of the scalar value.
        A single constant value can also be assigned.
        The gradient function is used to decrease the opacity
        in the "flat" regions of the volume while maintaining the opacity
        at the boundaries between material types.  The gradient is measured
        as the amount by which the intensity changes over unit distance.

        |read_vti| |read_vti.py|_
        """
        self._alphaGrad = alphaGrad
        volumeProperty = self.GetProperty()
        if alphaGrad is None:
            volumeProperty.DisableGradientOpacityOn()
            return self
        else:
            volumeProperty.DisableGradientOpacityOff()

        #smin, smax = self._imagedata.GetScalarRange()
        smin, smax = 0, 255
        gotf = vtk.vtkPiecewiseFunction()
        if utils.isSequence(alphaGrad):
            for i, al in enumerate(alphaGrad):
                xalpha = smin + (smax - smin) * i / (len(alphaGrad) - 1)
                # Create transfer mapping scalar value to gradient opacity
                gotf.AddPoint(xalpha, al)
                #colors.printc("alphaGrad at", round(xalpha, 1), "\tset to", al, c="b", bold=0)
        else:
            gotf.AddPoint(smin, alphaGrad) # constant alphaGrad
            gotf.AddPoint(smax, alphaGrad)

        volumeProperty.SetGradientOpacity(gotf)
        return self

    def interpolation(self, itype):
        """
        Set interpolation type.

        0 = nearest neighbour
        1 = linear
        """
        self.GetProperty().SetInterpolationType(itype)
        return self

    def componentWeight(self, i, weight):
        """
        Set the scalar component weight in range [0,1].
        """
        self.GetProperty().SetComponentWeight(i, weight)
        return self

    def threshold(self, vmin=None, vmax=None, replaceWith=0):
        """
        Binary or continuous volume thresholding.
        Find the voxels that contain the value below/above or inbetween
        [vmin, vmax] and replaces it with the provided value (default is 0).
        """
        th = vtk.vtkImageThreshold()
        th.SetInputData(self.imagedata())

        if vmin is not None and vmax is not None:
            th.ThresholdBetween(vmin, vmax)
        elif vmin is not None:
            th.ThresholdByLower(vmin)
        elif vmax is not None:
            th.ThresholdByUpper(vmax)

        th.SetInValue(replaceWith)
        th.Update()
        return self._update(th.GetOutput())

    def crop(self,
             top=None, bottom=None,
             right=None, left=None,
             front=None, back=None, VOI=()):
        """Crop a ``Volume`` object.

        :param float top:    fraction to crop from the top plane (positive z)
        :param float bottom: fraction to crop from the bottom plane (negative z)
        :param float front:  fraction to crop from the front plane (positive y)
        :param float back:   fraction to crop from the back plane (negative y)
        :param float right:  fraction to crop from the right plane (positive x)
        :param float left:   fraction to crop from the left plane (negative x)
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

            from vtkplotter import load, datadir
            vol = load(datadir+'embryo.tif')
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

    def cutWithPlane(self, origin=(0,0,0), normal=(1,0,0)):
        """
        Cuts ``Volume`` with the plane defined by a point and a normal
        creating a tetrahedral mesh object.
        Makes sense only if the plane is not along any of the cartesian planes,
        otherwise use ``crop()`` which is way faster.

        :param origin: the cutting plane goes through this point
        :param normal: normal of the cutting plane
        """
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        clipper = vtk.vtkClipVolume()
        clipper.SetInputData(self._imagedata)
        clipper.SetClipFunction(plane)
        clipper.GenerateClipScalarsOff()
        clipper.GenerateClippedOutputOff()
        clipper.Mixed3DCellGenerationOff() # generate only tets
        clipper.SetValue(0)
        clipper.Update()

        vol = Volume(clipper.GetOutput()).color(self._color)
        return vol #self._update(clipper.GetOutput())


    def resize(self, *newdims):
        """Increase or reduce the number of voxels of a Volume with interpolation."""
        old_dims = np.array(self.imagedata().GetDimensions())
        old_spac = np.array(self.imagedata().GetSpacing())
        rsz = vtk.vtkImageResize()
        rsz.SetResizeMethodToOutputDimensions()
        rsz.SetInputData(self.imagedata())
        rsz.SetOutputDimensions(newdims)
        rsz.Update()
        self._imagedata = rsz.GetOutput()
        new_spac = old_spac * old_dims/newdims  # keep aspect ratio
        self._imagedata.SetSpacing(new_spac)
        return self._update(self._imagedata)

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
            colors.printc("~times Error in mirror(): mirror must be set to x, y, z or n.", c=1)
            raise RuntimeError()
        ff.Update()
        return self._update(ff.GetOutput())

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


    def isosurface(self, threshold=True, connectivity=False):
        """Return an ``Mesh`` isosurface extracted from the ``Volume`` object.

        :param threshold: value or list of values to draw the isosurface(s)
        :type threshold: float, list
        :param bool connectivity: if True only keeps the largest portion of the polydata

        |isosurfaces| |isosurfaces.py|_
        """
        scrange = self._imagedata.GetScalarRange()
        cf = vtk.vtkContourFilter()
        cf.SetInputData(self._imagedata)
        cf.UseScalarTreeOn()
        cf.ComputeScalarsOn()
        cf.ComputeNormalsOn()

        if utils.isSequence(threshold):
            cf.SetNumberOfContours(len(threshold))
            for i, t in enumerate(threshold):
                cf.SetValue(i, t)
            cf.Update()
        else:
            if threshold is True:
                threshold = (2 * scrange[0] + scrange[1]) / 3.0
                print('automatic threshold set to ' + utils.precision(threshold, 3), end=' ')
                print('in [' + utils.precision(scrange[0], 3) + ', ' + utils.precision(scrange[1], 3)+']')
            cf.SetValue(0, threshold)
            cf.Update()

        clp = vtk.vtkCleanPolyData()
        clp.SetInputConnection(cf.GetOutputPort())
        clp.Update()
        poly = clp.GetOutput()

        if connectivity:
            conn = vtk.vtkPolyDataConnectivityFilter()
            conn.SetExtractionModeToLargestRegion()
            conn.SetInputData(poly)
            conn.Update()
            poly = conn.GetOutput()

        a = Mesh(poly, c=None).phong()
        a._mapper.SetScalarRange(scrange[0], scrange[1])
        return a


    def legosurface(self, vmin=None, vmax=None, invert=False, cmap='afmhot_r'):
        """
        Represent a ``Volume`` as lego blocks (voxels).
        By default colors correspond to the volume's scalar.
        Returns an ``Mesh``.

        :param float vmin: the lower threshold, voxels below this value are not shown.
        :param float vmax: the upper threshold, voxels above this value are not shown.
        :param str cmap: color mapping of the scalar associated to the voxels.

        |legosurface| |legosurface.py|_
        """
        dataset = vtk.vtkImplicitDataSet()
        dataset.SetDataSet(self._imagedata)
        window = vtk.vtkImplicitWindowFunction()
        window.SetImplicitFunction(dataset)

        srng = list(self._imagedata.GetScalarRange())
        if vmin is not None:
            srng[0] = vmin
        if vmax is not None:
            srng[1] = vmax
        tol = 0.00001*(srng[1]-srng[0])
        srng[0] -= tol
        srng[1] += tol
        window.SetWindowRange(srng)

        extract = vtk.vtkExtractGeometry()
        extract.SetInputData(self._imagedata)
        extract.SetImplicitFunction(window)
        extract.SetExtractInside(invert)
        extract.ExtractBoundaryCellsOff()
        extract.Update()

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(extract.GetOutput())
        gf.Update()

        a = Mesh(gf.GetOutput()).lw(0.1).flat()
        scalars = a.getPointArray(0).astype(np.float)
        a.pointColors(scalars, vmin=srng[0], vmax=srng[1], cmap=cmap)
        a.mapPointsToCells()
        return a


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
        image1 = self._imagedata

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
            mf.SetInput2Data(volume2._imagedata)
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
            mat.SetInput2Data(volume2._imagedata)

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
            colors.printc("~times Error in volumeOperation: unknown operation", operation, c=1)
            raise RuntimeError()
        mat.Update()
        return self._update(mat.GetOutput())


    def frequencyPassFilter(self, lowcutoff=None, highcutoff=None, order=1):
        """
        Low-pass and high-pass filtering become trivial in the frequency domain.
        A portion of the pixels/voxels are simply masked or attenuated.
        This function applies a high pass Butterworth filter that attenuates the frequency domain
        image with the function

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
        fft.SetInputData(self._imagedata)
        fft.Update()
        out = fft.GetOutput()

        if highcutoff:
            butterworthLowPass = vtk.vtkImageButterworthLowPass()
            butterworthLowPass.SetInputData(out)
            butterworthLowPass.SetCutOff(highcutoff) # actually inverted..(?)
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
        ver.SetInputData(self._imagedata)
        ver.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        ver.Update()
        return self._update(ver.GetOutput())


    def dilate(self, neighbours=(2,2,2)):
        """Replace a voxel with the maximum over an ellipsoidal neighborhood of voxels.
        If `neighbours` of an axis is 1, no processing is done on that axis.

        See example script: |erode_dilate.py|_
        """
        ver = vtk.vtkImageContinuousDilate3D()
        ver.SetInputData(self._imagedata)
        ver.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        ver.Update()
        return self._update(ver.GetOutput())


    def magnitude(self):
        """Colapses components with magnitude function."""
        imgm = vtk.vtkImageMagnitude()
        imgm.SetInputData(self.imagedata())
        imgm.Update()
        return self._update(imgm.GetOutput())


    def toPoints(self):
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
        euv.SetInputData(self._imagedata)
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
        imc.SetInput1Data(self._imagedata)
        imc.SetInput2Data(vol2._imagedata)
        imc.SetDimensionality(dim)
        imc.Update()
        return Volume(imc.GetOutput())




