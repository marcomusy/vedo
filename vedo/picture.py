import numpy as np
import vedo
import vedo.colors as colors
import vedo.utils as utils
import vtk

__doc__ = """
Submodule to work with common format images
.. image:: https://vedo.embl.es/images/basic/rotateImage.png
"""
__all__ = ["Picture"]


#################################################
def _get_img(obj, flip=False):
    # get vtkImageData from numpy array
    obj = np.asarray(obj)

    if obj.ndim == 3: # has shape (nx,ny, ncolor_alpha_chan)
        iac = vtk.vtkImageAppendComponents()
        nchan = obj.shape[2] # get number of channels in inputimage (L/LA/RGB/RGBA)
        for i in range(nchan):
            if flip:
                arr = np.flip(np.flip(obj[:,:,i], 0), 0).ravel()
            else:
                arr = np.flip(obj[:,:,i], 0).ravel()
            arr = np.clip(arr, 0, 255)
            varb = utils.numpy2vtk(arr, dtype=np.uint8, name="RGBA")
            imgb = vtk.vtkImageData()
            imgb.SetDimensions(obj.shape[1], obj.shape[0], 1)
            imgb.GetPointData().AddArray(varb)
            imgb.GetPointData().SetActiveScalars("RGBA")
            iac.AddInputData(imgb)
        iac.Update()
        img = iac.GetOutput()

    elif obj.ndim == 2: # black and white
        if flip:
            arr = np.flip(obj[:,:], 0).ravel()
        else:
            arr = obj.ravel()
        arr = np.clip(arr, 0, 255)
        varb = utils.numpy2vtk(arr, dtype=np.uint8, name="RGBA")
        img = vtk.vtkImageData()
        img.SetDimensions(obj.shape[1], obj.shape[0], 1)
        img.GetPointData().AddArray(varb)
        img.GetPointData().SetActiveScalars("RGBA")

    return img


#################################################
class Picture(vtk.vtkImageActor, vedo.base.Base3DProp):
    """
    Derived class of ``vtkImageActor``. Used to represent 2D pictures.
    Can be instantiated with a path file name or with a numpy array.

    By default the transparency channel is disabled.
    To enable it set channels=4.

    Use `Picture.dimensions()` to access the number of pixels in x and y.

    |rotateImage| |rotateImage.py|_

    :param int,list channels: only select these specific rgba channels (useful to remove alpha)
    :param bool flip: flip xy axis convention (when input is a numpy array)
    """
    def __init__(self, obj=None, channels=3, flip=False):

        vtk.vtkImageActor.__init__(self)
        vedo.base.Base3DProp.__init__(self)

        if utils.isSequence(obj) and len(obj): # passing array
            img = _get_img(obj, flip)

        elif isinstance(obj, vtk.vtkImageData):
            img = obj

        elif isinstance(obj, str):
            if "https://" in obj:
                obj = vedo.io.download(obj, verbose=False)

            fname = obj.lower()
            if fname.endswith(".png"):
                picr = vtk.vtkPNGReader()
            elif fname.endswith(".jpg") or fname.endswith(".jpeg"):
                picr = vtk.vtkJPEGReader()
            elif fname.endswith(".bmp"):
                picr = vtk.vtkBMPReader()
            elif fname.endswith(".tif") or fname.endswith(".tiff"):
                picr = vtk.vtkTIFFReader()
                picr.SetOrientationType(vedo.settings.tiffOrientationType)
            else:
                colors.printc("Cannot understand picture format", obj, c='r')
                return
            picr.SetFileName(obj)
            self.filename = obj
            picr.Update()
            img = picr.GetOutput()

        else:
            img = vtk.vtkImageData()

        # select channels
        if isinstance(channels, int):
            channels = list(range(channels))

        nchans = len(channels)
        n = img.GetPointData().GetScalars().GetNumberOfComponents()
        if nchans and n > nchans:
            pec = vtk.vtkImageExtractComponents()
            pec.SetInputData(img)
            if nchans == 4:
                pec.SetComponents(channels[0], channels[1], channels[2], channels[3])
            elif nchans == 3:
                pec.SetComponents(channels[0], channels[1], channels[2])
            elif nchans == 2:
                pec.SetComponents(channels[0], channels[1])
            elif nchans == 1:
                pec.SetComponents(channels[0])
            pec.Update()
            img = pec.GetOutput()

        self._data = img
        self.SetInputData(img)

        sx,sy,_ = img.GetDimensions()
        self.shape = np.array([sx,sy])

        self._mapper = self.GetMapper()


    def inputdata(self):
        """Return the underlying ``vtkImagaData`` object."""
        return self._data

    def dimensions(self):
        nx, ny, _ = self._data.GetDimensions()
        return np.array([nx, ny])

    def channels(self):
        return self._data.GetPointData().GetScalars().GetNumberOfComponents()

    def _update(self, data):
        """Overwrite the Picture data mesh with a new data."""
        self._data = data
        self._mapper.SetInputData(data)
        self._mapper.Modified()
        nx, ny, _ = self._data.GetDimensions()
        self.shape = np.array([nx,ny])
        return self

    def clone(self, transform=False):
        """Return an exact copy of the input Picture.
        If transform is True, it is given the same scaling and position."""
        img = vtk.vtkImageData()
        img.DeepCopy(self._data)
        pic = Picture(img)
        if transform:
            # assign the same transformation to the copy
            pic.SetOrigin(self.GetOrigin())
            pic.SetScale(self.GetScale())
            pic.SetOrientation(self.GetOrientation())
            pic.SetPosition(self.GetPosition())
        return pic

    def extent(self, ext=None):
        """
        Get or set the physical extent that the picture spans.
        Format is ext=[minx, maxx, miny, maxy].
        """
        if ext is None:
            return self._data.GetExtent()
        else:
            self._data.SetExtent(ext[0],ext[1],ext[2],ext[3],0,0)
            self._mapper.Modified()
            return self

    def alpha(self, a=None):
        """Set/get picture's transparency in the rendering scene."""
        if a is not None:
            self.GetProperty().SetOpacity(a)
            return self
        else:
            return self.GetProperty().GetOpacity()

    def level(self, value=None):
        """Get/Set the image color level (brightness) in the rendering scene."""
        if value is None:
            return self.GetProperty().GetColorLevel()
        self.GetProperty().SetColorLevel(value)
        return self

    def window(self, value=None):
        """Get/Set the image color window (contrast) in the rendering scene."""
        if value is None:
            return self.GetProperty().GetColorWindow()
        self.GetProperty().SetColorWindow(value)
        return self


    def crop(self, top=None, bottom=None, right=None, left=None, pixels=False):
        """Crop picture.

        :param float top: fraction to crop from the top margin
        :param float bottom: fraction to crop from the bottom margin
        :param float left: fraction to crop from the left margin
        :param float right: fraction to crop from the right margin
        :param bool pixels: units are pixels
        """
        extractVOI = vtk.vtkExtractVOI()
        extractVOI.SetInputData(self._data)
        extractVOI.IncludeBoundaryOn()

        d = self.GetInput().GetDimensions()
        if pixels:
            extractVOI.SetVOI(right, d[0]-left, bottom, d[1]-top, 0, 0)
        else:
            bx0, bx1, by0, by1 = 0, d[0]-1, 0, d[1]-1
            if left is not None:   bx0 = int((d[0]-1)*left)
            if right is not None:  bx1 = int((d[0]-1)*(1-right))
            if bottom is not None: by0 = int((d[1]-1)*bottom)
            if top is not None:    by1 = int((d[1]-1)*(1-top))
            extractVOI.SetVOI(bx0, bx1, by0, by1, 0, 0)
        extractVOI.Update()
        return self._update(extractVOI.GetOutput())

    def pad(self, pixels=10, value=255):
        """
        Add the specified number of pixels at the picture borders.
        Pixels can be a list formatted as [left,right,bottom,top].

        Parameters
        ----------
        pixels : int,list , optional
            number of pixels to be added (or a list of length 4). The default is 10.
        value : int, optional
            intensity value (gray-scale color) of the padding. The default is 255.
        """
        x0,x1,y0,y1,_z0,_z1 = self._data.GetExtent()
        pf = vtk.vtkImageConstantPad()
        pf.SetInputData(self._data)
        pf.SetConstant(value)
        if utils.isSequence(pixels):
            pf.SetOutputWholeExtent(x0-pixels[0],x1+pixels[1],
                                    y0-pixels[2],y1+pixels[3], 0,0)
        else:
            pf.SetOutputWholeExtent(x0-pixels,x1+pixels, y0-pixels,y1+pixels, 0,0)
        pf.Update()
        img = pf.GetOutput()
        return self._update(img)


    def tile(self, nx=4, ny=4, shift=(0,0)):
        """
        Generate a tiling from the current picture by mirroring and repeating it.

        Parameters
        ----------
        nx :  float, optional
            number of repeats along x. The default is 4.
        ny : float, optional
            number of repeats along x. The default is 4.
        shift : list, optional
            shift in x and y in pixels. The default is 4.
        """
        x0,x1,y0,y1,z0,z1 = self._data.GetExtent()
        constantPad = vtk.vtkImageMirrorPad()
        constantPad.SetInputData(self._data)
        constantPad.SetOutputWholeExtent(int(x0+shift[0]+0.5), int(x1*nx+shift[0]+0.5),
                                         int(y0+shift[1]+0.5), int(y1*ny+shift[1]+0.5), z0,z1)
        constantPad.Update()
        return Picture(constantPad.GetOutput())


    def append(self, pictures, axis='z', preserveExtents=False):
        """
        Append the input images to the current one along the specified axis.
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

            from vedo import Picture, dataurl
            pic = Picture(dataurl+'dog.jpg').pad()
            pic.append([pic,pic,pic], axis='y')
            pic.append([pic,pic,pic,pic], axis='x')
            pic.show(axes=1)
        """
        ima = vtk.vtkImageAppend()
        ima.SetInputData(self._data)
        if not utils.isSequence(pictures):
            pictures = [pictures]
        for p in pictures:
            if isinstance(p, vtk.vtkImageData):
                ima.AddInputData(p)
            else:
                ima.AddInputData(p._data)
        ima.SetPreserveExtents(preserveExtents)
        if axis   == "x":
            axis = 0
        elif axis == "y":
            axis = 1
        ima.SetAppendAxis(axis)
        ima.Update()
        return self._update(ima.GetOutput())


    def resize(self, newsize):
        """Resize the image resolution by specifying the number of pixels in width and height.
        If left to zero, it will be automatically calculated to keep the original aspect ratio.

        :param list,float newsize: shape of picture as [npx, npy], or as a fraction.
        """
        old_dims = np.array(self._data.GetDimensions())

        if not utils.isSequence(newsize):
            newsize = (old_dims * newsize + 0.5).astype(int)

        if not newsize[1]:
            ar = old_dims[1]/old_dims[0]
            newsize = [newsize[0], int(newsize[0]*ar+0.5)]
        if not newsize[0]:
            ar = old_dims[0]/old_dims[1]
            newsize = [int(newsize[1]*ar+0.5), newsize[1]]
        newsize = [newsize[0], newsize[1], old_dims[2]]

        rsz = vtk.vtkImageResize()
        rsz.SetInputData(self._data)
        rsz.SetResizeMethodToOutputDimensions()
        rsz.SetOutputDimensions(newsize)
        rsz.Update()
        out = rsz.GetOutput()
        out.SetSpacing(1,1,1)
        return self._update(out)

    def mirror(self, axis="x"):
        """Mirror picture along x or y axis. Same as flip()."""
        ff = vtk.vtkImageFlip()
        ff.SetInputData(self.inputdata())
        if axis.lower() == "x":
            ff.SetFilteredAxis(0)
        elif axis.lower() == "y":
            ff.SetFilteredAxis(1)
        else:
            colors.printc("Error in mirror(): mirror must be set to x or y.", c='r')
            raise RuntimeError()
        ff.Update()
        return self._update(ff.GetOutput())

    def flip(self, axis="y"):
        """Mirror picture along x or y axis. Same as mirror()."""
        return self.mirror(axis=axis)

    def rotate(self, angle, center=(), scale=1, mirroring=False, bc='w', alpha=1):
        """
        Rotate by the specified angle (anticlockwise).

        Parameters
        ----------
        angle : float
            rotation angle in degrees.
        center: list
            center of rotation (x,y) in pixels.
        """
        bounds = self.bounds()
        pc = [0,0,0]
        if center:
            pc[0] = center[0]
            pc[1] = center[1]
        else:
            pc[0] = (bounds[1] + bounds[0]) / 2.0
            pc[1] = (bounds[3] + bounds[2]) / 2.0
        pc[2] = (bounds[5] + bounds[4]) / 2.0

        transform = vtk.vtkTransform()
        transform.Translate(pc)
        transform.RotateWXYZ(-angle, 0, 0, 1)
        transform.Scale(1/scale,1/scale,1)
        transform.Translate(-pc[0], -pc[1], -pc[2])

        reslice = vtk.vtkImageReslice()
        reslice.SetMirror(mirroring)
        c = np.array(colors.getColor(bc))*255
        reslice.SetBackgroundColor([c[0],c[1],c[2], alpha*255])
        reslice.SetInputData(self._data)
        reslice.SetResliceTransform(transform)
        reslice.SetOutputDimensionality(2)
        reslice.SetInterpolationModeToCubic()
        reslice.SetOutputSpacing(self._data.GetSpacing())
        reslice.SetOutputOrigin(self._data.GetOrigin())
        reslice.SetOutputExtent(self._data.GetExtent())
        reslice.Update()
        return self._update(reslice.GetOutput())

    def select(self, component):
        """Select one single component of the rgb image"""
        ec = vtk.vtkImageExtractComponents()
        ec.SetInputData(self._data)
        ec.SetComponents(component)
        ec.Update()
        return Picture(ec.GetOutput())

    def bw(self):
        """Make it black and white using luminance calibration"""
        n = self._data.GetPointData().GetNumberOfComponents()
        if n==4:
            ecr = vtk.vtkImageExtractComponents()
            ecr.SetInputData(self._data)
            ecr.SetComponents(0,1,2)
            ecr.Update()
            img = ecr.GetOutput()
        else:
            img = self._data

        ecr = vtk.vtkImageLuminance()
        ecr.SetInputData(img)
        ecr.Update()
        return self._update(ecr.GetOutput())

    def smooth(self, sigma=3, radius=None):
        """
        Smooth a Picture with Gaussian kernel.

        Parameters
        ----------
        sigma : int, optional
            number of sigmas in pixel units. The default is 3.
        radius : TYPE, optional
            how far out the gaussian kernel will go before being clamped to zero. The default is None.
        """
        gsf = vtk.vtkImageGaussianSmooth()
        gsf.SetDimensionality(2)
        gsf.SetInputData(self._data)
        if radius is not None:
            if utils.isSequence(radius):
                gsf.SetRadiusFactors(radius[0],radius[1])
            else:
                gsf.SetRadiusFactor(radius)

        if utils.isSequence(sigma):
            gsf.SetStandardDeviations(sigma[0], sigma[1])
        else:
            gsf.SetStandardDeviation(sigma)
        gsf.Update()
        return self._update(gsf.GetOutput())

    def median(self):
        """Median filter that preserves thin lines and corners.
        It operates on a 5x5 pixel neighborhood. It computes two values initially:
        the median of the + neighbors and the median of the x neighbors.
        It then computes the median of these two values plus the center pixel.
        This result of this second median is the output pixel value.
        """
        medf = vtk.vtkImageHybridMedian2D()
        medf.SetInputData(self._data)
        medf.Update()
        return self._update(medf.GetOutput())

    def enhance(self):
        """
        Enhance a b&w picture using the laplacian, enhancing high-freq edges.

        Example:

            .. code-block:: python

                import vedo
                p = vedo.Picture(vedo.dataurl+'images/dog.jpg').bw()
                vedo.show(p, p.clone().enhance(), N=2, mode='image')
        """
        img = self._data
        scalarRange = img.GetPointData().GetScalars().GetRange()

        cast = vtk.vtkImageCast()
        cast.SetInputData(img)
        cast.SetOutputScalarTypeToDouble()
        cast.Update()

        laplacian = vtk.vtkImageLaplacian()
        laplacian.SetInputData(cast.GetOutput())
        laplacian.SetDimensionality(2)
        laplacian.Update()

        subtr = vtk.vtkImageMathematics()
        subtr.SetInputData(0, cast.GetOutput())
        subtr.SetInputData(1, laplacian.GetOutput())
        subtr.SetOperationToSubtract()
        subtr.Update()

        colorWindow = scalarRange[1] - scalarRange[0]
        colorLevel = colorWindow / 2
        originalColor = vtk.vtkImageMapToWindowLevelColors()
        originalColor.SetWindow(colorWindow)
        originalColor.SetLevel(colorLevel)
        originalColor.SetInputData(subtr.GetOutput())
        originalColor.Update()
        return self._update(originalColor.GetOutput())

    def fft(self, mode='magnitude', logscale=12, center=True):
        """Fast Fourier transform of a picture.

        :param float logscale: if non-zero, take the logarithm of the
            intensity and scale it by this factor.

        :param str mode: either [magnitude, real, imaginary, complex], compute the
            point array data accordingly.
        :param bool center: shift constant zero-frequency to the center of the image for display.
            (FFT converts spatial images into frequency space, but puts the zero frequency at the origin)
        """
        ffti = vtk.vtkImageFFT()
        ffti.SetInputData(self._data)
        ffti.Update()

        if 'mag' in mode:
            mag = vtk.vtkImageMagnitude()
            mag.SetInputData(ffti.GetOutput())
            mag.Update()
            out = mag.GetOutput()
        elif 'real' in mode:
            extractRealFilter = vtk.vtkImageExtractComponents()
            extractRealFilter.SetInputData(ffti.GetOutput())
            extractRealFilter.SetComponents(0)
            extractRealFilter.Update()
            out = extractRealFilter.GetOutput()
        elif 'imaginary' in mode:
            extractImgFilter = vtk.vtkImageExtractComponents()
            extractImgFilter.SetInputData(ffti.GetOutput())
            extractImgFilter.SetComponents(1)
            extractImgFilter.Update()
            out = extractImgFilter.GetOutput()
        elif 'complex' in mode:
            out = ffti.GetOutput()
        else:
            colors.printc("Error in fft(): unknown mode", mode)
            raise RuntimeError()

        if center:
            center = vtk.vtkImageFourierCenter()
            center.SetInputData(out)
            center.Update()
            out = center.GetOutput()

        if 'complex' not in mode:
            if logscale:
                ils = vtk.vtkImageLogarithmicScale()
                ils.SetInputData(out)
                ils.SetConstant(logscale)
                ils.Update()
                out = ils.GetOutput()

        return Picture(out)

    def rfft(self, mode='magnitude'):
        """Reverse Fast Fourier transform of a picture."""

        ffti = vtk.vtkImageRFFT()
        ffti.SetInputData(self._data)
        ffti.Update()

        if 'mag' in mode:
            mag = vtk.vtkImageMagnitude()
            mag.SetInputData(ffti.GetOutput())
            mag.Update()
            out = mag.GetOutput()
        elif 'real' in mode:
            extractRealFilter = vtk.vtkImageExtractComponents()
            extractRealFilter.SetInputData(ffti.GetOutput())
            extractRealFilter.SetComponents(0)
            extractRealFilter.Update()
            out = extractRealFilter.GetOutput()
        elif 'imaginary' in mode:
            extractImgFilter = vtk.vtkImageExtractComponents()
            extractImgFilter.SetInputData(ffti.GetOutput())
            extractImgFilter.SetComponents(1)
            extractImgFilter.Update()
            out = extractImgFilter.GetOutput()
        elif 'complex' in mode:
            out = ffti.GetOutput()
        else:
            colors.printc("Error in rfft(): unknown mode", mode)
            raise RuntimeError()

        return Picture(out)

    def filterpass(self, lowcutoff=None, highcutoff=None, order=3):
        """
        Low-pass and high-pass filtering become trivial in the frequency domain.
        A portion of the pixels/voxels are simply masked or attenuated.
        This function applies a high pass Butterworth filter that attenuates the
        frequency domain image with the function

        |G_Of_Omega|

        The gradual attenuation of the filter is important.
        A simple high-pass filter would simply mask a set of pixels in the frequency domain,
        but the abrupt transition would cause a ringing effect in the spatial domain.

        :param list lowcutoff:  the cutoff frequencies
        :param list highcutoff: the cutoff frequencies
        :param int order: order determines sharpness of the cutoff curve
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

        caster = vtk.vtkImageCast()
        caster. SetOutputScalarTypeToUnsignedChar()
        caster.SetInputData(butterworthReal.GetOutput())
        caster.Update()
        return self._update(caster.GetOutput())


    def blend(self, pic, alpha1=0.5, alpha2=0.5):
        """Take L, LA, RGB, or RGBA images as input and blends
        them according to the alpha values and/or the opacity setting for each input.
        """
        blf = vtk.vtkImageBlend()
        blf.AddInputData(self._data)
        blf.AddInputData(pic._data)
        blf.SetOpacity(0, alpha1)
        blf.SetOpacity(1, alpha2)
        blf.SetBlendModeToNormal()
        blf.Update()
        return self._update(blf.GetOutput())


    def warp(self, sourcePts=(), targetPts=(), transform=None, sigma=1,
             mirroring=False, bc='w', alpha=1):
        """
        Warp an image using thin-plate splines.

        Parameters
        ----------
        sourcePts : list, optional
            source points.
        targetPts : list, optional
            target points.
        transform : TYPE, optional
            a vtkTransform object can be supplied. The default is None.
        sigma : float, optional
            stiffness of the interpolation. The default is 1.
        mirroring : TYPE, optional
            fill the margins with a reflection of the original image. The default is False.
        bc : TYPE, optional
            fill the margins with a solid color. The default is 'w'.
        alpha : TYPE, optional
            opacity of the filled margins. The default is 1.
        """
        if transform is None:
            # source and target must be filled
            transform = vtk.vtkThinPlateSplineTransform()
            transform.SetBasisToR2LogR()
            if isinstance(sourcePts, vedo.Points):
                sourcePts = sourcePts.points()
            if isinstance(targetPts, vedo.Points):
                targetPts = targetPts.points()

            ns = len(sourcePts)
            nt = len(targetPts)
            if ns != nt:
                colors.printc("Error in picture.warp(): #source != #target points", ns, nt, c='r')
                raise RuntimeError()

            ptsou = vtk.vtkPoints()
            ptsou.SetNumberOfPoints(ns)

            pttar = vtk.vtkPoints()
            pttar.SetNumberOfPoints(nt)

            for i in range(ns):
                p = sourcePts[i]
                ptsou.SetPoint(i, [p[0],p[1],0])
                p = targetPts[i]
                pttar.SetPoint(i, [p[0],p[1],0])

            transform.SetSigma(sigma)
            transform.SetSourceLandmarks(pttar)
            transform.SetTargetLandmarks(ptsou)
        else:
            # ignore source and target
            pass

        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(self._data)
        reslice.SetOutputDimensionality(2)
        reslice.SetResliceTransform(transform)
        reslice.SetInterpolationModeToCubic()
        reslice.SetMirror(mirroring)
        c = np.array(colors.getColor(bc))*255
        reslice.SetBackgroundColor([c[0],c[1],c[2], alpha*255])
        reslice.Update()
        self.transform = transform
        return self._update(reslice.GetOutput())


    def invert(self):
        """
        Return an inverted picture (inverted in each color channel).
        """
        rgb = self.tonumpy()
        data = 255 - np.array(rgb)
        return self._update(_get_img(data))


    def binarize(self, thresh=None, invert=False):
        """Return a new Picture where pixel above threshold are set to 255
        and pixels below are set to 0.

        Parameters
        ----------
        invert : bool, optional
            Invert threshold. Default is False.

        Example
        -------
        .. code-block:: python

            from vedo import Picture, show
            pic1 = Picture("https://aws.glamour.es/prod/designs/v1/assets/620x459/547577.jpg")
            pic2 = pic1.clone().invert()
            pic3 = pic1.clone().binarize()
            show(pic1, pic2, pic3, N=3, bg="blue9")
        """
        rgb = self.tonumpy()
        if rgb.ndim == 3:
            intensity = np.sum(rgb, axis=2)/3
        else:
            intensity = rgb

        if thresh is None:
            vmin, vmax = np.min(intensity), np.max(intensity)
            thresh = (vmax+vmin)/2

        data = np.zeros_like(intensity).astype(np.uint8)
        mask = np.where(intensity>thresh)
        if invert:
            data += 255
            data[mask] = 0
        else:
            data[mask] = 255

        return self._update(_get_img(data, flip=True))


    def threshold(self, value=None, flip=False):
        """
        Create a polygonal Mesh from a Picture by filling regions with pixels
        luminosity above a specified value.

        Parameters
        ----------
        value : float, optional
            The default is None, e.i. 1/3 of the scalar range.

        flip: bool, optional
            Flip polygon orientations

        Returns
        -------
        Mesh
            A polygonal mesh.
        """
        mgf = vtk.vtkImageMagnitude()
        mgf.SetInputData(self._data)
        mgf.Update()
        msq = vtk.vtkMarchingSquares()
        msq.SetInputData(mgf.GetOutput())
        if value is None:
            r0,r1 = self._data.GetScalarRange()
            value = r0 + (r1-r0)/3
        msq.SetValue(0, value)
        msq.Update()
        if flip:
            rs = vtk.vtkReverseSense()
            rs.SetInputData(msq.GetOutput())
            rs.ReverseCellsOn()
            rs.ReverseNormalsOff()
            rs.Update()
            output = rs.GetOutput()
        else:
            output = msq.GetOutput()
        ctr = vtk.vtkContourTriangulator()
        ctr.SetInputData(output)
        ctr.Update()
        return vedo.Mesh(ctr.GetOutput(), c='k').bc('t').lighting('off')


    def tomesh(self):
        """
        Convert an image to polygonal data (quads),
        with each polygon vertex assigned a RGBA value.
        """
        dims = self._data.GetDimensions()
        gr = vedo.shapes.Grid(s=dims[:2], res=(dims[0]-1, dims[1]-1))
        gr.pos(int(dims[0]/2), int(dims[1]/2)).pickable(True).wireframe(False).lw(0)
        self._data.GetPointData().GetScalars().SetName("RGBA")
        gr.inputdata().GetPointData().AddArray(self._data.GetPointData().GetScalars())
        gr.inputdata().GetPointData().SetActiveScalars("RGBA")
        gr._mapper.SetArrayName("RGBA")
        gr._mapper.SetScalarModeToUsePointData()
        # gr._mapper.SetColorModeToDirectScalars()
        gr._mapper.ScalarVisibilityOn()
        gr.name = self.name
        gr.filename = self.filename
        return gr

    def tonumpy(self):
        """Get read-write access to pixels of a Picture object as a numpy array.
        Note that the shape is (nrofchannels, nx, ny).

        When you set values in the output image, you don't want numpy to reallocate the array
        but instead set values in the existing array, so use the [:] operator.
        Example: arr[:] = arr - 15

        If the array is modified call:
        ``picture.modified()``
        when all your modifications are completed.
        """
        nx, ny, _ = self._data.GetDimensions()
        nchan = self._data.GetPointData().GetScalars().GetNumberOfComponents()
        narray = utils.vtk2numpy(self._data.GetPointData().GetScalars()).reshape(ny,nx,nchan)
        narray = np.flip(narray, axis=0).astype(np.uint8)
        return narray

    def rectangle(self, xspan, yspan, c='green5', alpha=1):
        """Draw a rectangle box on top of current image. Units are pixels.

        .. code-block:: python

                import vedo
                pic = vedo.Picture("dog.jpg")
                pic.rectangle([100,300], [100,200], c='green4', alpha=0.7)
                pic.line([100,100],[400,500], lw=2, alpha=1)
                pic.triangle([250,300], [100,300], [200,400])
                vedo.show(pic, axes=1)
        """
        x1, x2 = xspan
        y1, y2 = yspan

        r,g,b = vedo.colors.getColor(c)
        c = np.array([r,g,b]) * 255
        c = c.astype(np.uint8)
        if alpha>1:
            alpha=1
        if alpha<=0:
            return self
        alpha2 = alpha
        alpha1 = 1-alpha

        nx, ny = self.dimensions()
        if x2>nx : x2=nx-1
        if y2>ny : y2=ny-1

        nchan = self.channels()
        narrayA = self.tonumpy()

        canvas_source = vtk.vtkImageCanvasSource2D()
        canvas_source.SetExtent(0, nx-1, 0, ny-1, 0, 0)
        canvas_source.SetScalarTypeToUnsignedChar()
        canvas_source.SetNumberOfScalarComponents(nchan)
        canvas_source.SetDrawColor(255,255,255)
        canvas_source.FillBox(x1, x2, y1, y2)
        canvas_source.Update()
        image_data = canvas_source.GetOutput()

        vscals = image_data.GetPointData().GetScalars()
        narrayB = vedo.utils.vtk2numpy(vscals).reshape(ny,nx,nchan)
        narrayB = np.flip(narrayB, axis=0)
        narrayC = np.where(narrayB < 255, narrayA, alpha1*narrayA+alpha2*c)
        return self._update(_get_img(narrayC))

    def line(self, p1, p2, lw=2, c='k2', alpha=1):
        """Draw a line on top of current image. Units are pixels."""
        x1, x2 = p1
        y1, y2 = p2

        r,g,b = vedo.colors.getColor(c)
        c = np.array([r,g,b]) * 255
        c = c.astype(np.uint8)
        if alpha>1:
            alpha=1
        if alpha<=0:
            return self
        alpha2 = alpha
        alpha1 = 1-alpha

        nx, ny = self.dimensions()
        if x2>nx : x2=nx-1
        if y2>ny : y2=ny-1

        nchan = self.channels()
        narrayA = self.tonumpy()

        canvas_source = vtk.vtkImageCanvasSource2D()
        canvas_source.SetExtent(0, nx-1, 0, ny-1, 0, 0)
        canvas_source.SetScalarTypeToUnsignedChar()
        canvas_source.SetNumberOfScalarComponents(nchan)
        canvas_source.SetDrawColor(255,255,255)
        canvas_source.FillTube(x1, x2, y1, y2, lw)
        canvas_source.Update()
        image_data = canvas_source.GetOutput()

        vscals = image_data.GetPointData().GetScalars()
        narrayB = vedo.utils.vtk2numpy(vscals).reshape(ny,nx,nchan)
        narrayB = np.flip(narrayB, axis=0)
        narrayC = np.where(narrayB < 255, narrayA, alpha1*narrayA+alpha2*c)
        return self._update(_get_img(narrayC))

    def triangle(self, p1, p2, p3, c='red3', alpha=1):
        """Draw a triangle on top of current image. Units are pixels."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        r,g,b = vedo.colors.getColor(c)
        c = np.array([r,g,b]) * 255
        c = c.astype(np.uint8)

        if alpha>1:
            alpha=1
        if alpha<=0:
            return self
        alpha2 = alpha
        alpha1 = 1-alpha

        nx, ny = self.dimensions()
        if x1>nx : x1=nx
        if x2>nx : x2=nx
        if x3>nx : x3=nx
        if y1>ny : y1=ny
        if y2>ny : y2=ny
        if y3>ny : y3=ny

        nchan = self.channels()
        narrayA = self.tonumpy()

        canvas_source = vtk.vtkImageCanvasSource2D()
        canvas_source.SetExtent(0, nx-1, 0, ny-1, 0, 0)
        canvas_source.SetScalarTypeToUnsignedChar()
        canvas_source.SetNumberOfScalarComponents(nchan)
        canvas_source.SetDrawColor(255,255,255)
        canvas_source.FillTriangle(x1, y1, x2, y2, x3, y3)
        canvas_source.Update()
        image_data = canvas_source.GetOutput()

        vscals = image_data.GetPointData().GetScalars()
        narrayB = vedo.utils.vtk2numpy(vscals).reshape(ny,nx,nchan)
        narrayB = np.flip(narrayB, axis=0)
        narrayC = np.where(narrayB < 255, narrayA, alpha1*narrayA+alpha2*c)
        return self._update(_get_img(narrayC))

#    def circle(self, center, radius, c='k3', alpha=1): # not working
#        """Draw a box."""
#        x1, y1 = center
#
#        r,g,b = vedo.colors.getColor(c)
#        c = np.array([r,g,b]) * 255
#        c = c.astype(np.uint8)
#
#        if alpha>1:
#            alpha=1
#        if alpha<=0:
#            return self
#        alpha2 = alpha
#        alpha1 = 1-alpha
#
#        nx, ny = self.dimensions()
#        nchan = self.channels()
#        narrayA = self.tonumpy()
#
#        canvas_source = vtk.vtkImageCanvasSource2D()
#        canvas_source.SetExtent(0, nx-1, 0, ny-1, 0, 0)
#        canvas_source.SetScalarTypeToUnsignedChar()
#        canvas_source.SetNumberOfScalarComponents(nchan)
#        canvas_source.SetDrawColor(255,255,255)
#        canvas_source.DrawCircle(x1, y1, radius)
#        canvas_source.Update()
#        image_data = canvas_source.GetOutput()
#
#        vscals = image_data.GetPointData().GetScalars()
#        narrayB = vedo.utils.vtk2numpy(vscals).reshape(ny,nx,nchan)
#        narrayB = np.flip(narrayB, axis=0)
#        narrayC = np.where(narrayB < 255, narrayA, alpha1*narrayA+alpha2*c)
#        return self._update(_get_img(narrayC))

    def text(self, txt,
                   pos=(0,0,0),
                   s=1,
                   c=None,
                   alpha=1,
                   bg=None,
                   font="Theemim",
                   dpi=500,
                   justify="bottom-left",
        ):
        """Build an image from a string."""

        if c is None: # automatic black or white
            if vedo.plotter_instance and vedo.plotter_instance.renderer:
                c = (0.9, 0.9, 0.9)
                if np.sum(vedo.plotter_instance.renderer.GetBackground()) > 1.5:
                    c = (0.1, 0.1, 0.1)
            else:
                c = (0.3, 0.3, 0.3)

        r = vtk.vtkTextRenderer()
        img = vtk.vtkImageData()

        tp = vtk.vtkTextProperty()
        tp.BoldOff()
        tp.SetColor(colors.getColor(c))
        tp.SetJustificationToLeft()
        if "top" in justify:
            tp.SetVerticalJustificationToTop()
        if "bottom" in justify:
            tp.SetVerticalJustificationToBottom()
        if "cent" in justify:
            tp.SetVerticalJustificationToCentered()
            tp.SetJustificationToCentered()
        if "left" in justify:
            tp.SetJustificationToLeft()
        if "right" in justify:
            tp.SetJustificationToRight()

        if font.lower() == "courier": tp.SetFontFamilyToCourier()
        elif font.lower() == "times": tp.SetFontFamilyToTimes()
        elif font.lower() == "arial": tp.SetFontFamilyToArial()
        else:
            tp.SetFontFamily(vtk.VTK_FONT_FILE)
            tp.SetFontFile(utils.getFontPath(font))

        if bg:
            bgcol = colors.getColor(bg)
            tp.SetBackgroundColor(bgcol)
            tp.SetBackgroundOpacity(alpha * 0.5)
            tp.SetFrameColor(bgcol)
            tp.FrameOn()

        #GetConstrainedFontSize (const vtkUnicodeString &str,
        # vtkTextProperty *tprop, int targetWidth, int targetHeight, int dpi)
        fs = r.GetConstrainedFontSize(txt, tp, 900, 1000, dpi)
        tp.SetFontSize(fs)

        r.RenderString(tp, txt, img, [1,1], dpi)
        # RenderString (vtkTextProperty *tprop, const vtkStdString &str,
        #   vtkImageData *data, int textDims[2], int dpi, int backend=Default)

        self.SetInputData(img)
        self.GetMapper().Modified()

        self.SetPosition(pos)
        x0, x1 = self.xbounds()
        if x1 != x0:
            sc = s/(x1-x0)
            self.SetScale(sc,sc,sc)
        return self

    def modified(self):
        """Use in conjunction with ``tonumpy()`` to update any modifications to the picture array"""
        self._data.GetPointData().GetScalars().Modified()
        return self

    def write(self, filename):
        """Write picture to file as png or jpg."""
        vedo.io.write(self._data, filename)
        return self
