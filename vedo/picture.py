import os
import numpy as np
import vtk
import vedo
import vedo.colors as colors
import vedo.docs as docs
import vedo.utils as utils
import vedo.settings as settings
from vtk.util.numpy_support import numpy_to_vtk

__doc__ = (
    """
Submodule extending the ``vtkImageActor`` object functionality.
"""
    + docs._defs
)

__all__ = ["Picture"]


#################################################
class Picture(vtk.vtkImageActor, vedo.base.Base3DProp):
    """
    Derived class of ``vtkImageActor``. Used to represent 2D pictures.
    Can be instantiated with a path file name or with a numpy array.

    Use `Picture.shape` to access the number of pixels in x and y.

    |rotateImage| |rotateImage.py|_
    """
    def __init__(self, obj=None):
        vtk.vtkImageActor.__init__(self)
        vedo.base.Base3DProp.__init__(self)


        if utils.isSequence(obj) and len(obj): # passing array
            obj = np.asarray(obj)

            if len(obj.shape) == 3: # has shape (nx,ny, ncolor_alpha_chan)
                iac = vtk.vtkImageAppendComponents()
                nchan = obj.shape[2] # get number of channels in inputimage (L/LA/RGB/RGBA)
                for i in range(nchan):
                    #arr = np.flip(np.flip(array[:,:,i], 0), 0).ravel()
                    arr = np.flip(obj[:,:,i], 0).ravel()
                    varb = numpy_to_vtk(arr, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
                    varb.SetName("RGBA")
                    imgb = vtk.vtkImageData()
                    imgb.SetDimensions(obj.shape[1], obj.shape[0], 1)
                    imgb.GetPointData().SetScalars(varb)
                    iac.AddInputData(imgb)
                iac.Update()
                img = iac.GetOutput()

            elif len(obj.shape) == 2: # black and white
                arr = np.flip(obj[:,:], 0).ravel()
                varb = numpy_to_vtk(arr, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
                varb.SetName("RGBA")
                img = vtk.vtkImageData()
                img.SetDimensions(obj.shape[1], obj.shape[0], 1)
                img.GetPointData().SetScalars(varb)

        elif isinstance(obj, vtk.vtkImageData):
            img = obj

        elif isinstance(obj, str):
            if "https://" in obj:
                obj = vedo.io.download(obj, verbose=False)

            if   ".png" in obj.lower():
                picr = vtk.vtkPNGReader()
            elif ".jpg" in obj.lower() or ".jpeg" in obj.lower():
                picr = vtk.vtkJPEGReader()
            elif ".bmp" in obj.lower():
                picr = vtk.vtkBMPReader()
            elif ".tif" in obj.lower():
                picr = vtk.vtkTIFFReader()
            else:
                colors.printc("Cannot understand picture format", obj, c='r')
                return
            picr.SetFileName(obj)
            self.filename = obj
            picr.Update()
            img = picr.GetOutput()

        else:
            img = vtk.vtkImageData()

        self._data = img
        self.SetInputData(img)

        sx,sy,_ = img.GetDimensions()
        self.shape = np.array([sx,sy])

        self._mapper = self.GetMapper()


    def inputdata(self):
        """Return the underlying ``vtkImagaData`` object."""
        return self._data

    def _update(self, data):
        """Overwrite the Picture data mesh with a new data."""
        self._data = data
        self._mapper.SetInputData(data)
        self._mapper.Modified()
        return self

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
            if settings.plotter_instance and settings.plotter_instance.renderer:
                c = (0.9, 0.9, 0.9)
                if np.sum(settings.plotter_instance.renderer.GetBackground()) > 1.5:
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
            if font in settings.fonts:
                tp.SetFontFile(settings.fonts_path + font + '.ttf')
            elif os.path.exists(font):
                tp.SetFontFile(font)
            else:
                colors.printc("\sad Font", font, "not found in", settings.fonts_path, c="r")
                colors.printc("\pin Available fonts are:", settings.fonts, c="m")
                return None

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


    def alpha(self, a=None):
        """Set/get picture's transparency."""
        if a is not None:
            self.GetProperty().SetOpacity(a)
            return self
        else:
            return self.GetProperty().GetOpacity()

    def crop(self, top=None, bottom=None, right=None, left=None):
        """Crop picture.

        :param float top: fraction to crop from the top margin
        :param float bottom: fraction to crop from the bottom margin
        :param float left: fraction to crop from the left margin
        :param float right: fraction to crop from the right margin
        """
        extractVOI = vtk.vtkExtractVOI()
        extractVOI.SetInputData(self.GetInput())
        extractVOI.IncludeBoundaryOn()

        d = self.GetInput().GetDimensions()
        bx0, bx1, by0, by1 = 0, d[0]-1, 0, d[1]-1
        if left is not None:   bx0 = int((d[0]-1)*left)
        if right is not None:  bx1 = int((d[0]-1)*(1-right))
        if bottom is not None: by0 = int((d[1]-1)*bottom)
        if top is not None:    by1 = int((d[1]-1)*(1-top))
        extractVOI.SetVOI(bx0, bx1, by0, by1, 0, 0)
        extractVOI.Update()
        return self._update(extractVOI.GetOutput())

    def mirror(self, axis="x"):
        """Mirror picture along x or y axis."""
        ff = vtk.vtkImageFlip()
        ff.SetInputData(self.inputdata())
        if axis.lower() == "x":
            ff.SetFilteredAxis(0)
        elif axis.lower() == "y":
            ff.SetFilteredAxis(1)
        else:
            colors.printc("\times Error in mirror(): mirror must be set to x or y.", c='r')
            raise RuntimeError()
        ff.Update()
        return self._update(ff.GetOutput())

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
        gr = vedo.shapes.Grid(sx=dims[0], sy=dims[1], resx=dims[0]-1, resy=dims[1]-1)
        gr.pos(int(dims[0]/2), int(dims[1]/2)).pickable(True).wireframe(False).lw(0)
        self._data.GetPointData().GetScalars().SetName("RGBA")
        gr.inputdata().GetPointData().AddArray(self._data.GetPointData().GetScalars())
        return gr


