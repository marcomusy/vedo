from __future__ import division, print_function

import numpy as np
import vtk
import vedo.colors as colors
import vedo.docs as docs
import vedo.utils as utils
import vedo.settings as settings
from vtk.util.numpy_support import numpy_to_vtk
from vedo.base import Base3DProp

__doc__ = (
    """
Submodule extending the ``vtkImageActor`` object functionality.
"""
    + docs._defs
)

__all__ = ["Picture"]


#################################################
class Picture(vtk.vtkImageActor, Base3DProp):
    """
    Derived class of ``vtkImageActor``. Used to represent 2D pictures.
    Can be instantiated with a path file name or with a numpy array.

    |rotateImage| |rotateImage.py|_
    """
    def __init__(self, obj=None):
        vtk.vtkImageActor.__init__(self)
        Base3DProp.__init__(self)

        if utils.isSequence(obj) and len(obj):
            iac = vtk.vtkImageAppendComponents()
            for i in range(3):
                #arr = np.flip(np.flip(array[:,:,i], 0), 0).ravel()
                arr = np.flip(obj[:,:,i], 0).ravel()
                varb = numpy_to_vtk(arr, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
                imgb = vtk.vtkImageData()
                imgb.SetDimensions(obj.shape[1], obj.shape[0], 1)
                imgb.GetPointData().SetScalars(varb)
                iac.AddInputData(0, imgb)
            iac.Update()
            img = iac.GetOutput()
            self.SetInputData(img)

        elif isinstance(obj, vtk.vtkImageData):
            self.SetInputData(obj)
            img = obj

        elif isinstance(obj, str):
            if "https://" in obj:
                import vedo.io as io
                obj = io.download(obj, verbose=False)

            if   ".png" in obj:
                picr = vtk.vtkPNGReader()
            elif ".jpg" in obj or ".jpeg" in obj:
                picr = vtk.vtkJPEGReader()
            elif ".bmp" in obj:
                picr = vtk.vtkBMPReader()
            elif ".tif" in obj:
                picr = vtk.vtkTIFFReader()
            else:
                colors.printc("Cannot understand picture format", obj, c='r')
                return
            picr.SetFileName(obj)
            picr.Update()
            img = picr.GetOutput()
            self.SetInputData(img)

        else:
            img = vtk.vtkImageData()
            self.SetInputData(img)

        self._data = img
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
                   font="Gula",
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
            import os
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








