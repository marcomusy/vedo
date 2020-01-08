from __future__ import division, print_function

import numpy as np
import vtk
import vtkplotter.colors as colors
import vtkplotter.docs as docs
import vtkplotter.utils as utils
from vtk.util.numpy_support import numpy_to_vtk
from vtkplotter.base import ActorBase

__doc__ = (
    """
Submodule extending the ``vtkImageActor`` object functionality.
"""
    + docs._defs
)

__all__ = ["Picture"]


#################################################
class Picture(vtk.vtkImageActor, ActorBase):
    """
    Derived class of ``vtkImageActor``. Used to represent 2D pictures.
    Can be instantiated with a path file name or with a numpy array.

    |rotateImage| |rotateImage.py|_
    """
    def __init__(self, obj=None):
        vtk.vtkImageActor.__init__(self)
        ActorBase.__init__(self)

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
            self.SetInputData(iac.GetOutput())
            #self.mirror()

        elif isinstance(obj, str):
            if ".png" in obj:
                picr = vtk.vtkPNGReader()
            elif ".jpg" in obj or ".jpeg" in obj:
                picr = vtk.vtkJPEGReader()
            elif ".bmp" in obj:
                picr = vtk.vtkBMPReader()
            picr.SetFileName(obj)
            picr.Update()
            self.SetInputData(picr.GetOutput())


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
        img = extractVOI.GetOutput()
        self.GetMapper().SetInputData(img)
        self.GetMapper().Modified()
        return self

    def mirror(self, axis="x"):
        """Mirror picture along x or y axis."""
        ff = vtk.vtkImageFlip()
        ff.SetInputData(self.inputdata())
        if axis.lower() == "x":
            ff.SetFilteredAxis(0)
        elif axis.lower() == "y":
            ff.SetFilteredAxis(1)
        else:
            colors.printc("~times Error in mirror(): mirror must be set to x or y.", c=1)
            raise RuntimeError()
        ff.Update()
        self.GetMapper().SetInputData(ff.GetOutput())
        self.GetMapper().Modified()
        return self
