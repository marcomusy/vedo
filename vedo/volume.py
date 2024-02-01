import glob
import os
import time
from weakref import ref as weak_ref_to

import numpy as np

import vedo.vtkclasses as vtki

import vedo
from vedo import transformations
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
class Volume(VolumeAlgorithms, VolumeVisual):
    """
    Class to describe dataset that are defined on "voxels",
    the 3D equivalent of 2D pixels.
    """
    def __init__(
        self,
        inputobj=None,
        dims=None,
        origin=None,
        spacing=None,
    ):
        """
        This class can be initialized with a numpy object,
        a `vtkImageData` or a list of 2D bmp files.

        Arguments:
            origin : (list)
                set volume origin coordinates
            spacing : (list)
                voxel dimensions in x, y and z.
            dims : (list)
                specify the dimensions of the volume.

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
        super().__init__()

        self.name = "Volume"
        self.filename = ""
        self.file_size = ""

        self.info = {}
        self.time =  time.time()

        self.actor = vtki.vtkVolume()
        self.actor.retrieve_object = weak_ref_to(self)
        self.properties = self.actor.GetProperty()

        self.transform = None
        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None

        ###################
        if isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = vedo.file_io.download(inputobj, verbose=False)  # fpath
            elif os.path.isfile(inputobj):
                self.filename = inputobj
            else:
                inputobj = sorted(glob.glob(inputobj))

        ###################
        inputtype = str(type(inputobj))

        # print('Volume inputtype', inputtype, c='b')

        if inputobj is None:
            img = vtki.vtkImageData()

        elif utils.is_sequence(inputobj):

            if isinstance(inputobj[0], str) and ".bmp" in inputobj[0].lower():
                # scan sequence of BMP files
                ima = vtki.new("ImageAppend")
                ima.SetAppendAxis(2)
                pb = utils.ProgressBar(0, len(inputobj))
                for i in pb.range():
                    f = inputobj[i]
                    if "_rec_spr" in f: # OPT specific
                        continue
                    picr = vtki.new("BMPReader")
                    picr.SetFileName(f)
                    picr.Update()
                    mgf = vtki.new("ImageMagnitude")
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

                img = vtki.vtkImageData()
                if dims is not None:
                    img.SetDimensions(dims[2], dims[1], dims[0])
                else:
                    if len(inputobj.shape) == 1:
                        vedo.logger.error("must set dimensions (dims keyword) in Volume")
                        raise RuntimeError()
                    img.SetDimensions(inputobj.shape)
                img.GetPointData().AddArray(varr)
                img.GetPointData().SetActiveScalars(varr.GetName())

        elif isinstance(inputobj, vtki.vtkImageData):
            img = inputobj

        elif isinstance(inputobj, str):
            if "https://" in inputobj:
                inputobj = vedo.file_io.download(inputobj, verbose=False)
            img = vedo.file_io.loadImageData(inputobj)
            self.filename = inputobj

        else:
            vedo.logger.error(f"cannot understand input type {inputtype}")
            return

        if dims is not None:
            img.SetDimensions(dims)

        if origin is not None:
            img.SetOrigin(origin)

        if spacing is not None:
            img.SetSpacing(spacing)

        self.dataset = img
        self.transform = None

        #####################################
        mapper = vtki.new("SmartVolumeMapper")
        mapper.SetInputData(img)
        self.actor.SetMapper(mapper)

        if img.GetPointData().GetScalars():
            if img.GetPointData().GetScalars().GetNumberOfComponents() == 1:
                self.properties.SetShade(True)
                self.properties.SetInterpolationType(1)
                self.cmap("RdBu_r")
                self.alpha([0.0, 0.0, 0.2, 0.4, 0.8, 1.0])
                self.alpha_gradient(None)
                self.properties.SetScalarOpacityUnitDistance(1.0)

        self.pipeline = utils.OperationNode(
            "Volume", comment=f"dims={tuple(self.dimensions())}", c="#4cc9f0"
        )
        #######################################################################

    @property
    def mapper(self):
        """Return the underlying `vtkMapper` object."""
        return self.actor.GetMapper()
    
    @mapper.setter
    def mapper(self, mapper):
        """
        Set the underlying `vtkMapper` object.
        
        Arguments:
            mapper : (str, vtkMapper)
                either 'gpu', 'opengl_gpu', 'fixed' or 'smart'
        """
        if isinstance(mapper, 
            (vtki.get_class("Mapper"), vtki.get_class("ImageResliceMapper"))
        ):
            pass
        elif mapper is None:
            mapper = vtki.new("SmartVolumeMapper")
        elif "gpu" in mapper:
            mapper = vtki.new("GPUVolumeRayCastMapper")
        elif "opengl_gpu" in mapper:
            mapper = vtki.new("OpenGLGPUVolumeRayCastMapper")
        elif "smart" in mapper:
            mapper = vtki.new("SmartVolumeMapper")
        elif "fixed" in mapper:
            mapper = vtki.new("FixedPointVolumeRayCastMapper")
        else:
            print("Error unknown mapper type", [mapper])
            raise RuntimeError()
        self.actor.SetMapper(mapper)

    def c(self, *args, **kwargs):
        """Deprecated. Use `Volume.cmap()` instead."""
        vedo.logger.warning("Volume.c() is deprecated, use Volume.cmap() instead")
        return self.cmap(*args, **kwargs)

    def _update(self, data, reset_locators=False):
        # reset_locators here is dummy
        self.dataset = data
        self.mapper.SetInputData(data)
        self.dataset.GetPointData().Modified()
        self.mapper.Modified()
        self.mapper.Update()
        return self

    def __str__(self):
        """Print a summary for the `Volume` object."""
        module = self.__class__.__module__
        name = self.__class__.__name__
        out = vedo.printc(
            f"{module}.{name} at ({hex(self.memory_address())})".ljust(75),
            c="c", bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m\x1b[36;1m"

        out+= "name".ljust(14) + ": " + str(self.name) + "\n"
        if self.filename:
            out+= "filename".ljust(14) + ": " + str(self.filename) + "\n"

        out+= "dimensions".ljust(14) + ": " + str(self.shape) + "\n"

        out+= "origin".ljust(14) + ": "
        out+= utils.precision(self.origin(), 6) + "\n"

        out+= "center".ljust(14) + ": "
        out+= utils.precision(self.center(), 6) + "\n"

        out+= "spacing".ljust(14)    + ": "
        out+= utils.precision(self.spacing(), 6) + "\n"

        bnds = self.bounds()
        bx1, bx2 = utils.precision(bnds[0], 3), utils.precision(bnds[1], 3)
        by1, by2 = utils.precision(bnds[2], 3), utils.precision(bnds[3], 3)
        bz1, bz2 = utils.precision(bnds[4], 3), utils.precision(bnds[5], 3)
        out+= "bounds".ljust(14) + ":"
        out+= " x=(" + bx1 + ", " + bx2 + "),"
        out+= " y=(" + by1 + ", " + by2 + "),"
        out+= " z=(" + bz1 + ", " + bz2 + ")\n"

        out+= "memory size".ljust(14) + ": "
        out+= str(int(self.dataset.GetActualMemorySize()/1024+0.5))+" MB\n"

        st = self.dataset.GetScalarTypeAsString()
        out+= "scalar size".ljust(14) + ": "
        out+= str(self.dataset.GetScalarSize()) + f" bytes ({st})\n"
        out+= "scalar range".ljust(14) + ": "
        out+= str(self.dataset.GetScalarRange()) + "\n"

        #utils.print_histogram(self, logscale=True, bins=8, height=15, c="b", bold=True)
        return out.rstrip() + "\x1b[0m"

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

        img = self.dataset

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

    def copy(self, deep=True):
        """Return a copy of the Volume. Alias of `clone()`."""
        return self.clone(deep=deep)

    def clone(self, deep=True):
        """Return a clone copy of the Volume. Alias of `copy()`."""
        if deep:
            newimg = vtki.vtkImageData()
            newimg.CopyStructure(self.dataset)
            newimg.CopyAttributes(self.dataset)
            newvol = Volume(newimg)
        else:
            newvol = Volume(self.dataset)

        prop = vtki.vtkVolumeProperty()
        prop.DeepCopy(self.properties)
        newvol.actor.SetProperty(prop)
        newvol.properties = prop

        newvol.pipeline = utils.OperationNode("clone", parents=[self], c="#bbd0ff", shape="diamond")
        return newvol
    
    def component_weight(self, i, weight):
        """Set the scalar component weight in range [0,1]."""
        self.properties.SetComponentWeight(i, weight)
        return self

    def xslice(self, i):
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

    def yslice(self, j):
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

    def zslice(self, k):
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

    def slice_plane(self, origin, normal, autocrop=False, border=0.5, mode="linear"):
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
        T.RotateWXYZ(np.rad2deg(angle), crossvec)
        T.Translate(pos)

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
    
    def slab(self, slice_range=(), axis='z', operation="mean"):
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
            if slice_range[0]  > dims[0]-1:
                slice_range[0] = dims[0]-1
            if slice_range[1]  > dims[0]-1:
                slice_range[1] = dims[0]-1
        elif axis == 'y':
            islab.SetOrientationToY()
            if slice_range[0]  > dims[1]-1:
                slice_range[0] = dims[1]-1
            if slice_range[1]  > dims[1]-1:
                slice_range[1] = dims[1]-1
        elif axis == 'z':
            islab.SetOrientationToZ()
            if slice_range[0]  > dims[2]-1:
                slice_range[0] = dims[2]-1
            if slice_range[1]  > dims[2]-1:
                slice_range[1] = dims[2]-1
        else:
            vedo.logger.error(f"Error in slab(): unknown axis {axis}")
            raise RuntimeError()
        
        islab.SetSliceRange(slice_range)
        islab.Update()

        msh = Mesh(islab.GetOutput()).lighting('off')
        msh.mapper.SetLookupTable(utils.ctf2lut(self, msh))
        msh.mapper.SetScalarRange(self.scalar_range())

        msh.metadata["slab_range"] = slice_range
        msh.metadata["slab_axis"]  = axis
        msh.metadata["slab_operation"] = operation

        # compute bounds of slab
        origin = self.origin()
        spacing = self.spacing()
        if axis == 'x':
            msh.metadata["slab_bounding_box"] = [
                origin[0] + slice_range[0]*spacing[0],
                origin[0] + slice_range[1]*spacing[0],
                origin[1],
                origin[1] + dims[1]*spacing[1],
                origin[2],
                origin[2] + dims[2]*spacing[2],
            ]
        elif axis == 'y':
            msh.metadata["slab_bounding_box"] = [
                origin[0],
                origin[0] + dims[0]*spacing[0],
                origin[1] + slice_range[0]*spacing[1],
                origin[1] + slice_range[1]*spacing[1],
                origin[2],
                origin[2] + dims[2]*spacing[2],
            ]
        elif axis == 'z':
            msh.metadata["slab_bounding_box"] = [
                origin[0],
                origin[0] + dims[0]*spacing[0],
                origin[1],
                origin[1] + dims[1]*spacing[1],
                origin[2] + slice_range[0]*spacing[2],
                origin[2] + slice_range[1]*spacing[2],
            ]

        msh.pipeline = utils.OperationNode(
            f"slab{slice_range}", 
            comment=f"axis={axis}, operation={operation}",
            parents=[self],
            c="#4cc9f0:#e9c46a",
        )
        msh.name = "SlabMesh"
        return msh


    def warp(self, source, target, sigma=1, mode="3d", fit=True):
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

        NLT = transformations.NonLinearTransform()
        NLT.source_points = source
        NLT.target_points = target
        NLT.sigma = sigma
        NLT.mode = mode

        self.apply_transform(NLT, fit=fit)
        self.pipeline = utils.OperationNode("warp", parents=[self], c="#4cc9f0")
        return self

    def apply_transform(self, T, fit=True, interpolation="cubic"):
        """
        Apply a transform to the scalars in the volume.

        Arguments:
            T : (LinearTransform, NonLinearTransform)
                The transformation to be applied
            fit : (bool)
                fit/adapt the old bounding box to the modified geometry
            interpolation : (str)
                one of the following: "nearest", "linear", "cubic"
        """
        if utils.is_sequence(T):
            T = transformations.LinearTransform(T)

        TI = T.compute_inverse()

        reslice = vtki.new("ImageReslice")
        reslice.SetInputData(self.dataset)
        reslice.SetResliceTransform(TI.T)
        reslice.SetOutputDimensionality(3)
        if "lin" in interpolation.lower():
            reslice.SetInterpolationModeToLinear()
        elif "near" in interpolation.lower():
            reslice.SetInterpolationModeToNearestNeighbor()
        elif "cubic" in interpolation.lower():
            reslice.SetInterpolationModeToCubic()
        else:
            vedo.logger.error(
                f"in apply_transform: unknown interpolation mode {interpolation}")
            raise ValueError()
        reslice.SetAutoCropOutput(fit)
        reslice.Update()
        self._update(reslice.GetOutput())
        self.transform = T
        self.pipeline = utils.OperationNode(
            "apply_transform", parents=[self], c="#4cc9f0")
        return self

    def imagedata(self):
        """
        DEPRECATED:
        Use `Volume.dataset` instead.

        Return the underlying `vtkImagaData` object.
        """
        print("Volume.imagedata() is deprecated, use Volume.dataset instead")
        return self.dataset
    
    def modified(self):
        """
        Mark the object as modified.

        Example:

        - [numpy2volume0.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/numpy2volume0.py)
        """
        scals = self.dataset.GetPointData().GetScalars()
        if scals:
            scals.Modified()
        return self

    def tonumpy(self):
        """
        Get read-write access to voxels of a Volume object as a numpy array.

        When you set values in the output image, you don't want numpy to reallocate the array
        but instead set values in the existing array, so use the [:] operator.

        Example:
            `arr[:] = arr*2 + 15`

        If the array is modified add a call to:
        `volume.modified()`
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

    @property
    def shape(self):
        """Return the nr. of voxels in the 3 dimensions."""
        return np.array(self.dataset.GetDimensions())

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
        """
        Set/get the origin of the volumetric dataset.

        The origin is the position in world coordinates of the point index (0,0,0).
        This point does not have to be part of the dataset, in other words,
        the dataset extent does not have to start at (0,0,0) and the origin 
        can be outside of the dataset bounding box. 
        The origin plus spacing determine the position in space of the points.
        """
        if s is not None:
            self.dataset.SetOrigin(s)
            return self
        return np.array(self.dataset.GetOrigin())
    
    def pos(self, p=None):
        """Set/get the position of the volumetric dataset."""
        if p is not None:
            self.origin(p)
            return self
        return self.origin()

    def center(self):
        """Get the center of the volumetric dataset."""
        # note that this does not have the set method like origin and spacing
        return np.array(self.dataset.GetCenter())
    
    def shift(self, s):
        """Shift the volumetric dataset by a vector."""
        self.origin(self.origin() + np.array(s))
        return self

    def rotate_x(self, angle, rad=False, around=None):
        """
        Rotate around x-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return self
        LT = transformations.LinearTransform().rotate_x(angle, rad, around)
        return self.apply_transform(LT, fit=True, interpolation="linear")

    def rotate_y(self, angle, rad=False, around=None):
        """
        Rotate around y-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return self
        LT = transformations.LinearTransform().rotate_y(angle, rad, around)
        return self.apply_transform(LT, fit=True, interpolation="linear")

    def rotate_z(self, angle, rad=False, around=None):
        """
        Rotate around z-axis. If angle is in radians set `rad=True`.

        Use `around` to define a pivoting point.
        """
        if angle == 0:
            return self
        LT = transformations.LinearTransform().rotate_z(angle, rad, around)
        return self.apply_transform(LT, fit=True, interpolation="linear")

    def get_cell_from_ijk(self, ijk):
        """
        Get the voxel id number at the given ijk coordinates.

        Arguments:
            ijk : (list)
                the ijk coordinates of the voxel
        """
        return self.ComputeCellId(ijk)
    
    def get_point_from_ijk(self, ijk):
        """
        Get the point id number at the given ijk coordinates.

        Arguments:
            ijk : (list)
                the ijk coordinates of the voxel
        """
        return self.ComputePointId(ijk)

    def permute_axes(self, x, y, z):
        """
        Reorder the axes of the Volume by specifying
        the input axes which are supposed to become the new X, Y, and Z.
        """
        imp = vtki.new("ImagePermute")
        imp.SetFilteredAxes(x, y, z)
        imp.SetInputData(self.dataset)
        imp.Update()
        self._update(imp.GetOutput())
        self.pipeline = utils.OperationNode(
            f"permute_axes({(x,y,z)})", parents=[self], c="#4cc9f0"
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
        rsp = vtki.new("ImageResample")
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
            "resample", comment=f"spacing: {tuple(new_spacing)}", parents=[self], c="#4cc9f0"
        )
        return self


    def threshold(self, above=None, below=None, replace=None, replace_value=None):
        """
        Binary or continuous volume thresholding.
        Find the voxels that contain a value above/below the input values
        and replace them with a new value (default is 0).
        """
        th = vtki.new("ImageThreshold")
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
        extractVOI = vtki.new("ExtractVOI")
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
        ima = vtki.new("ImageAppend")
        ima.SetInputData(self.dataset)
        if not utils.is_sequence(volumes):
            volumes = [volumes]
        for volume in volumes:
            if isinstance(volume, vtki.vtkImageData):
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
        pf = vtki.new("ImageConstantPad")
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

    def resize(self, newdims):
        """Increase or reduce the number of voxels of a Volume with interpolation."""
        rsz = vtki.new("ImageResize")
        rsz.SetResizeMethodToOutputDimensions()
        rsz.SetInputData(self.dataset)
        rsz.SetOutputDimensions(newdims)
        rsz.Update()
        self.dataset = rsz.GetOutput()
        self._update(self.dataset)
        self.pipeline = utils.OperationNode(
            "resize", parents=[self], c="#4cc9f0", comment=f"dims={tuple(self.dimensions())}"
        )
        return self

    def normalize(self):
        """Normalize that scalar components for each point."""
        norm = vtki.new("ImageNormalize")
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

        ff = vtki.new("ImageFlip")
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
        Keyword `volume2` can be a constant `float`.

        Possible operations are:
        ```
        and, or, xor, nand, nor, not,
        +, -, /, 1/x, sin, cos, exp, log,
        abs, **2, sqrt, min, max, atan, atan2, median,
        mag, dot, gradient, divergence, laplacian.
        ```

        Example:
        ```py
        from vedo import Box, show
        vol1 = Box(size=(35,10, 5)).binarize()
        vol2 = Box(size=( 5,10,35)).binarize()
        vol = vol1.operation("xor", vol2)
        show([[vol1, vol2], 
            ["vol1 xor vol2", vol]],
            N=2, axes=1, viewup="z",
        ).close()
        ```

        Note:
            For logic operations, the two volumes must have the same bounds.
            If they do not, a larger image is created to contain both and the
            volumes are resampled onto the larger image before the operation is
            performed. This can be slow and memory intensive.

        See also:
            - [volume_operations.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/volume_operations.py)
        """
        op = operation.lower()
        image1 = self.dataset

        if op in ["and", "or", "xor", "nand", "nor"]:

            if not np.allclose(image1.GetBounds(), volume2.dataset.GetBounds()):
                # create a larger image to contain both
                b1 = image1.GetBounds()
                b2 = volume2.dataset.GetBounds()
                b = [
                    min(b1[0], b2[0]),
                    max(b1[1], b2[1]),
                    min(b1[2], b2[2]),
                    max(b1[3], b2[3]),
                    min(b1[4], b2[4]),
                    max(b1[5], b2[5]),
                ]
                dims1 = image1.GetDimensions()
                dims2 = volume2.dataset.GetDimensions()
                dims = [max(dims1[0], dims2[0]), max(dims1[1], dims2[1]), max(dims1[2], dims2[2])]

                image = vtki.vtkImageData()
                image.SetDimensions(dims)
                spacing = (
                    (b[1] - b[0]) / dims[0],
                    (b[3] - b[2]) / dims[1],
                    (b[5] - b[4]) / dims[2],
                )
                image.SetSpacing(spacing)
                image.SetOrigin((b[0], b[2], b[4]))
                image.AllocateScalars(vtki.VTK_UNSIGNED_CHAR, 1)
                image.GetPointData().GetScalars().FillComponent(0, 0)

                interp1 = vtki.new("ImageReslice")
                interp1.SetInputData(image1)
                interp1.SetOutputExtent(image.GetExtent())
                interp1.SetOutputOrigin(image.GetOrigin())
                interp1.SetOutputSpacing(image.GetSpacing())
                interp1.SetInterpolationModeToNearestNeighbor()
                interp1.Update()
                imageA = interp1.GetOutput()

                interp2 = vtki.new("ImageReslice")
                interp2.SetInputData(volume2.dataset)
                interp2.SetOutputExtent(image.GetExtent())
                interp2.SetOutputOrigin(image.GetOrigin())
                interp2.SetOutputSpacing(image.GetSpacing())
                interp2.SetInterpolationModeToNearestNeighbor()
                interp2.Update()
                imageB = interp2.GetOutput()

            else:
                imageA = image1
                imageB = volume2.dataset

            img_logic = vtki.new("ImageLogic")
            img_logic.SetInput1Data(imageA)
            img_logic.SetInput2Data(imageB)
            img_logic.SetOperation(["and", "or", "xor", "nand", "nor"].index(op))
            img_logic.Update()

            out_vol = Volume(img_logic.GetOutput())
            out_vol.pipeline = utils.OperationNode(
                "operation", comment=f"{op}", parents=[self, volume2], c="#4cc9f0", shape="cylinder"
            )
            return out_vol  ######################################################

        if volume2 and isinstance(volume2, Volume):
            # assert image1.GetScalarType() == volume2.dataset.GetScalarType(), "volumes have different scalar types"
            # make sure they have the same bounds:
            assert np.allclose(image1.GetBounds(), volume2.dataset.GetBounds()), "volumes have different bounds"
            # make sure they have the same spacing:
            assert np.allclose(image1.GetSpacing(), volume2.dataset.GetSpacing()), "volumes have different spacing"
            # make sure they have the same origin:
            assert np.allclose(image1.GetOrigin(), volume2.dataset.GetOrigin()), "volumes have different origin"

        mf = None
        if op in ["median"]:
            mf = vtki.new("ImageMedian3D")
            mf.SetInputData(image1)
        elif op in ["mag"]:
            mf = vtki.new("ImageMagnitude")
            mf.SetInputData(image1)
        elif op in ["dot"]:
            mf = vtki.new("ImageDotProduct")
            mf.SetInput1Data(image1)
            mf.SetInput2Data(volume2.dataset)
        elif op in ["grad", "gradient"]:
            mf = vtki.new("ImageGradient")
            mf.SetDimensionality(3)
            mf.SetInputData(image1)
        elif op in ["div", "divergence"]:
            mf = vtki.new("ImageDivergence")
            mf.SetInputData(image1)
        elif op in ["laplacian"]:
            mf = vtki.new("ImageLaplacian")
            mf.SetDimensionality(3)
            mf.SetInputData(image1)
        elif op in ["not"]:
            mf = vtki.new("ImageLogic")
            mf.SetInput1Data(image1)
            mf.SetOperation(4)

        if mf is not None:
            mf.Update()
            vol = Volume(mf.GetOutput())
            vol.pipeline = utils.OperationNode(
                "operation", comment=f"{op}", parents=[self], c="#4cc9f0", shape="cylinder"
            )
            return vol  ######################################################

        mat = vtki.new("ImageMathematics")
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
            "operation", comment=f"{op}", parents=[self, volume2], shape="cylinder", c="#4cc9f0"
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
        fft = vtki.new("ImageFFT")
        fft.SetInputData(self.dataset)
        fft.Update()
        out = fft.GetOutput()

        if high_cutoff:
            blp = vtki.new("ImageButterworthLowPass")
            blp.SetInputData(out)
            blp.SetCutOff(high_cutoff)
            blp.SetOrder(order)
            blp.Update()
            out = blp.GetOutput()

        if low_cutoff:
            bhp = vtki.new("ImageButterworthHighPass")
            bhp.SetInputData(out)
            bhp.SetCutOff(low_cutoff)
            bhp.SetOrder(order)
            bhp.Update()
            out = bhp.GetOutput()

        rfft = vtki.new("ImageRFFT")
        rfft.SetInputData(out)
        rfft.Update()

        ecomp = vtki.new("ImageExtractComponents")
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
        gsf = vtki.new("ImageGaussianSmooth")
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
        imgm = vtki.new("ImageMedian3D")
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
        ver = vtki.new("ImageContinuousErode3D")
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
        ver = vtki.new("ImageContinuousDilate3D")
        ver.SetInputData(self.dataset)
        ver.SetKernelSize(neighbours[0], neighbours[1], neighbours[2])
        ver.Update()
        self._update(ver.GetOutput())
        self.pipeline = utils.OperationNode("dilate", parents=[self], c="#4cc9f0")
        return self

    def magnitude(self):
        """Colapses components with magnitude function."""
        imgm = vtki.new("ImageMagnitude")
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
        v2p = vtki.new("ImageToPoints")
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
        euv = vtki.new("ImageEuclideanDistance")
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
        imc = vtki.new("ImageCorrelation")
        imc.SetInput1Data(self.dataset)
        imc.SetInput2Data(vol2.dataset)
        imc.SetDimensionality(dim)
        imc.Update()
        vol = Volume(imc.GetOutput())

        vol.pipeline = utils.OperationNode("correlation_with", parents=[self, vol2], c="#4cc9f0")
        return vol

    def scale_voxels(self, scale=1):
        """Scale the voxel content by factor `scale`."""
        rsl = vtki.new("ImageReslice")
        rsl.SetInputData(self.dataset)
        rsl.SetScalarScale(scale)
        rsl.Update()
        self._update(rsl.GetOutput())
        self.pipeline = utils.OperationNode(
            "scale_voxels", comment=f"scale={scale}", parents=[self], c="#4cc9f0"
        )
        return self
