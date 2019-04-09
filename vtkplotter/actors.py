from __future__ import division, print_function
import vtk
import numpy as np
import vtkplotter.docs as docs
import vtkplotter.colors as colors
import vtkplotter.utils as utils
import vtkplotter.settings as settings
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

__doc__ = (
    """
Submodule extending the ``vtkActor``, ``vtkVolume`` 
and ``vtkImageActor`` objects functionality.
"""
    + docs._defs
)

__all__ = [
    'Actor',
    'Assembly',
    'ImageActor',
    'Volume',
    'mergeActors',
    'isosurface',
    'collection',
]


################################################# functions
def mergeActors(actors, tol=0):
    """
    Build a new actor formed by the fusion of the polydatas of input objects.
    Similar to Assembly, but in this case the input objects become a single mesh.

    .. hint:: |thinplate_grid| |thinplate_grid.py|_
    """
    polylns = vtk.vtkAppendPolyData()
    for a in actors:
        polylns.AddInputData(a.polydata())
    polylns.Update()
    pd = polylns.GetOutput()
    return Actor(pd)


def collection():
    """
    Return the list of actor which have been created so far,
    even without having assigned them a name.
    Useful in loops. E.g.:
        
        >>> from vtkplotter import Cone, collect, show
        >>> for i in range(10):
        >>>     Cone(pos=[3*i, 0, 0], axis=[i, i-5, 0])        
        >>> show(collection())
    """
    return settings.collectable_actors


def isosurface(image, smoothing=0, threshold=None, connectivity=False):
    """Return a ``vtkActor`` isosurface extracted from a ``vtkImageData`` object.

    :param float smoothing: gaussian filter to smooth vtkImageData, in units of sigmas
    :param threshold:    value or list of values to draw the isosurface(s)
    :type threshold: float, list
    :param bool connectivity: if True only keeps the largest portion of the polydata

    .. hint:: |isosurfaces| |isosurfaces.py|_
    """
    if smoothing:
        smImg = vtk.vtkImageGaussianSmooth()
        smImg.SetDimensionality(3)
        smImg.SetInputData(image)
        smImg.SetStandardDeviations(smoothing, smoothing, smoothing)
        smImg.Update()
        image = smImg.GetOutput()

    scrange = image.GetScalarRange()
    if scrange[1] > 1e10:
        print("Warning, high scalar range detected:", scrange)

    cf = vtk.vtkContourFilter()
    cf.SetInputData(image)
    cf.UseScalarTreeOn()
    cf.ComputeScalarsOn()

    if utils.isSequence(threshold):
        cf.SetNumberOfContours(len(threshold))
        for i, t in enumerate(threshold):
            cf.SetValue(i, t)
        cf.Update()
    else:
        if not threshold:
            threshold = (2 * scrange[0] + scrange[1]) / 3.0
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

    a = Actor(poly, c=None).phong()
    a.mapper.SetScalarRange(scrange[0], scrange[1])
    return a


################################################# classes
class Prop(object):
    """Adds functionality to ``Actor``, ``Assembly``,
    ``vtkImageData`` and ``vtkVolume`` objects."""

    def __init__(self):

        self.filename = ""
        self.trail = None
        self.trailPoints = []
        self.trailSegmentSize = 0
        self.trailOffset = None
        self.top = None
        self.base = None
        self.info = dict()
        self._time = 0
        self._legend = None
        self.scalarbar = None

    def show(
        self,
        at=None,
        shape=(1, 1),
        N=None,
        pos=(0, 0),
        size="auto",
        screensize="auto",
        title="",
        bg="blackboard",
        bg2=None,
        axes=4,
        infinity=False,
        verbose=True,
        interactive=None,
        offscreen=False,
        resetcam=True,
        zoom=None,
        viewup="",
        azimuth=0,
        elevation=0,
        roll=0,
        interactorStyle=0,
        newPlotter=False,
        depthpeeling=False,
        q=False,
    ):
        """
        Create on the fly an instance of class ``Plotter`` or use the last existing one to
        show one single object.

        Allowed input objects are: ``Actor`` or ``Volume``.

        This is meant as a shortcut. If more than one object needs to be visualised
        please use the syntax `show([actor1, actor2,...], options)`.

        :param bool newPlotter: if set to `True`, a call to ``show`` will instantiate
            a new ``Plotter`` object (a new window) instead of reusing the first created.
            See e.g.: |readVolumeAsIsoSurface.py|_
        :return: the current ``Plotter`` class instance.

        .. note:: E.g.:

            >>> from vtkplotter import *
            >>> s = Sphere()
            >>> s.show(at=1, N=2)
            >>> c = Cube()
            >>> c.show(at=0, interactive=True)
        """
        from vtkplotter.plotter import show

        return show(
            self,
            at=at,
            shape=shape,
            N=N,
            pos=pos,
            size=size,
            screensize=screensize,
            title=title,
            bg=bg,
            bg2=bg2,
            axes=axes,
            infinity=infinity,
            verbose=verbose,
            interactive=interactive,
            offscreen=offscreen,
            resetcam=resetcam,
            zoom=zoom,
            viewup=viewup,
            azimuth=azimuth,
            elevation=elevation,
            roll=roll,
            interactorStyle=interactorStyle,
            newPlotter=newPlotter,
            depthpeeling=depthpeeling,
            q=q,
        )

    def legend(self, txt=None):
        """Set/get ``Actor`` legend text.

        :param str txt: legend text.

        Size and positions can be modified by setting attributes
        ``Plotter.legendSize``, ``Plotter.legendBC`` and ``Plotter.legendPos``.

        .. hint:: |fillholes.py|_
        """
        if txt:
            self._legend = txt
        else:
            return self._legend
        return self

    def pos(self, p_x=None, y=None, z=None):
        """Set/Get actor position."""
        if p_x is None:
            return np.array(self.GetPosition())
        if z is None:  # assume p_x is of the form (x,y,z)
            self.SetPosition(p_x)
        else:
            self.SetPosition(p_x, y, z)
        if self.trail:
            self.updateTrail()
        return self  # return itself to concatenate methods

    def addPos(self, dp_x=None, dy=None, dz=None):
        """Add vector to current actor position."""
        p = np.array(self.GetPosition())
        if dz is None:  # assume dp_x is of the form (x,y,z)
            self.SetPosition(p + dp_x)
        else:
            self.SetPosition(p + [dp_x, dy, dz])
        if self.trail:
            self.updateTrail()
        return self

    def x(self, position=None):
        """Set/Get actor position along x axis."""
        p = self.GetPosition()
        if position is None:
            return p[0]
        self.SetPosition(position, p[1], p[2])
        if self.trail:
            self.updateTrail()
        return self

    def y(self, position=None):
        """Set/Get actor position along y axis."""
        p = self.GetPosition()
        if position is None:
            return p[1]
        self.SetPosition(p[0], position, p[2])
        if self.trail:
            self.updateTrail()
        return self

    def z(self, position=None):
        """Set/Get actor position along z axis."""
        p = self.GetPosition()
        if position is None:
            return p[2]
        self.SetPosition(p[0], p[1], position)
        if self.trail:
            self.updateTrail()
        return self

    def rotate(self, angle, axis=(1, 0, 0), axis_point=(0, 0, 0), rad=False):
        """Rotate ``Actor`` around an arbitrary `axis` passing through `axis_point`."""
        if rad:
            anglerad = angle
        else:
            anglerad = angle / 57.29578
        axis = utils.versor(axis)
        a = np.cos(anglerad / 2)
        b, c, d = -axis * np.sin(anglerad / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )
        rv = np.dot(R, self.GetPosition() - np.array(axis_point)) + axis_point

        if rad:
            angle *= 57.29578
        # this vtk method only rotates in the origin of the actor:
        self.RotateWXYZ(angle, axis[0], axis[1], axis[2])
        self.SetPosition(rv)
        if self.trail:
            self.updateTrail()
        return self

    def rotateX(self, angle, axis_point=(0, 0, 0), rad=False):
        """Rotate around x-axis. If angle is in radians set ``rad=True``."""
        if rad:
            angle *= 57.29578
        self.RotateX(angle)
        if self.trail:
            self.updateTrail()
        return self

    def rotateY(self, angle, axis_point=(0, 0, 0), rad=False):
        """Rotate around y-axis. If angle is in radians set ``rad=True``."""
        if rad:
            angle *= 57.29578
        self.RotateY(angle)
        if self.trail:
            self.updateTrail()
        return self

    def rotateZ(self, angle, axis_point=(0, 0, 0), rad=False):
        """Rotate around z-axis. If angle is in radians set ``rad=True``."""
        if rad:
            angle *= 57.29578
        self.RotateZ(angle)
        if self.trail:
            self.updateTrail()
        return self

    def orientation(self, newaxis=None, rotation=0, rad=False):
        """
        Set/Get actor orientation.

        :param rotation: If != 0 rotate actor around newaxis.
        :param rad: set to True if angle is in radians.

        .. hint:: |gyroscope2| |gyroscope2.py|_
        """
        if rad:
            rotation *= 57.29578
        initaxis = utils.versor(self.top - self.base)
        if newaxis is None:
            return initaxis
        newaxis = utils.versor(newaxis)
        pos = np.array(self.GetPosition())
        crossvec = np.cross(initaxis, newaxis)
        angle = np.arccos(np.dot(initaxis, newaxis))
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.Translate(-pos)
        if rotation:
            T.RotateWXYZ(rotation, initaxis)
        T.RotateWXYZ(angle * 57.29578, crossvec)
        T.Translate(pos)
        self.SetUserMatrix(T.GetMatrix())
        if self.trail:
            self.updateTrail()
        return self

    def scale(self, s=None):
        """Set/get actor's scaling factor.

        :param s: scaling factor(s).
        :type s: float, list

        .. note:: if `s==(sx,sy,sz)` scale differently in the three coordinates."""
        if s is None:
            return np.array(self.GetScale())
        self.SetScale(s)
        return self  # return itself to concatenate methods

    def time(self, t=None):
        """Set/get actor's absolute time."""
        if t is None:
            return self._time
        self._time = t
        return self  # return itself to concatenate methods

    def addTrail(self, offset=None, maxlength=None, n=25, c=None, alpha=None, lw=1):
        """Add a trailing line to actor.

        :param offset: set an offset vector from the object center.
        :param maxlength: length of trailing line in absolute units
        :param n: number of segments to control precision
        :param lw: line width of the trail

        .. hint:: |trail| |trail.py|_
        """
        if maxlength is None:
            maxlength = self.diagonalSize() * 20
            if maxlength == 0:
                maxlength = 1

        if self.trail is None:
            pos = self.GetPosition()
            self.trailPoints = [None] * n
            self.trailSegmentSize = maxlength / n
            self.trailOffset = offset

            ppoints = vtk.vtkPoints()  # Generate the polyline
            poly = vtk.vtkPolyData()
            ppoints.SetData(numpy_to_vtk([pos] * n))
            poly.SetPoints(ppoints)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(n)
            for i in range(n):
                lines.InsertCellPoint(i)
            poly.SetPoints(ppoints)
            poly.SetLines(lines)
            mapper = vtk.vtkPolyDataMapper()

            if c is None:
                if hasattr(self, "GetProperty"):
                    col = self.GetProperty().GetColor()
                else:
                    col = (0.1, 0.1, 0.1)
            else:
                col = colors.getColor(c)

            if alpha is None:
                alpha = 1
                if hasattr(self, "GetProperty"):
                    alpha = self.GetProperty().GetOpacity()

            mapper.SetInputData(poly)
            tline = Actor()
            tline.SetMapper(mapper)
            tline.GetProperty().SetColor(col)
            tline.GetProperty().SetOpacity(alpha)
            tline.GetProperty().SetLineWidth(lw)
            self.trail = tline  # holds the vtkActor
            return self

    def updateTrail(self):
        currentpos = np.array(self.GetPosition())
        if self.trailOffset:
            currentpos += self.trailOffset
        lastpos = self.trailPoints[-1]
        if lastpos is None:  # reset list
            self.trailPoints = [currentpos] * len(self.trailPoints)
            return
        if np.linalg.norm(currentpos - lastpos) < self.trailSegmentSize:
            return

        self.trailPoints.append(currentpos)  # cycle
        self.trailPoints.pop(0)

        tpoly = self.trail.polydata()
        tpoly.GetPoints().SetData(numpy_to_vtk(self.trailPoints))
        return self

    def print(self):
        """Print  ``Actor``, ``Assembly``, ``Volume`` or ``ImageActor`` infos."""
        utils.printInfo(self)
        return self

    def on(self):
        """Switch on actor visibility. Object is not removed."""
        self.VisibilityOn()
        return self

    def off(self):
        """Switch off actor visibility. Object is not removed."""
        self.VisibilityOff()
        return self


####################################################
# Actor inherits from vtkActor and Prop
class Actor(vtk.vtkActor, Prop):
    """Build an instance of object ``Actor`` derived from ``vtkActor``.

    Either ``vtkPolyData`` or ``vtkActor`` is expected as input.

    :param c: color in RGB format, hex, symbol or name
    :param float alpha: opacity value
    :param bool wire:  show surface as wireframe
    :param bc: backface color of internal surface
    :param str texture: jpg file name or surface texture name
    :param bool computeNormals: compute point and cell normals at creation

    |basicshapes|
    """

    def __init__(
        self,
        poly=None,
        c="gold",
        alpha=1,
        wire=False,
        bc=None,
        texture=None,
        computeNormals=False,
        u=None,
    ):
        vtk.vtkActor.__init__(self)
        Prop.__init__(self)

        inputtype = str(type(poly))
        if "vtkActor" in inputtype:
            self.mapper = poly.GetMapper()
            self.poly = self.mapper.GetInput()
        else:
            self.poly = None  # cache vtkPolyData and mapper for speed
            self.mapper = vtk.vtkPolyDataMapper()
            self.SetMapper(self.mapper)

        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None
        self._bfprop = None  # backface property holder

        self.GetProperty().SetInterpolationToFlat()

        if settings.computeNormals is not None:
            computeNormals = settings.computeNormals

        if poly:
            if computeNormals:
                pdnorm = vtk.vtkPolyDataNormals()
                pdnorm.SetInputData(poly)
                pdnorm.ComputePointNormalsOn()
                pdnorm.ComputeCellNormalsOn()
                pdnorm.FlipNormalsOff()
                pdnorm.ConsistencyOn()
                pdnorm.Update()
                self.poly = pdnorm.GetOutput()
            else:
                self.poly = poly

            self.mapper.SetInputData(self.poly)

        prp = self.GetProperty()
        
        if settings.renderPointsAsSpheres:
            prp.RenderPointsAsSpheresOn()

        if alpha is not None:
            prp.SetOpacity(alpha)

        if c is None:
            self.mapper.ScalarVisibilityOn()
            prp.SetColor(colors.getColor("gold"))
        else:
            self.mapper.ScalarVisibilityOff()
            c = colors.getColor(c)
            prp.SetColor(c)
            prp.SetAmbient(0.1)
            prp.SetAmbientColor(c)
            prp.SetDiffuse(1)

        if wire:
            prp.SetRepresentationToWireframe()

        if texture:
            prp.SetColor(1.0, 1.0, 1.0)
            self.mapper.ScalarVisibilityOff()
            self.texture(texture)
        if bc and alpha == 1:  # defines a specific color for the backface
            backProp = vtk.vtkProperty()
            backProp.SetDiffuseColor(colors.getColor(bc))
            backProp.SetOpacity(alpha)
            self.SetBackfaceProperty(backProp)


    def __add__(self, actors):
        if isinstance(actors, list):
            alist = [self]
            for l in actors:
                if isinstance(l, vtk.vtkAssembly):
                    alist += l.getActors()
                else:
                    alist += l
            return Assembly(alist)
        elif isinstance(actors, vtk.vtkAssembly):
            actors.AddPart(self)
            return actors
        return Assembly([self, actors])

    def pickable(self, value=None):
        """Set/get pickable property of actor."""
        if value is None:
            return self.GetPickable()
        else:
            self.SetPickable(value)
            return self

    def updateMesh(self, polydata):
        """
        Change or modify the polygonal mesh for the actor with a new one.
        """
        self.poly = polydata
        self.mapper.SetInputData(polydata)
        self.mapper.Modified()
        return self

    def addScalarBar(self, c=None, title="", horizontal=False, vmin=None, vmax=None):
        """
        Add a 2D scalar bar to actor.

        .. hint:: |mesh_bands| |mesh_bands.py|_
        """
        # book it, it will be created by Plotter.show() later
        self.scalarbar = [c, title, horizontal, vmin, vmax]
        return self

    def addScalarBar3D(
        self,
        pos=(0, 0, 0),
        normal=(0, 0, 1),
        sx=0.1,
        sy=2,
        nlabels=9,
        ncols=256,
        cmap=None,
        c="k",
        alpha=1,
    ):
        """
        Draw a 3D scalar bar to actor.

        .. hint:: |mesh_coloring| |mesh_coloring.py|_
        """
        # book it, it will be created by Plotter.show() later
        self.scalarbar = [pos, normal, sx, sy, nlabels, ncols, cmap, c, alpha]
        return self


    def texture(self, name, scale=1, falsecolors=False, mapTo=1):
        """Assign a texture to actor from image file or predefined texture name."""
        import os

        if mapTo == 1:
            tmapper = vtk.vtkTextureMapToCylinder()
        elif mapTo == 2:
            tmapper = vtk.vtkTextureMapToSphere()
        elif mapTo == 3:
            tmapper = vtk.vtkTextureMapToPlane()

        tmapper.SetInputData(self.polydata(False))
        if mapTo == 1:
            tmapper.PreventSeamOn()

        xform = vtk.vtkTransformTextureCoords()
        xform.SetInputConnection(tmapper.GetOutputPort())
        xform.SetScale(scale, scale, scale)
        if mapTo == 1:
            xform.FlipSOn()
        xform.Update()

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(xform.GetOutputPort())
        mapper.ScalarVisibilityOff()

        fn = settings.textures_path + name + ".jpg"
        if os.path.exists(name):
            fn = name
        elif not os.path.exists(fn):
            colors.printc("~sad Texture", name, "not found in", settings.textures_path, c="r")
            colors.printc("~target Available textures:", c="m", end=" ")
            for ff in os.listdir(settings.textures_path):
                colors.printc(ff.split(".")[0], end=" ", c="m")
            print()
            return

        jpgReader = vtk.vtkJPEGReader()
        jpgReader.SetFileName(fn)
        atext = vtk.vtkTexture()
        atext.RepeatOn()
        atext.EdgeClampOff()
        atext.InterpolateOn()
        if falsecolors:
            atext.MapColorScalarsThroughLookupTableOn()
        atext.SetInputConnection(jpgReader.GetOutputPort())
        self.GetProperty().SetColor(1, 1, 1)
        self.SetMapper(mapper)
        self.SetTexture(atext)
        self.Modified()
        return self

    def getPoint(self, i):
        """
        Retrieve specific `i-th` point coordinates in mesh.
        Actor transformation is reset to its mesh position/orientation.

        :param int i: index of vertex point.

        .. warning:: if used in a loop this can slow down the execution by a lot.

        .. seealso:: ``actor.Points()``
        """
        poly = self.polydata(True)
        p = [0, 0, 0]
        poly.GetPoints().GetPoint(i, p)
        return np.array(p)

    def setPoint(self, i, p):
        """
        Set specific `i-th` point coordinates in mesh.
        Actor transformation is reset to its original mesh position/orientation.

        :param int i: index of vertex point.
        :param list p: new coordinates of mesh point.

        .. warning:: if used in a loop this can slow down the execution by a lot.

        .. seealso:: ``actor.Points()``
        """
        poly = self.polydata(False)
        poly.GetPoints().SetPoint(i, p)
        poly.GetPoints().Modified()
        # reset actor to identity matrix position/rotation:
        self.PokeMatrix(vtk.vtkMatrix4x4())
        return self

    def getPoints(self, transformed=True, copy=True):
        """
        Return the list of vertex coordinates of the input mesh.
        Same as `actor.coordinates()`.

        :param bool transformed: if `False` ignore any previous trasformation
            applied to the mesh.
        :param bool copy: if `False` return the reference to the points
            so that they can be modified in place.
        """
        return self.coordinates(transformed, copy)

    def setPoints(self, pts):
        """
        Set specific points coordinates in mesh. Input is a python list.
        Actor transformation is reset to its mesh position/orientation.

        :param list pts: new coordinates of mesh vertices.
        """
        vpts = vtk.vtkPoints()
        vpts.SetData(numpy_to_vtk(pts, deep=True))
        self.poly.SetPoints(vpts)
        self.poly.GetPoints().Modified()
        # reset actor to identity matrix position/rotation:
        self.PokeMatrix(vtk.vtkMatrix4x4())
        return self

    def computeNormals(self):
        """Compute cell and vertex normals for the actor's mesh.

        .. warning:: Mesh gets modified, can have a different nr. of vertices.
        """
        poly = self.polydata(False)
        pnormals = poly.GetPointData().GetNormals()
        cnormals = poly.GetCellData().GetNormals()
        if pnormals and cnormals:
            return self

        pdnorm = vtk.vtkPolyDataNormals()
        pdnorm.SetInputData(poly)
        pdnorm.ComputePointNormalsOn()
        pdnorm.ComputeCellNormalsOn()
        pdnorm.FlipNormalsOff()
        pdnorm.ConsistencyOn()
        pdnorm.Update()
        return self.updateMesh(pdnorm.GetOutput())

    def alpha(self, opacity=None):
        """Set/get actor's transparency. Same as `actor.opacity()`."""
        if opacity is None:
            return self.GetProperty().GetOpacity()

        self.GetProperty().SetOpacity(opacity)
        bfp = self.GetBackfaceProperty()
        if bfp:
            if opacity < 1:
                self._bfprop = bfp
                self.SetBackfaceProperty(None)
            else:
                self.SetBackfaceProperty(self._bfprop)
        return self

    def opacity(self, alpha=None):
        """Set/get actor's transparency. Same as `actor.alpha()`."""
        return self.alpha(alpha)

    def wireframe(self, wire=True):
        """Set actor's representation as wireframe or solid surface. Same as `wire()`."""
        return self.wire(wire)

    def wire(self, wireframe=True):
        """Set actor's representation as wireframe or solid surface. Same as `wireframe()`."""
        if wireframe:
            self.GetProperty().SetRepresentationToWireframe()
        else:
            self.GetProperty().SetRepresentationToSurface()
        return self

    def flat(self):
        """Set surface interpolation to flat."""
        self.GetProperty().SetInterpolationToFlat()
        self.GetProperty().SetSpecular(0)
        return self
        
    def phong(self):
        """Set surface interpolation to Phong."""
        self.GetProperty().SetInterpolationToPhong()
        return self

    def gouraud(self):
        """Set surface interpolation to Gouraud."""
        self.GetProperty().SetInterpolationToGouraud()
        return self

    def backFaceCulling(self, value=True):
        """Set culling of polygons based on orientation of normal with respect to camera."""
        self.GetProperty().SetBackfaceCulling(value)
        return self

    def frontFaceCulling(self, value=True):
        """Set culling of polygons based on orientation of normal with respect to camera."""
        self.GetProperty().SetFrontfaceCulling(value)
        return self
    

    def pointSize(self, s=None):
        """Set/get actor's point size of vertices."""
        if s is not None:
            if isinstance(self, vtk.vtkAssembly):
                cl = vtk.vtkPropCollection()
                self.GetActors(cl)
                cl.InitTraversal()
                a = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
                a.GetProperty().SetRepresentationToPoints()
                a.GetProperty().SetPointSize(s)
            else:
                self.GetProperty().SetRepresentationToPoints()
                self.GetProperty().SetPointSize(s)
        else:
            return self.GetProperty().GetPointSize()
        return self

    def color(self, c=False):
        """
        Set/get actor's color.
        If None is passed as input, will use colors from active scalars.
        Same as `c()`.
        """
        if c is False:
            return np.array(self.GetProperty().GetColor())
        elif c is None:
            self.GetMapper().ScalarVisibilityOn()
            return self
        else:
            self.GetMapper().ScalarVisibilityOff()
            self.GetProperty().SetColor(colors.getColor(c))
            return self

    def c(self, color=False):
        """
        Shortcut for `actor.color()`.
        If None is passed as input, will use colors from active scalars.
        Same as `color()`.
        """
        return self.color(color)

    def backColor(self, bc=None):
        """
        Set/get actor's backface color.
        """
        backProp = self.GetBackfaceProperty()

        if bc is None:
            if backProp:
                return backProp.GetDiffuseColor()
            return None

        if self.GetProperty().GetOpacity() < 1:
            colors.printc("~noentry backColor(): only active for alpha=1", c="y")
            return self

        if not backProp:
            backProp = vtk.vtkProperty()

        backProp.SetDiffuseColor(colors.getColor(bc))
        backProp.SetOpacity(self.GetProperty().GetOpacity())
        self.SetBackfaceProperty(backProp)
        self.GetMapper().ScalarVisibilityOff()
        return self

    def bc(self, backColor=False):
        """Shortcut for `actor.backColor()`. """
        return self.backColor(backColor)
    

    def lineWidth(self, lw=None):
        """Set/get width of mesh edges. Same as lw()."""
        if lw is not None:
            if lw == 0:
                self.GetProperty().EdgeVisibilityOff()
                return
            self.GetProperty().EdgeVisibilityOn()
            self.GetProperty().SetLineWidth(lw)
        else:
            return self.GetProperty().GetLineWidth()
        return self

    def lw(self, lineWidth=None):
        """Set/get width of mesh edges. Same as lineWidth()"""
        return self.lineWidth(lineWidth)

    def clean(self, tol=None):
        """
        Clean actor's polydata. Can also be used to decimate a mesh if ``tol`` is large.
        If ``tol=None`` only removes coincident points.

        :param tol: defines how far should be the points from each other in terms of fraction
            of the bounding box length.

        .. hint:: |moving_least_squares1D| |moving_least_squares1D.py|_

            |recosurface| |recosurface.py|_
        """
        poly = self.polydata(False)
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.PointMergingOn()
        cleanPolyData.ConvertLinesToPointsOn()
        cleanPolyData.ConvertPolysToLinesOn()
        cleanPolyData.SetInputData(poly)
        if tol:
            cleanPolyData.SetTolerance(tol)
        cleanPolyData.Update()
        return self.updateMesh(cleanPolyData.GetOutput())

    def xbounds(self):
        """Get the actor bounds `[xmin,xmax]`."""
        b = self.polydata(True).GetBounds()
        return (b[0], b[1])

    def ybounds(self):
        """Get the actor bounds `[ymin,ymax]`."""
        b = self.polydata(True).GetBounds()
        return (b[2], b[3])

    def zbounds(self):
        """Get the actor bounds `[zmin,zmax]`."""
        b = self.polydata(True).GetBounds()
        return (b[4], b[5])

    def averageSize(self):
        """Calculate the average size of a mesh.
        This is the mean of the vertex distances from the center of mass."""
        cm = self.centerOfMass()
        coords = self.coordinates(copy=False)
        if not len(coords):
            return 0
        s, c = 0.0, 0.0
        n = len(coords)
        step = int(n / 10000.0) + 1
        for i in np.arange(0, n, step):
            s += utils.mag(coords[i] - cm)
            c += 1
        return s / c

    def diagonalSize(self):
        """Get the length of the diagonal of actor bounding box."""
        b = self.polydata().GetBounds()
        return np.sqrt((b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2)

    def maxBoundSize(self):
        """Get the maximum dimension in x, y or z of the actor bounding box."""
        b = self.polydata(True).GetBounds()
        return max(abs(b[1] - b[0]), abs(b[3] - b[2]), abs(b[5] - b[4]))

    def centerOfMass(self):
        """Get the center of mass of actor.

        .. hint:: |fatlimb| |fatlimb.py|_
        """
        cmf = vtk.vtkCenterOfMass()
        cmf.SetInputData(self.polydata(True))
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)

    def volume(self, value=None):
        """Get/set the volume occupied by actor."""
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        v = mass.GetVolume()
        if value is not None:
            if not v:
                colors.printc("~bomb Volume is zero: cannot rescale.", c=1, end="")
                colors.printc(" Consider adding actor.triangle()", c=1)
                return self
            self.scale(value / v)
            return self
        else:
            return v

    def area(self, value=None):
        """Get/set the surface area of actor.

        .. hint:: |largestregion.py|_
        """
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        ar = mass.GetSurfaceArea()
        if value is not None:
            if not ar:
                colors.printc("~bomb Area is zero: cannot rescale.", c=1, end="")
                colors.printc(" Consider adding actor.triangle()", c=1)
                return self
            self.scale(value / ar)
            return self
        else:
            return ar

    def closestPoint(self, pt, N=1, radius=None, returnIds=False):
        """
        Find the closest point on a mesh given from the input point `pt`.

        :param int N: if greater than 1, return a list of N ordered closest points.
        :param float radius: if given, get all points within that radius.
        :param bool returnIds: return points IDs instead of point coordinates.

        .. hint:: |fitplanes.py|_

            |align1| |align1.py|_

            |quadratic_morphing| |quadratic_morphing.py|_

        .. note:: The appropriate kd-tree search locator is built on the fly and cached for speed.
        """
        poly = self.polydata(True)

        if N > 1 or radius:
            plocexists = self.point_locator
            if not plocexists or (plocexists and self.point_locator is None):
                point_locator = vtk.vtkPointLocator()
                point_locator.SetDataSet(poly)
                point_locator.BuildLocator()
                self.point_locator = point_locator

            vtklist = vtk.vtkIdList()
            if N > 1:
                self.point_locator.FindClosestNPoints(N, pt, vtklist)
            else:
                self.point_locator.FindPointsWithinRadius(radius, pt, vtklist)
            if returnIds:
                return [int(vtklist.GetId(k)) for k in range(vtklist.GetNumberOfIds())]
            else:
                trgp = []
                for i in range(vtklist.GetNumberOfIds()):
                    trgp_ = [0, 0, 0]
                    vi = vtklist.GetId(i)
                    poly.GetPoints().GetPoint(vi, trgp_)
                    trgp.append(trgp_)
                return np.array(trgp)

        clocexists = self.cell_locator
        if not clocexists or (clocexists and self.cell_locator is None):
            cell_locator = vtk.vtkCellLocator()
            cell_locator.SetDataSet(poly)
            cell_locator.BuildLocator()
            self.cell_locator = cell_locator

        trgp = [0, 0, 0]
        cid = vtk.mutable(0)
        dist2 = vtk.mutable(0)
        subid = vtk.mutable(0)
        self.cell_locator.FindClosestPoint(pt, trgp, cid, subid, dist2)
        if returnIds:
            return int(cid)
        else:
            return np.array(trgp)


    def distanceToMesh(self, actor, signed=False, negate=False):
        '''
        Computes the (signed) distance from one mesh to another.
        
        Example: |distance2mesh.py|_
        '''
        poly1 = self.polydata()
        poly2 = actor.polydata()
        df = vtk.vtkDistancePolyDataFilter()
        df.SetInputData(0, poly1)
        df.SetInputData(1, poly2)
        if signed:
            df.SignedDistanceOn()
        if negate:
            df.NegateDistanceOn()
        df.Update()
        
        scals = df.GetOutput().GetPointData().GetScalars()
        poly1.GetPointData().AddArray(scals)

        poly1.GetPointData().SetActiveScalars(scals.GetName())
        rng = scals.GetRange()
        self.mapper.SetScalarRange(rng[0], rng[1])
        self.mapper.ScalarVisibilityOn()
        return self


    def clone(self, transformed=True):
        """
        Clone a ``Actor(vtkActor)`` and make an exact copy of it.

        :param transformed: if `False` ignore any previous trasformation applied to the mesh.

        .. hint:: |carcrash| |carcrash.py|_
        """
        poly = self.polydata(transformed=transformed)
        polyCopy = vtk.vtkPolyData()
        polyCopy.DeepCopy(poly)

        cloned = Actor()
        cloned.poly = polyCopy
        cloned.mapper.SetInputData(polyCopy)
        cloned.mapper.SetScalarVisibility(self.mapper.GetScalarVisibility())
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.GetProperty())
        cloned.SetProperty(pr)
        return cloned


    def transformPolydata(self, transformation):
        """Obsolete: use  transformMesh()."""
        print("~noentry Obsolete transformPolydata(): use transformMesh().\n")
        return self.transformMesh(transformation)

    def transformMesh(self, transformation):
        """
        Apply this transformation to the polygonal `mesh`, 
        not to the actor's transformation, which is reset.

        :param transformation: ``vtkTransform`` or ``vtkMatrix4x4`` object.
        """
        if isinstance(transformation, vtk.vtkMatrix4x4):
            tr = vtk.vtkTransform()
            tr.SetMatrix(transformation)
            transformation = tr

        tf = vtk.vtkTransformPolyDataFilter()

        tf.SetTransform(transformation)
        tf.SetInputData(self.polydata())
        tf.Update()
        self.PokeMatrix(vtk.vtkMatrix4x4())  # identity
        return self.updateMesh(tf.GetOutput())
    

    def normalize(self):
        """
        Shift actor's center of mass at origin and scale its average size to unit.
        """
        cm = self.centerOfMass()
        coords = self.coordinates()
        if not len(coords):
            return
        pts = coords - cm
        xyz2 = np.sum(pts * pts, axis=0)
        scale = 1 / np.sqrt(np.sum(xyz2) / len(pts))
        t = vtk.vtkTransform()
        t.Scale(scale, scale, scale)
        t.Translate(-cm)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(self.poly)
        tf.SetTransform(t)
        tf.Update()
        return self.updateMesh(tf.GetOutput())

    def mirror(self, axis="x"):
        """
        Mirror the actor polydata along one of the cartesian axes.

        .. note::  ``axis='n'``, will flip only mesh normals.

        .. hint:: |mirror| |mirror.py|_
        """
        poly = self.polydata(transformed=True)
        polyCopy = vtk.vtkPolyData()
        polyCopy.DeepCopy(poly)

        sx, sy, sz = 1, 1, 1
        dx, dy, dz = self.GetPosition()
        if axis.lower() == "x":
            sx = -1
        elif axis.lower() == "y":
            sy = -1
        elif axis.lower() == "z":
            sz = -1
        elif axis.lower() == "n":
            pass
        else:
            colors.printc("~times Error in mirror(): mirror must be set to x, y, z or n.", c=1)
            raise RuntimeError()

        if axis != "n":
            for j in range(polyCopy.GetNumberOfPoints()):
                p = [0, 0, 0]
                polyCopy.GetPoint(j, p)
                polyCopy.GetPoints().SetPoint(
                    j,
                    p[0] * sx - dx * (sx - 1),
                    p[1] * sy - dy * (sy - 1),
                    p[2] * sz - dz * (sz - 1),
                )
        rs = vtk.vtkReverseSense()
        rs.SetInputData(polyCopy)
        rs.ReverseNormalsOn()
        rs.Update()
        polyCopy = rs.GetOutput()

        pdnorm = vtk.vtkPolyDataNormals()
        pdnorm.SetInputData(polyCopy)
        pdnorm.ComputePointNormalsOn()
        pdnorm.ComputeCellNormalsOn()
        pdnorm.FlipNormalsOff()
        pdnorm.ConsistencyOn()
        pdnorm.Update()
        return self.updateMesh(pdnorm.GetOutput())

    def flipNormals(self):
        """
        Flip all mesh normals. Same as `actor.mirror('n')`.
        """
        return self.mirror("n")

    def shrink(self, fraction=0.85):
        """Shrink the triangle polydata in the representation of the input mesh.

        Example:
            .. code-block:: python

                from vtkplotter import load, Sphere, show
                pot = load('data/shapes/teapot.vtk').shrink(0.75)
                s = Sphere(r=0.2).pos(0,0,-0.5)
                show(pot, s)

            |shrink| |shrink.py|_
        """
        poly = self.polydata(True)
        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputData(poly)
        shrink.SetShrinkFactor(fraction)
        shrink.Update()
        return self.updateMesh(shrink.GetOutput())

    def stretch(self, q1, q2):
        """Stretch actor between points `q1` and `q2`. Mesh is not affected.

        .. hint:: |aspring| |aspring.py|_

        .. note:: for ``Actors`` like helices, Line, cylinders, cones etc., 
            two attributes ``actor.base``, and ``actor.top`` are already defined.
        """
        if self.base is None:
            colors.printc('~times Error in stretch(): Please define vectors \
                          actor.base and actor.top at creation. Exit.', c='r')
            exit(0)

        p1, p2 = self.base, self.top
        q1, q2, z = np.array(q1), np.array(q2), np.array([0, 0, 1])
        plength = np.linalg.norm(p2 - p1)
        qlength = np.linalg.norm(q2 - q1)
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.Translate(-p1)
        cosa = np.dot(p2 - p1, z) / plength
        n = np.cross(p2 - p1, z)
        T.RotateWXYZ(np.arccos(cosa) * 57.3, n)

        T.Scale(1, 1, qlength / plength)

        cosa = np.dot(q2 - q1, z) / qlength
        n = np.cross(q2 - q1, z)
        T.RotateWXYZ(-np.arccos(cosa) * 57.3, n)
        T.Translate(q1)

        self.SetUserMatrix(T.GetMatrix())
        if self.trail:
            self.updateTrail()
        return self

    def cutPlane(self, origin=(0, 0, 0), normal=(1, 0, 0), showcut=False):
        """Deprecated: use cutWithPlane"""
        colors.printc("~targetcut Plane deprecated: use cutWithPlane()", c=1, box="-")
        return self.cutWithPlane(origin, normal, showcut)

    def cutWithPlane(self, origin=(0, 0, 0), normal=(1, 0, 0), showcut=False):
        """
        Takes a ``vtkActor`` and cuts it with the plane defined by a point and a normal.

        :param origin: the cutting plane goes through this point
        :param normal: normal of the cutting plane
        :param showcut: if `True` show the cut off part of the mesh as thin wireframe.

        .. hint:: |trail| |trail.py|_
        """
        if normal is "x":
            normal = (1,0,0)
        elif normal is "y":
            normal = (0,1,0)
        elif normal is "z":
            normal = (0,0,1)
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        self.computeNormals()
        poly = self.polydata()
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(poly)
        clipper.SetClipFunction(plane)
        if showcut:
            clipper.GenerateClippedOutputOn()
        else:
            clipper.GenerateClippedOutputOff()
        clipper.GenerateClipScalarsOff()
        clipper.SetValue(0)
        clipper.Update()

        self.updateMesh(clipper.GetOutput())

        if showcut:
            c = self.GetProperty().GetColor()
            cpoly = clipper.GetClippedOutput()
            restActor = Actor(cpoly, c=c, alpha=0.05, wire=1)
            restActor.SetUserMatrix(self.GetMatrix())
            asse = Assembly([self, restActor])
            self = asse
            return asse
        else:
            return self

    def cutWithMesh(self, mesh, invert=False):
        """
        Cut an ``Actor`` mesh with another ``vtkPolyData`` or ``Actor``.

        :param bool invert: if True return cut off part of actor.

        .. hint:: |cutWithMesh| |cutWithMesh.py|_

            |cutAndCap| |cutAndCap.py|_
        """
        if isinstance(mesh, vtk.vtkPolyData):
            polymesh = mesh
        if isinstance(mesh, Actor):
            polymesh = mesh.polydata()
        else:
            polymesh = mesh.GetMapper().GetInput()
        poly = self.polydata()

        # Create an array to hold distance information
        signedDistances = vtk.vtkFloatArray()
        signedDistances.SetNumberOfComponents(1)
        signedDistances.SetName("SignedDistances")

        # implicit function that will be used to slice the mesh
        ippd = vtk.vtkImplicitPolyDataDistance()
        ippd.SetInput(polymesh)

        # Evaluate the signed distance function at all of the grid points
        for pointId in range(poly.GetNumberOfPoints()):
            p = poly.GetPoint(pointId)
            signedDistance = ippd.EvaluateFunction(p)
            signedDistances.InsertNextValue(signedDistance)

        # add the SignedDistances to the grid
        poly.GetPointData().SetScalars(signedDistances)

        # use vtkClipDataSet to slice the grid with the polydata
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(poly)
        if invert:
            clipper.InsideOutOff()
        else:
            clipper.InsideOutOn()
        clipper.SetValue(0.0)
        clipper.Update()
        return self.updateMesh(clipper.GetOutput())

    def cap(self, returnCap=False):
        """
        Generate a "cap" on a clipped actor, or caps sharp edges.

        .. hint:: |cutAndCap| |cutAndCap.py|_
        """
        poly = self.polydata(True)

        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(poly)
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ManifoldEdgesOff()
        fe.Update()

        stripper = vtk.vtkStripper()
        stripper.SetInputData(fe.GetOutput())
        stripper.Update()

        boundaryPoly = vtk.vtkPolyData()
        boundaryPoly.SetPoints(stripper.GetOutput().GetPoints())
        boundaryPoly.SetPolys(stripper.GetOutput().GetLines())

        tf = vtk.vtkTriangleFilter()
        tf.SetInputData(boundaryPoly)
        tf.Update()

        if returnCap:
            return Actor(tf.GetOutput())
        else:
            polyapp = vtk.vtkAppendPolyData()
            polyapp.AddInputData(poly)
            polyapp.AddInputData(tf.GetOutput())
            polyapp.Update()
            return self.updateMesh(polyapp.GetOutput()).clean()

    def threshold(self, scalars, vmin=None, vmax=None, useCells=False):
        """
        Extracts cells where scalar value satisfies threshold criterion.

        :param scalars: name of the scalars array.
        :type scalars: str, list
        :param float vmin: minimum value of the scalar
        :param float vmax: maximum value of the scalar
        :param bool useCells: if `True`, assume array scalars refers to cells.

        .. hint:: |mesh_threshold| |mesh_threshold.py|_
        """
        if utils.isSequence(scalars):
            self.addPointScalars(scalars, "threshold")
            scalars = "threshold"
        elif self.scalars(scalars) is None:
            colors.printc("~times No scalars found with name", scalars, c=1)
            exit()

        thres = vtk.vtkThreshold()
        thres.SetInputData(self.poly)

        if useCells:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
        else:
            asso = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        thres.SetInputArrayToProcess(0, 0, 0, asso, scalars)

        if vmin is None and vmax is not None:
            thres.ThresholdByLower(vmax)
        elif vmax is None and vmin is not None:
            thres.ThresholdByUpper(vmin)
        else:
            thres.ThresholdBetween(vmin, vmax)
        thres.Update()

        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(thres.GetOutput())
        gf.Update()
        return self.updateMesh(gf.GetOutput())

    def triangle(self, verts=True, lines=True):
        """
        Converts actor polygons and strips to triangles.
        """
        tf = vtk.vtkTriangleFilter()
        tf.SetPassLines(lines)
        tf.SetPassVerts(verts)
        tf.SetInputData(self.poly)
        tf.Update()
        return self.updateMesh(tf.GetOutput())

    def pointColors(self, scalars, cmap="jet", alpha=1, bands=None, vmin=None, vmax=None):
        """
        Set individual point colors by providing a list of scalar values and a color map.
        `scalars` can be a string name of the ``vtkArray``.

        :param cmap: color map scheme to transform a real number into a color.
        :type cmap: str, list, vtkLookupTable, matplotlib.colors.LinearSegmentedColormap
        :param alpha: mesh transparency. Can be a ``list`` of values one for each vertex.
        :type alpha: float, list
        :param int bands: group scalars in this number of bins, typically to form bands or stripes.
        :param float vmin: clip scalars to this minimum value
        :param float vmax: clip scalars to this maximum value

        .. hint:: |mesh_coloring| |mesh_coloring.py|_

            |mesh_alphas| |mesh_alphas.py|_

            |mesh_bands| |mesh_bands.py|_

            |mesh_custom| |mesh_custom.py|_
        """
        poly = self.polydata(False)

        if isinstance(scalars, str):  # if a name is passed
            scalars = vtk_to_numpy(poly.GetPointData().GetArray(scalars))

        n = len(scalars)
        useAlpha = False
        if n != poly.GetNumberOfPoints():
            colors.printc('~times pointColors Error: nr. of scalars != nr. of points',
                          n, poly.GetNumberOfPoints(), c=1)
        if utils.isSequence(alpha):
            useAlpha = True
            if len(alpha) > n:
                colors.printc('~times pointColors Error: nr. of scalars < nr. of alpha values',
                              n, len(alpha), c=1)
                exit()
        if bands:
            scalars = utils.makeBands(scalars, bands)

        if vmin is None:
            vmin = np.min(scalars)
        if vmax is None:
            vmax = np.max(scalars)

        lut = vtk.vtkLookupTable()  # build the look-up table

        if utils.isSequence(cmap):
            sname = "pointColors_custom"
            lut.SetNumberOfTableValues(len(cmap))
            lut.Build()
            for i, c in enumerate(cmap):
                col = colors.getColor(c)
                if len(col) == 4:
                    r, g, b, a = col
                else:
                    r, g, b = col
                    a = 1
                lut.SetTableValue(i, r, g, b, a)

        elif isinstance(cmap, vtk.vtkLookupTable):
            sname = "pointColors_lut"
            lut = cmap

        else:
            if isinstance(cmap, str):
                sname = "pointColors_" + cmap
            else:
                sname = "pointColors"
            lut.SetNumberOfTableValues(512)
            lut.Build()
            for i in range(512):
                r, g, b = colors.colorMap(i, cmap, 0, 512)
                if useAlpha:
                    idx = int(i / 512 * len(alpha))
                    lut.SetTableValue(i, r, g, b, alpha[idx])
                else:
                    lut.SetTableValue(i, r, g, b, alpha)

        arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
        arr.SetName(sname)
        self.mapper.SetScalarRange(vmin, vmax)
        self.mapper.SetLookupTable(lut)
        self.mapper.ScalarVisibilityOn()
        poly.GetPointData().SetScalars(arr)
        poly.GetPointData().SetActiveScalars(sname)
        return self

    def cellColors(self, scalars, cmap="jet", alpha=1, bands=None, vmin=None, vmax=None):
        """
        Set individual cell colors by setting a scalar.

        :param cmap: color map scheme to transform a real number into a color.
        :type cmap: str, list, vtkLookupTable, matplotlib.colors.LinearSegmentedColormap
        :param alpha: mesh transparency. Can be a ``list`` of values one for each vertex.
        :type alpha: float, list
        :param int bands: group scalars in this number of bins, typically to form bands of stripes.
        :param float vmin: clip scalars to this minimum value
        :param float vmax: clip scalars to this maximum value

        .. hint:: |mesh_coloring| |mesh_coloring.py|_
        """
        poly = self.polydata(False)

        if isinstance(scalars, str):  # if a name is passed
            scalars = vtk_to_numpy(poly.GetCellData().GetArray(scalars))

        n = len(scalars)
        useAlpha = False
        if n != poly.GetNumberOfCells():
            colors.printc('~times cellColors Error: nr. of scalars != nr. of cells',
                          n, poly.GetNumberOfCells(), c=1)
        if utils.isSequence(alpha):
            useAlpha = True
            if len(alpha) > n:
                colors.printc('~times cellColors Error: nr. of scalars != nr. of alpha values',
                              n, len(alpha), c=1)
                exit()
        if bands:
            scalars = utils.makeBands(scalars, bands)

        if vmin is None:
            vmin = np.min(scalars)
        if vmax is None:
            vmax = np.max(scalars)

        lut = vtk.vtkLookupTable()  # build the look-up table

        if utils.isSequence(cmap):
            sname = "cellColors_custom"
            lut.SetNumberOfTableValues(len(cmap))
            lut.Build()
            for i, c in enumerate(cmap):
                col = colors.getColor(c)
                if len(col) == 4:
                    r, g, b, a = col
                else:
                    r, g, b = col
                    a = 1
                lut.SetTableValue(i, r, g, b, a)

        elif isinstance(cmap, vtk.vtkLookupTable):
            sname = "cellColors_lut"
            lut = cmap

        else:
            if isinstance(cmap, str):
                sname = "cellColors_" + cmap
            else:
                sname = "cellColors"
            lut.SetNumberOfTableValues(512)
            lut.Build()
            for i in range(512):
                r, g, b = colors.colorMap(i, cmap, 0, 512)
                if useAlpha:
                    idx = int(i / 512 * len(alpha))
                    lut.SetTableValue(i, r, g, b, alpha[idx])
                else:
                    lut.SetTableValue(i, r, g, b, alpha)

        arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
        arr.SetName(sname)
        self.mapper.SetScalarRange(vmin, vmax)
        self.mapper.SetLookupTable(lut)
        self.mapper.ScalarVisibilityOn()
        poly.GetCellData().SetScalars(arr)
        poly.GetCellData().SetActiveScalars(sname)
        return self

    def addPointScalars(self, scalars, name):
        """
        Add point scalars to the actor's polydata assigning it a name.

        .. hint:: |mesh_coloring| |mesh_coloring.py|_
        """
        poly = self.polydata(False)
        if len(scalars) != poly.GetNumberOfPoints():
            colors.printc('~times pointScalars Error: Number of scalars != nr. of points',
                          len(scalars), poly.GetNumberOfPoints(), c=1)
            exit()
        arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)
        poly.GetPointData().SetActiveScalars(name)
        self.mapper.SetScalarRange(np.min(scalars), np.max(scalars))
        self.mapper.ScalarVisibilityOn()
        return self

    def addCellScalars(self, scalars, name):
        """
        Add cell scalars to the actor's polydata assigning it a name.
        """
        poly = self.polydata(False)
        if isinstance(scalars, str):
            scalars = vtk_to_numpy(poly.GetPointData().GetArray(scalars))

        if len(scalars) != poly.GetNumberOfCells():
            colors.printc("~times Number of scalars != nr. of cells", c=1)
            exit()
        arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
        arr.SetName(name)
        poly.GetCellData().AddArray(arr)
        poly.GetCellData().SetActiveScalars(name)
        self.mapper.SetScalarRange(np.min(scalars), np.max(scalars))
        self.mapper.ScalarVisibilityOn()
        return self

    def addPointVectors(self, vectors, name):
        """
        Add a point vector field to the actor's polydata assigning it a name.
        """
        poly = self.polydata(False)
        if len(vectors) != poly.GetNumberOfPoints():
            colors.printc('~times addPointVectors Error: Number of vectors != nr. of points',
                          len(vectors), poly.GetNumberOfPoints(), c=1)
            exit()
        arr = vtk.vtkDoubleArray()
        arr.SetNumberOfComponents(3)
        arr.SetName(name)
        for v in vectors:
            arr.InsertNextTuple(v)
        poly.GetPointData().AddArray(arr)
        poly.GetPointData().SetActiveVectors(name)
        return self

    def addIDs(self, asfield=False):
        """
        Generate point and cell ids.

        :param bool asfield: flag to control whether to generate scalar or field data.
        """
        ids = vtk.vtkIdFilter()
        ids.SetInputData(self.poly)
        ids.PointIdsOn()
        ids.CellIdsOn()
        if asfield:
            ids.FieldDataOn()
        else:
            ids.FieldDataOff()
        ids.Update()
        return self.updateMesh(ids.GetOutput())

    def addCurvatureScalars(self, method=0, lut=None):
        """
        Build an ``Actor`` that contains the color coded surface
        curvature calculated in three different ways.

        :param int method: 0-gaussian, 1-mean, 2-max, 3-min curvature.
        :param float lut: optional look up table.

        :Example:
            .. code-block:: python
            
                from vtkplotter import *
                t = Torus().addCurvatureScalars()
                show(t)

            |curvature|
        """
        curve = vtk.vtkCurvatures()
        curve.SetInputData(self.poly)
        curve.SetCurvatureType(method)
        curve.Update()
        self.poly = curve.GetOutput()

        scls = self.poly.GetPointData().GetScalars().GetRange()
        print("curvature(): scalar range is", scls)

        self.mapper.SetInputData(self.poly)
        if lut:
            self.mapper.SetLookupTable(lut)
            self.mapper.SetUseLookupTableScalarRange(1)
        self.mapper.Update()
        self.Modified()
        self.mapper.ScalarVisibilityOn()
        return self

    def scalars(self, name_or_idx=None, datatype="point"):
        """
        Retrieve point or cell scalars using array name or index number.
        If no ``name`` is given return the list of names of existing arrays.

        .. hint:: |mesh_coloring.py|_
        """
        poly = self.polydata(False)

        if name_or_idx is None:  # get mode behaviour
            ncd = poly.GetCellData().GetNumberOfArrays()
            npd = poly.GetPointData().GetNumberOfArrays()
            nfd = poly.GetFieldData().GetNumberOfArrays()
            arrs = []
            for i in range(npd):
                print(i, "PointData", poly.GetPointData().GetArrayName(i))
                arrs.append(["PointData", poly.GetPointData().GetArrayName(i)])
            for i in range(ncd):
                print(i, "CellData", poly.GetCellData().GetArrayName(i))
                arrs.append(["CellData", poly.GetCellData().GetArrayName(i)])
            for i in range(nfd):
                print(i, "FieldData", poly.GetFieldData().GetArrayName(i))
                arrs.append(["FieldData", poly.GetFieldData().GetArrayName(i)])
            return arrs
        else:  # set mode
            if "point" in datatype.lower():
                data = poly.GetPointData()
            elif "cell" in datatype.lower():
                data = poly.GetCellData()
            elif "field" in datatype.lower():
                data = poly.GetFieldData()
            else:
                colors.printc("~times Error in scalars(): unknown datatype", datatype, c=1)
                exit()

            if isinstance(name_or_idx, int):
                name = data.GetArrayName(name_or_idx)
                if name is None:
                    return None
                data.SetActiveScalars(name)
                return vtk_to_numpy(data.GetArray(name))
            elif isinstance(name_or_idx, str):
                arr = data.GetArray(name_or_idx)
                if arr is None:
                    return None
                data.SetActiveScalars(name_or_idx)
                return vtk_to_numpy(arr)

            return None

    def subdivide(self, N=1, method=0):
        """Increase the number of vertices of a surface mesh.

        :param int N: number of subdivisions.
        :param int method: Loop(0), Linear(1), Adaptive(2), Butterfly(3)

        .. hint:: |tutorial_subdivide| |tutorial.py|_
        """
        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputData(self.polydata())
        triangles.Update()
        originalMesh = triangles.GetOutput()
        if method == 0:
            sdf = vtk.vtkLoopSubdivisionFilter()
        elif method == 1:
            sdf = vtk.vtkLinearSubdivisionFilter()
        elif method == 2:
            sdf = vtk.vtkAdaptiveSubdivisionFilter()
        elif method == 3:
            sdf = vtk.vtkButterflySubdivisionFilter()
        else:
            colors.printc("~times Error in subdivide: unknown method.", c="r")
            exit()
        if method != 2:
            sdf.SetNumberOfSubdivisions(N)
        sdf.SetInputData(originalMesh)
        sdf.Update()
        return self.updateMesh(sdf.GetOutput())

    def decimate(self, fraction=0.5, N=None, boundaries=False, verbose=True):
        """
        Downsample the number of vertices in a mesh.

        :param float fraction: the desired target of reduction.
        :param int N: the desired number of final points (**fraction** is recalculated based on it).
        :param bool boundaries: (True), decide whether to leave boundaries untouched or not.

        .. note:: Setting ``fraction=0.1`` leaves 10% of the original nr of vertices.

        .. hint:: |skeletonize| |skeletonize.py|_
        """
        poly = self.polydata(True)
        if N:  # N = desired number of points
            Np = poly.GetNumberOfPoints()
            fraction = float(N) / Np
            if fraction >= 1:
                return self

        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(poly)
        decimate.SetTargetReduction(1 - fraction)
        decimate.PreserveTopologyOff()
        if boundaries:
            decimate.BoundaryVertexDeletionOff()
        else:
            decimate.BoundaryVertexDeletionOn()
        decimate.Update()
        if verbose:
            print("Nr. of pts, input:", poly.GetNumberOfPoints(), end="")
            print(" output:", decimate.GetOutput().GetNumberOfPoints())
        return self.updateMesh(decimate.GetOutput())

    def addGaussNoise(self, sigma):
        """
        Add gaussian noise.

        :param float sigma: sigma is expressed in percent of the diagonal size of actor.
        """
        sz = self.diagonalSize()
        pts = self.coordinates()
        n = len(pts)
        ns = np.random.randn(n, 3) * sigma * sz / 100
        vpts = vtk.vtkPoints()
        vpts.SetNumberOfPoints(n)
        vpts.SetData(numpy_to_vtk(pts + ns, deep=True))
        self.poly.SetPoints(vpts)
        self.poly.GetPoints().Modified()
        return self

    def smoothLaplacian(self, niter=15, relaxfact=0.1, edgeAngle=15, featureAngle=60):
        """
        Adjust mesh point positions using `Laplacian` smoothing.

        :param int niter: number of iterations.
        :param float relaxfact: relaxation factor. Small `relaxfact` and large `niter` are more stable.
        :param float edgeAngle: edge angle to control smoothing along edges (either interior or boundary).
        :param float featureAngle: specifies the feature angle for sharp edge identification.

        .. hint:: |mesh_smoothers.py|_
        """
        poly = self.poly
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(poly)
        cl.Update()
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(cl.GetOutput())
        smoothFilter.SetNumberOfIterations(niter)
        smoothFilter.SetRelaxationFactor(relaxfact)
        smoothFilter.SetEdgeAngle(edgeAngle)
        smoothFilter.SetFeatureAngle(featureAngle)
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.FeatureEdgeSmoothingOn()
        smoothFilter.GenerateErrorScalarsOn()
        smoothFilter.Update()
        return self.updateMesh(smoothFilter.GetOutput())

    def smoothWSinc(self, niter=15, passBand=0.1, edgeAngle=15, featureAngle=60):
        """
        Adjust mesh point positions using the `Windowed Sinc` function interpolation kernel.

        :param int niter: number of iterations.
        :param float passBand: set the passband value for the windowed sinc filter.
        :param float edgeAngle: edge angle to control smoothing along edges (either interior or boundary).
        :param float featureAngle: specifies the feature angle for sharp edge identification.

        .. hint:: |mesh_smoothers| |mesh_smoothers.py|_
        """
        poly = self.poly
        cl = vtk.vtkCleanPolyData()
        cl.SetInputData(poly)
        cl.Update()
        smoothFilter = vtk.vtkWindowedSincPolyDataFilter()
        smoothFilter.SetInputData(cl.GetOutput())
        smoothFilter.SetNumberOfIterations(niter)
        smoothFilter.SetEdgeAngle(edgeAngle)
        smoothFilter.SetFeatureAngle(featureAngle)
        smoothFilter.SetPassBand(passBand)
        smoothFilter.NormalizeCoordinatesOn()
        smoothFilter.NonManifoldSmoothingOn()
        smoothFilter.FeatureEdgeSmoothingOn()
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.Update()
        return self.updateMesh(smoothFilter.GetOutput())


    def fillHoles(self, size=None):
        """Identifies and fills holes in input mesh. 
        Holes are identified by locating boundary edges, linking them together into loops, 
        and then triangulating the resulting loops. 

        :param float size: approximate limit to the size of the hole that can be filled.
        """
        fh = vtk.vtkFillHolesFilter()
        if not size:
            mb = self.maxBoundSize()
            size = mb / 10
        fh.SetHoleSize(size)
        fh.SetInputData(self.poly)
        fh.Update()
        return self.updateMesh(fh.GetOutput())


    def write(self, filename="mesh.vtk", binary=True):
        """
        Write actor's polydata in its current transformation to file.
        """
        import vtkplotter.vtkio as vtkio

        return vtkio.write(self, filename, binary)
    
    
    ########################################################################
    ### stuff that is not returning the input (sometimes modified) actor ###

    def normalAt(self, i):
        """Calculate normal at vertex point `i`."""
        normals = self.polydata(True).GetPointData().GetNormals()
        return np.array(normals.GetTuple(i))

    def normals(self, cells=False):
        """Retrieve vertex normals as a numpy array.

        :params bool cells: if `True` return cell normals.
        """
        if cells:
            vtknormals = self.polydata(True).GetCellData().GetNormals()
        else:
            vtknormals = self.polydata(True).GetPointData().GetNormals()
        return vtk_to_numpy(vtknormals)

    def polydata(self, transformed=True):
        """
        Returns the ``vtkPolyData`` of an ``Actor``.

        .. note:: If ``transformed=True`` returns a copy of polydata that corresponds
            to the current actor's position in space.

        .. hint:: |quadratic_morphing| |quadratic_morphing.py|_
        """
        if not transformed:
            if not self.poly:
                self.poly = self.GetMapper().GetInput()  # cache it for speed
            return self.poly
        M = self.GetMatrix()
        if utils.isIdentity(M):
            if not self.poly:
                self.poly = self.GetMapper().GetInput()  # cache it for speed
            return self.poly
        # if identity return the original polydata
        # otherwise make a copy that corresponds to
        # the actual position in space of the actor
        transform = vtk.vtkTransform()
        transform.SetMatrix(M)
        tp = vtk.vtkTransformPolyDataFilter()
        tp.SetTransform(transform)
        tp.SetInputData(self.poly)
        tp.Update()
        return tp.GetOutput()

    def coordinates(self, transformed=True, copy=True):
        """
        Return the list of vertex coordinates of the input mesh.

        :param bool transformed: if `False` ignore any previous trasformation applied to the mesh.
        :param bool copy: if `False` return the reference to the points 
            so that they can be modified in place.

        .. hint:: |align1.py|_
        """
        poly = self.polydata(transformed)
        if copy:
            return np.array(vtk_to_numpy(poly.GetPoints().GetData()))
        else:
            return vtk_to_numpy(poly.GetPoints().GetData())

    def N(self):
        """Retrieve number of mesh vertices. Shortcut for `actor.NPoints()`."""
        return self.polydata(False).GetNumberOfPoints()

    def NPoints(self):
        """Retrieve number of mesh vertices."""
        return self.polydata(False).GetNumberOfPoints()

    def NCells(self):
        """Retrieve number of mesh cells."""
        return self.polydata(False).GetNumberOfCells()


    def move(self, u_function):
        """
        Move a mesh by using an external function which prescribes the displacement
        at any point in space.
        Useful for manipulating ``dolfin`` meshes.
        """
        if self.mesh:
            self.u = u_function
            delta = [u_function(p) for p in self.mesh.coordinates()]
            movedpts = self.mesh.coordinates() + delta
            self.polydata(False).GetPoints().SetData(numpy_to_vtk(movedpts))
            self.poly.GetPoints().Modified()
            self.u_values = delta
        else:
            colors.printc("Warning: calling move() but actor.mesh is", self.mesh, c=3)
        return self
    

    def getTransform(self):
        """
        Check if ``info['transform']`` exists and returns it.
        Otherwise return current user transformation 
        (where the actor is currently placed).
        """
        if "transform" in self.info.keys():
            T = self.info["transform"]
            return T
        else:
            T = self.GetMatrix()
            tr = vtk.vtkTransform()
            tr.SetMatrix(T)
            return tr

    def setTransform(self, T):
        """
        Transform actor position and orientation wrt to its polygonal mesh,
        which remains unmodified.
        """
        if isinstance(T, vtk.vtkMatrix4x4):
            self.SetUserMatrix(T)
        else:
            try:
                self.SetUserTransform(T)
            except TypeError:
                colors.printc('~time Error in setTransform(): consider transformPolydata() instead.', c=1)
        return self


    def isInside(self, point, tol=0.0001):
        """Return True if point is inside a polydata closed surface."""
        poly = self.polydata(True)
        points = vtk.vtkPoints()
        points.InsertNextPoint(point)
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(points)
        sep = vtk.vtkSelectEnclosedPoints()
        sep.SetTolerance(tol)
        sep.CheckSurfaceOff()
        sep.SetInputData(pointsPolydata)
        sep.SetSurfaceData(poly)
        sep.Update()
        return sep.IsInside(0)

    def insidePoints(self, points, invert=False, tol=1e-05):
        """Return the sublist of points that are inside a polydata closed surface."""
        poly = self.polydata(True)
        # check if the stl file is closed
        
        #featureEdge = vtk.vtkFeatureEdges()
        #featureEdge.FeatureEdgesOff()
        #featureEdge.BoundaryEdgesOn()
        #featureEdge.NonManifoldEdgesOn()
        #featureEdge.SetInputData(poly)
        #featureEdge.Update()
        #openEdges = featureEdge.GetOutput().GetNumberOfCells()
        #if openEdges != 0:
        #    colors.printc("~lightning Warning: polydata is not a closed surface", c=5)

        vpoints = vtk.vtkPoints()
        vpoints.SetData(numpy_to_vtk(points, deep=True))
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(vpoints)
        sep = vtk.vtkSelectEnclosedPoints()
        sep.SetTolerance(tol)
        sep.SetInputData(pointsPolydata)
        sep.SetSurfaceData(poly)
        sep.Update()

        mask1, mask2 = [], []
        for i, p in enumerate(points):
            if sep.IsInside(i):
                mask1.append(p)
            else:
                mask2.append(p)
        if invert:
            return mask2
        else:
            return mask1

    def cellCenters(self):
        """Get the list of cell centers of the mesh surface.

        .. hint:: |delaunay2d| |delaunay2d.py|_
        """
        vcen = vtk.vtkCellCenters()
        vcen.SetInputData(self.polydata(True))
        vcen.Update()
        return vtk_to_numpy(vcen.GetOutput().GetPoints().GetData())

    def boundaries(self, boundaryEdges=True, featureAngle=65, nonManifoldEdges=True):
        """
        Return an ``Actor`` that shows the boundary lines of an input mesh.
        
        :param bool boundaryEdges: Turn on/off the extraction of boundary edges.
        :param float featureAngle: Specify the feature angle for extracting feature edges.
        :param bool nonManifoldEdges: Turn on/off the extraction of non-manifold edges.
        """
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(self.polydata())
        fe.SetBoundaryEdges(boundaryEdges)
        
        if featureAngle:
            fe.FeatureEdgesOn()
            fe.SetFeatureAngle(featureAngle)
        else:
            fe.FeatureEdgesOff()
        
        fe.SetNonManifoldEdges(nonManifoldEdges)
        fe.ColoringOff()
        fe.Update()
        return Actor(fe.GetOutput(), c="p").lw(5)


    def connectedVertices(self, index, returnIds=False):
        """Find all vertices connected to an input vertex specified by its index.

        :param bool returnIds: return vertex IDs instead of vertex coordinates.

        .. hint:: |connVtx| |connVtx.py|_
        """
        mesh = self.polydata()

        cellIdList = vtk.vtkIdList()
        mesh.GetPointCells(index, cellIdList)

        idxs = []
        for i in range(cellIdList.GetNumberOfIds()):
            pointIdList = vtk.vtkIdList()
            mesh.GetCellPoints(cellIdList.GetId(i), pointIdList)
            for j in range(pointIdList.GetNumberOfIds()):
                idj = pointIdList.GetId(j)
                if idj == index:
                    continue
                if idj in idxs:
                    continue
                idxs.append(idj)

        if returnIds:
            return idxs
        else:
            trgp = []
            for i in idxs:
                p = [0, 0, 0]
                mesh.GetPoints().GetPoint(i, p)
                trgp.append(p)
            return np.array(trgp)


    def connectedCells(self, index, returnIds=False):
        """Find all cellls connected to an input vertex specified by its index."""
        
        # Find all cells connected to point index
        dpoly = self.polydata()
        cellPointIds = vtk.vtkIdList()
        dpoly.GetPointCells(index, cellPointIds)
    
        ids = vtk.vtkIdTypeArray()
        ids.SetNumberOfComponents(1)
        rids = []
        for k in range(cellPointIds.GetNumberOfIds()):
            cid = cellPointIds.GetId(k)
            ids.InsertNextValue(cid)
            rids.append(int(cid))
        if returnIds:
            return rids
           
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(ids)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, dpoly)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()
        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(extractSelection.GetOutput())
        gf.Update()
        return Actor(gf.GetOutput()).lw(1)
                

    def intersectWithLine(self, p0, p1):
        """Return the list of points intersecting the actor along segment p0 and p1.

        .. hint:: |spherical_harmonics1.py|_ |spherical_harmonics2.py|_
        """
        if not self.line_locator:
            line_locator = vtk.vtkOBBTree()
            line_locator.SetDataSet(self.polydata(True))
            line_locator.BuildLocator()
            self.line_locator = line_locator

        intersectPoints = vtk.vtkPoints()
        intersection = [0, 0, 0]
        self.line_locator.IntersectWithLine(p0, p1, intersectPoints, None)
        pts = []
        for i in range(intersectPoints.GetNumberOfPoints()):
            intersectPoints.GetPoint(i, intersection)
            pts.append(list(intersection))
        return pts



#################################################
class Assembly(vtk.vtkAssembly, Prop):
    """Group many actors as a single new actor as a ``vtkAssembly``.

    .. hint:: |gyroscope1| |gyroscope1.py|_

         |icon| |icon.py|_
    """

    def __init__(self, actors):

        vtk.vtkAssembly.__init__(self)
        Prop.__init__(self)

        self.actors = actors

        if len(actors) and hasattr(actors[0], "base"):
            self.base = actors[0].base
            self.top = actors[0].top
        else:
            self.base = None
            self.top = None

        for a in actors:
            if a:
                self.AddPart(a)

    def __add__(self, actors):
        if isinstance(actors, list):
            for a in actors:
                self.AddPart(self)
        elif isinstance(actors, vtk.vtkAssembly):
            acts = actors.getActors()
            for a in acts:
                self.AddPart(a)
        else:  # actors=one actor
            self.AddPart(actors)
        return self

    def getActors(self):
        """Unpack a list of ``vtkActor`` objects from a ``vtkAssembly``."""
        cl = vtk.vtkPropCollection()
        self.GetActors(cl)
        self.actors = []
        cl.InitTraversal()
        for i in range(self.GetNumberOfPaths()):
            act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
            if act.GetPickable():
                self.actors.append(act)
        return self.actors

    def getActor(self, i):
        """Get `i-th` ``vtkActor`` object from a ``vtkAssembly``."""
        return self.getActors()[i]

    def diagonalSize(self):
        """Return the maximum diagonal size of the ``Actors`` of the ``Assembly``."""
        szs = [a.diagonalSize() for a in self.actors]
        return np.max(szs)


#################################################
class ImageActor(vtk.vtkImageActor, Prop):
    """
    Derived class of ``vtkImageActor``.

    .. hint:: |rotateImage| |rotateImage.py|_
    """

    def __init__(self):
        vtk.vtkImageActor.__init__(self)
        Prop.__init__(self)

    def alpha(self, a=None):
        """Set/get actor's transparency."""
        if a is not None:
            self.GetProperty().SetOpacity(a)
            return self
        else:
            return self.GetProperty().GetOpacity()
        
    def crop(self, top=None, bottom=None, left=None, right=None):
        """Crop image.
        
        :param float top: fraction to crop from the top margin
        :param float bottom: fraction to crop from the bottom margin
        :param float left: fraction to crop from the left margin
        :param float right: fraction to crop from the right margin
        """
        extractVOI = vtk.vtkExtractVOI()
        extractVOI.SetInputData(self.GetInput())
        extractVOI.IncludeBoundaryOn ()

        d = self.GetInput().GetDimensions()
        bx0, bx1, by0, by1 = 0, d[0]-1, 0, d[1]-1
        if left is not None:   bx0 = int((d[0]-1)*left)
        if right is not None:  bx1 = int((d[0]-1)*(1-right))
        if bottom is not None: by0 = int((d[1]-1)*bottom)
        if top is not None:    by1 = int((d[1]-1)*(1-top))
        extractVOI.SetVOI(bx0, bx1, by0, by1, 0, 0)
        extractVOI.Update()
        img = extractVOI.GetOutput()
        img.SetOrigin(-bx0, -by0, 0)
        self.GetMapper().SetInputData(img)
        self.GetMapper().Modified()
        return self
        

##########################################################################
class Volume(vtk.vtkVolume, Prop):
    """Derived class of ``vtkVolume``.

    :param c: sets colors along the scalar range
    :type c: list, str
    :param alphas: sets transparencies along the scalar range
    :type c: float, list

    .. hint:: if a `list` of values is used for `alphas` this is interpreted
        as a transfer function along the range.

        |read_vti| |read_vti.py|_
    """

    def __init__(self, img, c="blue", alphas=(0.0, 0.4, 0.9, 1)):
        """Derived class of ``vtkVolume``.

        :param c: sets colors along the scalar range
        :type c: list, str
        :param alphas: sets transparencies along the scalar range
        :type c: float, list

        if a `list` of values is used for `alphas` this is interpreted
        as a transfer function along the range.
        """
        vtk.vtkVolume.__init__(self)
        Prop.__init__(self)

        if utils.isSequence(img):
            nx, ny, nz = img.shape
            vtkimg = vtk.vtkImageData()
            vtkimg.SetDimensions(nx, ny, nz)  # range is [0, bins-1]
            vtkimg.AllocateScalars(vtk.VTK_FLOAT, 1)
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        vtkimg.SetScalarComponentFromFloat(ix, iy, iz, 0, img[ix, iy, iz])
            img = vtkimg

        self.image = img

        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.mapper.SetBlendModeToMaximumIntensity()
        self.mapper.UseJitteringOn()
        self.mapper.SetInputData(img)
        colors.printc("scalar range is", np.round(img.GetScalarRange(), 4), c="b", bold=0)
        smin, smax = img.GetScalarRange()
        if smax > 1e10:
            print("~lightning Warning, high scalar range detected:", smax)
            smax = abs(10 * smin) + 0.1
            print("         reset to:", smax)

        colorTransferFunction = vtk.vtkColorTransferFunction()
        if utils.isSequence(c):
            for i, ci in enumerate(c):
                r, g, b = colors.getColor(ci)
                xalpha = smin + (smax - smin) * i / (len(c) - 1)
                colorTransferFunction.AddRGBPoint(xalpha, r, g, b)
                colors.printc('\tcolor at', round(xalpha, 1),
                              '\tset to', colors.getColorName((r, g, b)), c='b', bold=0)
        else:
            # Create transfer mapping scalar value to color
            r, g, b = colors.getColor(c)
            colorTransferFunction.AddRGBPoint(smin, 1.0, 1.0, 1.0)
            colorTransferFunction.AddRGBPoint((smax + smin) / 3, r / 2, g / 2, b / 2)
            colorTransferFunction.AddRGBPoint(smax, 0.0, 0.0, 0.0)

        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        for i, al in enumerate(alphas):
            xalpha = smin + (smax - smin) * i / (len(alphas) - 1)
            # Create transfer mapping scalar value to opacity
            opacityTransferFunction.AddPoint(xalpha, al)
            colors.printc("    alpha at", round(xalpha, 1), "\tset to", al, c="b", bold=0)

        # The property describes how the data will look
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.SetInterpolationTypeToLinear()
        # volumeProperty.SetScalarOpacityUnitDistance(1)

#        volumeMapper.SetBlendModeToComposite()
#        volumeProperty.ShadeOn()
#        volumeProperty.SetAmbient(0.1)
#        volumeProperty.SetDiffuse(1)
#        volumeProperty.SetSpecular(1)
#        volumeProperty.SetSpecularPower(2.0)

        # volume holds the mapper and the property and can be used to position/orient it
        self.SetMapper(self.mapper)
        self.SetProperty(volumeProperty)
    
    def imagedata(self):
        return self.image
    
