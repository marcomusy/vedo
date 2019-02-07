from __future__ import division, print_function
import vtkplotter.docs as docs
import vtk
import numpy as np
import vtkplotter.colors as colors
import vtkplotter.utils as utils
import vtkplotter.settings as settings
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

__doc__="""
Submodule extending the ``vtkActor``, ``vtkVolume`` 
and ``vtkImageActor`` objects functionality.
"""+docs._defs

__all__ = [
    'Actor',
    'Assembly',
    'ImageActor',
    'Volume',
    'mergeActors',
    'isosurface',
]



################################################# functions
def mergeActors(actors, c=None, alpha=1,
                wire=False, bc=None, legend=None, texture=None):
    '''
    Build a new actor formed by the fusion of the polydatas of input objects.
    Similar to Assembly, but in this case the input objects become a single mesh.

    .. hint:: |thinplate_grid| |thinplate_grid.py|_ 
    '''
    polylns = vtk.vtkAppendPolyData()
    for a in actors:
        polylns.AddInputData(a.polydata(True))
    polylns.Update()
    pd = polylns.GetOutput()
    return Actor(pd, c, alpha, wire, bc, legend, texture)

def isosurface(image, smoothing=0, threshold=None, connectivity=False):
    '''Return a ``vtkActor`` isosurface extracted from a ``vtkImageData`` object.
    
    :param float smoothing: gaussian filter to smooth vtkImageData, in units of sigmas
    :param threshold:    value or list of values to draw the isosurface(s)
    :type threshold: float, list
    :param bool connectivity: if True only keeps the largest portion of the polydata
    
    .. hint:: |isosurfaces| |isosurfaces.py|_
    '''
    if smoothing:
        #print('  gaussian smoothing data with volume_smoothing =', smoothing)
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
        for i,t in enumerate(threshold):
            cf.SetValue(i, t)
        cf.Update()
    else:
        if not threshold:            
            threshold = (2*scrange[0]+scrange[1])/3.
        cf.SetValue(0, threshold)
        cf.Update()

    clp = vtk.vtkCleanPolyData()
    clp.SetInputConnection(cf.GetOutputPort())
    clp.Update()
    poly = clp.GetOutput()

    if connectivity:
        #print('applying connectivity filter, select largest region')
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetExtractionModeToLargestRegion()
        conn.SetInputData(poly)
        conn.Update()
        poly = conn.GetOutput()
    
    a = Actor(poly, c=None)
    a.mapper.SetScalarRange(scrange[0], scrange[1])
    return a


################################################# classes
class Prop(object): 
    '''Adds functionality to ``Actor``, ``Assembly``, ``vtkImageData`` and ``vtkVolume`` objects.'''

    def __init__(self):

        self.filename = ''
        self.trail = None
        self.trailPoints = []
        self.trailSegmentSize = 0
        self.trailOffset = None
        self.top = None
        self.base = None
        self.info = dict()
        self._time = 0
        self._legend = None



    def legend(self, txt=None):
        '''Set/get ``Actor`` legend text.

        :param str txt: legend text.

        Size and positions can be modified by setting attributes 
        ``Plotter.legendSize``, ``Plotter.legendBC`` and ``Plotter.legendPos``.

        .. hint:: |fillholes.py|_ 
        ''' 
        if txt:
            self._legend = txt
        else:
            return self._legend
        return self


    def pos(self, p_x=None, y=None, z=None):
        '''Set/Get actor position.'''
        if p_x is None:
            return np.array(self.GetPosition())
        if z is None:  # assume p_x is of the form (x,y,z)
            self.SetPosition(p_x)
        else:
            self.SetPosition(p_x, y, z)
        if self.trail:
            self.updateTrail()
        return self    # return itself to concatenate methods

    def addPos(self, dp_x=None, dy=None, dz=None):
        '''Add vector to current actor position.'''
        p = np.array(self.GetPosition())
        if dz is None: # assume dp_x is of the form (x,y,z)
            self.SetPosition(p + dp_x)
        else:
            self.SetPosition(p + [dp_x, dy, dz])
        if self.trail:
            self.updateTrail()
        return self

    def x(self, position=None):
        '''Set/Get actor position along x axis.'''
        p = self.GetPosition()
        if position is None:
            return p[0]
        self.SetPosition(position, p[1], p[2])
        if self.trail:
            self.updateTrail()
        return self

    def y(self, position=None):
        '''Set/Get actor position along y axis.'''
        p = self.GetPosition()
        if position is None:
            return p[1]
        self.SetPosition(p[0], position, p[2])
        if self.trail:
            self.updateTrail()
        return self

    def z(self, position=None):
        '''Set/Get actor position along z axis.'''
        p = self.GetPosition()
        if position is None:
            return p[2]
        self.SetPosition(p[0], p[1], position)
        if self.trail:
            self.updateTrail()
        return self

    def rotate(self, angle, axis=[1, 0, 0], axis_point=[0, 0, 0], rad=False):
        '''Rotate ``Actor`` around an arbitrary `axis` passing through `axis_point`.'''
        if rad:
            anglerad = angle
        else:
            anglerad = angle/57.29578
        axis = utils.norm(axis)
        a = np.cos(anglerad / 2)
        b, c, d = -axis * np.sin(anglerad / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
        rv = np.dot(R, self.GetPosition()-np.array(axis_point)) + axis_point

        if rad:
            angle *= 57.29578
        # this vtk method only rotates in the origin of the actor:
        self.RotateWXYZ(angle, axis[0], axis[1], axis[2])
        self.SetPosition(rv)
        if self.trail:
            self.updateTrail()
        return self

    def rotateX(self, angle, axis_point=[0, 0, 0], rad=False):
        '''Rotate around x-axis. If angle is in radians set ``rad=True``.'''
        if rad: angle *= 57.29578
        self.RotateX(angle)
        if self.trail:
            self.updateTrail()
        return self

    def rotateY(self, angle, axis_point=[0, 0, 0], rad=False):
        '''Rotate around y-axis. If angle is in radians set ``rad=True``.'''
        if rad: angle *= 57.29578
        self.RotateY(angle)
        if self.trail:
            self.updateTrail()
        return self

    def rotateZ(self, angle, axis_point=[0, 0, 0], rad=False):
        '''Rotate around z-axis. If angle is in radians set ``rad=True``.'''
        if rad: angle *= 57.29578
        self.RotateZ(angle)
        if self.trail:
            self.updateTrail()
        return self

    def orientation(self, newaxis=None, rotation=0, rad=False):
        '''
        Set/Get actor orientation.
        
        :param rotation: If != 0 rotate actor around newaxis.
        :param rad: set to True if angle is in radians.

        .. hint:: |gyroscope2| |gyroscope2.py|_ 
        '''
        if rad: rotation *= 57.29578
        initaxis = utils.norm(self.top - self.base)
        if newaxis is None:
            return initaxis
        newaxis = utils.norm(newaxis)
        pos = np.array(self.GetPosition())
        crossvec = np.cross(initaxis, newaxis)
        angle = np.arccos(np.dot(initaxis, newaxis))
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.Translate(-pos)
        if rotation:
            T.RotateWXYZ(rotation, initaxis)
        T.RotateWXYZ(angle*57.29578, crossvec)
        T.Translate(pos)
        self.SetUserMatrix(T.GetMatrix())
        if self.trail:
            self.updateTrail()
        return self

    def scale(self, s=None):
        '''Set/get actor's scaling factor.
        
        :param s: scaling factor(s).
        :type s: float, list
        
        .. note:: if `s==[sx,sy,sz]` scale differently in the three coordinates.'''
        if s is None:
            return np.array(self.GetScale())
        self.SetScale(s)
        return self  # return itself to concatenate methods

    
    def transform(self, trans):
        '''
        Apply this transformation to the actor.
        
        :param trans: ``vtkTransform`` or ``vtkMatrix4x4`` object.
        '''

        if isinstance(trans, vtk.vtkMatrix4x4):
            tr = vtk.vtkTransform()
            tr.SetMatrix(trans)
        elif isinstance(trans, vtk.vtkTransform):
            tr = trans
        
        self.SetUserTransform(tr)
        return self


    def time(self, t=None):
        '''Set/get actor's absolute time.'''
        if t is None:
            return self._time
        self._time = t
        return self  # return itself to concatenate methods


    def addTrail(self, offset=None, maxlength=None, n=25, c=None, alpha=None, lw=1):
        '''Add a trailing line to actor.

        :param offset: set an offset vector from the object center.
        :param maxlength: length of trailing line in absolute units
        :param n: number of segments to control precision
        :param lw: line width of the trail

        .. hint:: |trail| |trail.py|_ 
        '''
        if maxlength is None:
            maxlength = self.diagonalSize()*20
            if maxlength == 0: maxlength=1

        if self.trail is None:
            pos = self.GetPosition()
            self.trailPoints = [None]*n
            self.trailSegmentSize = maxlength/n
            self.trailOffset = offset

            ppoints = vtk.vtkPoints()  # Generate the polyline
            poly = vtk.vtkPolyData()
            ppoints.SetData(numpy_to_vtk([pos]*n))
            poly.SetPoints(ppoints)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(n)
            for i in range(n):
                lines.InsertCellPoint(i)
            poly.SetPoints(ppoints)
            poly.SetLines(lines)
            mapper = vtk.vtkPolyDataMapper()

            if c is None:
                if hasattr(self, 'GetProperty'):
                    col = self.GetProperty().GetColor()
                else:
                    col = (0.1,0.1,0.1)
            else:
                col = colors.getColor(c)
            al = colors._getAlpha(c)
            if al:
                alpha = al
            if alpha is None:
                alpha = 1
                if hasattr(self, 'GetProperty'):
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
        if self.trailOffset: currentpos += self.trailOffset
        lastpos = self.trailPoints[-1]
        if lastpos is None:  # reset list
            self.trailPoints = [currentpos]*len(self.trailPoints)
            return
        if np.linalg.norm(currentpos-lastpos) < self.trailSegmentSize:
            return

        self.trailPoints.append(currentpos)  # cycle
        self.trailPoints.pop(0)

        tpoly = self.trail.polydata()
        tpoly.GetPoints().SetData(numpy_to_vtk(self.trailPoints))
        return self

    def print(self):
        '''Print  ``Actor``, ``Assembly``, ``Volume`` or ``ImageActor`` infos.'''
        utils.printInfo(self)
        return self


####################################################
# Actor inherits from vtkActor and Prop
class Actor(vtk.vtkActor, Prop):
    '''Build an instance of object ``Actor`` derived from ``vtkActor``.
    
    A ``vtkPolyData`` is expected as input.

    :param c: color in RGB format, hex, symbol or name
    :param float alpha: opacity value
    :param bool wire:  show surface as wireframe
    :param bc: backface color of internal surface
    :param str legend:  optional string
    :param str texture: jpg file name or surface texture name
    :param bool computeNormals: compute point and cell normals at creation
    '''
    def __init__(self, poly=None, c='gold', alpha=1,
                 wire=False, bc=None, legend=None, texture=None, 
                 computeNormals=False):
        vtk.vtkActor.__init__(self)
        Prop.__init__(self)

        self.point_locator = None
        self.cell_locator = None
        self.line_locator = None
        self.poly = None  # cache vtkPolyData and mapper for speed
        self._bfprop = None # backface property holder
        
        self.mapper = vtk.vtkPolyDataMapper()
        self.SetMapper(self.mapper)

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
        # On some vtk versions/platforms points are redered as ugly squares
        prp.RenderPointsAsSpheresOn()

        # check if color string contains a float, in this case ignore alpha
        if alpha is None:
            alpha = 1
        al = colors._getAlpha(c)
        if al:
            alpha = al
        prp.SetOpacity(alpha)

        if c is None:
            self.mapper.ScalarVisibilityOn()
            prp.SetColor(colors.getColor('gold'))
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
            prp.SetColor(1.,1.,1.)
            self.mapper.ScalarVisibilityOff()
            self.texture(texture)
        if bc and alpha==1:  # defines a specific color for the backface
            backProp = vtk.vtkProperty()
            backProp.SetDiffuseColor(colors.getColor(bc))
            backProp.SetOpacity(alpha)
            self.SetBackfaceProperty(backProp)

        if legend:
            self._legend = legend


    def polydata(self, rebuild=True):
        '''
        Returns the ``vtkPolyData`` of an ``Actor``.

        .. note:: If ``rebuild=True`` returns a copy of polydata that corresponds 
            to the current actor's position in space.

        .. hint:: |quadratic_morphing| |quadratic_morphing.py|_ 
        '''
        if not rebuild:
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


    def coordinates(self, rebuild=True, copy=True):
        """
        Return the list of vertex coordinates of the input mesh.

        :param bool rebuild: if `False` ignore any previous trasformation applied to the mesh.
        :param bool copy: if `False` return the reference to the points 
            so that they can be modified in place.

        .. hint:: |align1.py|_ 
        """
        poly = self.polydata(rebuild)
        if copy:
            return np.array(vtk_to_numpy(poly.GetPoints().GetData()))
        else:
            return vtk_to_numpy(poly.GetPoints().GetData())

    def N(self):
        '''Retrieve number of mesh vertices.'''
        return self.polydata(False).GetNumberOfPoints()

    def Ncells(self):
        '''Retrieve number of mesh cells.'''
        return self.polydata(False).GetNumberOfCells()


    def texture(self, name, scale=1, falsecolors=False, mapTo=1):
        '''Assign a texture to actor from image file or predefined texture name.'''
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

        fn = utils.textures_path + name + ".jpg"
        if os.path.exists(name):
            fn = name
        elif not os.path.exists(fn):
            colors.printc('Texture', name, 'not found in', utils.textures_path, c='r')
            colors.printc('Available textures:', c='m', end=' ')
            for ff in os.listdir(utils.textures_path):
                colors.printc(ff.split('.')[0], end=' ', c='m')
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


    def clone(self, rebuild=True):
        '''
        Clone a ``Actor(vtkActor)`` and make an exact copy of it.

        :param rebuild: if `False` ignore any previous trasformation applied to the mesh.

        .. hint:: |carcrash| |carcrash.py|_ 
        '''
        poly = self.polydata(rebuild=rebuild)
        polyCopy = vtk.vtkPolyData()
        polyCopy.DeepCopy(poly)

        cact = Actor()
        cact.poly = polyCopy
        newmapper = vtk.vtkPolyDataMapper()
        newmapper.SetInputData(polyCopy)
        newmapper.SetScalarVisibility(self.mapper.GetScalarVisibility())
        cact.mapper = newmapper
        cact.SetMapper(newmapper)
        pr = vtk.vtkProperty()
        pr.DeepCopy(self.GetProperty())
        cact.SetProperty(pr)
        return cact


    def normalize(self):
        '''
        Shift actor's center of mass at origin and scale its average size to unit.
        '''
        cm = self.centerOfMass()
        coords = self.coordinates()
        if not len(coords):
            return
        pts = coords - cm
        xyz2 = np.sum(pts * pts, axis=0)
        scale = 1/np.sqrt(np.sum(xyz2)/len(pts))
        t = vtk.vtkTransform()
        t.Scale(scale, scale, scale)
        t.Translate(-cm)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(self.poly)
        tf.SetTransform(t)
        tf.Update()
        self.mapper.SetInputData(tf.GetOutput())
        self.mapper.Update()
        self.poly = tf.GetOutput()
        self.Modified()
        return self  # return same obj for concatenation


    def mirror(self, axis='x'):
        '''
        Mirror the actor polydata along one of the cartesian axes.

        .. note::  ``axis='n'``, will flip only mesh normals.

        .. hint:: |mirror| |mirror.py|_ 
        '''
        poly = self.polydata(rebuild=True)
        polyCopy = vtk.vtkPolyData()
        polyCopy.DeepCopy(poly)

        sx, sy, sz = 1, 1, 1
        dx, dy, dz = self.GetPosition()
        if axis.lower() == 'x':
            sx = -1
        elif axis.lower() == 'y':
            sy = -1
        elif axis.lower() == 'z':
            sz = -1
        elif axis.lower() == 'n':
            pass
        else:
            colors.printc(
                "Error in mirror(): mirror must be set to x, y, z or n.", c=1)
            exit()
        if axis != 'n':
            for j in range(polyCopy.GetNumberOfPoints()):
                p = [0, 0, 0]
                polyCopy.GetPoint(j, p)
                polyCopy.GetPoints().SetPoint(j, p[0]*sx-dx*(sx-1),
                                                 p[1]*sy-dy*(sy-1),
                                                 p[2]*sz-dz*(sz-1))
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
                
        polynorm = pdnorm.GetOutput()
        self.mapper.SetInputData(polynorm)
        self.mapper.Update()
        self.poly = polynorm
        self.Modified()       
        return self


    def shrink(self, fraction=0.85):   # N.B. input argument gets modified
        '''Shrink the triangle polydata in the representation of the input mesh.

        Example:
            
            >>> from vtkplotter import load, sphere, show
            >>> pot = load('data/shapes/teapot.vtk').shrink(0.75)
            >>> s = sphere(r=0.2).pos(0,0,-0.5)
            >>> show([pot, s])
            
            |shrink| |shrink.py|_ 
        '''
        poly = self.polydata(True)
        shrink = vtk.vtkShrinkPolyData()
        shrink.SetInputData(poly)
        shrink.SetShrinkFactor(fraction)
        shrink.Update()
        self.mapper.SetInputData(shrink.GetOutput())
        self.mapper.Update()
        self.Modified()
        return self


    def stretch(self, q1, q2):
        '''Stretch actor between points `q1` and `q2`.

        .. hint:: |aspring| |aspring.py|_ 

        .. note:: for ``Actors`` like helices, line, cylinders, cones etc., 
            two attributes ``actor.base``, and ``actor.top`` are already defined.
        '''
        if self.base is None:
            colors.printc('stretch(): Please define vectors \
                          actor.base and actor.top at creation. Exit.', c='r')
            exit(0)

        p1, p2 = self.base, self.top
        q1, q2, z = np.array(q1), np.array(q2), np.array([0, 0, 1])
        plength = np.linalg.norm(p2-p1)
        qlength = np.linalg.norm(q2-q1)
        T = vtk.vtkTransform()
        T.PostMultiply()
        T.Translate(-p1)
        cosa = np.dot(p2-p1, z)/plength
        n = np.cross(p2-p1, z)
        T.RotateWXYZ(np.arccos(cosa)*57.3, n)

        T.Scale(1, 1, qlength/plength)

        cosa = np.dot(q2-q1, z)/qlength
        n = np.cross(q2-q1, z)
        T.RotateWXYZ(-np.arccos(cosa)*57.3, n)
        T.Translate(q1)

        self.SetUserMatrix(T.GetMatrix())
        if self.trail:
            self.updateTrail()
        return self


    def cutPlane(self, origin=(0, 0, 0), normal=(1, 0, 0), showcut=False):
        '''
        Takes a ``vtkActor`` and cuts it with the plane defined by a point and a normal.
        
        :param origin: the cutting plane goes through this point
        :param normal: normal of the cutting plane
        :param showcut: if `True` show the cut off part of the mesh as thin wireframe.

        .. hint:: |trail| |trail.py|_ 
        '''
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
        clipped = clipper.GetOutput()

        self.mapper.SetInputData(clipped)
        self.mapper.Update()
        self.poly = clipped
        self.Modified()

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
        '''
        Cut an ``Actor`` mesh with another ``vtkPolyData`` or ``Actor``.
        
        :param bool invert: if True return cut off part of actor.
        
        .. hint:: |cutWithMesh| |cutWithMesh.py|_ 
        
            |cutAndCap| |cutAndCap.py|_ 
        '''
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

        self.mapper.SetInputData(clipper.GetOutput())
        self.mapper.ScalarVisibilityOff()
        self.mapper.Update()
        self.poly = clipper.GetOutput()
        self.Modified()
        return self


    def cap(self, returnCap=False):
        '''
        Generate a "cap" on a clipped actor, or caps sharp edges.
        
        .. hint:: |cutAndCap| |cutAndCap.py|_ 
        '''
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
            pd = polyapp.GetOutput()
            self.mapper.SetInputData(pd)
            self.mapper.Update()
            self.poly = pd
            self.Modified()
            self.clean()
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
        featureEdge = vtk.vtkFeatureEdges()
        featureEdge.FeatureEdgesOff()
        featureEdge.BoundaryEdgesOn()
        featureEdge.NonManifoldEdgesOn()
        featureEdge.SetInputData(poly)
        featureEdge.Update()
        openEdges = featureEdge.GetOutput().GetNumberOfCells()
        if openEdges != 0:
            colors.printc("Warning: polydata is not a closed surface", c=5)

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
        '''Get the list of cell centers of the mesh surface.

        .. hint:: |delaunay2d| |delaunay2d.py|_ 
        '''
        vcen = vtk.vtkCellCenters()
        vcen.SetInputData(self.polydata(True))
        vcen.Update()
        return vtk_to_numpy(vcen.GetOutput().GetPoints().GetData())


    def clean(self, tol=None):
        '''
        Clean actor's polydata. Can also be used to decimate a mesh if ``tol`` is large. 
        If ``tol=None`` only removes coincident points.

        :param tol: defines how far should be the points from each other in terms of fraction 
            of the bounding box length.

        .. hint:: |moving_least_squares1D| |moving_least_squares1D.py|_ 

            |recosurface| |recosurface.py|_ 
        '''
        poly = self.polydata(False)
        cleanPolyData = vtk.vtkCleanPolyData()
        cleanPolyData.PointMergingOn()
        cleanPolyData.SetInputData(poly)
        if tol:
            cleanPolyData.SetTolerance(tol)
        cleanPolyData.Update()
        self.mapper.SetInputData(cleanPolyData.GetOutput())
        self.mapper.Update()
        self.poly = cleanPolyData.GetOutput()
        self.Modified()
        return self  


    def xbounds(self):
        '''Get the actor bounds `[xmin,xmax]`.'''
        b = self.polydata(True).GetBounds()
        return (b[0], b[1])


    def ybounds(self):
        '''Get the actor bounds `[ymin,ymax]`.'''
        b = self.polydata(True).GetBounds()
        return (b[2], b[3])


    def zbounds(self):
        '''Get the actor bounds `[zmin,zmax]`.'''
        b = self.polydata(True).GetBounds()
        return (b[4], b[5])


    def averageSize(self):
        '''Calculate the average size of a mesh.
        This is the mean of the vertex distances from the center of mass.'''
        cm = self.centerOfMass()
        coords = self.coordinates(copy=False)
        if not len(coords):
            return 0
        s, c = 0.0, 0.0
        n = len(coords)
        step = int(n/10000.)+1
        for i in np.arange(0, n, step):
            s += utils.mag(coords[i] - cm)
            c += 1
        return s/c


    def diagonalSize(self):
        '''Get the length of the diagonal of actor bounding box.'''
        b = self.polydata().GetBounds()
        return np.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)


    def maxBoundSize(self):
        '''Get the maximum dimension in x, y or z of the actor bounding box.'''
        b = self.polydata(True).GetBounds()
        return max(abs(b[1]-b[0]), abs(b[3]-b[2]), abs(b[5]-b[4]))


    def centerOfMass(self):
        '''Get the center of mass of actor.

        .. hint:: |fatlimb| |fatlimb.py|_ 
        '''
        cmf = vtk.vtkCenterOfMass()
        cmf.SetInputData(self.polydata(True))
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)


    def volume(self):
        '''Get the volume occupied by actor.'''
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        return mass.GetVolume()


    def area(self):
        '''Get the surface area of actor.

        .. hint:: |largestregion.py|_ 
        '''
        mass = vtk.vtkMassProperties()
        mass.SetGlobalWarningDisplay(0)
        mass.SetInputData(self.polydata())
        mass.Update()
        return mass.GetSurfaceArea()


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


    def pointColors(self, scalars, cmap='jet', alpha=1, bands=None, vmin=None, vmax=None):
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

        if isinstance(scalars, str): # if a name is passed
            scalars = vtk_to_numpy(poly.GetPointData().GetArray(scalars))

        n = len(scalars)
        useAlpha = False
        if n != poly.GetNumberOfPoints():
            colors.printc('pointColors Error: nr. of scalars != nr. of points', 
                          n, poly.GetNumberOfPoints(), c=1)
        if utils.isSequence(alpha):
            useAlpha = True
            if len(alpha) > n:
                colors.printc('pointColors Error: nr. of scalars < nr. of alpha values', 
                              n, len(alpha), c=1) 
                exit()
        if bands:
            scalars = utils.makeBands(scalars, bands)

        if vmin is None: 
            vmin = np.min(scalars)
        if vmax is None: 
            vmax = np.max(scalars)
    
        lut = vtk.vtkLookupTable() # build the look-up table
        
        if utils.isSequence(cmap):
            sname = 'pointColors_custom'
            lut.SetNumberOfTableValues(len(cmap))
            lut.Build()
            for i,c in enumerate(cmap):
                col = colors.getColor(c)
                if len(col)==4:
                    r,g,b,a = col
                else:
                    r,g,b = col
                    a = colors._getAlpha(c)
                    if not a:
                        a = 1
                lut.SetTableValue(i, r,g,b, a)
                
        elif isinstance(cmap, vtk.vtkLookupTable):
            sname = 'pointColors_lut'
            lut = cmap
            
        else:
            if isinstance(cmap, str):
                sname = 'pointColors_'+cmap
            else:
                sname = 'pointColors'
            lut.SetNumberOfTableValues(512)
            lut.Build()            
            for i in range(512):
                r,g,b = colors.colorMap(i, cmap, 0, 512)
                if useAlpha:
                    idx = int(i/512*len(alpha))
                    lut.SetTableValue(i, r,g,b, alpha[idx])
                else:
                    lut.SetTableValue(i, r,g,b, alpha)

        arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
        arr.SetName(sname)
        self.mapper.SetScalarRange(vmin, vmax)
        self.mapper.SetLookupTable(lut)
        self.mapper.ScalarVisibilityOn()
        poly.GetPointData().SetScalars(arr)
        poly.GetPointData().SetActiveScalars(sname)
        return self


    def cellColors(self, scalars, cmap='jet', alpha=1, bands=None, vmin=None, vmax=None):
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

        if isinstance(scalars, str): # if a name is passed
            scalars = vtk_to_numpy(poly.GetCellData().GetArray(scalars))

        n = len(scalars)
        useAlpha = False
        if n != poly.GetNumberOfCells():
            colors.printc('cellColors Error: nr. of scalars != nr. of cells', 
                          n, poly.GetNumberOfCells(), c=1)
        if utils.isSequence(alpha):
            useAlpha = True
            if len(alpha) > n:
                colors.printc('cellColors Error: nr. of scalars != nr. of alpha values', 
                              n, len(alpha), c=1)
                exit()
        if bands:
            scalars = utils.makeBands(scalars, bands)

        if vmin is None: 
            vmin = np.min(scalars)
        if vmax is None: 
            vmax = np.max(scalars)
            
        lut = vtk.vtkLookupTable() # build the look-up table
        
        if utils.isSequence(cmap):
            sname = 'cellColors_custom'
            lut.SetNumberOfTableValues(len(cmap))
            lut.Build()
            for i,c in enumerate(cmap):
                col = colors.getColor(c)
                if len(col)==4:
                    r,g,b,a = col
                else:
                    r,g,b = col
                    a = colors._getAlpha(c)
                    if not a:
                        a = 1
                lut.SetTableValue(i, r,g,b, a)
                
        elif isinstance(cmap, vtk.vtkLookupTable):
            sname = 'cellColors_lut'
            lut = cmap
            
        else:
            if isinstance(cmap, str):
                sname = 'cellColors_'+cmap
            else:
                sname = 'cellColors'
            lut.SetNumberOfTableValues(512)
            lut.Build()            
            for i in range(512):
                r,g,b = colors.colorMap(i, cmap, 0, 512)
                if useAlpha:
                    idx = int(i/512*len(alpha))
                    lut.SetTableValue(i, r,g,b, alpha[idx])
                else:
                    lut.SetTableValue(i, r,g,b, alpha)

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
            colors.printc('pointScalars Error: Number of scalars != nr. of points',
                          len(scalars), poly.GetNumberOfPoints(), c=1)
            exit()
        arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)
        poly.GetPointData().SetActiveScalars(name)
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
            colors.printc('Number of scalars != nr. of cells', c=1)
            exit()
        arr = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
        arr.SetName(name)
        poly.GetCellData().AddArray(arr)
        poly.GetCellData().SetActiveScalars(name)
        self.mapper.ScalarVisibilityOn()
        return self


    def addPointField(self, vectors, name):
        """
        Add point vector field to the actor's polydata assigning it a name.
        """
        poly = self.polydata(False)
        if len(vectors) != poly.GetNumberOfPoints():
            colors.printc('pointvectors Error: Number of vectors != nr. of points',
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
    

    def scalars(self, name=None):
        """
        Retrieve point or cell scalars using array name or index number.
        If no ``name`` is given return the list of names of existing arrays.

        .. hint:: |mesh_coloring.py|_ 
        """
        poly = self.polydata(False)

        if name is None:
            ncd = poly.GetCellData().GetNumberOfArrays()
            npd = poly.GetPointData().GetNumberOfArrays()
            nfd = poly.GetFieldData().GetNumberOfArrays()
            arrs=[]
            for i in range(npd):
                print(i,'GetPointData',poly.GetPointData().GetArrayName(i))
                arrs.append(poly.GetPointData().GetArrayName(i))
            for i in range(ncd):
                print('GetcellData')
                arrs.append(poly.GetCellData().GetArrayName(i))
            for i in range(nfd):
                print('GetfieldData')
                arrs.append(poly.GetFieldData().GetArrayName(i))
            return arrs
        arr = poly.GetPointData().GetArray(name)
        if arr:
            if isinstance(name, int):
                name = poly.GetPointData().GetArrayName(name)
            poly.GetPointData().SetActiveScalars(name)
            return vtk_to_numpy(arr)
        else:
            if isinstance(name, int):
                name = poly.GetCellData().GetArrayName(name)
            arr = poly.GetCellData().GetArray(name)
            if arr:
                poly.GetCellData().SetActiveScalars(name)
                return vtk_to_numpy(arr)
        return None


    def connectedVertices(self, index, returnIds=False):
        '''Find all vertices connected to an input vertex specified by its index.
        
        :param bool returnIds: return vertex IDs instead of vertex coordinates.
        
        .. hint:: |connVtx| |connVtx.py|_ 
        '''
        mesh = self.polydata()
                
        cellIdList = vtk.vtkIdList()
        mesh.GetPointCells(index, cellIdList)
        
        idxs = []
        for i in range(cellIdList.GetNumberOfIds()):
            pointIdList = vtk.vtkIdList()
            mesh.GetCellPoints(cellIdList.GetId(i), pointIdList)
            for j in range(pointIdList.GetNumberOfIds()):
                idj = pointIdList.GetId(j)
                if idj == index: continue
                if idj in idxs: continue
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
            

    def intersectWithLine(self, p0, p1):
        '''Return the list of points intersecting the actor along segment p0 and p1.

        .. hint:: |spherical_harmonics1.py|_  |spherical_harmonics2.py|_
        '''
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


    def subdivide(self, N=1, method=0, legend=None):
        '''Increase the number of vertices of a surface mesh.

        :param int N: number of subdivisions.
        :param int method: Loop(0), Linear(1), Adaptive(2), Butterfly(3)

        .. hint:: |tutorial_subdivide| |tutorial.py|_ 
        '''
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
            colors.printc('Error in subdivide: unknown method.', c='r')
            exit(1)
        if method != 2:
            sdf.SetNumberOfSubdivisions(N)
        sdf.SetInputData(originalMesh)
        sdf.Update()
        out = sdf.GetOutput()
        if legend is None:
            legend = self.legend
        self.mapper.SetInputData(out)
        self.mapper.Update()
        self.poly = out
        self.Modified()
        return self


    def decimate(self, fraction=0.5, N=None, boundaries=False, verbose=True):
        '''
        Downsample the number of vertices in a mesh.
        
        :param float fraction: the desired target of reduction.
        :param int N: the desired number of final points (**fraction** is recalculated based on it).
        :param bool boundaries: (True), decide whether to leave boundaries untouched or not.

        .. note:: Setting ``fraction=0.1`` leaves 10% of the original nr of vertices.

        .. hint:: |skeletonize| |skeletonize.py|_ 
        '''
        poly = self.polydata(True)
        if N:  # N = desired number of points
            Np = poly.GetNumberOfPoints()
            fraction = float(N)/Np
            if fraction >= 1:
                return self

        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(poly)
        decimate.SetTargetReduction(1-fraction)
        decimate.PreserveTopologyOff()
        if boundaries:
            decimate.BoundaryVertexDeletionOff()
        else:
            decimate.BoundaryVertexDeletionOn()
        decimate.Update()
        if verbose:
            print('Nr. of pts, input:', poly.GetNumberOfPoints(), end='')
            print(' output:', decimate.GetOutput().GetNumberOfPoints())
        self.mapper.SetInputData(decimate.GetOutput())
        self.mapper.Update()
        self.poly = decimate.GetOutput()
        self.Modified()
        return self  # return same obj for concatenation


    def gaussNoise(self, sigma):
        '''
        Add gaussian noise.
        
        :param float sigma: sigma is expressed in percent of the diagonal size of actor.
        '''
        sz = self.diagonalSize()
        pts = self.coordinates()
        n = len(pts)
        ns = np.random.randn(n, 3)*sigma*sz/100
        vpts = vtk.vtkPoints()
        vpts.SetNumberOfPoints(n)
        vpts.SetData(numpy_to_vtk(pts+ns, deep=True))
        self.poly.SetPoints(vpts)
        self.poly.GetPoints().Modified()
        return self


    def point(self, i, p=None):
        '''
        Retrieve/set specific `i-th` point coordinates in mesh. 
        Actor transformation is reset to its mesh position/orientation.

        :param int i: index of vertex point.
        :param list p: new coordinates of mesh point.

        .. warning:: if used in a loop this can slow down the execution by a lot.
        
        .. seealso:: ``actor.points()``
        '''
        if p is None:
            poly = self.polydata(True)
            p = [0, 0, 0]
            poly.GetPoints().GetPoint(i, p)
            return np.array(p)
        else:
            poly = self.polydata(False)
            poly.GetPoints().SetPoint(i, p)
            # reset actor to identity matrix position/rotation:
            self.PokeMatrix(vtk.vtkMatrix4x4())
        return self

    def points(self, pts):
        '''
        Set specific points coordinates in mesh. Input is a python list.
        Actor transformation is reset to its mesh position/orientation.

        :param list pts: new coordinates of mesh vertices.
        '''
        vpts = vtk.vtkPoints()
        vpts.SetData(numpy_to_vtk(pts, deep=True))
        self.poly.SetPoints(vpts)
        # reset actor to identity matrix position/rotation:
        self.PokeMatrix(vtk.vtkMatrix4x4())
        return self


    def normalAt(self, i):
        '''Calculate normal at vertex point `i`.'''
        normals = self.polydata(True).GetPointData().GetNormals()
        return np.array(normals.GetTuple(i))

    
    def normals(self, cells=False):
        '''Retrieve vertex normals as a numpy array.
        
        :params bool cells: if `True` return cell normals.
        '''
        if cells:            
            vtknormals = self.polydata(True).GetCellData().GetNormals()
        else: 
            vtknormals = self.polydata(True).GetPointData().GetNormals()
        return vtk_to_numpy(vtknormals)
    
    
    def computeNormals(self):
        '''Compute cell and vertex normals for the actor's mesh.
        
        .. warning:: mesh is modified, can have a different nr. of vertices.
        '''
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
        self.poly = pdnorm.GetOutput()

        self.mapper.SetInputData(self.poly)
        self.mapper.Modified()
        return self


    def alpha(self, a=None):
        '''Set/get actor's transparency.'''
        if a is None:
            return self.GetProperty().GetOpacity()
        else:
            self.GetProperty().SetOpacity(a)
            bfp = self.GetBackfaceProperty()
            if bfp :
                if a<1:
                    self._bfprop = bfp
                    self.SetBackfaceProperty(None)
                else:
                    self.SetBackfaceProperty(self._bfprop)
            return self


    def wire(self, w=True):
        '''Set actor's representation as wireframe or solid surface.'''
        if w:
            self.GetProperty().SetRepresentationToWireframe()
        else:
            self.GetProperty().SetRepresentationToSurface()
        return self


    def pointSize(self, s=None):
        '''Set/get actor's point size of vertices.'''
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
        '''
        Set/get actor's color.
        If None is passed as input, will use colors from active scalars.
        '''
        if c is False:
            return np.array(self.GetProperty().GetColor())
        elif c is None:
            self.GetMapper().ScalarVisibilityOn()
            return self
        else:
            self.GetMapper().ScalarVisibilityOff()
            self.GetProperty().SetColor(colors.getColor(c))
            return self


    def backColor(self, bc=None):
        '''
        Set/get actor's backface color.
        '''
        backProp = self.GetBackfaceProperty()
        
        if bc is None:
            if backProp:
                return backProp.GetDiffuseColor()
            return None
        
        if self.GetProperty().GetOpacity() < 1:
            colors.printc('backColor(): only active for alpha=1', c='y')
            return self

        if not backProp:
            backProp = vtk.vtkProperty()
            
        backProp.SetDiffuseColor(colors.getColor(bc))
        backProp.SetOpacity(self.GetProperty().GetOpacity())
        self.SetBackfaceProperty(backProp)
        return self


    def lineWidth(self, lw=None):
        '''Set/get width of mesh edges.'''
        if lw is not None:
            if lw==0:
                self.GetProperty().EdgeVisibilityOff()
                return
            self.GetProperty().EdgeVisibilityOn()
            self.GetProperty().SetLineWidth(lw)
        else:
            return self.GetProperty().GetLineWidth()
        return self


#################################################
class Assembly(vtk.vtkAssembly, Prop):
    '''Group many actors as a single new actor as a ``vtkAssembly``.

    .. hint:: |gyroscope1| |gyroscope1.py|_ 
    
         |icon| |icon.py|_ 
    '''

    def __init__(self, actors, legend=None):

        vtk.vtkAssembly.__init__(self)
        Prop.__init__(self)

        self.actors = actors
        self._legend = legend

        if len(actors) and hasattr(actors[0], 'base'):
            self.base = actors[0].base
            self.top = actors[0].top
        else:
            self.base = None
            self.top = None

        for a in actors:
            if a:
                self.AddPart(a)

    def getActors(self):
        '''Unpack a list of ``vtkActor`` objects from a ``vtkAssembly``.'''
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
        '''Get `i-th` ``vtkActor`` object from a ``vtkAssembly``.'''
        return self.getActors()[i]

    def diagonalSize(self):
        '''Return the maximum diagonal size of the ``Actors`` of the ``Assembly``.'''
        szs = [a.diagonalSize() for a in self.actors]
        return np.max(szs)


#################################################
class ImageActor(vtk.vtkImageActor, Prop):
    '''
    Derived class of ``vtkImageActor``.
    '''

    def __init__(self):
        vtk.vtkImageActor.__init__(self)
        Prop.__init__(self)

    def alpha(self, a=None):
        '''Set/get actor's transparency.'''
        if a is not None:
            self.GetProperty().SetOpacity(a)
            return self
        else:
            return self.GetProperty().GetOpacity()


##########################################################################
class Volume(vtk.vtkVolume, Prop):
    '''Derived class of ``vtkVolume``.

    :param c: sets colors along the scalar range
    :type c: list, str
    :param alphas: sets transparencies along the scalar range
    :type c: float, list
    
    .. hint:: if a `list` of values is used for `alphas` this is interpreted
        as a transfer function along the range.
        
        |read_vti| |read_vti.py|_
    '''

    def __init__(self, img, c='blue', alphas=[0.0, 0.4, 0.9, 1]):
        '''Derived class of ``vtkVolume``.
    
        :param c: sets colors along the scalar range
        :type c: list, str
        :param alphas: sets transparencies along the scalar range
        :type c: float, list
        
        if a `list` of values is used for `alphas` this is interpreted
        as a transfer function along the range.
        '''
        vtk.vtkVolume.__init__(self)
        Prop.__init__(self)

        if utils.isSequence(img):
            nx,ny,nz = img.shape
            vtkimg = vtk.vtkImageData()
            vtkimg.SetDimensions(nx,ny,nz) # range is [0, bins-1]
            vtkimg.AllocateScalars(vtk.VTK_FLOAT, 1)
            for ix in range(nx):
               for iy in range(ny):
                   for iz in range(nz):
                       vtkimg.SetScalarComponentFromFloat(ix, iy, iz, 0, img[ix, iy, iz]) 
            img = vtkimg
            
        self.image = img

        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetBlendModeToMaximumIntensity()
        volumeMapper.SetInputData(img)
        colors.printc('scalar range is', np.round(img.GetScalarRange(),4), c='b', bold=0)
        smin, smax = img.GetScalarRange()
        if smax > 1e10:
            print("Warning, high scalar range detected:", smax)
            smax = abs(10*smin)+.1
            print("         reset to:", smax)

        colorTransferFunction = vtk.vtkColorTransferFunction()
        if utils.isSequence(c):
            for i,ci in enumerate(c):
                r, g, b = colors.getColor(ci)
                xalpha = smin+(smax-smin)*i/(len(c)-1)
                colorTransferFunction.AddRGBPoint(xalpha, r,g,b)
                colors.printc('\tcolor at', round(xalpha, 1),
                              '\tset to', colors.getColorName((r,g,b)), c='b', bold=0)
        else:
            # Create transfer mapping scalar value to color
            r, g, b = colors.getColor(c)
            colorTransferFunction.AddRGBPoint(smin, 1.0, 1.0, 1.0)
            colorTransferFunction.AddRGBPoint((smax+smin)/3, r/2, g/2, b/2)
            colorTransferFunction.AddRGBPoint(smax, 0.0, 0.0, 0.0)

        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        for i, al in enumerate(alphas):
            xalpha = smin+(smax-smin)*i/(len(alphas)-1)
            # Create transfer mapping scalar value to opacity
            opacityTransferFunction.AddPoint(xalpha, al)
            colors.printc('\talpha at', round(xalpha, 1),
                          '\tset to', al, c='b', bold=0)

        # The property describes how the data will look
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.SetInterpolationTypeToLinear()
        #volumeProperty.SetScalarOpacityUnitDistance(1)

        # volume holds the mapper and the property and can be used to position/orient it
        self.SetMapper(volumeMapper)
        self.SetProperty(volumeProperty)
