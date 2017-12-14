# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:10:27 2017

@author: marco musy
"""
from __future__ import division, print_function
import os, sys, types
import numpy as np
import colors
import vtk
import time


vtkMV = vtk.vtkVersion().GetVTKMajorVersion() > 5
def setInput(vtkobj, p):
    if vtkMV: vtkobj.SetInputData(p)
    else: vtkobj.SetInput(p)

    
##############################################################################
def makeActor(poly, c='gold', alpha=0.5, 
              wire=False, bc=None, edges=False, legend=None, texture=None):
    '''Return a vtkActor from an input vtkPolyData, optional args:
       c,       color in RGB format, hex, symbol or name
       alpha,   transparency (0=invisible)
       wire,    show surface as wireframe
       bc,      backface color of internal surface
       edges,   show edges as line on top of surface
       legend   optional string
       texture  jpg file name of surface texture, eg. 'metalfloor1'
    '''
    dataset = vtk.vtkPolyDataNormals()
    setInput(dataset, poly)
    dataset.SetFeatureAngle(60.0)
    dataset.ComputePointNormalsOn()
    dataset.ComputeCellNormalsOn()
    dataset.FlipNormalsOff()
    dataset.ConsistencyOn()
    dataset.Update()
    mapper = vtk.vtkPolyDataMapper()

#    mapper.ScalarVisibilityOff()    
#    mapper.ScalarVisibilityOn ()
#    mapper.SetScalarMode(2)
#    mapper.SetColorModeToDefault()
#    mapper.SelectColorArray("Colors")
#    mapper.SetScalarRange(0,255)
#    mapper.SetScalarModeToUsePointData ()
#    mapper.UseLookupTableScalarRangeOff ()

    setInput(mapper, dataset.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    c = colors.getColor(c)
    actor.GetProperty().SetColor(c)
    actor.GetProperty().SetOpacity(alpha)

    actor.GetProperty().SetSpecular(0)
    actor.GetProperty().SetSpecularColor(c)
    actor.GetProperty().SetSpecularPower(1)

    actor.GetProperty().SetAmbient(0)
    actor.GetProperty().SetAmbientColor(c)

    actor.GetProperty().SetDiffuse(1)
    actor.GetProperty().SetDiffuseColor(c)

    if edges: actor.GetProperty().EdgeVisibilityOn()
    if wire: actor.GetProperty().SetRepresentationToWireframe()
    if texture: assignTexture(actor, texture)
    if bc: # defines a specific color for the backface
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor(colors.getColor(bc))
        backProp.SetOpacity(alpha)
        actor.SetBackfaceProperty(backProp)

    assignPhysicsMethods(actor)    
    assignConvenienceMethods(actor, legend)    
    return actor


def makeAssembly(actors, legend=None):
    '''Treat many actors as a single new actor'''
    assembly = vtk.vtkAssembly()
    for a in actors: assembly.AddPart(a)
    setattr(assembly, 'legend', legend) 
#    if hasattr(actors[0], 'legend'): assembly.legend = actors[0].legend
    assignPhysicsMethods(assembly)
    return assembly


def assignConvenienceMethods(actor, legend):
    
    if not hasattr(actor, 'legend'):
        setattr(actor, 'legend', legend)

    def _frotate(self, angle, axis, rad=False): 
        return rotate(self, angle, axis, rad)
    actor.rotate = types.MethodType( _frotate, actor )

    def _frotateX(self, angle, rad=False): 
        return rotate(self, angle, [1,0,0], rad)
    actor.rotateX = types.MethodType( _frotateX, actor )

    def _frotateY(self, angle, rad=False): 
        return rotate(self, angle, [0,1,0], rad)
    actor.rotateY = types.MethodType( _frotateY, actor )

    def _frotateZ(self, angle, rad=False): 
        return rotate(self, angle, [0,0,1], rad)
    actor.rotateZ = types.MethodType( _frotateZ, actor )

    def _fclone(self, c='gold', alpha=1, wire=False, bc=None,
                edges=False, legend=None, texture=None): 
        return clone(self, c, alpha, wire, bc, edges, legend, texture)
    actor.clone = types.MethodType( _fclone, actor )

    def _fpoint(self, i, p=None): 
        poly = getPolyData(self)
        if p is None:
            p = [0,0,0]
            poly.GetPoints().GetPoint(i, p)
            return np.array(p)
        else:
            poly.GetPoints().SetPoint(i, p)
            #actor.GetMapper().Update()
        return 
    actor.point = types.MethodType( _fpoint, actor )

    def _fnormalize(self): return normalize(self)
    actor.normalize = types.MethodType( _fnormalize, actor )

    def _fshrink(self, fraction=0.85): return shrink(self, fraction)
    actor.shrink = types.MethodType( _fshrink, actor )

    def _fcutterw(self): return cutterWidget(self)
    actor.cutterWidget = types.MethodType( _fcutterw, actor )
    
    def _fvisible(self, alpha=1): self.GetProperty().SetOpacity(alpha)
    actor.visible = types.MethodType( _fvisible, actor )
    


def assignPhysicsMethods(actor):
    
    apos = np.array(actor.GetPosition())
    setattr(actor, '_pos',  apos)         # position  
    def _fpos(self, p=None): 
        if p is None: return self._pos
        self.SetPosition(p)
        self._pos = np.array(p)
    actor.pos = types.MethodType( _fpos, actor )

    def _faddpos(self, dp): 
        self.AddPosition(dp)
        self._pos += dp        
    actor.addpos = types.MethodType( _faddpos, actor )

    def _fpx(self, px=None):               # X  
        if px is None: return self._pos[0]
        newp = [px, self._pos[1], self._pos[2]]
        self.SetPosition(newp)
        self._pos = newp
    actor.x = types.MethodType( _fpx, actor )

    def _fpy(self, py=None):               # Y  
        if py is None: return self._pos[1]
        newp = [self._pos[0], py, self._pos[2]]
        self.SetPosition(newp)
        self._pos = newp
    actor.y = types.MethodType( _fpy, actor )

    def _fpz(self, pz=None):               # Z  
        if pz is None: return self._pos[2]
        newp = [self._pos[0], self._pos[1], pz]
        self.SetPosition(newp)
        self._pos = newp
    actor.z = types.MethodType( _fpz, actor )
     
    setattr(actor, '_vel',  np.array([0,0,0]))  # velocity
    def _fvel(self, v=None): 
        if v is None: return self._vel
        self._vel = v
    actor.vel = types.MethodType( _fvel, actor )
    
    def _fvx(self, vx=None):               # VX  
        if vx is None: return self._vel[0]
        newp = [vx, self._vel[1], self._vel[2]]
        self.SetPosition(newp)
        self._vel = newp
    actor.vx = types.MethodType( _fvx, actor )

    def _fvy(self, vy=None):               # VY  
        if vy is None: return self._vel[1]
        newp = [self._vel[0], vy, self._vel[2]]
        self.SetPosition(newp)
        self._vel = newp
    actor.vy = types.MethodType( _fvy, actor )

    def _fvz(self, vz=None):               # VZ  
        if vz is None: return self._vel[2]
        newp = [self._vel[0], self._vel[1], vz]
        self.SetPosition(newp)
        self._vel = newp
    actor.vz = types.MethodType( _fvz, actor )
     
    setattr(actor, '_mass',  1.0)               # mass
    def _fmass(self, m=None): 
        if m is None: return self._mass
        self._mass = m
    actor.mass = types.MethodType( _fmass, actor )

    setattr(actor, '_axis',  np.array([0,0,1]))  # axis
    def _faxis(self, a=None): 
        if a is None: return self._axis
        self._axis = a
    actor.axis = types.MethodType( _faxis, actor )

    setattr(actor, '_omega', 0.0)     # angular velocity
    def _fomega(self, o=None): 
        if o is None: return self._omega
        self._omega = o
    actor.omega = types.MethodType( _fomega, actor )

    def _fmomentum(self): 
        return self._mass * self._vel
    actor.momentum = types.MethodType( _fmomentum, actor )

    def _fgamma(self):                 # Lorentz factor
        v2 = np.sum( self._vel*self._vel )
        return 1./np.sqrt(1. - v2/299792.48**2)
    actor.gamma = types.MethodType( _fgamma, actor )

    return actor ########### >>


######################################################### 
def normalize(actor): 
    cm = getCM(actor)
    coords = getCoordinates(actor)
    if not len(coords) : return
    pts = getCoordinates(actor) - cm
    xyz2 = np.sum(pts * pts, axis=0)
    scale = 1./np.sqrt(np.sum(xyz2)/len(pts))
    actor.SetPosition(0,0,0)
    actor.SetScale(scale, scale, scale)
    poly = getPolyData(actor)
    for i,p in enumerate(pts): 
        poly.GetPoints().SetPoint(i, p)


def clone(actor, c='gold', alpha=0.5, wire=False, bc=None,
          edges=False, legend=None, texture=None): 
    poly = getPolyData(actor)
    if not len(getCoordinates(actor)):
        printc('Limitation: cannot clone textured obj. Returning input.', 'red')
        return actor
    polyCopy = vtk.vtkPolyData()
    polyCopy.DeepCopy(poly)
    a = makeActor(polyCopy, c, alpha, wire, bc, edges, legend, texture)
    return a
    

def rotate(actor, angle, axis, rad=False):
    l = np.linalg.norm(axis)
    if not l: return
    axis /= l
    if rad: angle *= 57.3
    actor.RotateWXYZ(-angle, axis[0], axis[1], axis[2])


def shrink(actor, fraction=0.85):
    poly = getPolyData(actor)
    shrink = vtk.vtkShrinkPolyData()
    setInput(shrink, poly)
    shrink.SetShrinkFactor(fraction)
    shrink.Update()
    mapper = actor.GetMapper()
    setInput(mapper, shrink.GetOutput())
    mapper.Update()


#########################################################
# Useful Functions
######################################################### 
def makePolyData(spoints, addLines=True):
    """Try to workout a polydata from points"""
    sourcePoints = vtk.vtkPoints()
    sourceVertices = vtk.vtkCellArray()
    for pt in spoints:
        if len(pt)==3: #it's 3D!
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], pt[2])
        else:
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], 0)
        sourceVertices.InsertNextCell(1)
        sourceVertices.InsertCellPoint(aid)
    source = vtk.vtkPolyData()
    source.SetPoints(sourcePoints)
    source.SetVerts(sourceVertices)
    if addLines:
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(spoints))
        for i in range(len(spoints)): lines.InsertCellPoint(i)
        source.SetLines(lines)
    source.Update()
    return source


def isInside(poly, point):
    """Return True if point is inside a polydata closed surface"""
    points = vtk.vtkPoints()
    points.InsertNextPoint(point)
    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)
    sep = vtk.vtkSelectEnclosedPoints()
    setInput(sep, pointsPolydata)
    sep.SetSurface(poly)
    sep.Update()
    return sep.IsInside(0)


#################################################################### get stuff
def getPolyData(obj, index=0): 
    '''
    Returns vtkPolyData from an other object (vtkActor, vtkAssembly, int)
    '''
    if   isinstance(obj, list) and len(obj)==1: obj = obj[0]
    if   isinstance(obj, vtk.vtkPolyData): return obj
    elif isinstance(obj, vtk.vtkActor):    return obj.GetMapper().GetInput()
    elif isinstance(obj, vtk.vtkActor2D):  return obj.GetMapper().GetInput()
    elif isinstance(obj, vtk.vtkAssembly):
        cl = vtk.vtkPropCollection()
        obj.GetActors(cl)
        cl.InitTraversal()
        for i in range(index+1):
            act = vtk.vtkActor.SafeDownCast(cl.GetNextProp())
        return act.GetMapper().GetInput()
    printc("Fatal Error in getPolyData(): ", 'red', end='')
    printc(("input is neither a poly nor an actor int or assembly.", [obj]), 'red')
    exit(1)


def getCoordinates(actors):
    """Return a merged list of coordinates of actors or polys"""
    if not isinstance(actors, list): actors = [actors]
    pts = []
    for i in range(len(actors)):
        apoly = getPolyData(actors[i])
        for j in range(apoly.GetNumberOfPoints()):
            p = [0, 0, 0]
            apoly.GetPoint(j, p)
            pts.append(p)
    return np.array(pts)


def getMaxOfBounds(actor):
    '''Get the maximum dimension of the actor bounding box'''
    poly = getPolyData(actor)
    b = poly.GetBounds()
    maxb = max(abs(b[1]-b[0]), abs(b[3]-b[2]), abs(b[5]-b[4]))
    return maxb


def getCM(actor):
    '''Get the Center of Mass of the actor'''
    if vtkMV: #faster
        cmf = vtk.vtkCenterOfMass()
        setInput(cmf, getPolyData(actor))
        #cmf.UseScalarsAsWeightsOff()
        cmf.Update()
        c = cmf.GetCenter()
        return np.array(c)
    else:
        pts = getCoordinates(actor)
        if not len(pts): return np.array([0,0,0])
        return np.mean(pts, axis=0)       


def getVolume(actor):
    '''Get the volume occupied by actor'''
    mass = vtk.vtkMassProperties()
    setInput(mass, getPolyData(actor))
    mass.Update() 
    return mass.GetVolume()


def getArea(actor):
    '''Get the surface area of actor'''
    mass = vtk.vtkMassProperties()
    setInput(mass, getPolyData(actor))
    mass.Update() 
    return mass.GetSurfaceArea()


def assignTexture(actor, name, scale=1, falsecolors=False, mapTo=1):
    '''Assign a texture to actro from file or name in /textures directory'''
    if   mapTo == 1: tmapper = vtk.vtkTextureMapToCylinder()
    elif mapTo == 2: tmapper = vtk.vtkTextureMapToSphere()
    elif mapTo == 3: tmapper = vtk.vtkTextureMapToPlane()
    
    setInput(tmapper, getPolyData(actor))
    if mapTo == 1:  tmapper.PreventSeamOn()
    
    xform = vtk.vtkTransformTextureCoords()
    xform.SetInputConnection(tmapper.GetOutputPort())
    xform.SetScale(scale,scale,scale)
    if mapTo == 1: xform.FlipSOn()
    
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(xform.GetOutputPort())
    
    cdir = os.path.dirname(__file__)     
    fn = cdir + '/textures/'+name+".jpg"
    if os.path.exists(name): 
        fn = name
    elif not os.path.exists(fn):
        printc(('Texture', name, 'not found in', cdir+'/textures'), 'red')
        return 
        
    jpgReader = vtk.vtkJPEGReader()
    jpgReader.SetFileName(fn)
    atext = vtk.vtkTexture()
    atext.RepeatOn()
    atext.EdgeClampOff()
    atext.InterpolateOn()
    if falsecolors: atext.MapColorScalarsThroughLookupTableOn()
    atext.SetInputConnection(jpgReader.GetOutputPort())
    actor.GetProperty().SetColor(1,1,1)
    actor.SetMapper(mapper)
    actor.SetTexture(atext)
    
    
def writeVTK(obj, fileoutput):
    wt = vtk.vtkPolyDataWriter()
    setInput(wt, getPolyData(obj))
    wt.SetFileName(fileoutput)
    wt.Write()
    printc(("Saved vtk file:", fileoutput), 'green')
    

########################################################################
def closestPoint(surf, pt, locator=None, N=None, radius=None):
    """
    Find the closest point on a polydata given an other point.
    If N is given, return a list of N ordered closest points.
    If radius is given, pick only within specified radius.
    """
    polydata = getPolyData(surf)
    trgp  = [0,0,0]
    cid   = vtk.mutable(0)
    dist2 = vtk.mutable(0)
    if not locator:
        if N: locator = vtk.vtkPointLocator()
        else: locator = vtk.vtkCellLocator()
        locator.SetDataSet(polydata)
        locator.BuildLocator()
    if N:
        vtklist = vtk.vtkIdList()
        vmath = vtk.vtkMath()
        locator.FindClosestNPoints(N, pt, vtklist)
        trgp_, trgp, dists2 = [0,0,0], [], []
        for i in range(vtklist.GetNumberOfIds()):
            vi = vtklist.GetId(i)
            polydata.GetPoints().GetPoint(vi, trgp_ )
            trgp.append( trgp_ )
            dists2.append(vmath.Distance2BetweenPoints(trgp_, pt))
        dist2 = dists2
    elif radius:
        cell = vtk.mutable(0)
        r = locator.FindClosestPointWithinRadius(pt, radius, trgp, cell, cid, dist2)
        if not r: 
            trgp = pt
            dist2 = 0.0
    else: 
        subid = vtk.mutable(0)
        locator.FindClosestPoint(pt, trgp, cid, subid, dist2)
    return trgp

def cutterWidget(obj, outputname='clipped.vtk', c=(0.2, 0.2, 1), alpha=1, 
                 wire=False, bc=(0.7, 0.8, 1), edges=False, legend=None):
    '''Pop up a box widget to cut parts of actor. Return largest part.'''
    apd = getPolyData(obj)
    planes  = vtk.vtkPlanes()
    planes.SetBounds(apd.GetBounds())
    clipper = vtk.vtkClipPolyData()
    setInput(clipper, apd)
    clipper.SetClipFunction(planes)
    clipper.InsideOutOn()
    clipper.GenerateClippedOutputOn()
    clipper.SetValue(0.)
    clipper.Update()

    confilter = vtk.vtkPolyDataConnectivityFilter()
    setInput(confilter, clipper.GetOutput())
    confilter.SetExtractionModeToLargestRegion()
    confilter.Update()
    cpd = vtk.vtkCleanPolyData()
    setInput(cpd, confilter.GetOutput())

    cpoly = clipper.GetClippedOutput() # cut away part
    restActor = makeActor(cpoly, c=c, alpha=0.05, wire=1)
    
    actor = makeActor(clipper.GetOutput(), c, alpha, wire, bc, edges, legend)
    actor.GetProperty().SetInterpolationToFlat()

    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    ren.AddActor(actor)
    ren.AddActor(restActor)

    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(800, 800)
    renWin.AddRenderer(ren)
    
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    istyl = vtk.vtkInteractorStyleSwitch()
    istyl.SetCurrentStyleToTrackballCamera()
    iren.SetInteractorStyle(istyl)
    
    def selectPolygons(object, event): object.GetPlanes(planes)
    boxWidget = vtk.vtkBoxWidget()
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1,0,1)
    boxWidget.GetOutlineProperty().SetColor(0.1,0.1,0.1)
    boxWidget.GetOutlineProperty().SetOpacity(0.8)
    boxWidget.SetPlaceFactor(1.05)
    boxWidget.SetInteractor(iren)
    setInput(boxWidget, apd)
    boxWidget.PlaceWidget()
    boxWidget.AddObserver("InteractionEvent", selectPolygons)
    boxWidget.On()
    
    printc(("Press X to save file:", outputname), 'blue')
    def cwkeypress(obj, event):
        if obj.GetKeySym() == "X":
            writeVTK(cpd.GetOutput(), outputname)
        elif obj.GetKeySym() == "Escape": exit()
            
    iren.Initialize()
    iren.AddObserver("KeyPressEvent", cwkeypress)
    iren.Start()
    boxWidget.Off()
    return actor


###########################################################################
class ProgressBar: 
    '''Class to print a progress bar with optional text on its right'''
    # import time                        ### Usage example:
    # pb = ProgressBar(0,400, c='red')
    # for i in pb.range():
    #     time.sleep(.1)
    #     pb.print('some message')       # or pb.print(counts=i) 
    def __init__(self, start, stop, step=1, c=None, ETA=True, width=25):
        self.start  = start
        self.stop   = stop
        self.step   = step
        self.color  = c
        self.width  = width
        self.bar    = ""  
        self.percent= 0
        self._counts= 0
        self._oldbar= ""
        self._lentxt= 0
        self._range = range(start, stop, step) 
        self._len   = len(self._range)
        self.clock0 = 0
        self.ETA    = ETA
        self.clock0 = time.time()
        self._update(0)
        
    def print(self, txt='', counts=None):
        if counts: self._update(counts)
        else:      self._update(self._counts + self.step)
        if self.bar != self._oldbar:
            self._oldbar = self.bar
            eraser = [' ']*self._lentxt + ['\b']*self._lentxt 
            eraser = ''.join(eraser)
            if self.ETA and self._counts>10:
                vel  = self._counts/(time.time() - self.clock0)
                remt =  (self.stop-self._counts)/vel
                if remt>60:
                    mins = int(remt/60)
                    secs = remt - 60*mins
                    mins = str(mins)+'m'
                    secs = str(int(secs))+'s '
                else:
                    mins = ''
                    secs= str(int(remt))+'s '
                vel = str(round(vel,1))
                eta = 'ETA: '+mins+secs+'('+vel+' it/s) '
            else: eta = ''
            txt = eta + txt 
            s = self.bar + ' ' + eraser + txt + '\r'
            if self.color: 
                printc(s, c=self.color, end='')
            else: 
                sys.stdout.write(s)
                sys.stdout.flush()
            if self.percent==100: print ('')
            self._lentxt = len(txt)

    def range(self): return self._range
    def len(self): return self._len
 
    def _update(self, counts):
        if counts < self.start: counts = self.start
        elif counts > self.stop: counts = self.stop
        self._counts = counts
        self.percent = (self._counts - self.start)*100
        self.percent /= self.stop - self.start
        self.percent = int(round(self.percent))
        af = self.width - 2
        nh = int(round( self.percent/100 * af ))
        if   nh==0:  self.bar = "[>%s]" % (' '*(af-1))
        elif nh==af: self.bar = "[%s]" % ('='*af)
        else:        self.bar = "[%s>%s]" % ('='*(nh-1), ' '*(af-nh))
        ps = str(self.percent) + "%"
        self.bar = ' '.join([self.bar, ps])
        

################################################################### color print
def printc(strings, c='black', bold=True, separator=' ', end='\n'):
    '''Print to terminal in color. Available colors:
    black, red, green, yellow, blue, magenta, cyan, white
    E.g.:
    cprint( 'anything', c='red', bold=False, end='' )
    cprint( ['anything', 455.5, vtkObject], 'green', separator='-')
    cprint(299792.48, c=4) #blue
    '''
    if isinstance(strings, tuple): strings = list(strings)
    elif not isinstance(strings, list): strings = [str(strings)]
    txt = str()
    for i,s in enumerate(strings):
        if i == len(strings)-1: separator=''
        txt = txt + str(s) + separator
    
    if _terminal_has_colors:
        try:
            if isinstance(c, int): 
                ncol = c % 8
            else: 
                cols = {'black':0, 'red':1, 'green':2, 'yellow':3, 
                        'blue':4, 'magenta':5, 'cyan':6, 'white':7}
                ncol = cols[c.lower()]
            if bold: seq = "\x1b[1;%dm" % (30+ncol)
            else:    seq = "\x1b[0;%dm" % (30+ncol)
            sys.stdout.write(seq + txt + "\x1b[0m" +end)
            sys.stdout.flush()
        except: print (txt, end=end)
    else:
        print (txt, end=end)
        
def _has_colors(stream):
    if not hasattr(stream, "isatty"): return False
    if not stream.isatty(): return False # auto color only on TTYs
    try:
        import curses
        curses.setupterm()
        return curses.tigetnum("colors") > 2
    except:
        return False
_terminal_has_colors = _has_colors(sys.stdout)

