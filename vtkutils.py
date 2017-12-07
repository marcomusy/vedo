# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:10:27 2017

@author: mmusy
"""
from __future__ import division, print_function
from glob import glob
import os, vtk
from colors import *

vtkMV = vtk.vtkVersion().GetVTKMajorVersion() > 5
def setInput(vtkobj, p):
    if vtkMV: vtkobj.SetInputData(p)
    else: vtkobj.SetInput(p)

####################################### LOADER
def load(filesOrDirs, c='gold', alpha=0.2, 
          wire=False, bc=None, edges=False, legend=True):
    '''Returns a vtkActor from reading a file or directory. 
       Optional args:
       c,     color in RGB format, hex, symbol or name
       alpha, transparency (0=invisible)
       wire,  show surface as wireframe
       bc,    backface color of internal surface
       legend, text to show on legend, if True picks filename.
    '''
    acts = []
    if isinstance(legend, int): legend = bool(legend)
    for fod in sorted(glob(filesOrDirs)):
        if os.path.isfile(fod): 
            a = _loadFile(fod, c, alpha, wire, bc, edges, legend)
            acts.append(a)
        elif os.path.isdir(fod):
            acts = _loadDir(fod, c, alpha, wire, bc, edges, legend)
    if not len(acts):
        print ('Cannot find:', filesOrDirs)
        exit(0) 
    if len(acts) == 1: return acts[0]
    else: return acts


def _loadFile(filename, c, alpha, wire, bc, edges, legend):
    fl = filename.lower()
    if '.xml' in fl or '.xml.gz' in fl: # Fenics tetrahedral mesh file
        actor = _loadXml(filename, c, alpha, wire, bc, edges, legend)
    elif '.pcd' in fl:                  # PCL point-cloud format
        actor = _loadPCD(filename, c, alpha, legend)
    else:
        poly = _loadPoly(filename)
        if not poly:
            print ('Unable to load', filename)
            return False
        if legend is True: legend = os.path.basename(filename)
        actor = makeActor(poly, c, alpha, wire, bc, edges, legend)
        if '.txt' in fl or '.xyz' in fl: 
            actor.GetProperty().SetPointSize(4)
    return actor
    
def _loadDir(mydir, c, alpha, wire, bc, edges, legend):
    acts = []
    for ifile in sorted(os.listdir(mydir)):
        _loadFile(mydir+'/'+ifile, c, alpha, wire, bc, edges)
    return acts

def _loadPoly(filename):
    '''Return a vtkPolyData object, NOT a vtkActor'''
    if not os.path.exists(filename): 
        print ('Cannot find file', filename)
        exit(0)
    fl = filename.lower()
    if   '.vtk' in fl: reader = vtk.vtkPolyDataReader()
    elif '.ply' in fl: reader = vtk.vtkPLYReader()
    elif '.obj' in fl: reader = vtk.vtkOBJReader()
    elif '.stl' in fl: reader = vtk.vtkSTLReader()
    elif '.byu' in fl or '.g' in fl: reader = vtk.vtkBYUReader()
    elif '.vtp' in fl: reader = vtk.vtkXMLPolyDataReader()
    elif '.vts' in fl: reader = vtk.vtkXMLStructuredGridReader()
    elif '.vtu' in fl: reader = vtk.vtkXMLUnstructuredGridReader()
    elif '.txt' in fl: reader = vtk.vtkParticleReader() # (x y z scalar) 
    elif '.xyz' in fl: reader = vtk.vtkParticleReader()
    else: reader = vtk.vtkDataReader()
    reader.SetFileName(filename)
    reader.Update()
    if '.vts' in fl: # structured grid
        gf = vtk.vtkStructuredGridGeometryFilter()
        gf.SetInputConnection(reader.GetOutputPort())
        gf.Update()
        poly = gf.GetOutput()
    elif '.vtu' in fl: # unstructured grid
        gf = vtk.vtkGeometryFilter()
        gf.SetInputConnection(reader.GetOutputPort())
        gf.Update()    
        poly = gf.GetOutput()
    else: poly = reader.GetOutput()
    
    if not poly: 
        print ('Unable to load', filename)
        return False
    
    mergeTriangles = vtk.vtkTriangleFilter()
    setInput(mergeTriangles, poly)
    mergeTriangles.Update()
    poly = mergeTriangles.GetOutput()
    return poly


def _loadXml(filename, c, alpha, wire, bc, edges, legend):
    '''Reads a Fenics/Dolfin file format'''
    if not os.path.exists(filename): 
        print ('Cannot find file', filename)
        exit(0)
    try:
        import xml.etree.ElementTree as et
        if '.gz' in filename:
            import gzip
            inF = gzip.open(filename, 'rb')
            outF = open('/tmp/filename.xml', 'wb')
            outF.write( inF.read() )
            outF.close()
            inF.close()
            tree = et.parse('/tmp/filename.xml')
        else: tree = et.parse(filename)
        coords, connectivity = [], []
        for mesh in tree.getroot():
            for elem in mesh:
                for e in elem.findall('vertex'):
                    x = float(e.get('x'))
                    y = float(e.get('y'))
                    z = float(e.get('z'))
                    coords.append([x,y,z])
                for e in elem.findall('tetrahedron'):
                    v0 = int(e.get('v0'))
                    v1 = int(e.get('v1'))
                    v2 = int(e.get('v2'))
                    v3 = int(e.get('v3'))
                    connectivity.append([v0,v1,v2,v3])
        points = vtk.vtkPoints()
        for p in coords: points.InsertNextPoint(p)

        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        cellArray = vtk.vtkCellArray()
        for itet in range(len(connectivity)):
            tetra = vtk.vtkTetra()
            for k,j in enumerate(connectivity[itet]):
                tetra.GetPointIds().SetId(k, j)
            cellArray.InsertNextCell(tetra)
        ugrid.SetCells(vtk.VTK_TETRA, cellArray)
        # 3D cells are mapped only if they are used by only one cell,
        #  i.e., on the boundary of the data set
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(ugrid.GetProducerPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        actor.GetProperty().SetColor(getColor(c))
        actor.GetProperty().SetOpacity(alpha/2.)
        #actor.GetProperty().VertexVisibilityOn()
        if edges: actor.GetProperty().EdgeVisibilityOn()
        if wire:  actor.GetProperty().SetRepresentationToWireframe()
        vpts = vtk.vtkPointSource()
        vpts.SetNumberOfPoints(len(coords))
        vpts.Update()
        vpts.GetOutput().SetPoints(points)
        pts_act = makeActor(vpts.GetOutput(), c='b', alpha=alpha)
        pts_act.GetProperty().SetPointSize(3)
        pts_act.GetProperty().SetRepresentationToPoints()
        actor2 = makeAssembly([pts_act, actor])
        if legend: setattr(actor2, 'legend', legend)
        if legend is True: 
            setattr(actor2, 'legend', os.path.basename(filename))
        return actor2
    except:
        print ("Cannot parse xml file. Skip.", filename)
        return False
 

def _loadPCD(filename, c, alpha, legend):
    '''Return vtkActor from Point Cloud file format'''            
    if not os.path.exists(filename): 
        print ('Cannot find file', filename)
        exit(0)
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    start = False
    pts = []
    N, expN = 0, 0
    for text in lines:
        if start:
            if N >= expN: break
            l = text.split()
            pts.append([float(l[0]),float(l[1]),float(l[2])])
            N += 1
        if not start and 'POINTS' in text:
            expN= int(text.split()[1])
        if not start and 'DATA ascii' in text:
            start = True
    if expN != N:
        print ('Mismatch in pcd file', expN, len(pts))
    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(len(pts))
    src.Update()
    poly = src.GetOutput()
    for i,p in enumerate(pts): poly.GetPoints().SetPoint(i, p)
    if not poly:
        print ('Unable to load', filename)
        return False
    actor = makeActor(poly, getColor(c), alpha)
    actor.GetProperty().SetPointSize(4)
    if legend: setattr(actor, 'legend', legend)
    if legend is True: setattr(actor, 'legend', os.path.basename(filename))
    return actor
    
##############################################################################
def makeActor(poly, c='gold', alpha=0.5, 
              wire=False, bc=None, edges=False, legend=None):
    '''Return a vtkActor from an input vtkPolyData, optional args:
       c,     color in RGB format, hex, symbol or name
       alpha, transparency (0=invisible)
       wire,  show surface as wireframe
       bc,    backface color of internal surface
       edges, show edges as line on top of surface
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
    setInput(mapper, dataset.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetSpecular(0.025)
    if edges: actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetColor(getColor(c))
    actor.GetProperty().SetOpacity(alpha)
    if wire: actor.GetProperty().SetRepresentationToWireframe()
    if bc: # defines a specific color for the backface
        backProp = vtk.vtkProperty()
        backProp.SetDiffuseColor(getColor(bc))
        backProp.SetOpacity(alpha)
        actor.SetBackfaceProperty(backProp)
    if legend: setattr(actor, 'legend', legend) 
    return actor


def makeAssembly(actors, legend=None):
    '''Treat many actors as a single new actor'''
    assembly = vtk.vtkAssembly()
    for a in actors: assembly.AddPart(a)
    if legend:
        setattr(assembly, 'legend', legend) 
    elif hasattr(actors[0], 'legend'): 
        setattr(assembly, 'legend', actors[0].legend) 
    return assembly


#########################################################
# Useful Functions
######################################################### 
def screenshot(filename='screenshot.png'):
    try:
        import gtk.gdk
        w = gtk.gdk.get_default_root_window().get_screen().get_active_window()
        sz = w.get_size()
        pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, sz[0], sz[1])
        pb = pb.get_from_drawable(w,w.get_colormap(),0,0,0,0, sz[0], sz[1])
        if pb is not None:
            pb.save(filename, "png")
            #print ("Screenshot saved to", filename)
        else: print ("Unable to save the screenshot. Skip.")
    except:
        print ("Unable to take the screenshot. Skip.")


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
    selectEnclosedPoints = vtk.vtkSelectEnclosedPoints()
    selectEnclosedPoints.SetInput(pointsPolydata)
    selectEnclosedPoints.SetSurface(poly)
    selectEnclosedPoints.Update()
    return selectEnclosedPoints.IsInside(0)


def getPolyData(obj, index=0): # get PolyData
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
    print ("Error: input is neither a poly nor an actor int or assembly.", obj)
    return False


def getPoint(i, actor):
    poly = getPolyData(actor)
    p = [0,0,0]
    poly.GetPoints().GetPoint(i, p)
    return np.array(p)


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
    return pts


def getMaxOfBounds(actor):
    poly = getPolyData(actor)
    b = poly.GetBounds()
    maxb = max(abs(b[1]-b[0]), abs(b[3]-b[2]), abs(b[5]-b[4]))
    return maxb


def assignTexture(actor, name, scale=1, falsecolors=False, mapTo=1):

    if   mapTo == 1: tmapper = vtk.vtkTextureMapToCylinder()
    elif mapTo == 2: tmapper = vtk.vtkTextureMapToSphere()
    elif mapTo == 3: tmapper = vtk.vtkTextureMapToPlane()
    
    tmapper.SetInput(getPolyData(actor))
    tmapper.PreventSeamOn()
    
    xform = vtk.vtkTransformTextureCoords()
    xform.SetInputConnection(tmapper.GetOutputPort())
    xform.SetScale(scale,scale,scale)
    
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(xform.GetOutputPort())
    
    cdir = os.path.dirname(__file__)     
    fn = cdir + '/textures/'+name+".jpg"
    if os.path.exists(name): 
        fn = name
    elif not os.path.exists(fn):
        print ('Texture', name, 'not found in', cdir+'/textures')
        return actor
        
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
    return actor
    
    
####################################
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
        

####################################
def writeVTK(obj, fileoutput):
    wt = vtk.vtkPolyDataWriter()
    setInput(wt, getPolyData(obj))
    wt.SetFileName(fileoutput)
    wt.Write()
    print ("Saved vtk file:", fileoutput)
    

####################################
def cutterWidget(obj, outputname='clipped.vtk', c=(0.2, 0.2, 1), alpha=1, 
                 wire=False, bc=(0.7, 0.8, 1), edges=False, legend=None):
                
    apd = getPolyData(obj)
    planes  = vtk.vtkPlanes()
    planes.SetBounds(apd.GetBounds())
    clipper = vtk.vtkClipPolyData()
    setInput(clipper, apd)
    clipper.SetClipFunction(planes)
    clipper.InsideOutOn()
    clipper.GenerateClippedOutputOn()

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
    
    def SelectPolygons(object, event): object.GetPlanes(planes)
    boxWidget = vtk.vtkBoxWidget()
    boxWidget.OutlineCursorWiresOn()
    boxWidget.GetSelectedOutlineProperty().SetColor(1,0,1)
    boxWidget.GetOutlineProperty().SetColor(0.1,0.1,0.1)
    boxWidget.GetOutlineProperty().SetOpacity(0.8)
    boxWidget.SetPlaceFactor(1.05)
    boxWidget.SetInteractor(iren)
    boxWidget.SetInput(apd)
    boxWidget.PlaceWidget()
    boxWidget.AddObserver("InteractionEvent", SelectPolygons)
    boxWidget.On()
    
    print ("Press X to save file:", outputname)
    def cwkeypress(obj, event):
        if obj.GetKeySym() == "X":
            writeVTK(cpd.GetOutput(), outputname)
            
    iren.Initialize()
    iren.AddObserver("KeyPressEvent", cwkeypress)
    iren.Start()
    boxWidget.Off()
    


###################################################################### Video
def openVideo(name='movie.avi', fps=12, duration=None, format="XVID"):
    global _videoname
    global _videoformat
    global _videoduration
    global _fps
    global _frames
    try:
        import cv2 #just check existence
        cv2.__version__
    except:
        print ("openVideo: cv2 not installed? Skip.")
        return
    _videoname = name
    _videoformat = format
    _videoduration = duration
    _fps = float(fps) # if duration is given, will be recalculated
    _frames = []
    if not os.path.exists('/tmp/v'): os.mkdir('/tmp/v')
    for fl in glob("/tmp/v/*.png"): os.remove(fl)
    print ("Video", name, "is open. Press q to continue.")
    
def addFrameVideo():
    global _videoname, _frames
    if not _videoname: return
    fr = '/tmp/v/'+str(len(_frames))+'.png'
    screenshot(fr)
    _frames.append(fr)

def pauseVideo(pause):
    '''insert a pause, in seconds'''
    global _frames
    if not _videoname: return
    fr = _frames[-1]
    n = int(_fps*pause)
    for i in range(n): 
        fr2='/tmp/v/'+str(len(_frames))+'.png'
        _frames.append(fr2)
        os.system("cp -f %s %s" % (fr, fr2))
        
def releaseGif(): #untested
    global _videoname, _frames
    if not _videoname: return
    try: import imageio
    except: 
        print ("release_gif: imageio not installed? Skip.")
        return
    images = []
    for fl in _frames:
        images.append(imageio.imread(fl))
    imageio.mimsave('animation.gif', images)

def releaseVideo():      
    global _videoname, _fps, _videoduration, _videoformat, _frames
    if not _videoname: return
    import cv2
    if _videoduration:
        _fps = len(_frames)/float(_videoduration)
        print ("Recalculated video FPS to", round(_fps,3))
    else: _fps = int(_fps)
    fourcc = cv2.cv.CV_FOURCC(*_videoformat)
    vid = None
    size = None
    for image in _frames:
        if not os.path.exists(image):
            print ('Image not found:', image)
            continue
        img = cv2.imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = cv2.VideoWriter(_videoname, fourcc, _fps, size, True)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = cv2.resize(img, size)
        vid.write(img)
    vid.release()
    print ('Video saved as', _videoname)
    _videoname = False


