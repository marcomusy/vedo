from __future__ import division, print_function
import vtk, os, sys, time
import vtkutils as vu
import vtkcolors as vc

def loadFile(filename, c, alpha, wire, bc, edges, legend, texture,
              smoothing, threshold, connectivity, scaling):
    fl = filename.lower()
    if '.xml' in fl or '.xml.gz' in fl: # Fenics tetrahedral mesh file
        actor = loadXml(filename, c, alpha, wire, bc, edges, legend)
    elif '.pcd' in fl:                  # PCL point-cloud format
        actor = loadPCD(filename, c, alpha, legend)
    elif '.tif' in fl or '.slc' in fl:  # tiff stack or slc
        actor = loadVolume(filename, c, alpha, wire, bc, edges, legend, texture,
                            smoothing, threshold, connectivity, scaling)
    elif '.png' in fl or '.jpg' in fl or '.jpeg' in fl:  # regular image
        actor = load2Dimage(filename, alpha)
    else:
        poly = loadPoly(filename)
        if not poly:
            printc(('Unable to load', filename), c=1)
            return False
        if legend is True: legend = os.path.basename(filename)
        actor = vu.makeActor(poly, c, alpha, wire, bc, edges, legend, texture)
        if '.txt' in fl or '.xyz' in fl: 
            actor.GetProperty().SetPointSize(4)
    return actor
    
def loadDir(mydir, c, alpha, wire, bc, edges, legend, texture,
             smoothing, threshold, connectivity, scaling):
    if not os.path.exists(mydir): 
        printc(('Error in loadDir: Cannot find', mydir), c=1)
        exit(0)
    acts = []
    for ifile in sorted(os.listdir(mydir)):
        loadFile(mydir+'/'+ifile, c, alpha, wire, bc, edges, legend, texture,
                       smoothing, threshold, connectivity, scaling)
    return acts

def loadPoly(filename):
    '''Return a vtkPolyData object, NOT a vtkActor'''
    if not os.path.exists(filename): 
        printc(('Error in loadPoly: Cannot find', filename), c=1)
        return None
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
        printc(('Unable to load', filename), c=1)
        return False
    
    cleanpd = vtk.vtkCleanPolyData()
    vu.setInput(cleanpd, poly)
    cleanpd.Update()
    return cleanpd.GetOutput()


def loadXml(filename, c, alpha, wire, bc, edges, legend):
    '''Reads a Fenics/Dolfin file format'''
    if not os.path.exists(filename): 
        printc(('Error in loadXml: Cannot find', filename), c=1)
        return None
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
    print('..loading',filename)
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
    if vu.vtkMV: 
        mapper.SetInputData(ugrid)
    else:
        mapper.SetInputConnection(ugrid.GetProducerPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToFlat()
    actor.GetProperty().SetColor(vc.getColor(c))
    actor.GetProperty().SetOpacity(alpha/2.)
    #actor.GetProperty().VertexVisibilityOn()
    if edges: actor.GetProperty().EdgeVisibilityOn()
    if wire:  actor.GetProperty().SetRepresentationToWireframe()
    vpts = vtk.vtkPointSource()
    vpts.SetNumberOfPoints(len(coords))
    vpts.Update()
    vpts.GetOutput().SetPoints(points)
    pts_act = vu.makeActor(vpts.GetOutput(), c='b', alpha=alpha)
    pts_act.GetProperty().SetPointSize(3)
    pts_act.GetProperty().SetRepresentationToPoints()
    actor2 = vu.makeAssembly([pts_act, actor])
    setattr(actor2, 'legend', legend)
    if legend is True: 
        setattr(actor2, 'legend', os.path.basename(filename))
    return actor2
 

def loadPCD(filename, c, alpha, legend):
    '''Return vtkActor from Point Cloud file format'''            
    if not os.path.exists(filename): 
        printc(('Error in loadPCD: Cannot find file', filename), c=1)
        return None
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
        printc(('Mismatch in pcd file', expN, len(pts)), 'red')
    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(len(pts))
    src.Update()
    poly = src.GetOutput()
    for i,p in enumerate(pts): poly.GetPoints().SetPoint(i, p)
    if not poly:
        printc(('Unable to load', filename), 'red')
        return False
    actor = vu.makeActor(poly, vc.getColor(c), alpha)
    actor.GetProperty().SetPointSize(4)
    if legend: setattr(actor, 'legend', legend)
    if legend is True: setattr(actor, 'legend', os.path.basename(filename))
    return actor


def loadVolume(filename, c, alpha, wire, bc, edges, legend, texture, 
              smoothing, threshold, connectivity, scaling):
    '''Return vtkActor from a TIFF stack or SLC file'''            
    if not os.path.exists(filename): 
        printc(('Error in loadVolume: Cannot find file', filename), c=1)
        return None
    
    print ('..reading file:', filename)
    if   '.tif' in filename.lower(): 
        reader = vtk.vtkTIFFReader() 
    elif '.slc' in filename.lower(): 
        reader = vtk.vtkSLCReader() 
        if not reader.CanReadFile(filename):
            printc('Sorry bad SLC file '+filename, 1)
            exit(1)
    reader.SetFileName(filename) 
    reader.Update() 
    image = reader.GetOutput()

    if smoothing:
        print ('  gaussian smoothing data with volume_smoothing =',smoothing)
        smImg = vtk.vtkImageGaussianSmooth()
        smImg.SetDimensionality(3)
        vu.setInput(smImg, image)
        smImg.SetStandardDeviations(smoothing, smoothing, smoothing)
        smImg.Update()
        image = smImg.GetOutput()
    
    scrange = image.GetScalarRange()
    if not threshold:
        threshold = (2*scrange[0]+scrange[1])/3.
        a = '  isosurfacing volume with automatic iso_threshold ='
    else: a='  isosurfacing volume with iso_threshold ='
    print (a, round(threshold,2), scrange)
    cf= vtk.vtkContourFilter()
    vu.setInput(cf, image)
    cf.UseScalarTreeOn()
    cf.ComputeScalarsOff()
    cf.SetValue(0, threshold)
    cf.Update()
    
    clp = vtk.vtkCleanPolyData()
    vu.setInput(clp, cf.GetOutput())
    clp.Update()
    image = clp.GetOutput()
    
    if connectivity:
        print ('  applying connectivity filter, select largest region')
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetExtractionModeToLargestRegion() 
        vu.setInput(conn, image)
        conn.Update()
        image = conn.GetOutput()

    if scaling:
        print ('  scaling xyz by factors', scaling)
        tf = vtk.vtkTransformPolyDataFilter()
        vu.setInput(tf, image)
        trans = vtk.vtkTransform()
        trans.Scale(scaling)
        tf.SetTransform(trans)
        tf.Update()
        image = tf.GetOutput()
    return vu.makeActor(image, c, alpha, wire, bc, edges, legend, texture)


def load2Dimage(filename, alpha):
    fl = filename.lower()
    if   '.png' in fl:
        picr = vtk.vtkPNGReader()
    elif '.jpg' in fl or '.jpeg' in fl:
        picr = vtk.vtkJPEGReader()
    else:
        print('file must end with .png or .jpg')
        exit(1)
    picr.SetFileName(filename)
    picr.Update()
    vactor = vtk.vtkImageActor()
    vu.setInput(vactor, picr.GetOutput())
    vactor.SetOpacity(alpha)
    vu.assignConvenienceMethods(vactor, False)    
    vu.assignPhysicsMethods(vactor)    
    return vactor

 
def write(obj, fileoutput):
    '''
    Write 3D object to file.
    Possile extensions are: .vtk, .ply, .obj, .stl, .byu, .vtp
    '''
    fr = fileoutput.lower()
    if   '.vtk' in fr: w = vtk.vtkPolyDataWriter()
    elif '.ply' in fr: w = vtk.vtkPLYWriter()
    elif '.obj' in fr: 
        w = vtk.vtkOBJExporter()
        w.SetFilePrefix(fileoutput.replace('.obj',''))
        printc('Please use write(vp.renderWin)',3)
        w.SetInput(obj)
        w.Update()
        printc("Saved file: "+fileoutput, 'g')
        return
    elif '.stl' in fr: w = vtk.vtkSTLWriter()
    elif '.byu' in fr or '.g' in fr: w = vtk.vtkBYUWriter()
    elif '.vtp' in fr: w = vtk.vtkXMLPolyDataWriter()
    else:
        printc('Unavailable format in file '+fileoutput, c='r')
        exit(1)
    try:
        vu.setInput(w, vu.polydata(obj, True))
        w.SetFileName(fileoutput)
        w.Write()
        printc("Saved file: "+fileoutput, 'g')
    except:
        printc("Error saving: "+fileoutput, 'r')


################################################################### Video
def screenshot(renderWin, filename='screenshot.png'):
    '''Take a screenshot of current rendering window'''
    w2if = vtk.vtkWindowToImageFilter()
    w2if.ShouldRerenderOff()
    w2if.SetInput(renderWin)
    w2if.ReadFrontBufferOff() # read from the back buffer
    w2if.Update()         
    pngwriter = vtk.vtkPNGWriter()
    pngwriter.SetFileName(filename)
    pngwriter.SetInputConnection(w2if.GetOutputPort())
    pngwriter.Write()
 

class Video:
    def __init__(self, renderWindow, name='movie.avi', fps=12, duration=None):
        # if duration is given, fps will be recalculated
        
        import glob  
        self.renderWindow = renderWindow
        self.name = name
        self.duration = duration
        self.fps = float(fps) 
        self.frames = []
        if not os.path.exists('/tmp/vpvid'): os.mkdir('/tmp/vpvid')
        for fl in glob.glob("/tmp/vpvid/*.png"): os.remove(fl)
        printc(("Video", name, "is open..."), 'm')
        
    def addFrame(self):
        fr = '/tmp/vpvid/'+str(len(self.frames))+'.png'
        screenshot(self.renderWindow, fr)
        self.frames.append(fr)
    
    def pause(self, pause=0):
        '''inserts a pause, in seconds'''
        import os
        fr = self.frames[-1]
        n = int(self.fps*pause)
        for i in range(n): 
            fr2='/tmp/vpvid/'+str(len(self.frames))+'.png'
            self.frames.append(fr2)
            os.system("cp -f %s %s" % (fr, fr2))
    
    def close(self):      
        if self.duration:
            _fps = len(self.frames)/float(self.duration)
            printc(("Recalculated video FPS to", round(_fps,3)), 'yellow')
        else: _fps = int(_fps)
        self.name = self.name.split('.')[0]+'.mp4'
        out = os.system("ffmpeg -loglevel panic -y -r " + str(_fps)
                        + " -i /tmp/vpvid/%01d.png "+self.name)
        if out: printc("ffmpeg returning error",1)
        printc(('Video saved as', self.name), 'green')
        return
    


###########################################################################
class ProgressBar: 
    '''Class to print a progress bar with optional text on its right
    
    Usage example:
        import time                        
        pb = ProgressBar(0,400, c='red')
        for i in pb.range():
            time.sleep(.1)
            pb.print('some message') # or pb.print(counts=i)
    ''' 

    def __init__(self, start, stop, step=1, c=None, ETA=True, width=25):
        self.start  = start
        self.stop   = stop
        self.step   = step
        self.color  = c
        self.width  = width
        self.bar    = ""  
        self.percent= 0
        self.clock0 = 0
        self.ETA    = ETA
        self.clock0 = time.time()
        self._update(0)
        self._counts= 0
        self._oldbar= ""
        self._lentxt= 0
        self._range = vu.arange(start, stop, step)
        self._len   = len(self._range)
        
    def print(self, txt='', counts=None):
        if counts: self._update(counts)
        else:      self._update(self._counts + self.step)
        if self.bar != self._oldbar:
            self._oldbar = self.bar
            eraser = [' ']*self._lentxt + ['\b']*self._lentxt 
            eraser = ''.join(eraser)
            if self.ETA:
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
            txt = eta + str(txt) 
            s = self.bar + ' ' + eraser + txt + '\r'
            if self.color: 
                printc(s, c=self.color, end='')
            else: 
                sys.stdout.write(s)
                sys.stdout.flush()
            if self.percent==100: print('')
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
    '''
    Print to terminal in color. 
    
    Available colors:
        black, red, green, yellow, blue, magenta, cyan, white
    Usage example:
        cprint( 'anything', c='red', bold=False, end='' )
        cprint( ['anything', 455.5, vtkObject], 'green')
        cprint(299792.48, c=4) # 4 is blue
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
                        'blue':4, 'magenta':5, 'cyan':6, 'white':7,
                        'k':0, 'r':1, 'g':2, 'y':3,
                        'b':4, 'm':5, 'c':6, 'w':7}
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


    
