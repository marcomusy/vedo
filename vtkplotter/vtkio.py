"""
Submodule to load meshes of different formats, and other I/O functionalities.
"""

from __future__ import division, print_function
import vtk
import os
import sys
import time

import vtkplotter.utils as utils
import vtkplotter.colors as colors
import numpy
from vtkplotter.actors import Actor, Assembly, ImageActor



def load(inputobj, c='gold', alpha=1,
         wire=False, bc=None, legend=True, texture=None,
         smoothing=None, threshold=None, connectivity=False):
    
        ''' Returns a vtkActor from reading a file, directory or vtkPolyData.

            Optional args:

                c,       color in RGB format, hex, symbol or name

                alpha,   transparency (0=invisible)

                wire,    show surface as wireframe      

                bc,      backface color of internal surface      

                legend,  text to show on legend, True picks filename

                texture, any png/jpg file can be used as texture

            For volumetric data (tiff, slc, vti files), an isosurface is calculated:

                smoothing,    gaussian filter to smooth vtkImageData

                threshold,    value to draw the isosurface

                connectivity, if True only keeps the largest portion of the polydata
        '''
        import os
        if isinstance(inputobj, vtk.vtkPolyData):
            a = Actor(inputobj, c, alpha, wire, bc, legend, texture)
            if inputobj and inputobj.GetNumberOfPoints() == 0:
                colors.printc('Warning: actor has zero points.', c=5)
            return a

        acts = []
        if isinstance(legend, int):
            legend = bool(legend)
        if isinstance(inputobj, list):
            flist = inputobj
        else:
            import glob
            flist = sorted(glob.glob(inputobj))
        for fod in flist:
            if os.path.isfile(fod):
                a = loadFile(fod, c, alpha, wire, bc, legend, texture,
                             smoothing, threshold, connectivity)
                acts.append(a)
            elif os.path.isdir(fod):
                acts = loadDir(fod, c, alpha, wire, bc, legend, texture,
                               smoothing, threshold, connectivity)
        if not len(acts):
            colors.printc('Error in load(): cannot find', inputobj, c=1)
            return None

        if len(acts) == 1:
            return acts[0]
        else:
            return acts
    

def loadFile(filename, c, alpha, wire, bc, legend, texture,
             smoothing, threshold, connectivity):
    '''Load a file of various formats.'''
    fl = filename.lower()
    if legend is True:
        legend = os.path.basename(filename)
    if fl.endswith('.xml') or fl.endswith('.xml.gz'):     # Fenics tetrahedral file
        actor = loadDolfin(filename, c, alpha, wire, bc, legend)
    elif fl.endswith('.neutral') or fl.endswith('.neu'):  # neutral tetrahedral file
        actor = loadNeutral(filename, c, alpha, wire, bc, legend)
    elif fl.endswith('.gmsh'):                            # gmesh file
        actor = loadGmesh(filename, c, alpha, wire, bc, legend)
    elif fl.endswith('.pcd'):                             # PCL point-cloud format
        actor = loadPCD(filename, c, alpha, legend)
    elif fl.endswith('.3ds'):                             # PCL point-cloud format
        actor = load3DS(filename, legend)
    elif fl.endswith('.tif') or fl.endswith('.slc') or fl.endswith('.vti'): 
        # tiff stack or slc or vti
        img = loadImageData(filename)
        actor = utils.isosurface(img, c, alpha, wire, bc, legend, texture,
                                 smoothing, threshold, connectivity)
    elif fl.endswith('.png') or fl.endswith('.jpg') or fl.endswith('.jpeg'):
        actor = load2Dimage(filename, alpha)
    else:
        poly = loadPolyData(filename)
        if not poly:
            colors.printc('Unable to load', filename, c=1)
            return None
        actor = Actor(poly, c, alpha, wire, bc, legend, texture)
        if fl.endswith('.txt') or fl.endswith('.xyz'):
            actor.GetProperty().SetPointSize(4)
    actor.filename = filename
    return actor


def loadDir(mydir, c, alpha, wire, bc, legend, texture,
            smoothing, threshold, connectivity):
    '''Load files contained in directory of various formats.'''
    if not os.path.exists(mydir):
        colors.printc('Error in loadDir: Cannot find', mydir, c=1)
        exit(0)
    acts = []
    flist = os.listdir(mydir)
    utils.humansort(flist)
    for ifile in flist:
        a = loadFile(mydir+'/'+ifile, c, alpha, wire, bc, legend, texture,
                     smoothing, threshold, connectivity)
        acts.append(a)
    return acts


def loadPolyData(filename):
    '''Load a file and return a vtkPolyData object (not a vtkActor).'''
    if not os.path.exists(filename):
        colors.printc('Error in loadPolyData: Cannot find', filename, c=1)
        return None
    fl = filename.lower()
    if fl.endswith('.vtk') or fl.endswith('.vtp'):
        reader = vtk.vtkPolyDataReader()
    elif fl.endswith('.ply'):
        reader = vtk.vtkPLYReader()
    elif fl.endswith('.obj'):
        reader = vtk.vtkOBJReader()
    elif fl.endswith('.stl'):
        reader = vtk.vtkSTLReader()
    elif fl.endswith('.byu') or fl.endswith('.g'):
        reader = vtk.vtkBYUReader()
    elif fl.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    elif fl.endswith('.vts'):
        reader = vtk.vtkXMLStructuredGridReader()
    elif fl.endswith('.vtu'):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif fl.endswith('.txt'):
        reader = vtk.vtkParticleReader()  # (x y z scalar)
    elif fl.endswith('.xyz'):
        reader = vtk.vtkParticleReader()
    else:
        reader = vtk.vtkDataReader()
    reader.SetFileName(filename)
    if fl.endswith('.vts'):  # structured grid
        reader.Update()
        gf = vtk.vtkStructuredGridGeometryFilter()
        gf.SetInputConnection(reader.GetOutputPort())
        gf.Update()
        poly = gf.GetOutput()
    elif fl.endswith('.vtu'):  # unstructured grid
        reader.Update()
        gf = vtk.vtkGeometryFilter()
        gf.SetInputConnection(reader.GetOutputPort())
        gf.Update()
        poly = gf.GetOutput()
    else:
        try:
            reader.Update()
            poly = reader.GetOutput()
        except:
            poly = None

    if not poly:
        return None

    cleanpd = vtk.vtkCleanPolyData()
    cleanpd.SetInputData(poly)
    cleanpd.Update()
    return cleanpd.GetOutput()


def loadXMLData(filename):  # not tested
    '''Read any type of vtk data object encoded in XML format.'''
    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def loadStructuredPoints(filename):
    '''Load a vtkStructuredPoints object from file and return a vtkActor.

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/readStructuredPoints.py)

    ![atomp2](https://user-images.githubusercontent.com/32848391/48198462-3b393700-e359-11e8-8272-670bd5f2db42.jpg)
    '''
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    gf = vtk.vtkImageDataGeometryFilter()
    gf.SetInputConnection(reader.GetOutputPort())
    gf.Update()
    return Actor(gf.GetOutput())


def loadStructuredGrid(filename):  # not tested
    '''Load a vtkStructuredGrid object from file and return a vtkActor.'''
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    gf = vtk.vtkStructuredGridGeometryFilter()
    gf.SetInputConnection(reader.GetOutputPort())
    gf.Update()
    return Actor(gf.GetOutput())


def loadUnStructuredGrid(filename):  # not tested
    '''Load a vtkunStructuredGrid object from file and return a vtkActor.'''
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    gf = vtk.vtkUnstructuredGridGeometryFilter()
    gf.SetInputConnection(reader.GetOutputPort())
    gf.Update()
    return Actor(gf.GetOutput())


def loadRectilinearGrid(filename):  # not tested
    '''Load a vtkRectilinearGrid object from file and return a vtkActor.'''
    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(filename)
    reader.Update()
    gf = vtk.vtkRectilinearGridGeometryFilter()
    gf.SetInputConnection(reader.GetOutputPort())
    gf.Update()
    return Actor(gf.GetOutput())


def load3DS(filename, legend):
    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)

    importer = vtk.vtk3DSImporter()
    importer.SetFileName(filename)
    importer.ComputeNormalsOn()
    importer.SetRenderWindow(renWin)
    importer.Update()

    actors = renderer.GetActors()  # vtkActorCollection
    acts = []
    for i in range(actors.GetNumberOfItems()):
        a = actors.GetItemAsObject(i)
        acts.append(a)
    del renWin
    return Assembly(acts, legend=legend)


def loadDolfin(filename, c, alpha, wire, bc, legend):
    '''Reads a Fenics/Dolfin file format'''
    if not os.path.exists(filename):
        colors.printc('Error in loadDolfin: Cannot find', filename, c=1)
        return None
    import xml.etree.ElementTree as et
    if filename.endswith('.gz'):
        import gzip
        inF = gzip.open(filename, 'rb')
        outF = open('/tmp/filename.xml', 'wb')
        outF.write(inF.read())
        outF.close()
        inF.close()
        tree = et.parse('/tmp/filename.xml')
    else:
        tree = et.parse(filename)
    coords, connectivity = [], []
    for mesh in tree.getroot():
        for elem in mesh:
            for e in elem.findall('vertex'):
                x = float(e.get('x'))
                y = float(e.get('y'))
                ez = e.get('z')
                if ez is None:
                    coords.append([x, y])
                else:
                    z = float(ez)
                    coords.append([x, y, z])

            tets = elem.findall('tetrahedron')
            if not len(tets):
                tris = elem.findall('triangle')
                for e in tris:
                    v0 = int(e.get('v0'))
                    v1 = int(e.get('v1'))
                    v2 = int(e.get('v2'))
                    connectivity.append([v0, v1, v2])
            else:
                for e in tets:
                    v0 = int(e.get('v0'))
                    v1 = int(e.get('v1'))
                    v2 = int(e.get('v2'))
                    v3 = int(e.get('v3'))
                    connectivity.append([v0, v1, v2, v3])
                    
    # this builds it as vtkUnstructuredGrid
    # points = vtk.vtkPoints()
    # for p in coords: points.InsertNextPoint(p)
    # import vtkplotter.shapes
    # pts_act = vtkplotter.shapes.points(coords, c=c, r=4, legend=legend)
    # if edges:
    #     # 3D cells are mapped only if they are used by only one cell,
    #     #  i.e., on the boundary of the data set
    #     ugrid = vtk.vtkUnstructuredGrid()
    #     ugrid.SetPoints(points)
    #     cellArray = vtk.vtkCellArray()
    #     for itet in range(len(connectivity)):
    #         tetra = vtk.vtkTetra()
    #         for k,j in enumerate(connectivity[itet]):
    #             tetra.GetPointIds().SetId(k, j)
    #         cellArray.InsertNextCell(tetra)
    #     ugrid.SetCells(vtk.VTK_TETRA, cellArray)
    #     mapper = vtk.vtkDataSetMapper()
    #     mapper.SetInputData(ugrid)
    #     actor = vtk.vtkActor()
    #     actor.SetMapper(mapper)
    #     actor.GetProperty().SetInterpolationToFlat()
    #     actor.GetProperty().SetColor(colors.getColor(c))
    #     actor.GetProperty().SetOpacity(alpha/2.)
    #     if wire: actor.GetProperty().SetRepresentationToWireframe()
    # else:
    #     return pts_act
    # ass = Assembly([pts_act, actor])
    # ass.legend = legend
    # return ass
    
    poly = buildPolyData(coords, connectivity)
    return Actor(poly, c, alpha, wire, bc, legend)


def loadNeutral(filename, c, alpha, wire, bc, legend):
    '''Reads a Neutral tetrahedral file format'''
    if not os.path.exists(filename):
        colors.printc('Error in loadNeutral: Cannot find', filename, c=1)
        return None

    coords, connectivity = convertNeutral2Xml(filename)
    poly = buildPolyData(coords, connectivity, indexOffset=0)
    return Actor(poly, c, alpha, wire, bc, legend)


def loadGmesh(filename, c, alpha, wire, bc, legend):
    '''
    Reads a gmesh file format
    '''
    if not os.path.exists(filename):
        colors.printc('Error in loadGmesh: Cannot find', filename, c=1)
        return None

    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    nnodes = 0
    index_nodes = 0
    for i, line in enumerate(lines):
        if '$Nodes' in line:
            index_nodes = i+1
            nnodes = int(lines[index_nodes])
            break
    node_coords = []
    for i in range(index_nodes+1, index_nodes+1 + nnodes):
        cn = lines[i].split()
        node_coords.append([float(cn[1]), float(cn[2]), float(cn[3])])

    nelements = 0
    index_elements = 0
    for i, line in enumerate(lines):
        if '$Elements' in line:
            index_elements = i+1
            nelements = int(lines[index_elements])
            break
    elements = []
    for i in range(index_elements+1, index_elements+1 + nelements):
        ele = lines[i].split()
        elements.append([int(ele[-3]), int(ele[-2]), int(ele[-1])])

    poly = buildPolyData(node_coords, elements, indexOffset=1)

    return Actor(poly, c, alpha, wire, bc, legend)


def loadPCD(filename, c, alpha, legend):
    '''Return vtkActor from Point Cloud file format'''
    if not os.path.exists(filename):
        colors.printc('Error in loadPCD: Cannot find file', filename, c=1)
        return None
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    start = False
    pts = []
    N, expN = 0, 0
    for text in lines:
        if start:
            if N >= expN:
                break
            l = text.split()
            pts.append([float(l[0]), float(l[1]), float(l[2])])
            N += 1
        if not start and 'POINTS' in text:
            expN = int(text.split()[1])
        if not start and 'DATA ascii' in text:
            start = True
    if expN != N:
        colors.printc('Mismatch in pcd file', expN, len(pts), c='red')
    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(len(pts))
    src.Update()
    poly = src.GetOutput()
    for i, p in enumerate(pts):
        poly.GetPoints().SetPoint(i, p)
    if not poly:
        colors.printc('Unable to load', filename, c='red')
        return False
    actor = Actor(poly, colors.getColor(c), alpha)
    actor.GetProperty().SetPointSize(4)
    if legend is True:
        actor.legend = os.path.basename(filename)
    return actor


def loadImageData(filename, spacing=[]):

    if not os.path.isfile(filename):
        colors.printc('File not found:', filename, c=1)
        return None

    if '.tif' in filename.lower():
        reader = vtk.vtkTIFFReader()
    elif '.slc' in filename.lower():
        reader = vtk.vtkSLCReader()
        if not reader.CanReadFile(filename):
            colors.printc('Sorry bad SLC/TIFF file '+filename, c=1)
            exit(1)
    elif '.vti' in filename.lower():
        reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    image = reader.GetOutput()
    if len(spacing) == 3:
        image.SetSpacing(spacing[0], spacing[1], spacing[2])
    colors.printc('voxel spacing is', image.GetSpacing(), c='b', bold=0)
    return image


###########################################################
def load2Dimage(filename, alpha):
    fl = filename.lower()
    if '.png' in fl:
        picr = vtk.vtkPNGReader()
    elif '.jpg' in fl or '.jpeg' in fl:
        picr = vtk.vtkJPEGReader()
    else:
        print('file must end with .png or .jpg')
        exit(1)
    picr.SetFileName(filename)
    picr.Update()
    vactor = ImageActor() #vtk.vtkImageActor()
    vactor.SetInputData(picr.GetOutput())
    if alpha is None:
        alpha = 1
    vactor.SetOpacity(alpha)
    return vactor


def write(obj, fileoutput):
    '''
    Write 3D object to file.

    Possile extensions are: .vtk, .ply, .obj, .stl, .byu, .vtp, .xyz, .tif
    '''
    if isinstance(obj, Actor):
        obj = obj.polydata(True)
    elif isinstance(obj, vtk.vtkActor):
        obj = obj.GetMapper().GetInput()

    fr = fileoutput.lower()
    if '.vtk' in fr:
        w = vtk.vtkPolyDataWriter()
    elif '.ply' in fr:
        w = vtk.vtkPLYWriter()
    elif '.stl' in fr:
        w = vtk.vtkSTLWriter()
    elif '.vtp' in fr:
        w = vtk.vtkXMLPolyDataWriter()
    elif '.xyz' in fr:
        w = vtk.vtkSimplePointsWriter()
    elif '.byu' in fr or fr.endswith('.g'):
        w = vtk.vtkBYUWriter()
    elif '.obj' in fr:
        obj = obj.polydata(True)
        w = vtk.vtkOBJExporter()
        w.SetFilePrefix(fileoutput.replace('.obj', ''))
        colors.printc('Please use write(vp.renderWin)', c=3)
        w.SetInputData(obj)
        w.Update()
        colors.printc("Saved file: "+fileoutput, c='g')
        return
    elif '.tif' in fr:
        w = vtk.vtkTIFFWriter()
        w.SetFileDimensionality(len(obj.GetDimensions()))
    elif '.vti' in fr:
        w = vtk.vtkXMLImageDataWriter()
    elif '.png' in fr:
        w = vtk.vtkPNGWriter()
    elif '.jpg' in fr:
        w = vtk.vtkJPEGWriter()
    elif '.bmp' in fr:
        w = vtk.vtkBMPWriter()
    else:
        colors.printc('Unknown format', fileoutput, 'file not saved.', c='r')
        return

    try:
        w.SetInputData(obj)
        w.SetFileName(fileoutput)
        w.Write()
        colors.printc("Saved file: "+fileoutput, c='g')
    except:
        colors.printc("Error saving: "+fileoutput, c='r')
    return


########################################################## Video
def screenshot(renderWin, filename='screenshot.png'):
    '''Take a screenshot of current rendering window'''
    w2if = vtk.vtkWindowToImageFilter()
    w2if.ShouldRerenderOff()
    w2if.SetInput(renderWin)
    w2if.ReadFrontBufferOff()  # read from the back buffer
    w2if.Update()
    pngwriter = vtk.vtkPNGWriter()
    pngwriter.SetFileName(filename)
    pngwriter.SetInputConnection(w2if.GetOutputPort())
    pngwriter.Write()


class Video:
    def __init__(self, renderWindow, name='movie.avi', fps=12, duration=None):
        '''Class to generate a video from the specified rendering window.

        Options:

            name, name of the output file

            fps,  frames per second

            duration, total duration of the video. If given, fps will be recalculated.

        [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/makeVideo.py)    
        '''
        import glob
        self.renderWindow = renderWindow
        self.name = name
        self.duration = duration
        self.fps = float(fps)
        self.frames = []
        if not os.path.exists('/tmp/vpvid'):
            os.mkdir('/tmp/vpvid')
        for fl in glob.glob("/tmp/vpvid/*.png"):
            os.remove(fl)
        colors.printc("Video", name, "is open...", c='m')

    def addFrame(self):
        '''Add frame to current video'''
        fr = '/tmp/vpvid/'+str(len(self.frames))+'.png'
        screenshot(self.renderWindow, fr)
        self.frames.append(fr)

    def pause(self, pause=0):
        '''Iinsert a pause, in seconds'''
        import os
        fr = self.frames[-1]
        n = int(self.fps*pause)
        for i in range(n):
            fr2 = '/tmp/vpvid/'+str(len(self.frames))+'.png'
            self.frames.append(fr2)
            os.system("cp -f %s %s" % (fr, fr2))

    def close(self):
        '''Render the video and write to file.'''
        if self.duration:
            _fps = len(self.frames)/float(self.duration)
            colors.printc("Recalculated video FPS to", round(_fps, 3), c='yellow')
        else:
            _fps = int(_fps)
        self.name = self.name.split('.')[0]+'.mp4'
        out = os.system("ffmpeg -loglevel panic -y -r " + str(_fps)
                        + " -i /tmp/vpvid/%01d.png "+self.name)
        if out:
            colors.printc("ffmpeg returning error", c=1)
        colors.printc('Video saved as', self.name, c='green')
        return


###########################################################################
class ProgressBar:
    '''Class to print a progress bar with optional text message.

    Basic usage example:

        import time                        

        pb = ProgressBar(0,400, c='red')

        for i in pb.range():
            time.sleep(.1)            
            pb.print('some message') # or pb.print(counts=i)

    [**Example**](https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/diffusion.py)    
    '''

    def __init__(self, start, stop, step=1, c=None, ETA=True, width=24, char='='):
        self.start = start
        self.stop = stop
        self.step = step
        self.color = c
        self.width = width
        self.char = char
        self.bar = ""
        self.percent = 0
        self.clock0 = 0
        self.ETA = ETA
        self.clock0 = time.time()
        self._remt = 1e10
        self._update(0)
        self._counts = 0
        self._oldbar = ""
        self._lentxt = 0
        self._range = numpy.arange(start, stop, step)
        self._len = len(self._range)

    def print(self, txt='', counts=None):
        if counts:
            self._update(counts)
        else:
            self._update(self._counts + self.step)
        if self.bar != self._oldbar:
            self._oldbar = self.bar
            eraser = [' ']*self._lentxt + ['\b']*self._lentxt
            eraser = ''.join(eraser)
            if self.ETA:
                vel = self._counts/(time.time() - self.clock0)
                self._remt = (self.stop-self._counts)/vel
                if self._remt > 60:
                    mins = int(self._remt/60)
                    secs = self._remt - 60*mins
                    mins = str(mins)+'m'
                    secs = str(int(secs+0.5))+'s '
                else:
                    mins = ''
                    secs = str(int(self._remt+0.5))+'s '
                vel = str(round(vel, 1))
                eta = 'ETA: '+mins+secs+'('+vel+' it/s) '
                if self._remt < 1:
                    dt = time.time() - self.clock0
                    if dt > 60:
                        mins = int(dt/60)
                        secs = dt - 60*mins
                        mins = str(mins)+'m'
                        secs = str(int(secs+0.5))+'s '
                    else:
                        mins = ''
                        secs = str(int(dt+0.5))+'s '
                    eta = 'Elapsed time: '+mins+secs+'('+vel+' it/s)        '
                    txt = ''
            else:
                eta = ''
            txt = eta + str(txt)
            s = self.bar + ' ' + eraser + txt + '\r'
            if self.color:
                colors.printc(s, c=self.color, end='')
            else:
                sys.stdout.write(s)
                sys.stdout.flush()
            if self.percent == 100:
                print('')
            self._lentxt = len(txt)

    def range(self): return self._range

    def len(self): return self._len

    def _update(self, counts):
        if counts < self.start:
            counts = self.start
        elif counts > self.stop:
            counts = self.stop
        self._counts = counts
        self.percent = (self._counts - self.start)*100
        self.percent /= self.stop - self.start
        self.percent = int(round(self.percent))
        af = self.width - 2
        nh = int(round(self.percent/100 * af))
        if nh == 0:
            self.bar = "[>%s]" % (' '*(af-1))
        elif nh == af:
            self.bar = "[%s]" % (self.char*af)
        else:
            self.bar = "[%s>%s]" % (self.char*(nh-1), ' '*(af-nh))
        if self.percent < 100:# and self._remt > 1:
            ps = ' '+str(self.percent) + "%"
        else: 
            ps = ''
        self.bar += ps


def convertNeutral2Xml(infile, outfile=None):

    f = open(infile, 'r')
    lines = f.readlines()
    f.close()

    ncoords = int(lines[0])
    fdolf_coords = []
    for i in range(1, ncoords+1):
        x, y, z = lines[i].split()
        fdolf_coords.append([float(x), float(y), float(z)])

    ntets = int(lines[ncoords+1])
    idolf_tets = []
    for i in range(ncoords+2, ncoords+ntets+2):
        text = lines[i].split()
        v0, v1, v2, v3 = text[1], text[2], text[3], text[4]
        idolf_tets.append([int(v0)-1, int(v1)-1, int(v2)-1, int(v3)-1])

    if outfile:  # write dolfin xml
        outF = open(outfile, 'w')
        outF.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        outF.write('<dolfin xmlns:dolfin="http://www.fenicsproject.org">\n')
        outF.write('  <mesh celltype="tetrahedron" dim="3">\n')

        outF.write('    <vertices size="'+str(ncoords)+'">\n')
        for i in range(ncoords):
            x, y, z = fdolf_coords[i]
            outF.write('      <vertex index="'+str(i)
                       + '" x="'+str(x)+'" y="'+str(y)+'" z="'+str(z)+'"/>\n')
        outF.write('    </vertices>\n')

        outF.write('    <cells size="'+str(ntets)+'">\n')
        for i in range(ntets):
            v0, v1, v2, v3 = idolf_tets[i]
            outF.write('      <tetrahedron index="'+str(i)
                       + '" v0="'+str(v0)+'" v1="'+str(v1)+'" v2="'+str(v2)+'" v3="'+str(v3)+'"/>\n')
        outF.write('    </cells>\n')

        outF.write('  </mesh>\n')
        outF.write('</dolfin>\n')
        outF.close()
    return fdolf_coords, idolf_tets


def buildPolyData(vertices, faces=None, indexOffset=0):
    '''
    Build a vtkPolyData object from a list of vertices
    and the connectivity representing the faces of the polygonal mesh.

    E.g. faces=[[0,1,2], [1,2,3], ...], 
         vertices=[[x1,y1,z1],[x2,y2,z2], ...] 

    Use indexOffset=1 if face numbering starts from 1 instead of 0.
    '''
    sourcePoints = vtk.vtkPoints()
    sourceVertices = vtk.vtkCellArray()
    sourcePolygons = vtk.vtkCellArray()
    for pt in vertices:
        if len(pt) > 2:
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], pt[2])
        else:
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], 0)
        sourceVertices.InsertNextCell(1)
        sourceVertices.InsertCellPoint(aid)
    if faces:
        for f in faces:
            n = len(f)
            if n == 4:
                plg = vtk.vtkTetra()
            elif n == 3:
                plg = vtk.vtkTriangle()
            else:
                plg = vtk.vtkPolygon()
                plg.GetPointIds().SetNumberOfIds(n)
            for i in range(n):
                plg.GetPointIds().SetId(i, f[i] - indexOffset)
            sourcePolygons.InsertNextCell(plg)

    poly = vtk.vtkPolyData()
    poly.SetPoints(sourcePoints)
    poly.SetVerts(sourceVertices)
    if faces:
        poly.SetPolys(sourcePolygons)
    clp = vtk.vtkCleanPolyData()
    clp.SetInputData(poly)
    clp.PointMergingOn()
    clp.Update()
    return clp.GetOutput()


#############
class Button:
    '''
    Build a Button object to be shown in the rendering window
    '''

    def __init__(self, fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle):

        self._status = 0
        self.states = states
        self.colors = c
        self.bcolors = bc
        self.function = fnc
        self.actor = vtk.vtkTextActor()
        self.actor.SetDisplayPosition(pos[0], pos[1])
        self.framewidth = 3
        self.offset = 5
        self.spacer = ' '

        self.textproperty = self.actor.GetTextProperty()
        self.textproperty.SetJustificationToCentered()
        if font.lower() == 'courier':
            self.textproperty.SetFontFamilyToCourier()
        elif font.lower() == 'times':
            self.textproperty.SetFontFamilyToTimes()
        else:
            self.textproperty.SetFontFamilyToArial()
        self.textproperty.SetFontSize(size)
        self.textproperty.SetBackgroundOpacity(alpha)
        self.textproperty.BoldOff()
        if bold:
            self.textproperty.BoldOn()
        self.textproperty.ItalicOff()
        if italic:
            self.textproperty.ItalicOn()
        self.textproperty.ShadowOff()
        self.textproperty.SetOrientation(angle)
        self.showframe = hasattr(self.textproperty, 'FrameOn')
        self.status(0)

    def status(self, s=None):
        '''
        Set/Get the status of the button
        '''
        if s is None:
            return self.states[self._status]
        if isinstance(s, str):
            s = self.states.index(s)
        self._status = s
        self.textproperty.SetLineOffset(self.offset)
        self.actor.SetInput(self.spacer + self.states[s] + self.spacer)
        s = s % len(self.colors)  # to avoid mismatch
        self.textproperty.SetColor(colors.getColor(self.colors[s]))
        bcc = numpy.array(colors.getColor(self.bcolors[s]))
        self.textproperty.SetBackgroundColor(bcc)
        if self.showframe:
            self.textproperty.FrameOn()
            self.textproperty.SetFrameWidth(self.framewidth)
            self.textproperty.SetFrameColor(numpy.sqrt(bcc))

    def switch(self):
        '''
        Change/cycle button status to the next defined status in states list.
        '''
        self._status = (self._status+1) % len(self.states)
        self.status(self._status)


# ############################################################### Events
# mouse event
def _mouseleft(vp, obj, event):

    x, y = vp.interactor.GetEventPosition()
    #print ('mouse at',x,y)

    vp.renderer = obj.FindPokedRenderer(x, y)
    vp.renderWin = obj.GetRenderWindow()
    clickedr = vp.renderers.index(vp.renderer)
    picker = vtk.vtkPropPicker()
    picker.PickProp(x, y, vp.renderer)
    clickedActor = picker.GetActor()

    # check if any button objects were created
    clickedActor2D = picker.GetActor2D()
    if clickedActor2D:
        for bt in vp.buttons:
            if clickedActor2D == bt.actor:
                bt.function()
                break

    clickedActorIsAssembly = False
    if not clickedActor:
        clickedActor = picker.GetAssembly()
        clickedActorIsAssembly = True
    vp.picked3d = picker.GetPickPosition()
    vp.justremoved = None

    if vp.verbose:
        if len(vp.renderers) > 1 or clickedr > 0 and vp.clickedr != clickedr:
            print('Current Renderer:', clickedr, end='')
            print(', nr. of actors =', len(vp.getActors()))

        leg, oldleg = '', ''
        if hasattr(clickedActor, 'legend'):
            leg = clickedActor.legend
        if hasattr(vp.clickedActor, 'legend'):
            oldleg = vp.clickedActor.legend
        # detect if clickin the same obj
        if leg and isinstance(leg, str) and len(leg) and oldleg != leg:
            try:
                indx = str(vp.getActors().index(clickedActor))
            except ValueError:
                indx = None
            try:
                indx = str(vp.actors.index(clickedActor))
            except ValueError:
                indx = None
            try:
                rgb = list(clickedActor.GetProperty().GetColor())
                cn = colors.getColorName(rgb)
                cn = '('+cn+'),'
            except:
                cn = ''
            if indx and isinstance(clickedActor, vtk.vtkAssembly):
                colors.printc('-> assembly', indx+':', clickedActor.legend, cn, end=' ')
            elif indx:
                colors.printc('-> actor', indx+':', leg, cn, dim=1, end=' ')
            if not clickedActorIsAssembly:
                n = clickedActor.GetMapper().GetInput().GetNumberOfPoints()
            else:
                n = vp.getActors(clickedActor)[
                    0].GetMapper().GetInput().GetNumberOfPoints()
            colors.printc('N='+str(n), dim=1, end='')
            px, py, pz = vp.picked3d
            px, py, pz = str(round(px, 1)), str(
                round(py, 1)), str(round(pz, 1))
            colors.printc(', p=('+px+','+py+','+pz+')', dim=1)

    vp.clickedActor = clickedActor
    vp.clickedRenderer = clickedr

    if vp.mouseLeftClickFunction:
        vp.mouseLeftClickFunction(clickedActor)


def _mouseright(vp, obj, event):

    x, y = vp.interactor.GetEventPosition()

    vp.renderer = obj.FindPokedRenderer(x, y)
    vp.renderWin = obj.GetRenderWindow()
    clickedr = vp.renderers.index(vp.renderer)
    picker = vtk.vtkPropPicker()
    picker.PickProp(x, y, vp.renderer)
    clickedActor = picker.GetActor()

    # check if any button objects were created
    clickedActor2D = picker.GetActor2D()
    if clickedActor2D:
        for bt in vp.buttons:
            if clickedActor2D == bt.actor:
                bt.function()
                break

    if not clickedActor:
        clickedActor = picker.GetAssembly()
    vp.picked3d = picker.GetPickPosition()

    vp.clickedActor = clickedActor
    vp.clickedRenderer = clickedr

    if vp.mouseRightClickFunction:
        vp.mouseRightClickFunction(clickedActor)


def _mousemiddle(vp, obj, event):

    x, y = vp.interactor.GetEventPosition()

    vp.renderer = obj.FindPokedRenderer(x, y)
    vp.renderWin = obj.GetRenderWindow()
    clickedr = vp.renderers.index(vp.renderer)
    picker = vtk.vtkPropPicker()
    picker.PickProp(x, y, vp.renderer)
    clickedActor = picker.GetActor()

    # check if any button objects were created
    clickedActor2D = picker.GetActor2D()
    if clickedActor2D:
        for bt in vp.buttons:
            if clickedActor2D == bt.actor:
                bt.function()
                break

    if not clickedActor:
        clickedActor = picker.GetAssembly()
    vp.picked3d = picker.GetPickPosition()

    vp.clickedActor = clickedActor
    vp.clickedRenderer = clickedr

    if vp.mouseMiddleClickFunction:
        vp.mouseMiddleClickFunction(vp.clickedActor)


# keystroke event
def _keypress(vp, obj, event):

    key = obj.GetKeySym()
    #print ('Pressed key:', key, event)

    if key in ["q", "Q", "space", "Return"]:
        vp.interactor.ExitCallback()
        return

    elif key == "e":
        if vp.verbose:
            print("closing window...")
        vp.interactor.GetRenderWindow().Finalize()
        vp.interactor.TerminateApp()
        return

    elif key == "Escape":
        print()
        sys.stdout.flush()
        vp.interactor.TerminateApp()
        vp.interactor.GetRenderWindow().Finalize()
        vp.interactor.TerminateApp()
        sys.exit(0)

    elif key == "m":
        if vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetOpacity(0.02)
            bfp = vp.clickedActor.GetBackfaceProperty()
            if bfp:
                bfp.SetOpacity(0.02)
        else:
            for a in vp.getActors():
                if a.GetPickable():
                    a.GetProperty().SetOpacity(0.02)
                    bfp = a.GetBackfaceProperty()
                    if bfp:
                        bfp.SetOpacity(0.02)

    elif key == "slash":
        if vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetOpacity(1)
            bfp = vp.clickedActor.GetBackfaceProperty()
            if bfp:
                bfp.SetOpacity(1)
        else:
            for a in vp.getActors():
                if a.GetPickable():
                    a.GetProperty().SetOpacity(1)
                    bfp = a.GetBackfaceProperty()
                    if bfp:
                        bfp.SetOpacity(1)

    elif key == "comma":
        if vp.clickedActor in vp.getActors():
            ap = vp.clickedActor.GetProperty()
            ap.SetOpacity(max([ap.GetOpacity()*0.75, 0.01]))
            bfp = vp.clickedActor.GetBackfaceProperty()
            if bfp:
                bfp.SetOpacity(ap.GetOpacity())
        else:
            for a in vp.getActors():
                if a.GetPickable():
                    ap = a.GetProperty()
                    ap.SetOpacity(max([ap.GetOpacity()*0.75, 0.01]))
                    bfp = a.GetBackfaceProperty()
                    if bfp:
                        bfp.SetOpacity(ap.GetOpacity())

    elif key == "period":
        if vp.clickedActor in vp.getActors():
            ap = vp.clickedActor.GetProperty()
            ap.SetOpacity(min([ap.GetOpacity()*1.25, 1.0]))
            bfp = vp.clickedActor.GetBackfaceProperty()
            if bfp:
                bfp.SetOpacity(ap.GetOpacity())
        else:
            for a in vp.getActors():
                if a.GetPickable():
                    ap = a.GetProperty()
                    ap.SetOpacity(min([ap.GetOpacity()*1.25, 1.0]))
                    bfp = a.GetBackfaceProperty()
                    if bfp:
                        bfp.SetOpacity(ap.GetOpacity())

    elif key == "P":
        if vp.clickedActor in vp.getActors():
            acts = [vp.clickedActor]
        else:
            acts = vp.getActors()
        for ia in acts:
            if ia.GetPickable():
                try:
                    ps = ia.GetProperty().GetPointSize()
                    if ps > 1:
                        ia.GetProperty().SetPointSize(ps-1)
                    ia.GetProperty().SetRepresentationToPoints()
                except AttributeError:
                    pass

    elif key == "p":
        if vp.clickedActor in vp.getActors():
            acts = [vp.clickedActor]
        else:
            acts = vp.getActors()
        for ia in acts:
            if ia.GetPickable():
                try:
                    ps = ia.GetProperty().GetPointSize()
                    ia.GetProperty().SetPointSize(ps+2)
                    ia.GetProperty().SetRepresentationToPoints()
                except AttributeError:
                    pass

    elif key == "w":
        if vp.clickedActor and vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetRepresentationToWireframe()
        else:
            for a in vp.getActors():
                if a and a.GetPickable():
                    a.GetProperty().SetRepresentationToWireframe()

    elif key == "r":
        vp.renderer.ResetCamera()

    ### now intercept custom observer ###########################
    if vp.keyPressFunction:
        if key not in ['Shift_L', 'Control_L', 'Super_L', 'Alt_L']:
            if key not in ['Shift_R', 'Control_R', 'Super_R', 'Alt_R']:
                vp.verbose = False
                vp.keyPressFunction(key)
                return

    if key == "S":
        vp.screenshot('screenshot.png')
        colors.printc('Saved rendering window as screenshot.png', c='blue')
        return

    elif key == "C":
        cam = vp.renderer.GetActiveCamera()
        print('\n### Example code to position this vtkCamera:')
        print('vp = vtkplotter.Plotter()\n...')
        print('vp.camera.SetPosition(',   [round(e, 3) for e in cam.GetPosition()],  ')')
        print('vp.camera.SetFocalPoint(', [round(e, 3) for e in cam.GetFocalPoint()], ')')
        print('vp.camera.SetViewUp(',     [round(e, 3) for e in cam.GetViewUp()], ')')
        print('vp.camera.SetDistance(',   round(cam.GetDistance(), 3), ')')
        print('vp.camera.SetClippingRange(', [round(e, 3) for e in cam.GetClippingRange()], ')')
        return

    elif key == "s":
        if vp.clickedActor and vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetRepresentationToSurface()
        else:
            for a in vp.getActors():
                if a and a.GetPickable():
                    a.GetProperty().SetRepresentationToSurface()

    elif key == "V":
        if not(vp.verbose):
            vp._tips()
        vp.verbose = not(vp.verbose)
        print("Verbose: ", vp.verbose)

    elif key in ["1", "KP_End", "KP_1"]:
        if vp.clickedActor and hasattr(vp.clickedActor, 'GetProperty'):
            vp.clickedActor.GetProperty().SetColor(colors.colors1[(vp.icol) % 10])
        else:
            for i, ia in enumerate(vp.getActors()):
                if not ia.GetPickable(): continue
                ia.GetProperty().SetColor(colors.colors1[(i+vp.icol) % 10])
                ia.GetMapper().ScalarVisibilityOff()
        vp.icol += 1
        vp._draw_legend()

    elif key in ["2", "KP_Down", "KP_2"]:
        if vp.clickedActor and hasattr(vp.clickedActor, 'GetProperty'):
            vp.clickedActor.GetProperty().SetColor(colors.colors2[(vp.icol) % 10])
        else:
            for i, ia in enumerate(vp.getActors()):
                if not ia.GetPickable(): continue
                ia.GetProperty().SetColor(colors.colors2[(i+vp.icol) % 10])
                ia.GetMapper().ScalarVisibilityOff()
        vp.icol += 1
        vp._draw_legend()

    elif key in ["3", "KP_Left", "KP_4"]:
        c = colors.getColor('gold')
        acs = vp.getActors()
        alpha = 1./len(acs)
        for ia in acs:
            if not ia.GetPickable(): continue
            ia.GetProperty().SetColor(c)
            ia.GetProperty().SetOpacity(alpha)
            ia.GetMapper().ScalarVisibilityOff()
        vp._draw_legend()


    elif key in ["k", "K"]:
        for a in vp.getActors():
            ptdata = a.polydata().GetPointData()
            cldata = a.polydata().GetCellData()

            arrtypes = dict()
            arrtypes[vtk.VTK_UNSIGNED_CHAR] = 'VTK_UNSIGNED_CHAR'
            arrtypes[vtk.VTK_UNSIGNED_INT] = 'VTK_UNSIGNED_INT'
            arrtypes[vtk.VTK_FLOAT] = 'VTK_FLOAT'
            arrtypes[vtk.VTK_DOUBLE] = 'VTK_DOUBLE'
            foundarr = 0

            if key == 'k':
                for i in range(ptdata.GetNumberOfArrays()):
                    name = ptdata.GetArrayName(i)
                    if name == 'Normals':
                        continue
                    ptdata.SetActiveScalars(name)
                    foundarr = 1
                if not foundarr:
                    print('No vtkArray is associated to points', end='')
                    if hasattr(a, 'legend'):
                        print(' for actor:', a.legend)
                    else:
                        print()

            if key == 'K':
                for i in range(cldata.GetNumberOfArrays()):
                    name = cldata.GetArrayName(i)
                    if name == 'Normals':
                        continue
                    cldata.SetActiveScalars(name)
                    foundarr = 1
                if not foundarr:
                    print('No vtkArray is associated to cells', end='')
                    if hasattr(a, 'legend'):
                        print(' for actor:', a.legend)
                    else:
                        print()

            a.GetMapper().ScalarVisibilityOn()

    elif key == "L":
        if vp.clickedActor in vp.getActors():
            acts = [vp.clickedActor]
        else:
            acts = vp.getActors()
        for ia in acts:
            if not ia.GetPickable(): continue
            try:
                ia.GetProperty().SetRepresentationToSurface()
                ls = ia.GetProperty().GetLineWidth()
                if ls <= 1:
                    ls = 1
                    ia.GetProperty().EdgeVisibilityOff()
                else:
                    ia.GetProperty().EdgeVisibilityOn()
                    ia.GetProperty().SetLineWidth(ls-1)
            except AttributeError:
                pass

    elif key == "l":
        if vp.clickedActor in vp.getActors():
            acts = [vp.clickedActor]
        else:
            acts = vp.getActors()
        for ia in acts:
            if not ia.GetPickable(): continue
            try:
                ia.GetProperty().EdgeVisibilityOn()
                c = ia.GetProperty().GetColor()
                ia.GetProperty().SetEdgeColor(c)
                ls = ia.GetProperty().GetLineWidth()
                ia.GetProperty().SetLineWidth(ls+1)
            except AttributeError:
                pass

    elif key == "n":  # show normals to an actor
        from vtkplotter.analysis import normals as ana_normals
        if vp.clickedActor in vp.getActors():
            acts = [vp.clickedActor]
        else:
            acts = vp.getActors()
        for ia in acts:
            if not ia.GetPickable(): continue
            alpha = ia.GetProperty().GetOpacity()
            c = ia.GetProperty().GetColor()
            a = ana_normals(ia, ratio=1, c=c, alpha=alpha)
            try:
                i = vp.actors.index(ia)
                vp.actors[i] = a
                vp.renderer.RemoveActor(ia)
                vp.interactor.Render()
            except ValueError:
                pass
            vp.render(a)

    elif key == "x":
        if vp.justremoved is None:
            if vp.clickedActor in vp.getActors() or isinstance(vp.clickedActor, vtk.vtkAssembly):
                vp.justremoved = vp.clickedActor
                vp.renderer.RemoveActor(vp.clickedActor)
            if hasattr(vp.clickedActor, 'legend') and vp.clickedActor.legend:
                colors.printc('...removing actor: ' +
                              str(vp.clickedActor.legend)+', press x to put it back')
            else:
                if vp.verbose:
                    colors.printc('Click an actor and press x to toggle it.', c=5)
        else:
            vp.renderer.AddActor(vp.justremoved)
            vp.renderer.Render()
            vp.justremoved = None
        vp._draw_legend()

    elif key == "X":
        if len(vp.actors):
            vp.clickedActor = vp.actors[-1]
        if vp.clickedActor:
            if not vp.cutterWidget:
                vp.cutterWidget = vp.addCutterTool(vp.clickedActor)
            else:
                fname = 'clipped.vtk'
                confilter = vtk.vtkPolyDataConnectivityFilter()
                confilter.SetInputData(vp.clickedActor.polydata(True))
                confilter.SetExtractionModeToLargestRegion()
                confilter.Update()
                cpd = vtk.vtkCleanPolyData()
                cpd.SetInputData(confilter.GetOutput())
                cpd.Update()
                w = vtk.vtkPolyDataWriter()
                w.SetInputData(cpd.GetOutput())
                w.SetFileName(fname)
                w.Write()
                colors.printc("  -> Saved file:", fname, c='m')
                vp.cutterWidget.Off()
                vp.cutterWidget = None

        else:
            colors.printc('Click an actor and press X to open the cutter box widget.', c=4)

    elif key == 'i':  # print info
        utils.printInfo(vp.clickedActor)

    if vp.interactor:
        vp.interactor.Render()
