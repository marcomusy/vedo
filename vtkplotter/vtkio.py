from __future__ import division, print_function
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import numpy as np

import vtkplotter.utils as utils
import vtkplotter.colors as colors
from vtkplotter.actors import Actor, Volume, Assembly, Picture
import vtkplotter.docs as docs
import vtkplotter.settings as settings

__doc__ = (
    """
Submodule to load meshes of different formats, and other I/O functionalities.
"""
    + docs._defs
)

__all__ = [
    "load",
    "download",
    "gunzip",
    "loadStructuredPoints",
    "loadStructuredGrid",
    "loadUnStructuredGrid",
    "loadRectilinearGrid",
    "write",
    "save",
    "exportWindow",
    "importWindow",
    "screenshot",
    "Video",
]


def load(inputobj, c=None, alpha=1, threshold=False, spacing=(), unpack=True):
    """
    Load ``Actor`` and ``Volume`` from file.

    The output will depend on the file extension. See examples below.

    :param c: color in RGB format, hex, symbol or name
    :param alpha: transparency/opacity of the polygonal data.

    For volumetric data `(tiff, slc, vti etc..)`:

    :param list c: can be a list of any length of colors. This list represents the color
        transfer function values equally spaced along the range of the volumetric scalar.
    :param list alpha: can be a list of any length of tranparencies. This list represents the
        transparency transfer function values equally spaced along the range of the volumetric scalar.
    :param float threshold: value to draw the isosurface, False by default to return a ``Volume``.
        If set to True will return an ``Actor`` with automatic choice of the isosurfacing threshold.
    :param list spacing: specify the voxel spacing in the three dimensions
    :param bool unpack: only for multiblock data, if True returns a flat list of objects.

    :Examples:
        .. code-block:: python

            from vtkplotter import datadir, load, show

            # Return an Actor
            g = load(datadir+'250.vtk')
            show(g)

            # Return a list of 2 Actors
            g = load([datadir+'250.vtk', datadir+'270.vtk'])
            show(g)

            # Return a list of actors by reading all files in a directory
            # (if directory contains DICOM files then a Volume is returned)
            g = load(datadir+'timecourse1d/')
            show(g)

            # Return a Volume. Color/Opacity transfer functions can be specified too.
            g = load(datadir+'embryo.slc')
            g.c(['y','lb','w']).alpha((0.0, 0.4, 0.9, 1))
            show(g)

            # Return an Actor from a SLC volume with automatic thresholding
            g = load(datadir+'embryo.slc', threshold=True)
            show(g)
    """
    acts = []
    if utils.isSequence(inputobj):
        flist = inputobj
    else:
        import glob
        flist = sorted(glob.glob(inputobj))

    for fod in flist:

        if os.path.isfile(fod): ### it's a file

            a = _load_file(fod, c, alpha, threshold, spacing, unpack)
            acts.append(a)

        elif os.path.isdir(fod):### it's a directory or DICOM
            flist = os.listdir(fod)
            if '.dcm' in flist[0]: ### it's DICOM
                reader = vtk.vtkDICOMImageReader()
                reader.SetDirectoryName(fod)
                reader.Update()
                image = reader.GetOutput()
                if len(spacing) == 3:
                    image.SetSpacing(spacing[0], spacing[1], spacing[2])
                if threshold is False:
                    if c is None and alpha == 1:
                        c = ['b','lb','lg','y','r'] # good for blackboard background
                        alpha = (0.0, 0.0, 0.2, 0.4, 0.8, 1)
                        #c = ['lb','db','dg','dr']  # good for white backgr
                        #alpha = (0.0, 0.0, 0.2, 0.6, 0.8, 1)
                    actor = Volume(image, c, alpha)
                else:
                    actor = Volume(image).isosurface(threshold=threshold)
                    actor.color(c).alpha(alpha)
                acts.append(actor)
            else: ### it's a normal directory
                utils.humansort(flist)
                for ifile in flist:
                    a = _load_file(fod+'/'+ifile, c, alpha, threshold, spacing, unpack)
                    acts.append(a)
        else:
            colors.printc("~times Error in load(): cannot find", fod, c=1)

    if len(acts) == 1:
        if not acts[0]:
            colors.printc("~times Error in load(): cannot load", inputobj, c=1)
        return acts[0]
    elif len(acts) == 0:
        colors.printc("~times Error in load(): cannot load", inputobj, c=1)
        return None
    else:
        return acts


def _load_file(filename, c, alpha, threshold, spacing, unpack):
    fl = filename.lower()

    ################################################################# other formats:
    if fl.endswith(".xml") or fl.endswith(".xml.gz") or fl.endswith(".xdmf"):
        # Fenics tetrahedral file
        actor = loadDolfin(filename)
    elif fl.endswith(".neutral") or fl.endswith(".neu"):  # neutral tetrahedral file
        actor = loadNeutral(filename)
    elif fl.endswith(".gmsh"):  # gmesh file
        actor = loadGmesh(filename)
    elif fl.endswith(".pcd"):  # PCL point-cloud format
        actor = loadPCD(filename)
        actor.GetProperty().SetPointSize(2)
    elif fl.endswith(".off"):
        actor = loadOFF(filename)
    elif fl.endswith(".3ds"):  # 3ds format
        actor = load3DS(filename)
    elif fl.endswith(".wrl"):
        importer = vtk.vtkVRMLImporter()
        importer.SetFileName(filename)
        importer.Read()
        importer.Update()
        actors = importer.GetRenderer().GetActors() #vtkActorCollection
        actors.InitTraversal()
        wacts = []
        for i in range(actors.GetNumberOfItems()):
            act = actors.GetNextActor()
            wacts.append(act)
        actor = Assembly(wacts)

        ################################################################# volumetric:
    elif fl.endswith(".tif") or fl.endswith(".slc") or fl.endswith(".vti") \
        or fl.endswith(".mhd") or fl.endswith(".nrrd") or fl.endswith(".nii") \
        or fl.endswith(".dem"):
        img = loadImageData(filename, spacing)
        if threshold is False:
            if c is None and alpha == 1:
                c = ['b','lb','lg','y','r'] # good for blackboard background
                alpha = (0.0, 0.0, 0.2, 0.4, 0.8, 1)
            actor = Volume(img, c, alpha)
        else:
            actor = Volume(img).isosurface(threshold=threshold)
            actor.color(c).alpha(alpha)

        ################################################################# 2D images:
    elif fl.endswith(".png") or fl.endswith(".jpg") or fl.endswith(".bmp") or fl.endswith(".jpeg"):
        if ".png" in fl:
            picr = vtk.vtkPNGReader()
        elif ".jpg" in fl or ".jpeg" in fl:
            picr = vtk.vtkJPEGReader()
        elif ".bmp" in fl:
            picr = vtk.vtkBMPReader()
        picr.SetFileName(filename)
        picr.Update()
        actor = Picture()  # object derived from vtk.vtkImageActor()
        actor.SetInputData(picr.GetOutput())
        if alpha is None:
            alpha = 1
        actor.SetOpacity(alpha)

        ################################################################# multiblock:
    elif fl.endswith(".vtm") or fl.endswith(".vtmb"):
        read = vtk.vtkXMLMultiBlockDataReader()
        read.SetFileName(filename)
        read.Update()
        mb = read.GetOutput()
        if unpack:
            acts = []
            for i in range(mb.GetNumberOfBlocks()):
                b =  mb.GetBlock(i)
                if isinstance(b, (vtk.vtkPolyData,
                                  vtk.vtkImageData,
                                  vtk.vtkUnstructuredGrid,
                                  vtk.vtkStructuredGrid,
                                  vtk.vtkRectilinearGrid)):
                    acts.append(b)
            return acts
        else:
            return mb

        ################################################################# numpy:
    elif fl.endswith(".npy"):
        acts = loadNumpy(filename)
        if unpack == False:
            return Assembly(acts)
        return acts

    elif fl.endswith(".geojson") or fl.endswith(".geojson.gz"):
        return loadGeoJSON(fl)

        ################################################################# polygonal mesh:
    else:
        if fl.endswith(".vtk"): # read all legacy vtk types

            #output can be:
            # PolyData, StructuredGrid, StructuredPoints, UnstructuredGrid, RectilinearGrid
            reader = vtk.vtkDataSetReader()
            reader.ReadAllScalarsOn()
            reader.ReadAllVectorsOn()
            reader.ReadAllTensorsOn()
            reader.ReadAllFieldsOn()
            reader.ReadAllNormalsOn()
            reader.ReadAllColorScalarsOn()

        elif fl.endswith(".ply"):
            reader = vtk.vtkPLYReader()
        elif fl.endswith(".obj"):
            reader = vtk.vtkOBJReader()
        elif fl.endswith(".stl"):
            reader = vtk.vtkSTLReader()
        elif fl.endswith(".byu") or fl.endswith(".g"):
            reader = vtk.vtkBYUReader()
        elif fl.endswith(".foam"):  # OpenFoam
            reader = vtk.vtkOpenFOAMReader()
        elif fl.endswith(".pvd"):
            reader = vtk.vtkXMLGenericDataObjectReader()
        elif fl.endswith(".vtp"):
            reader = vtk.vtkXMLPolyDataReader()
        elif fl.endswith(".vts"):
            reader = vtk.vtkXMLStructuredGridReader()
        elif fl.endswith(".vtu"):
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif fl.endswith(".vtr"):
            reader = vtk.vtkXMLRectilinearGridReader()
        elif fl.endswith(".pvtk"):
            reader = vtk.vtkPDataSetReader()
        elif fl.endswith(".pvtr"):
            reader = vtk.vtkXMLPRectilinearGridReader()
        elif fl.endswith("pvtu"):
            reader = vtk.vtkXMLPUnstructuredGridReader()
        elif fl.endswith(".txt") or fl.endswith(".xyz"):
            reader = vtk.vtkParticleReader()  # (format is x, y, z, scalar)
        elif fl.endswith(".facet"):
            reader = vtk.vtkFacetReader()
        else:
            return None

        reader.SetFileName(filename)
        reader.Update()
        routput = reader.GetOutput()

        if not routput:
            colors.printc("~noentry Unable to load", filename, c=1)
            return None

        actor = Actor(routput, c, alpha)
        if fl.endswith(".txt") or fl.endswith(".xyz"):
            actor.GetProperty().SetPointSize(4)

    actor.filename = filename
    return actor

def download(url, prefix=''):
    """Retrieve a file from a url, save it locally and return its path."""

    if "https://" not in url and "http://" not in url:
    #    colors.printc('Invalid URL:\n', url, c=1)
        return url

    basename = os.path.basename(url)
    if os.path.exists(basename):
        return basename

    colors.printc('..downloading:\n', url)
    try:
        from urllib.request import urlopen
    except ImportError:
        import urllib2
        import contextlib
        urlopen = lambda url_: contextlib.closing(urllib2.urlopen(url_))

    basename += prefix
    with urlopen(url) as response, open(basename, 'wb') as output:
        output.write(response.read())
    return basename

def gunzip(filename):
    """Unzip a ``.gz`` file to a temporary file and returns its path."""
    if not filename.endswith('.gz'):
        #colors.printc("gunzip() error: file must end with .gz", c=1)
        return filename
    from tempfile import NamedTemporaryFile
    import gzip

    tmp_file = NamedTemporaryFile(delete=False)
    tmp_file.name = os.path.join(os.path.dirname(tmp_file.name),
                                 os.path.basename(filename).replace('.gz',''))
    inF = gzip.open(filename, "rb")
    outF = open(tmp_file.name, "wb")
    outF.write(inF.read())
    outF.close()
    inF.close()
    return tmp_file.name


###################################################################
def loadStructuredPoints(filename):
    """Load a ``vtkStructuredPoints`` object from file."""
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def loadStructuredGrid(filename):
    """Load a ``vtkStructuredGrid`` object from file."""
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def loadUnStructuredGrid(filename):
    """Load a ``vtkunStructuredGrid`` object from file."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def loadRectilinearGrid(filename):
    """Load a ``vtkRectilinearGrid`` object from file."""
    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def loadXMLGenericData(filename):
    """Read any type of vtk data object encoded in XML format."""
    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


###################################################################
def load3DS(filename):
    """Load ``3DS`` file format from file. Return an ``Assembly(vtkAssembly)`` object."""
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
    return Assembly(acts)


def loadOFF(filename):
    """Read OFF file format."""
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    vertices = []
    faces = []
    NumberOfVertices = None
    i = -1
    for text in lines:
        if len(text) == 0:
            continue
        if text == '\n':
            continue
        if "#" in text:
            continue
        if "OFF" in text:
            continue

        ts = text.split()
        n = len(ts)

        if not NumberOfVertices and n > 1:
            NumberOfVertices, NumberOfFaces = int(ts[0]), int(ts[1])
            continue
        i += 1

        if i < NumberOfVertices and n == 3:
            x, y, z = float(ts[0]), float(ts[1]), float(ts[2])
            vertices.append([x, y, z])

        ids = []
        if NumberOfVertices <= i < (NumberOfVertices + NumberOfFaces + 1) and n > 2:
            ids += [int(xx) for xx in ts[1:]]
            faces.append(ids)

    return Actor(utils.buildPolyData(vertices, faces))


def loadGeoJSON(filename):
    """Load GeoJSON files."""
    if filename.endswith('.gz'):
        filename = gunzip(filename)
    jr = vtk.vtkGeoJSONReader()
    jr.SetFileName(filename)
    jr.Update()
    return Actor(jr.GetOutput())


def loadDolfin(filename, exterior=False):
    """Reads a `Fenics/Dolfin` file format (.xml or .xdmf).
    Return an ``Actor(vtkActor)`` object."""
    import sys
    if sys.version_info[0] < 3:
        return _loadDolfin_old(filename)

    import dolfin

    if filename.lower().endswith('.xdmf'):
        f = dolfin.XDMFFile(filename)
        m = dolfin.Mesh()
        f.read(m)
    else:
        m = dolfin.Mesh(filename)

    bm = dolfin.BoundaryMesh(m, "exterior")

    if exterior:
        poly = utils.buildPolyData(bm.coordinates(), bm.cells(), fast=True)
    else:
        polyb = utils.buildPolyData(bm.coordinates(), bm.cells(), fast=True)
        polym = utils.buildPolyData(m.coordinates(), m.cells(), fast=True)
        app = vtk.vtkAppendPolyData()
        app.AddInputData(polym)
        app.AddInputData(polyb)
        app.Update()
        poly = app.GetOutput()
    return Actor(poly).lw(0.1)


def _loadDolfin_old(filename, exterior='dummy'):
    import xml.etree.ElementTree as et

    if filename.endswith(".gz"):
        import gzip

        inF = gzip.open(filename, "rb")
        outF = open("/tmp/filename.xml", "wb")
        outF.write(inF.read())
        outF.close()
        inF.close()
        tree = et.parse("/tmp/filename.xml")
    else:
        tree = et.parse(filename)

    coords, connectivity = [], []
    for mesh in tree.getroot():
        for elem in mesh:
            for e in elem.findall("vertex"):
                x = float(e.get("x"))
                y = float(e.get("y"))
                ez = e.get("z")
                if ez is None:
                    coords.append([x, y])
                else:
                    z = float(ez)
                    coords.append([x, y, z])

            tets = elem.findall("tetrahedron")
            if not len(tets):
                tris = elem.findall("triangle")
                for e in tris:
                    v0 = int(e.get("v0"))
                    v1 = int(e.get("v1"))
                    v2 = int(e.get("v2"))
                    connectivity.append([v0, v1, v2])
            else:
                for e in tets:
                    v0 = int(e.get("v0"))
                    v1 = int(e.get("v1"))
                    v2 = int(e.get("v2"))
                    v3 = int(e.get("v3"))
                    connectivity.append([v0, v1, v2, v3])

    poly = utils.buildPolyData(coords, connectivity)
    return Actor(poly)


def loadNeutral(filename):
    """Reads a `Neutral` tetrahedral file format. Return an ``Actor(vtkActor)`` object."""
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    ncoords = int(lines[0])
    coords = []
    for i in range(1, ncoords + 1):
        x, y, z = lines[i].split()
        coords.append([float(x), float(y), float(z)])

    ntets = int(lines[ncoords + 1])
    idolf_tets = []
    for i in range(ncoords + 2, ncoords + ntets + 2):
        text = lines[i].split()
        v0, v1, v2, v3 = int(text[1])-1, int(text[2])-1, int(text[3])-1, int(text[4])-1
#        p0, p1, p2, p3 = np.array(coords[v1]), np.array(coords[v0]), coords[v3], coords[v2]
#        d10 = p1-p0
#        d21 = p2-p1
#        dc = np.cross(d10, d21)
#        print(np.dot(dc,p3-p0))
        idolf_tets.append([v0, v1, v2, v3])

    poly = utils.buildPolyData(coords, idolf_tets)
    return Actor(poly)


def loadGmesh(filename):
    """Reads a `gmesh` file format. Return an ``Actor(vtkActor)`` object."""
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    nnodes = 0
    index_nodes = 0
    for i, line in enumerate(lines):
        if "$Nodes" in line:
            index_nodes = i + 1
            nnodes = int(lines[index_nodes])
            break
    node_coords = []
    for i in range(index_nodes + 1, index_nodes + 1 + nnodes):
        cn = lines[i].split()
        node_coords.append([float(cn[1]), float(cn[2]), float(cn[3])])

    nelements = 0
    index_elements = 0
    for i, line in enumerate(lines):
        if "$Elements" in line:
            index_elements = i + 1
            nelements = int(lines[index_elements])
            break
    elements = []
    for i in range(index_elements + 1, index_elements + 1 + nelements):
        ele = lines[i].split()
        elements.append([int(ele[-3]), int(ele[-2]), int(ele[-1])])

    poly = utils.buildPolyData(node_coords, elements, indexOffset=1)
    return Actor(poly)


def loadPCD(filename):
    """Return ``vtkActor`` from `Point Cloud` file format. Return an ``Actor(vtkActor)`` object."""
    f = open(filename, "r")
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
        if not start and "POINTS" in text:
            expN = int(text.split()[1])
        if not start and "DATA ascii" in text:
            start = True
    if expN != N:
        colors.printc("~!? Mismatch in pcd file", expN, len(pts), c="red")
    poly = utils.buildPolyData(pts)
    return Actor(poly).pointSize(4)


def loadNumpy(inobj):
    import numpy as np

    if isinstance(inobj, str):
        data = np.load(inobj, allow_pickle=True)
    else:
        data = inobj

    def loadcommon(obj, d):
        keys = d.keys()
        if 'time' in keys: obj.time(d['time'])
        if 'transform' in keys and len(d['transform']) == 4:
            vm = vtk.vtkMatrix4x4()
            for i in [0, 1, 2, 3]:
               for j in [0, 1, 2, 3]:
                   vm.SetElement(i, j, d['transform'][i,j])
            obj.setTransform(vm)
        elif 'position' in keys:
            obj.pos(d['position'])
        if hasattr(obj, 'GetProperty'):
            prp = obj.GetProperty()
            if 'ambient' in keys: prp.SetAmbient(d['ambient'])
            if 'diffuse' in keys: prp.SetDiffuse(d['diffuse'])

    ##################
    def _buildactor(d):
        vertices = d['points']
        cells = None
        lines = None
        keys = d.keys()
        if 'cells' in keys:
            cells = d['cells']
        if 'lines' in keys:
            lines = d['lines']

        poly = utils.buildPolyData(vertices, cells, lines)
        act = Actor(poly)
        loadcommon(act, d)

        act.mapper.ScalarVisibilityOff()
        if 'celldata' in keys:
            for csc, cscname in d['celldata']:
                act.addCellScalars(csc, cscname)
                if not 'normal' in cscname.lower():
                    act.scalars(cscname) # activate
        if 'pointdata' in keys:
            for psc, pscname in d['pointdata']:
                act.addPointScalars(psc, pscname)
                if not 'normal' in pscname.lower():
                    act.scalars(pscname) # activate

        prp = act.GetProperty()
        if 'specular' in keys:      prp.SetSpecular(d['specular'])
        if 'specularpower' in keys: prp.SetSpecularPower(d['specularpower'])
        if 'specularcolor' in keys: prp.SetSpecularColor(d['specularcolor'])
        if 'shading' in keys:       prp.SetInterpolation(d['shading'])
        if 'alpha' in keys:         prp.SetOpacity(d['alpha'])
        if 'opacity' in keys:       prp.SetOpacity(d['opacity']) # synomym
        if 'pointsize' in keys and d['pointsize']: prp.SetPointSize(d['pointsize'])
        if 'texture' in keys and d['texture']:     act.texture(d['texture'])
        if 'linewidth' in keys and d['linewidth']: act.lineWidth(d['linewidth'])
        if 'linecolor' in keys and d['linecolor']: act.lineColor(d['linecolor'])
        if 'representation' in keys: prp.SetRepresentation(d['representation'])
        if 'color' in keys and d['color']: act.color(d['color'])
        if 'backColor' in keys and d['backColor']: act.backColor(d['backColor'])

        if 'activedata' in keys and d['activedata'] is not None:
            act.mapper.ScalarVisibilityOn()
            if d['activedata'][0] == 'celldata':
                poly.GetCellData().SetActiveScalars(d['activedata'][1])
            if d['activedata'][0] == 'pointdata':
                poly.GetPointData().SetActiveScalars(d['activedata'][1])

        return act
        ##################

    objs = []
    for d in data:
        #print('loadNumpy type is:', d['type'])

        if 'mesh' == d['type']:
            objs.append(_buildactor(d))

        elif 'assembly' == d['type']:
            assacts = []
            for ad in d['actors']:
                assacts.append(_buildactor(ad))
            asse = Assembly(assacts)
            loadcommon(asse, d)
            objs.append(asse)

        elif 'image' == d['type']:
            shp = d['shape'][1], d['shape'][0]
            arr0 = d['array']
            rcv =   arr0[:,0].reshape(shp)
            gcv =   arr0[:,1].reshape(shp)
            bcv =   arr0[:,2].reshape(shp)
            arr = np.array([rcv, gcv, bcv])
            arr = np.swapaxes(arr, 0, 2)
            vimg = Picture(arr)
            loadcommon(vimg, d)
            objs.append(vimg)

        elif 'volume' == d['type']:
            vol = Volume(d['array'])
            loadcommon(vol, d)
            vol.jittering(d['jittering'])
            vol.mode(d['mode'])
            vol.color(d['color'])
            vol.alpha(d['alpha'])
            vol.alphaGradient(d['alphagrad'])
            objs.append(vol)

    if len(objs) == 1:
        return objs[0]
    elif len(objs) == 0:
        return None
    else:
        return objs


def _np_dump(obj):
    '''dump a vtkplotter obj to a numpy dictionary'''
    adict = dict()

    def fillcommon(obj, adict):
        adict['filename'] = obj.filename
        adict['legend'] = obj.legend()
        adict['time'] = obj.time()
        adict['rendered_at'] = obj.renderedAt
        adict['position'] = obj.pos()
        m = np.eye(4)
        vm = obj.getTransform().GetMatrix()
        for i in [0, 1, 2, 3]:
            for j in [0, 1, 2, 3]:
                m[i,j] = vm.GetElement(i, j)
        adict['transform'] = m
        minv = np.eye(4)
        vm.Invert()
        for i in [0, 1, 2, 3]:
            for j in [0, 1, 2, 3]:
                minv[i,j] = vm.GetElement(i, j)
        adict['transform_inverse'] = minv
        if hasattr(obj, 'GetProperty'): # assembly doesnt
            prp = obj.GetProperty()
            adict['ambient'] = prp.GetAmbient()
            adict['diffuse'] = prp.GetDiffuse()


    def _doactor(obj, adict):
        adict['points'] = obj.coordinates(transformed=0).astype(np.float32)
        poly = obj.polydata()
        adict['cells'] = None
        adict['lines'] = None

        if poly.GetNumberOfCells():
            try:
                adict['cells'] = np.array(obj.faces(), dtype=np.uint32)
            except ValueError:
                adict['cells'] = obj.faces()

        if poly.GetNumberOfLines():
            adict['lines'] = vtk_to_numpy(poly.GetLines().GetData()).astype(np.uint32)

        fillcommon(obj, adict)
        adict['pointdata'] = []
        adict['celldata'] = []
        adict['activedata'] = None
        if poly.GetCellData().GetScalars():
            adict['activedata'] = ['celldata', poly.GetCellData().GetScalars().GetName()]
        if poly.GetPointData().GetScalars():
            adict['activedata'] = ['pointdata', poly.GetPointData().GetScalars().GetName()]

        for itype, iname in obj.scalars():
            if itype == 'PointData':
                adict['pointdata'].append([obj.getPointArray(iname), iname])
            if itype == 'CellData':
                adict['celldata'].append([obj.getCellArray(iname), iname])

        prp = obj.GetProperty()
        adict['alpha'] = prp.GetOpacity()
        adict['representation'] = prp.GetRepresentation()
        adict['texture'] = None
        adict['pointsize'] = prp.GetPointSize()
        if prp.GetEdgeVisibility():
            adict['linewidth'] = prp.GetLineWidth()
            if hasattr(prp, 'GetLineColor'):
                adict['linecolor'] = prp.GetLineColor()
        else:
            adict['linewidth'] = 0
            adict['linecolor'] = 0
        adict['specular'] = prp.GetSpecular()
        adict['specularpower'] = prp.GetSpecularPower()
        adict['specularcolor'] = prp.GetSpecularColor()
        adict['shading'] = prp.GetInterpolation()
        adict['color'] = prp.GetColor()

        adict['backColor'] = None
        if obj.GetBackfaceProperty():
            adict['backColor'] = obj.GetBackfaceProperty().GetColor()


    ############################
    adict['type'] = 'unknown'
    if isinstance(obj, Actor):
        adict['type'] = 'mesh'
        _doactor(obj, adict)

    elif isinstance(obj, Assembly):
        adict['type'] = 'assembly'
        adict['actors'] = []
        for a in obj.getActors():
            assdict = dict()
            if not isinstance(a, Actor): #normal vtkActor
                b = Actor(a) # promote it to a Actor
                pra = vtk.vtkProperty()
                pra.DeepCopy(a.GetProperty())
                b.SetProperty(pra)
                a = b
            _doactor(a, assdict)
            adict['actors'].append(assdict)
        fillcommon(obj, adict)

    elif isinstance(obj, Picture):
        adict['type'] = 'image'
        arr = vtk_to_numpy(obj.inputdata().GetPointData().GetScalars())
        adict['array'] = arr
        adict['shape'] = obj.inputdata().GetDimensions()
        fillcommon(obj, adict)
        #print('image', arr, arr.shape, obj.inputdata().GetDimensions())

    elif isinstance(obj, Volume):
        adict['type'] = 'volume'
        imgdata = obj.inputdata()
        arr = vtk_to_numpy(imgdata.GetPointData().GetScalars())
        adict['array'] = arr.reshape(imgdata.GetDimensions())
        adict['mode'] = obj.mode()
        adict['jittering'] = obj.mapper.GetUseJittering()

        prp = obj.GetProperty()
        ctf = prp.GetRGBTransferFunction()
        otf = prp.GetScalarOpacity()
        gotf = prp.GetGradientOpacity()
        smin, smax = ctf.GetRange()
        xs = np.linspace(smin, smax, num=100, endpoint=True)
        cols, als, algrs = [], [], []
        for x in xs:
            cols.append(ctf.GetColor(x))
            als.append(otf.GetValue(x))
            if gotf:
                algrs.append(gotf.GetValue(x))
        adict['color'] = cols
        adict['alpha'] = als
        adict['alphagrad'] = algrs
        fillcommon(obj, adict)

    return adict


def loadImageData(filename, spacing=()):
    """Read and return a ``vtkImageData`` object from file.
    Use ``load`` instead.
    E.g. `img = load('myfile.tif').imagedata()`
    """
    if ".tif" in filename.lower():
        reader = vtk.vtkTIFFReader()
    elif ".slc" in filename.lower():
        reader = vtk.vtkSLCReader()
        if not reader.CanReadFile(filename):
            colors.printc("~prohibited Sorry bad slc file " + filename, c=1)
            return None
    elif ".vti" in filename.lower():
        reader = vtk.vtkXMLImageDataReader()
    elif ".mhd" in filename.lower():
        reader = vtk.vtkMetaImageReader()
    elif ".dem" in filename.lower():
        reader = vtk.vtkDEMReader()
    elif ".nii" in filename.lower():
        reader = vtk.vtkNIFTIImageReader()
    elif ".nrrd" in filename.lower():
        reader = vtk.vtkNrrdReader()
        if not reader.CanReadFile(filename):
            colors.printc("~prohibited Sorry bad nrrd file " + filename, c=1)
            return None
    reader.SetFileName(filename)
    reader.Update()
    image = reader.GetOutput()
    if len(spacing) == 3:
        image.SetSpacing(spacing[0], spacing[1], spacing[2])
    return image



###########################################################
def write(objct, fileoutput, binary=True):
    """
    Write 3D object to file. (same as `save()`).

    Possile extensions are:
        - vtk, vti, npy, ply, obj, stl, byu, vtp, vti, mhd, xyz, tif, png, bmp.
    """
    obj = objct
    if isinstance(obj, Actor): # picks transformation
        obj = objct.polydata(True)
    elif isinstance(obj, (vtk.vtkActor, vtk.vtkVolume)):
        obj = objct.GetMapper().GetInput()
    elif isinstance(obj, (vtk.vtkPolyData, vtk.vtkImageData)):
        obj = objct

    fr = fileoutput.lower()
    if   ".vtk" in fr:
        writer = vtk.vtkPolyDataWriter()
    elif ".ply" in fr:
        writer = vtk.vtkPLYWriter()
        pscal = obj.GetPointData().GetScalars()
        if not pscal:
            pscal = obj.GetCellData().GetScalars()
        if pscal and pscal.GetName():
            writer.SetArrayName(pscal.GetName())
            #writer.SetColorMode(0)
        lut = objct.GetMapper().GetLookupTable()
        if lut:
            writer.SetLookupTable(lut)
    elif ".stl" in fr:
        writer = vtk.vtkSTLWriter()
    elif ".vtp" in fr:
        writer = vtk.vtkXMLPolyDataWriter()
    elif ".vtm" in fr:
        g = vtk.vtkMultiBlockDataGroupFilter()
        for ob in objct:
            g.AddInputData(ob)
        g.Update()
        mb = g.GetOutputDataObject(0)
        wri = vtk.vtkXMLMultiBlockDataWriter()
        wri.SetInputData(mb)
        wri.SetFileName(fileoutput)
        wri.Write()
        return mb
    elif ".xyz" in fr:
        writer = vtk.vtkSimplePointsWriter()
    elif ".facet" in fr:
        writer = vtk.vtkFacetWriter()
    elif ".tif" in fr:
        writer = vtk.vtkTIFFWriter()
        writer.SetFileDimensionality(len(obj.GetDimensions()))
    elif ".vti" in fr:
        writer = vtk.vtkXMLImageDataWriter()
    elif ".mhd" in fr:
        writer = vtk.vtkMetaImageWriter()
    elif ".nii" in fr:
        writer = vtk.vtkNIFTIImageWriter()
    elif ".png" in fr:
        writer = vtk.vtkPNGWriter()
    elif ".jpg" in fr:
        writer = vtk.vtkJPEGWriter()
    elif ".bmp" in fr:
        writer = vtk.vtkBMPWriter()
    elif ".npy" in fr:
        if utils.isSequence(objct):
            objslist = objct
        else:
            objslist = [objct]
        dicts2save = []
        for obj in objslist:
            dicts2save.append( _np_dump(obj) )
        np.save(fileoutput, dicts2save)
        return dicts2save

    elif ".xml" in fr:  # write tetrahedral dolfin xml
        vertices = objct.coordinates().astype(str)
        faces = np.array(objct.faces()).astype(str)
        ncoords = vertices.shape[0]
        outF = open(fileoutput, "w")
        outF.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        outF.write('<dolfin xmlns:dolfin="http://www.fenicsproject.org">\n')

        if len(faces[0]) == 4:# write tetrahedral mesh
            ntets = faces.shape[0]
            outF.write('  <mesh celltype="tetrahedron" dim="3">\n')
            outF.write('    <vertices size="' + str(ncoords) + '">\n')
            for i in range(ncoords):
                x, y, z = vertices[i]
                outF.write('      <vertex index="'+str(i)+'" x="'+x+'" y="'+y+'" z="'+z+'"/>\n')
            outF.write('    </vertices>\n')
            outF.write('    <cells size="' + str(ntets) + '">\n')
            for i in range(ntets):
                v0, v1, v2, v3 = faces[i]
                outF.write('     <tetrahedron index="'+str(i)
                           + '" v0="'+v0+'" v1="'+v1+'" v2="'+v2+'" v3="'+v3+'"/>\n')

        elif len(faces[0]) == 3:# write triangle mesh
            ntri = faces.shape[0]
            outF.write('  <mesh celltype="triangle" dim="2">\n')
            outF.write('    <vertices size="' + str(ncoords) + '">\n')
            for i in range(ncoords):
                x, y, dummy_z = vertices[i]
                outF.write('      <vertex index="'+str(i)+'" x="'+x+'" y="'+y+'"/>\n')
            outF.write('    </vertices>\n')
            outF.write('    <cells size="' + str(ntri) + '">\n')
            for i in range(ntri):
                v0, v1, v2 = faces[i]
                outF.write('     <triangle index="'+str(i)+'" v0="'+v0+'" v1="'+v1+'" v2="'+v2+'"/>\n')

        outF.write('    </cells>\n')
        outF.write("  </mesh>\n")
        outF.write("</dolfin>\n")
        outF.close()
        return objct

    else:
        colors.printc("~noentry Unknown format", fileoutput, "file not saved.", c="r")
        return objct

    try:
        if hasattr(writer, 'SetFileTypeToBinary'):
            if binary:
                writer.SetFileTypeToBinary()
            else:
                writer.SetFileTypeToASCII()
        writer.SetInputData(obj)
        writer.SetFileName(fileoutput)
        writer.Write()
        colors.printc("~save Saved file: " + fileoutput, c="g")
    except Exception as e:
        colors.printc("~noentry Error saving: " + fileoutput, "\n", e, c="r")
    return objct

def save(objct, fileoutput, binary=True):
    """
    Save 3D object to file. (same as `write()`).

    Possile extensions are:
        - vtk, vti, npy, ply, obj, stl, vtp, xyz, tif, vti, mhd, png, jpg, bmp.
    """
    return write(objct, fileoutput, binary)


###########################################################
def exportWindow(fileoutput, binary=False, speed=None, html=True):
    '''
    Exporter which writes out the renderered scene into an OBJ, X3D or Numpy file.
    X3D is an XML-based format for representation 3D scenes (similar to VRML).
    Check out http://www.web3d.org/x3d for more details.

    :param float speed: set speed for x3d files.
    :param bool html: generate a test html page for x3d files.

    |export_x3d| |export_x3d.py|_

        `generated webpage <https://vtkplotter.embl.es/examples/embryo.html>`_

        See also: FEniCS test `webpage <https://vtkplotter.embl.es/examples/fenics_elasticity.html>`_.

    .. note:: the rendering window can also be exported to `numpy` file `scene.npy`
        by pressing ``E`` keyboard at any moment during visualization.
    '''
    fr = fileoutput.lower()

    if ".obj" in fr:
        w = vtk.vtkOBJExporter()
        w.SetInputData(settings.plotter_instance.window)
        w.Update()
        colors.printc("~save Saved file:", fileoutput, c="g")

    elif ".x3d" in fr:
        exporter = vtk.vtkX3DExporter()
        exporter.SetBinary(binary)
        exporter.FastestOff()
        if speed:
            exporter.SetSpeed(speed)
        exporter.SetInput(settings.plotter_instance.window)
        exporter.SetFileName(fileoutput)
        exporter.Update()
        exporter.Write()
        if not html:
            return
        from vtkplotter.docs import x3d_html
        x3d_html = x3d_html.replace("~fileoutput", fileoutput)
        wsize = settings.plotter_instance.window.GetSize()
        x3d_html = x3d_html.replace("~width", str(wsize[0]))
        x3d_html = x3d_html.replace("~height", str(wsize[1]))
        #b = settings.plotter_instance.renderer.ComputeVisiblePropBounds()
        #s = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        #c = (b[1] + b[0])/2, (b[3] + b[2])/2, (b[5] + b[4])/2
        #x3d_html = x3d_html.replace("~size", str(s*2))
        #x3d_html = x3d_html.replace("~center", str(c[0])+" "+str(c[1])+" "+str(c[2]))
        outF = open(fileoutput.replace('.x3d', '.html'), "w")
        outF.write(x3d_html)
        outF.close()
        colors.printc("~save Saved files:", fileoutput,
                      fileoutput.replace('.x3d', '.html'), c="g")
    elif ".npy" in fr:
        sdict = dict()
        vp = settings.plotter_instance
        sdict['shape'] = vp.shape #todo
        sdict['sharecam'] = vp.sharecam #todo
        sdict['camera'] = None #todo
        sdict['position'] = vp.pos
        sdict['size'] = vp.size
        sdict['axes'] = vp.axes
        sdict['title'] = vp.title
        sdict['xtitle'] = vp.xtitle
        sdict['ytitle'] = vp.ytitle
        sdict['ztitle'] = vp.ztitle
        sdict['backgrcol'] = colors.getColor(vp.backgrcol)
        sdict['useDepthPeeling'] = settings.useDepthPeeling
        sdict['renderPointsAsSpheres'] = settings.renderPointsAsSpheres
        sdict['renderLinesAsTubes'] = settings.renderLinesAsTubes
        sdict['hiddenLineRemoval'] = settings.hiddenLineRemoval
        sdict['visibleGridEdges'] = settings.visibleGridEdges
        sdict['interactorStyle'] = settings.interactorStyle
        sdict['useParallelProjection'] = settings.useParallelProjection
        sdict['objects'] = []
        for a in vp.getActors() + vp.getVolumes():
            sdict['objects'].append(_np_dump(a))
        np.save(fileoutput, [sdict])

    return

def importWindow(fileinput):
    """Import a whole scene from a Numpy file.
    Return ``Plotter`` instance."""
    import numpy as np
    from vtkplotter import Plotter

    data = np.load(fileinput, allow_pickle=True)[0]

    if 'renderPointsAsSpheres' in data.keys():
        settings.renderPointsAsSpheres = data['renderPointsAsSpheres']
    if 'renderLinesAsTubes' in data.keys():
        settings.renderLinesAsTubes = data['renderLinesAsTubes']
    if 'hiddenLineRemoval' in data.keys():
        settings.hiddenLineRemoval = data['hiddenLineRemoval']
    if 'visibleGridEdges' in data.keys():
        settings.visibleGridEdges = data['visibleGridEdges']
    if 'interactorStyle' in data.keys():
        settings.interactorStyle = data['interactorStyle']
    if 'useParallelProjection' in data.keys():
        settings.useParallelProjection = data['useParallelProjection']

    pos = data.pop('position', (0, 0))
    axes = data.pop('axes', 4)
    title = data.pop('title', '')
    backgrcol = data.pop('backgrcol', "blackboard")

    vp = Plotter(pos=pos,
                 #size=data['size'], # not necessarily a good idea to set it
                 #shape=data['shape'],
                 axes=axes,
                 title=title,
                 bg=backgrcol,
    )
    vp.xtitle = data.pop('xtitle', 'x')
    vp.ytitle = data.pop('ytitle', 'y')
    vp.ztitle = data.pop('ztitle', 'z')

    objs = loadNumpy(data['objects'])
    if not utils.isSequence(objs):
       objs = [objs]
    vp.actors = objs

#    if vp.shape==(1,1):
#        vp.actors = loadNumpy(data['objects'])
#    else:
#        print(objs, )
#        for a in objs:
#            for ar in a.renderedAt:
#                print(vp.shape, [a], ar )
#                vp.show(a, at=ar)
    return vp


##########################################################
def screenshot(filename="screenshot.png"):
    """
    Save a screenshot of the current rendering window.
    """
    if not settings.plotter_instance.window:
        colors.printc('~bomb screenshot(): Rendering window is not present, skip.', c=1)
        return
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(settings.plotter_instance.window)
    s = settings.screeshotScale
    w2if.SetScale(s, s)
    if settings.screenshotTransparentBackground:
        w2if.SetInputBufferTypeToRGBA()
    w2if.ReadFrontBufferOff()  # read from the back buffer
    w2if.Update()
    if filename.endswith('.png'):
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
    elif filename.endswith('.jpg'):
        writer = vtk.vtkJPEGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
    elif filename.endswith('.svg'):
        writer = vtk.vtkGL2PSExporter()
        #writer.SetFileFormatToPDF()
        #writer.SetFileFormatToTeX()
        writer.SetFileFormatToSVG()
        writer.CompressOff()
        writer.SetInput(settings.plotter_instance.window)
        writer.SetFilePrefix(filename.split('.')[0])
        writer.Write()


class Video:
    """
    Class to generate a video from the specified rendering window.
    Program ``ffmpeg`` is used to create video from each generated frame.
    :param str name: name of the output file.
    :param int fps: set the number of frames per second.
    :param float duration: set the total `duration` of the video and recalculates `fps` accordingly.
    :param str ffmpeg: set path to ffmpeg program. Default value considers ffmpeg is in the path.

    |makeVideo| |makeVideo.py|_
    """

    def __init__(self, name="movie.avi", **kwargs):

        from tempfile import TemporaryDirectory

        self.name = name
        self.duration = kwargs.pop('duration', None)
        self.fps = float(kwargs.pop('fps', 12))
        self.ffmpeg = kwargs.pop('ffmpeg', 'ffmpeg')
        self.frames = []
        self.tmp_dir = TemporaryDirectory()
        self.get_filename = lambda x: os.path.join(self.tmp_dir.name, x)
        colors.printc("~video Video", name, "is open...", c="m")

    def addFrame(self):
        """Add frame to current video."""
        fr = self.get_filename(str(len(self.frames)) + ".png")
        screenshot(fr)
        self.frames.append(fr)

    def pause(self, pause=0):
        """Insert a `pause`, in seconds."""
        fr = self.frames[-1]
        n = int(self.fps * pause)
        for _ in range(n):
            fr2 = self.get_filename(str(len(self.frames)) + ".png")
            self.frames.append(fr2)
            os.system("cp -f %s %s" % (fr, fr2))

    def close(self):
        """Render the video and write to file."""
        if self.duration:
            self.fps = len(self.frames) / float(self.duration)
            colors.printc("Recalculated video FPS to", round(self.fps, 3), c="yellow")
        else:
            self.fps = int(self.fps)
        self.name = self.name.split('.')[0]+'.mp4'
        out = os.system(self.ffmpeg + " -loglevel panic -y -r " + str(self.fps)
                        + " -i " + self.tmp_dir.name + os.sep + "%01d.png " + self.name)
        if out:
            colors.printc("ffmpeg returning error", c=1)
        colors.printc("~save Video saved as", self.name, c="green")
        self.tmp_dir.cleanup()
        return


