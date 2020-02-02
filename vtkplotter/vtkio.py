from __future__ import division, print_function
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import numpy as np

import vtkplotter.utils as utils
import vtkplotter.colors as colors

from vtkplotter.assembly import Assembly
from vtkplotter.mesh import Mesh
from vtkplotter.picture import Picture
from vtkplotter.volume import Volume

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
    "loadRectilinearGrid",
    "loadUnStructuredGrid",
    "write",
    "save",
    "exportWindow",
    "importWindow",
    "screenshot",
    "Video",
]


def load(inputobj, c=None, alpha=1, threshold=False, spacing=(), unpack=True):
    """
    Load ``Mesh``, ``Volume`` and ``Picture`` objects from file.

    The output will depend on the file extension. See examples below.
    Unzip on the fly, if it ends with `.gz`.

    :param c: color in RGB format, hex, symbol or name
    :param alpha: transparency/opacity of the polygonal data.

    For volumetric data `(tiff, slc, vti etc..)`:

    :param list c: can be a list of any length of colors. This list represents the color
        transfer function values equally spaced along the range of the volumetric scalar.
    :param list alpha: can be a list of any length of tranparencies. This list represents the
        transparency transfer function values equally spaced along the range of the volumetric scalar.
    :param float threshold: value to draw the isosurface, False by default to return a ``Volume``.
        If set to True will return an ``Mesh`` with automatic choice of the isosurfacing threshold.
    :param list spacing: specify the voxel spacing in the three dimensions
    :param bool unpack: only for multiblock data, if True returns a flat list of objects.

    :Examples:
        .. code-block:: python

            from vtkplotter import datadir, load, show

            # Return a Mesh object
            g = load(datadir+'250.vtk')
            show(g)

            # Return a list of 2 meshes
            g = load([datadir+'250.vtk', datadir+'270.vtk'])
            show(g)

            # Return a list of meshes by reading all files in a directory
            # (if directory contains DICOM files then a Volume is returned)
            g = load(datadir+'timecourse1d/')
            show(g)

            # Return a Volume. Color/Opacity transfer functions can be specified too.
            g = load(datadir+'embryo.slc')
            g.c(['y','lb','w']).alpha((0.0, 0.4, 0.9, 1))
            show(g)

            # Return a Mesh from a SLC volume with automatic thresholding
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

            if fod.endswith('.gz'):
                fod = gunzip(fod)

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
        settings.collectable_actors.append(acts[0])
        return acts[0]
    elif len(acts) == 0:
        colors.printc("~times Error in load(): cannot load", inputobj, c=1)
        return None
    else:
        settings.collectable_actors += acts
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
            actor = Volume(img)
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
                                  vtk.vtkUnstructuredGrid,
                                  vtk.vtkStructuredGrid,
                                  vtk.vtkRectilinearGrid)):
                    acts.append(Mesh(b, c=c, alpha=alpha))
                if isinstance(b, vtk.vtkImageData):
                    acts.append(Volume(b))
            return acts
        else:
            return mb

        ################################################################# numpy:
    elif fl.endswith(".npy"):
        acts = loadNumpy(filename)
        if unpack is False:
            return Assembly(acts)
        return acts

    elif fl.endswith(".geojson") or fl.endswith(".geojson.gz"):
        return loadGeoJSON(fl)

    elif fl.endswith(".pvd"):
        return loadPVD(fl)

    elif fl.endswith(".pdb"):
        return loadPDB(fl)

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

        actor = Mesh(routput, c, alpha)
        if fl.endswith(".txt") or fl.endswith(".xyz"):
            actor.GetProperty().SetPointSize(4)

    actor.filename = filename
    return actor

def download(url, prefix=''):
    """Retrieve a file from a url, save it locally and return its path."""

    if "https://" not in url and "http://" not in url:
        #colors.printc('Invalid URL:\n', url, c=1)
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

    return Mesh(utils.buildPolyData(vertices, faces))


def loadGeoJSON(filename):
    """Load GeoJSON files."""
    if filename.endswith('.gz'):
        filename = gunzip(filename)
    jr = vtk.vtkGeoJSONReader()
    jr.SetFileName(filename)
    jr.Update()
    return Mesh(jr.GetOutput())


def loadDolfin(filename, exterior=False):
    """Reads a `Fenics/Dolfin` file format (.xml or .xdmf).
    Return an ``Mesh(vtkActor)`` object."""
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
        poly = utils.buildPolyData(bm.points(), bm.cells(), fast=True)
    else:
        polyb = utils.buildPolyData(bm.points(), bm.cells(), fast=True)
        polym = utils.buildPolyData(m.points(), m.cells(), fast=True)
        app = vtk.vtkAppendPolyData()
        app.AddInputData(polym)
        app.AddInputData(polyb)
        app.Update()
        poly = app.GetOutput()
    return Mesh(poly).lw(0.1)


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
    return Mesh(poly)


def loadPVD(filename):
    """Reads a paraview set of files."""
    import xml.etree.ElementTree as et

    tree = et.parse(filename)

    dname = os.path.dirname(filename)
    if not dname:
        dname = '.'

    listofobjs = []
    for coll in tree.getroot():
        for dataset in coll:
            fname = dataset.get("file")
            ob = load(dname+'/'+fname)
            tm = dataset.get("timestep")
            if tm:
                ob.time(tm)
            listofobjs.append(ob)
    if len(listofobjs) == 1:
        return listofobjs[0]
    elif len(listofobjs) == 0:
        return None
    else:
        return listofobjs


def loadPDB(filename, bondScale=1, hydrogenBondScale=1, coilWidth=0.3, helixWidth=1.3):
    """Reads a molecule Protein Data Bank file."""
    rr = vtk.vtkPDBReader()
    rr.SetFileName('1btn.pdb')
    rr.SetBScale(bondScale)
    rr.SetHBScale(hydrogenBondScale)
    rr.Update()
    prf = vtk.vtkProteinRibbonFilter()
    prf.SetCoilWidth(coilWidth)
    prf.SetHelixWidth(helixWidth)
    prf.SetInputData(rr.GetOutput())
    prf.Update()
    return Mesh(prf.GetOutput())


def loadNeutral(filename):
    """Reads a `Neutral` tetrahedral file format. Return an ``Mesh(vtkActor)`` object."""
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
    return Mesh(poly)


def loadGmesh(filename):
    """Reads a `gmesh` file format. Return an ``Mesh(vtkActor)`` object."""
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
    return Mesh(poly)


def loadPCD(filename):
    """Return a ``Mesh`` made of only vertex points
    from `Point Cloud` file format. Return an ``Mesh(vtkActor)`` object."""
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
    return Mesh(poly).pointSize(4)


def loadNumpy(inobj):
    """Load a vtkplotter format file."""
    if isinstance(inobj, str):
        data = np.load(inobj, allow_pickle=True, encoding='latin1')
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
        act = Mesh(poly)
        loadcommon(act, d)

        act.mapper().ScalarVisibilityOff()
        if 'celldata' in keys:
            for csc, cscname in d['celldata']:
                act.addCellScalars(csc, cscname)
                if 'normal' not in cscname.lower():
                    act.getCellArray(cscname) # activate
        if 'pointdata' in keys:
            for psc, pscname in d['pointdata']:
                act.addPointScalars(psc, pscname)
                if 'normal' not in pscname.lower():
                    act.getPointArray(pscname) # activate

        prp = act.GetProperty()
        if 'specular' in keys:      prp.SetSpecular(d['specular'])
        if 'specularpower' in keys: prp.SetSpecularPower(d['specularpower'])
        if 'specularcolor' in keys: prp.SetSpecularColor(d['specularcolor'])
        if 'shading' in keys:       prp.SetInterpolation(d['shading'])
        if 'alpha' in keys:         prp.SetOpacity(d['alpha'])
        if 'opacity' in keys:       prp.SetOpacity(d['opacity']) # synonym
        if 'pointsize' in keys and d['pointsize']: prp.SetPointSize(d['pointsize'])
        if 'texture' in keys and d['texture']:     act.texture(d['texture'])
        if 'linewidth' in keys and d['linewidth']: act.lineWidth(d['linewidth'])
        if 'linecolor' in keys and d['linecolor']: act.lineColor(d['linecolor'])
        if 'representation' in keys: prp.SetRepresentation(d['representation'])
        if 'color' in keys and d['color']: act.color(d['color'])
        if 'backColor' in keys and d['backColor']: act.backColor(d['backColor'])

        if 'activedata' in keys and d['activedata'] is not None:
            act.mapper().ScalarVisibilityOn()
            if d['activedata'][0] == 'celldata':
                poly.GetCellData().SetActiveScalars(d['activedata'][1])
            if d['activedata'][0] == 'pointdata':
                poly.GetPointData().SetActiveScalars(d['activedata'][1])

        return act
        ##################

    objs = []
    for d in data:
        #print('loadNumpy:', d)

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
        if hasattr(obj, 'GetProperty'): # assembly doesn't
            prp = obj.GetProperty()
            adict['ambient'] = prp.GetAmbient()
            adict['diffuse'] = prp.GetDiffuse()


    def _doactor(obj, adict):
        adict['points'] = obj.points(transformed=0).astype(np.float32)
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

        for iname in obj.getArrayNames()['PointData']:
            adict['pointdata'].append([obj.getPointArray(iname), iname])
        for iname in obj.getArrayNames()['CellData']:
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
    if isinstance(obj, Mesh):
        adict['type'] = 'mesh'
        _doactor(obj, adict)

    elif isinstance(obj, Assembly):
        adict['type'] = 'assembly'
        adict['actors'] = []
        for a in obj.getMeshes():
            assdict = dict()
            if not isinstance(a, Mesh): #normal vtkActor
                b = Mesh(a) # promote it to a Actor
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
        adict['jittering'] = obj.mapper().GetUseJittering()

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
    if isinstance(obj, Mesh): # picks transformation
        obj = objct.polydata(True)
    elif isinstance(obj, (vtk.vtkActor, vtk.vtkVolume)):
        obj = objct.GetMapper().GetInput()
    elif isinstance(obj, (vtk.vtkPolyData, vtk.vtkImageData)):
        obj = objct

    fr = fileoutput.lower()
    if   fr.endswith(".vtk"):
        writer = vtk.vtkPolyDataWriter()
    elif fr.endswith(".ply"):
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
    elif fr.endswith(".stl"):
        writer = vtk.vtkSTLWriter()
    elif fr.endswith(".vtp"):
        writer = vtk.vtkXMLPolyDataWriter()
    elif fr.endswith(".vtm"):
        g = vtk.vtkMultiBlockDataGroupFilter()
        for ob in objct:
            if isinstance(ob, Mesh): # picks transformation
                ob = ob.polydata(True)
            elif isinstance(ob, (vtk.vtkActor, vtk.vtkVolume)):
                ob = ob.GetMapper().GetInput()
            g.AddInputData(ob)
        g.Update()
        mb = g.GetOutputDataObject(0)
        wri = vtk.vtkXMLMultiBlockDataWriter()
        wri.SetInputData(mb)
        wri.SetFileName(fileoutput)
        wri.Write()
        return mb
    elif fr.endswith(".xyz"):
        writer = vtk.vtkSimplePointsWriter()
    elif fr.endswith(".facet"):
        writer = vtk.vtkFacetWriter()
    elif fr.endswith(".tif"):
        writer = vtk.vtkTIFFWriter()
        writer.SetFileDimensionality(len(obj.GetDimensions()))
    elif fr.endswith(".vti"):
        writer = vtk.vtkXMLImageDataWriter()
    elif fr.endswith(".mhd"):
        writer = vtk.vtkMetaImageWriter()
    elif fr.endswith(".nii"):
        writer = vtk.vtkNIFTIImageWriter()
    elif fr.endswith(".png"):
        writer = vtk.vtkPNGWriter()
    elif fr.endswith(".jpg"):
        writer = vtk.vtkJPEGWriter()
    elif fr.endswith(".bmp"):
        writer = vtk.vtkBMPWriter()
    elif fr.endswith(".npy"):
        if utils.isSequence(objct):
            objslist = objct
        else:
            objslist = [objct]
        dicts2save = []
        for obj in objslist:
            dicts2save.append( _np_dump(obj) )
        np.save(fileoutput, dicts2save)
        return dicts2save

    elif fr.endswith(".obj"):
        outF = open(fileoutput, "w")
        outF.write('# OBJ file format with ext .obj\n')
        outF.write('# File Created by vtkplotter\n')
        cobjct = objct.clone().clean()

        for p in cobjct.points():
            outF.write('v '+ str(p[0]) +" "+ str(p[1])+" "+ str(p[2])+'\n')

        for vn in cobjct.normals(cells=False):
            outF.write('vn '+str(vn[0])+" "+str(vn[1])+" "+str(vn[2])+'\n')

        #pdata = cobjct.polydata().GetPointData().GetScalars()
        #if pdata:
        #    ndata = vtk_to_numpy(pdata)
        #    for vd in ndata:
        #        outF.write('vp '+ str(vd) +'\n')

        #ptxt = cobjct.polydata().GetPointData().GetTCoords() # not working
        #if ptxt:
        #    ntxt = vtk_to_numpy(ptxt)
        #    print(len(cobjct.faces()), cobjct.points().shape, ntxt.shape)
        #    for vt in ntxt:
        #        outF.write('vt '+ str(vt[0]) +" "+ str(vt[1])+ ' 0\n')

        for f in cobjct.faces():
            fs = ''
            for fi in f:
                fs += " "+str(fi+1)
            outF.write('f' + fs + '\n')

        #ldata = cobjct.polydata().GetLines().GetData()
        #print(cobjct.polydata().GetLines())
        #if ldata:
        #    ndata = vtk_to_numpy(ldata)
        #    print(ndata)
        #    for l in ndata:
        #        ls = ''
        #        for li in l:
        #            ls += str(li+1)+" "
        #        outF.write('l '+ ls + '\n')

        outF.close()
        return objct

    elif fr.endswith(".xml"):  # write tetrahedral dolfin xml
        vertices = objct.points().astype(str)
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

    if fr.endswith(".obj"):
        w = vtk.vtkOBJExporter()
        w.SetInputData(settings.plotter_instance.window)
        w.Update()
        colors.printc("~save Saved file:", fileoutput, c="g")

    elif fr.endswith(".obj"):
        writer = vtk.vtkOBJWriter()
        writer.SetInputData(obj)
        writer.SetFileName(fileoutput)
        writer.Write()

    elif fr.endswith(".x3d"):
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
    elif fr.endswith(".npy"):
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
        for a in vp.getMeshes() + vp.getVolumes():
            sdict['objects'].append(_np_dump(a))
        np.save(fileoutput, [sdict])

    return

def importWindow(fileinput, mtlFile=None, texturePath=None):
    """Import a whole scene from a Numpy or OBJ wavefront file.
    Return ``Plotter`` instance.

    :param str mtlFile: MTL file for OBJ wavefront files.
    :param str texturePath: path of the texture files directory.
    """
    from vtkplotter import Plotter

    if '.npy' in fileinput:
        data = np.load(fileinput, allow_pickle=True, encoding="latin1").flatten()[0]

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

        axes = data.pop('axes', 4)
        title = data.pop('title', '')
        backgrcol = data.pop('backgrcol', "blackboard")

        vp = Plotter(#size=data['size'], # not necessarily a good idea to set it
                     #shape=data['shape'],
                     axes=axes,
                     title=title,
                     bg=backgrcol,
        )
        vp.xtitle = data.pop('xtitle', 'x')
        vp.ytitle = data.pop('ytitle', 'y')
        vp.ztitle = data.pop('ztitle', 'z')

        if 'objects' in data.keys():
            objs = loadNumpy(data['objects'])
            if not utils.isSequence(objs):
               objs = [objs]
        else:
            colors.printc("Trying to import a that was not exported.", c=1)
            colors.printc(" -> try to load a single object with load().", c=1)
            return loadNumpy(fileinput)
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

    elif '.obj' in fileinput.lower():

        vp = Plotter()

        importer = vtk.vtkOBJImporter()
        importer.SetFileName(fileinput)
        if mtlFile is not False:
            if mtlFile is None:
                mtlFile = fileinput.replace('.obj', '.mtl').replace('.OBJ', '.MTL')
            importer.SetFileNameMTL(mtlFile)
        if texturePath is not False:
            if texturePath is None:
                texturePath = fileinput.replace('.obj', '.txt').replace('.OBJ', '.TXT')
            importer.SetTexturePath(texturePath)
        importer.SetRenderWindow(vp.window)
        importer.Update()

        actors = vp.renderer.GetActors()
        actors.InitTraversal()
        for i in range(actors.GetNumberOfItems()):
            vactor = actors.GetNextActor()
            act = Mesh(vactor)
            act_tu = vactor.GetTexture()
            if act_tu:
                act_tu.InterpolateOn()
                act.texture(act_tu)
            vp.actors.append( act )
        return vp


##########################################################
def screenshot(filename="screenshot.png", scale=None, returnNumpy=False):
    """
    Save a screenshot of the current rendering window.

    :param int scale: set image magnification
    :param bool returnNumpy: return a numpy array of the image
    """
    if not settings.plotter_instance or not settings.plotter_instance.window:
        colors.printc('~bomb screenshot(): Rendering window is not present, skip.', c=1)
        return

    if scale is None:
        scale = settings.screeshotScale

    if settings.screeshotLargeImage:
       w2if = vtk.vtkRenderLargeImage()
       w2if.SetInput(settings.plotter_instance.renderer)
       w2if.SetMagnification(scale)
    else:
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(settings.plotter_instance.window)
        if hasattr(w2if, 'SetScale'):
            w2if.SetScale(scale, scale)
        if settings.screenshotTransparentBackground:
            w2if.SetInputBufferTypeToRGBA()
        w2if.ReadFrontBufferOff()  # read from the back buffer
    w2if.Update()

    if returnNumpy:
        w2ifout = w2if.GetOutput()
        npdata = vtk_to_numpy(w2ifout.GetPointData().GetArray("ImageScalars"))
        npdata = npdata[:,[0,1,2]]
        ydim, xdim, _ = w2ifout.GetDimensions()
        npdata = npdata.reshape([xdim, ydim, -1])
        npdata = np.flip(npdata, axis=0)
        return npdata

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
            colors.printc("Recalculated video FPS to", round(self.fps, 3), c="m")
        else:
            self.fps = int(self.fps)
        self.name = self.name.split('.')[0]+'.mp4'
        out = os.system(self.ffmpeg + " -loglevel panic -y -r " + str(self.fps)
                        + " -i " + self.tmp_dir.name + os.sep + "%01d.png " + self.name)
        if out:
            colors.printc("ffmpeg returning error", c=1)
        colors.printc("~save Video saved as", self.name, c="m")
        self.tmp_dir.cleanup()
        return


