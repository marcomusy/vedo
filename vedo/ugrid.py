import numpy as np
import vedo
import vedo.colors as colors
import vedo.utils as utils
import vtk
from vedo import settings
from vedo.base import BaseGrid

__doc__ = """
Work with unstructured grid datasets
"""

__all__ = ["UGrid"]

#########################################################################
class UGrid(vtk.vtkActor, BaseGrid):
    """Support for UnstructuredGrid objects."""
    def __init__(self, inputobj=None):

        vtk.vtkActor.__init__(self)
        BaseGrid.__init__(self)

        inputtype = str(type(inputobj))
        self._data = None
        self._polydata = None
        self.name = "UGrid"

        ###################
        if inputobj is None:
            self._data = vtk.vtkUnstructuredGrid()

        elif utils.isSequence(inputobj):

            pts, cells, celltypes = inputobj

            self._data = vtk.vtkUnstructuredGrid()

            if not utils.isSequence(cells[0]):
                tets=[]
                nf=cells[0]+1
                for i, cl in enumerate(cells):
                    if i==nf or i==0:
                        k = i+1
                        nf = cl+k
                        cell = [cells[j+k] for j in range(cl)]
                        tets.append(cell)
                cells = tets

            # This would fill the points and use those to define orientation
            vpts = utils.numpy2vtk(pts, dtype=float)
            points = vtk.vtkPoints()
            points.SetData(vpts)
            self._data.SetPoints(points)

            # This fill the points and use cells to define orientation
            # points = vtk.vtkPoints()
            # for c in cells:
            #       for pid in c:
            #           points.InsertNextPoint(pts[pid])
            # self._data.SetPoints(points)

            # Fill cells
            # https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
            for i, ct in enumerate(celltypes):
                cell_conn =  cells[i]
                if ct == vtk.VTK_HEXAHEDRON:
                    cell = vtk.vtkHexahedron()
                elif ct == vtk.VTK_TETRA:
                    cell = vtk.vtkTetra()
                elif ct == vtk.VTK_VOXEL:
                    cell = vtk.vtkVoxel()
                elif ct == vtk.VTK_WEDGE:
                    cell = vtk.vtkWedge()
                elif ct == vtk.VTK_PYRAMID:
                    cell = vtk.vtkPyramid()
                elif ct == vtk.VTK_HEXAGONAL_PRISM:
                    cell = vtk.vtkHexagonalPrism()
                elif ct == vtk.VTK_PENTAGONAL_PRISM:
                    cell = vtk.vtkPentagonalPrism()
                else:
                    print("UGrid: cell type", ct, "not implemented. Skip.")
                    continue
                cpids = cell.GetPointIds()
                for j, pid in enumerate(cell_conn):
                    cpids.SetId(j, pid)
                self._data.InsertNextCell(ct, cpids)

        elif "UnstructuredGrid" in inputtype:
            self._data = inputobj

        elif isinstance(inputobj, str):
            from vedo.io import download, loadUnStructuredGrid
            if "https://" in inputobj:
                inputobj = download(inputobj, verbose=False)
            self._data = loadUnStructuredGrid(inputobj)
            self.filename = inputobj

        else:
            vedo.logger.error(f"cannot understand input type {inputtype}")
            return

        # self._mapper = vtk.vtkDataSetMapper()
        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetInterpolateScalarsBeforeMapping(settings.interpolateScalarsBeforeMapping)

        if settings.usePolygonOffset:
            self._mapper.SetResolveCoincidentTopologyToPolygonOffset()
            pof, pou = settings.polygonOffsetFactor, settings.polygonOffsetUnits
            self._mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(pof, pou)
        self.GetProperty().SetInterpolationToFlat()

        if not self._data:
            return

        # now fill the representation of the vtk unstr grid
        sf = vtk.vtkShrinkFilter()
        sf.SetInputData(self._data)
        sf.SetShrinkFactor(1.0)
        sf.Update()
        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(sf.GetOutput())
        gf.Update()
        self._polydata = gf.GetOutput()

        self._mapper.SetInputData(self._polydata)
        sc = None
        if self.useCells:
            sc = self._polydata.GetCellData().GetScalars()
        else:
            sc = self._polydata.GetPointData().GetScalars()
        if sc:
            self._mapper.SetScalarRange(sc.GetRange())

        self.SetMapper(self._mapper)
        self.property = self.GetProperty()

    # ------------------------------------------------------------------


    def clone(self):
        """Clone the UGrid object to yield an exact copy."""
        ugCopy = vtk.vtkUnstructuredGrid()
        ugCopy.DeepCopy(self._data)

        cloned = UGrid(ugCopy)
        pr = self.GetProperty()
        if isinstance(pr, vtk.vtkVolumeProperty):
            prv = vtk.vtkVolumeProperty()
        else:
            prv = vtk.vtkProperty()
        prv.DeepCopy(pr)
        cloned.SetProperty(prv)
        cloned.property = prv

        #assign the same transformation to the copy
        cloned.SetOrigin(self.GetOrigin())
        cloned.SetScale(self.GetScale())
        cloned.SetOrientation(self.GetOrientation())
        cloned.SetPosition(self.GetPosition())
        cloned.name = self.name
        return cloned


    def color(self, c=False, alpha=None):
        """
        Set/get UGrid color.
        If None is passed as input, will use colors from active scalars.
        Same as `ugrid.c()`.
        """
        if c is False:
            return np.array(self.GetProperty().GetColor())
        elif c is None:
            self._mapper.ScalarVisibilityOn()
            return self
        self._mapper.ScalarVisibilityOff()
        cc = colors.getColor(c)
        self.property.SetColor(cc)
        if self.trail:
            self.trail.GetProperty().SetColor(cc)
        if alpha is not None:
            self.alpha(alpha)
        return self


    def alpha(self, opacity=None):
        """Set/get mesh's transparency. Same as `mesh.opacity()`."""
        if opacity is None:
            return self.property.GetOpacity()

        self.property.SetOpacity(opacity)
        bfp = self.GetBackfaceProperty()
        if bfp:
            if opacity < 1:
                self._bfprop = bfp
                self.SetBackfaceProperty(None)
            else:
                self.SetBackfaceProperty(self._bfprop)
        return self

    def opacity(self, alpha=None):
        """Set/get mesh's transparency. Same as `mesh.alpha()`."""
        return self.alpha(alpha)

    def wireframe(self, value=True):
        """Set mesh's representation as wireframe or solid surface.
        Same as `mesh.wireframe()`."""
        if value:
            self.property.SetRepresentationToWireframe()
        else:
            self.property.SetRepresentationToSurface()
        return self

    def lineWidth(self, lw=None):
        """Set/get width of mesh edges. Same as `lw()`."""
        if lw is not None:
            if lw == 0:
                self.property.EdgeVisibilityOff()
                self.property.SetRepresentationToSurface()
                return self
            self.property.EdgeVisibilityOn()
            self.property.SetLineWidth(lw)
        else:
            return self.property.GetLineWidth()
        return self

    def lw(self, lineWidth=None):
        """Set/get width of mesh edges. Same as `lineWidth()`."""
        return self.lineWidth(lineWidth)

    def lineColor(self, lc=None):
        """Set/get color of mesh edges. Same as `lc()`."""
        if lc is not None:
            if "ireframe" in self.property.GetRepresentationAsString():
                self.property.EdgeVisibilityOff()
                self.color(lc)
                return self
            self.property.EdgeVisibilityOn()
            self.property.SetEdgeColor(colors.getColor(lc))
        else:
            return self.property.GetEdgeColor()
        return self

    def lc(self, lineColor=None):
        """Set/get color of mesh edges. Same as `lineColor()`."""
        return self.lineColor(lineColor)


    def extractCellType(self, ctype):
        """Extract a specific cell type and return a new UGrid."""
        uarr = self._data.GetCellTypesArray()
        ctarrtyp = np.where(utils.vtk2numpy(uarr)==ctype)[0]
        uarrtyp = utils.numpy2vtk(ctarrtyp, deep=False, dtype='id')
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(uarrtyp)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        es = vtk.vtkExtractSelection()
        es.SetInputData(0, self._data)
        es.SetInputData(1, selection)
        es.Update()
        return UGrid(es.GetOutput())
