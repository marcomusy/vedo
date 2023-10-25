#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset of vtk classes to be imported directly
"""
import importlib


location = dict()
module_cache = {}


def get(module_name="", cls_name=""):
    if not cls_name:
        cls_name = module_name
        module_name = location[cls_name]
    module_name = "vtkmodules." + module_name
    if module_name not in module_cache:
        module = importlib.import_module(module_name)
        module_cache[module_name] = module
    if cls_name:
        return getattr(module_cache[module_name], cls_name)
    else:
        return module_cache[module_name]


####################################################


import vtkmodules.vtkCommonComputationalGeometry

from vtkmodules.vtkCommonColor import vtkNamedColors

location["vtkNamedColors"] = "vtkCommonColor"


from vtkmodules.vtkCommonCore import (
    mutable,
    VTK_UNSIGNED_CHAR,
    VTK_UNSIGNED_SHORT,
    VTK_UNSIGNED_INT,
    VTK_UNSIGNED_LONG,
    VTK_UNSIGNED_LONG_LONG,
    VTK_UNSIGNED_CHAR,
    VTK_CHAR,
    VTK_SHORT,
    VTK_INT,
    VTK_LONG,
    VTK_LONG_LONG,
    VTK_FLOAT,
    VTK_DOUBLE,
    VTK_SIGNED_CHAR,
    VTK_ID_TYPE,
    VTK_VERSION_NUMBER,
    VTK_FONT_FILE,
    vtkArray,
    vtkIdTypeArray,
    vtkBitArray,
    vtkCharArray,
    vtkDoubleArray,
    vtkFloatArray,
    vtkIdList,
    vtkIntArray,
    vtkLookupTable,
    vtkMath,
    vtkPoints,
    vtkStringArray,
    vtkUnsignedCharArray,
    vtkVariant,
    vtkVariantArray,
    vtkVersion,
)

as_strings = [
    "mutable",
    "VTK_UNSIGNED_CHAR",
    "VTK_UNSIGNED_SHORT",
    "VTK_UNSIGNED_INT",
    "VTK_UNSIGNED_LONG",
    "VTK_UNSIGNED_LONG_LONG",
    "VTK_UNSIGNED_CHAR",
    "VTK_CHAR",
    "VTK_SHORT",
    "VTK_INT",
    "VTK_LONG",
    "VTK_LONG_LONG",
    "VTK_FLOAT",
    "VTK_DOUBLE",
    "VTK_SIGNED_CHAR",
    "VTK_ID_TYPE",
    "VTK_VERSION_NUMBER",
    "VTK_FONT_FILE",
    "vtkArray",
    "vtkIdTypeArray",
    "vtkBitArray",
    "vtkCharArray",
    "vtkDoubleArray",
    "vtkFloatArray",
    "vtkIdList",
    "vtkIntArray",
    "vtkLookupTable",
    "vtkMath",
    "vtkPoints",
    "vtkStringArray",
    "vtkUnsignedCharArray",
    "vtkVariant",
    "vtkVariantArray",
    "vtkVersion",
]
for name in as_strings:
    location[name] = "vtkCommonCore"


from vtkmodules.vtkCommonDataModel import (
    VTK_HEXAHEDRON,
    VTK_TETRA,
    VTK_VOXEL,
    VTK_WEDGE,
    VTK_PYRAMID,
    VTK_HEXAGONAL_PRISM,
    VTK_PENTAGONAL_PRISM,
    vtkCellArray,
    vtkBox,
    vtkCellLocator,
    vtkCylinder,
    vtkDataSetAttributes,
    vtkDataObject,
    vtkDataSet,
    vtkFieldData,
    vtkHexagonalPrism,
    vtkHexahedron,
    vtkImageData,
    vtkImplicitDataSet,
    vtkImplicitSelectionLoop,
    vtkImplicitWindowFunction,
    vtkIterativeClosestPointTransform,
    vtkLine,
    vtkMultiBlockDataSet,
    vtkMutableDirectedGraph,
    vtkPentagonalPrism,
    vtkPlane,
    vtkPlanes,
    vtkPointLocator,
    vtkPolyData,
    vtkPolyLine,
    vtkPolyPlane,
    vtkPolygon,
    vtkPyramid,
    vtkQuadric,
    vtkRectilinearGrid,
    vtkSelection,
    vtkSelectionNode,
    vtkSphere,
    vtkStaticCellLocator,
    vtkStaticPointLocator,
    vtkStructuredGrid,
    vtkTetra,
    vtkTriangle,
    vtkUnstructuredGrid,
    vtkVoxel,
    vtkWedge,
)

as_strings = [
    "VTK_HEXAHEDRON",
    "VTK_TETRA",
    "VTK_VOXEL",
    "VTK_WEDGE",
    "VTK_PYRAMID",
    "VTK_HEXAGONAL_PRISM",
    "VTK_PENTAGONAL_PRISM",
    "vtkCellArray",
    "vtkBox",
    "vtkCellLocator",
    "vtkCylinder",
    "vtkDataSetAttributes",
    "vtkDataObject",
    "vtkDataSet",
    "vtkFieldData",
    "vtkHexagonalPrism",
    "vtkHexahedron",
    "vtkImageData",
    "vtkImplicitDataSet",
    "vtkImplicitSelectionLoop",
    "vtkImplicitWindowFunction",
    "vtkIterativeClosestPointTransform",
    "vtkLine",
    "vtkMultiBlockDataSet",
    "vtkMutableDirectedGraph",
    "vtkPentagonalPrism",
    "vtkPlane",
    "vtkPlanes",
    "vtkPointLocator",
    "vtkPolyData",
    "vtkPolyLine",
    "vtkPolyPlane",
    "vtkPolygon",
    "vtkPyramid",
    "vtkQuadric",
    "vtkRectilinearGrid",
    "vtkSelection",
    "vtkSelectionNode",
    "vtkSphere",
    "vtkStaticCellLocator",
    "vtkStaticPointLocator",
    "vtkStructuredGrid",
    "vtkTetra",
    "vtkTriangle",
    "vtkUnstructuredGrid",
    "vtkVoxel",
    "vtkWedge",
]
for name in as_strings:
    location[name] = "vtkCommonDataModel"


from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm
location["vtkAlgorithm"] = "vtkCommonExecutionModel"

from vtkmodules.vtkCommonMath import vtkMatrix4x4, vtkQuaternion
location["vtkMatrix4x4"] = "vtkCommonMath"
location["vtkQuaternion"] = "vtkCommonMath"

from vtkmodules.vtkCommonTransforms import (
    vtkHomogeneousTransform,
    vtkLandmarkTransform,
    vtkLinearTransform,
    vtkThinPlateSplineTransform,
    vtkTransform,
)
location["vtkHomogeneousTransform"] = "vtkCommonTransforms"
location["vtkLandmarkTransform"] = "vtkCommonTransforms"
location["vtkLinearTransform"] = "vtkCommonTransforms"
location["vtkThinPlateSplineTransform"] = "vtkCommonTransforms"
location["vtkTransform"] = "vtkCommonTransforms"

from vtkmodules.vtkFiltersCore import (
    VTK_BEST_FITTING_PLANE,
    vtk3DLinearGridCrinkleExtractor,
    vtkAppendPolyData,
    vtkCellCenters,
    vtkCellDataToPointData,
    vtkCenterOfMass,
    vtkCleanPolyData,
    vtkClipPolyData,
    vtkPolyDataConnectivityFilter,
    vtkPolyDataEdgeConnectivityFilter,
    vtkContourFilter,
    vtkContourGrid,
    vtkCutter,
    vtkDecimatePro,
    vtkDelaunay2D,
    vtkDelaunay3D,
    vtkElevationFilter,
    vtkFeatureEdges,
    vtkFlyingEdges3D,
    vtkGlyph3D,
    vtkIdFilter,
    vtkImageAppend,
    vtkImplicitPolyDataDistance,
    vtkMarchingSquares,
    vtkMaskPoints,
    vtkMassProperties,
    vtkPointDataToCellData,
    vtkPolyDataNormals,
    vtkProbeFilter,
    vtkQuadricDecimation,
    vtkResampleWithDataSet,
    vtkReverseSense,
    vtkStripper,
    vtkTensorGlyph,
    vtkThreshold,
    vtkTriangleFilter,
    vtkTubeFilter,
    vtkUnstructuredGridQuadricDecimation,
    vtkVoronoi2D,
    vtkWindowedSincPolyDataFilter,
)

as_strings = [
    "VTK_BEST_FITTING_PLANE",
    "vtk3DLinearGridCrinkleExtractor",
    "vtkAppendPolyData",
    "vtkCellCenters",
    "vtkCellDataToPointData",
    "vtkCenterOfMass",
    "vtkCleanPolyData",
    "vtkClipPolyData",
    "vtkPolyDataConnectivityFilter",
    "vtkPolyDataEdgeConnectivityFilter",
    "vtkContourFilter",
    "vtkContourGrid",
    "vtkCutter",
    "vtkDecimatePro",
    "vtkDelaunay2D",
    "vtkDelaunay3D",
    "vtkElevationFilter",
    "vtkFeatureEdges",
    "vtkFlyingEdges3D",
    "vtkGlyph3D",
    "vtkIdFilter",
    "vtkImageAppend",
    "vtkImplicitPolyDataDistance",
    "vtkMarchingSquares",
    "vtkMaskPoints",
    "vtkMassProperties",
    "vtkPointDataToCellData",
    "vtkPolyDataNormals",
    "vtkProbeFilter",
    "vtkQuadricDecimation",
    "vtkResampleWithDataSet",
    "vtkReverseSense",
    "vtkStripper",
    "vtkTensorGlyph",
    "vtkThreshold",
    "vtkTriangleFilter",
    "vtkTubeFilter",
    "vtkUnstructuredGridQuadricDecimation",
    "vtkVoronoi2D",
    "vtkWindowedSincPolyDataFilter",
]
for name in as_strings:
    location[name] = "vtkFiltersCore"


try:
    from vtkmodules.vtkFiltersCore import (
        vtkStaticCleanUnstructuredGrid,
        vtkPolyDataPlaneCutter,
    )
    location["vtkStaticCleanUnstructuredGrid"] = "vtkFiltersCore"
    location["vtkPolyDataPlaneCutter"] = "vtkFiltersCore"
except ImportError:
    pass


from vtkmodules.vtkFiltersExtraction import (
    vtkExtractCellsByType,
    vtkExtractGeometry,
    vtkExtractPolyDataGeometry,
    vtkExtractSelection,
)
as_strings = [
    "vtkExtractCellsByType",
    "vtkExtractGeometry",
    "vtkExtractPolyDataGeometry",
    "vtkExtractSelection",
]
for name in as_strings:
    location[name] = "vtkFiltersExtraction"

try:
    from vtkmodules.vtkFiltersExtraction import vtkExtractEdges  # vtk9.0
    location["vtkExtractEdges"] = "vtkFiltersExtraction"
except ImportError:
    from vtkmodules.vtkFiltersCore import vtkExtractEdges  # vtk9.2
    location["vtkExtractEdges"] = "vtkFiltersCore"

from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
location["vtkStreamTracer"] = "vtkFiltersFlowPaths"


from vtkmodules.vtkFiltersGeneral import (
    vtkBooleanOperationPolyDataFilter,
    vtkBoxClipDataSet,
    vtkCellValidator,
    vtkClipDataSet,
    vtkCountVertices,
    vtkContourTriangulator,
    vtkCurvatures,
    vtkDataSetTriangleFilter,
    vtkDensifyPolyData,
    vtkDistancePolyDataFilter,
    vtkGradientFilter,
    vtkIntersectionPolyDataFilter,
    vtkLoopBooleanPolyDataFilter,
    vtkMultiBlockDataGroupFilter,
    vtkTransformPolyDataFilter,
    vtkOBBTree,
    vtkQuantizePolyDataPoints,
    vtkRandomAttributeGenerator,
    vtkShrinkFilter,
    vtkShrinkPolyData,
    vtkRectilinearGridToTetrahedra,
    vtkVertexGlyphFilter,
)
as_strings = [
    "vtkBooleanOperationPolyDataFilter",
    "vtkBoxClipDataSet",
    "vtkCellValidator",
    "vtkClipDataSet",
    "vtkCountVertices",
    "vtkContourTriangulator",
    "vtkCurvatures",
    "vtkDataSetTriangleFilter",
    "vtkDensifyPolyData",
    "vtkDistancePolyDataFilter",
    "vtkGradientFilter",
    "vtkIntersectionPolyDataFilter",
    "vtkLoopBooleanPolyDataFilter",
    "vtkMultiBlockDataGroupFilter",
    "vtkTransformPolyDataFilter",
    "vtkOBBTree",
    "vtkQuantizePolyDataPoints",
    "vtkRandomAttributeGenerator",
    "vtkShrinkFilter",
    "vtkShrinkPolyData",
    "vtkRectilinearGridToTetrahedra",
    "vtkVertexGlyphFilter",
]
for name in as_strings:
    location[name] = "vtkFiltersGeneral"

try:
    from vtkmodules.vtkCommonDataModel import vtkCellTreeLocator
    location["vtkCellTreeLocator"] = "vtkCommonDataModel"
except ImportError:
    from vtkmodules.vtkFiltersGeneral import vtkCellTreeLocator
    location["vtkCellTreeLocator"] = "vtkFiltersGeneral"

from vtkmodules.vtkFiltersGeometry import (
    vtkGeometryFilter,
    vtkDataSetSurfaceFilter,
    vtkImageDataGeometryFilter,
)
location["vtkGeometryFilter"] = "vtkFiltersGeometry"
location["vtkDataSetSurfaceFilter"] = "vtkFiltersGeometry"
location["vtkImageDataGeometryFilter"] = "vtkFiltersGeometry"

try:
    from vtkmodules.vtkFiltersGeometry import vtkMarkBoundaryFilter
    location["vtkMarkBoundaryFilter"] = "vtkFiltersGeometry"
except ImportError:
    pass


from vtkmodules.vtkFiltersHybrid import (
    vtkFacetReader,
    vtkImplicitModeller,
    vtkPolyDataSilhouette,
    vtkProcrustesAlignmentFilter,
    vtkRenderLargeImage,
)
as_strings = [
    "vtkFacetReader",
    "vtkImplicitModeller",
    "vtkPolyDataSilhouette",
    "vtkProcrustesAlignmentFilter",
    "vtkRenderLargeImage",
]
for name in as_strings:
    location[name] = "vtkFiltersHybrid"


from vtkmodules.vtkFiltersModeling import (
    vtkAdaptiveSubdivisionFilter,
    vtkBandedPolyDataContourFilter,
    vtkButterflySubdivisionFilter,
    vtkContourLoopExtraction,
    vtkCookieCutter,
    vtkDijkstraGraphGeodesicPath,
    vtkFillHolesFilter,
    vtkHausdorffDistancePointSetFilter,
    vtkLinearExtrusionFilter,
    vtkLinearSubdivisionFilter,
    vtkLoopSubdivisionFilter,
    vtkRibbonFilter,
    vtkRotationalExtrusionFilter,
    vtkRuledSurfaceFilter,
    vtkSectorSource,
    vtkSelectEnclosedPoints,
    vtkSelectPolyData,
    vtkSubdivideTetra,
)
as_strings = [
    "vtkAdaptiveSubdivisionFilter",
    "vtkBandedPolyDataContourFilter",
    "vtkButterflySubdivisionFilter",
    "vtkContourLoopExtraction",
    "vtkCookieCutter",
    "vtkDijkstraGraphGeodesicPath",
    "vtkFillHolesFilter",
    "vtkHausdorffDistancePointSetFilter",
    "vtkLinearExtrusionFilter",
    "vtkLinearSubdivisionFilter",
    "vtkLoopSubdivisionFilter",
    "vtkRibbonFilter",
    "vtkRotationalExtrusionFilter",
    "vtkRuledSurfaceFilter",
    "vtkSectorSource",
    "vtkSelectEnclosedPoints",
    "vtkSelectPolyData",
    "vtkSubdivideTetra",
]
for name in as_strings:
    location[name] = "vtkFiltersModeling"

try:
    from vtkmodules.vtkFiltersModeling import vtkCollisionDetectionFilter
    location["vtkCollisionDetectionFilter"] = "vtkFiltersModeling"
except ImportError:
    pass

try:
    from vtkmodules.vtkFiltersModeling import vtkImprintFilter
    location["vtkImprintFilter"] = "vtkFiltersModeling"
except ImportError:
    pass

from vtkmodules.vtkFiltersPoints import (
    vtkConnectedPointsFilter,
    vtkDensifyPointCloudFilter,
    vtkEuclideanClusterExtraction,
    vtkExtractEnclosedPoints,
    vtkExtractSurface,
    vtkGaussianKernel,
    vtkLinearKernel,
    vtkPCANormalEstimation,
    vtkPointDensityFilter,
    vtkPointInterpolator,
    vtkRadiusOutlierRemoval,
    vtkShepardKernel,
    vtkSignedDistance,
    vtkVoronoiKernel,
)
as_strings = [
    "vtkConnectedPointsFilter",
    "vtkDensifyPointCloudFilter",
    "vtkEuclideanClusterExtraction",
    "vtkExtractEnclosedPoints",
    "vtkExtractSurface",
    "vtkGaussianKernel",
    "vtkLinearKernel",
    "vtkPCANormalEstimation",
    "vtkPointDensityFilter",
    "vtkPointInterpolator",
    "vtkRadiusOutlierRemoval",
    "vtkShepardKernel",
    "vtkSignedDistance",
    "vtkVoronoiKernel",
]
for name in as_strings:
    location[name] = "vtkFiltersPoints"


from vtkmodules.vtkFiltersSources import (
    vtkArcSource,
    vtkArrowSource,
    vtkConeSource,
    vtkCubeSource,
    vtkCylinderSource,
    vtkDiskSource,
    vtkFrustumSource,
    vtkGlyphSource2D,
    vtkGraphToPolyData,
    vtkLineSource,
    vtkOutlineCornerFilter,
    vtkParametricFunctionSource,
    vtkPlaneSource,
    vtkPointSource,
    vtkProgrammableSource,
    vtkSphereSource,
    vtkTexturedSphereSource,
    vtkTessellatedBoxSource,
)
as_strings = [
    "vtkArcSource",
    "vtkArrowSource",
    "vtkConeSource",
    "vtkCubeSource",
    "vtkCylinderSource",
    "vtkDiskSource",
    "vtkFrustumSource",
    "vtkGlyphSource2D",
    "vtkGraphToPolyData",
    "vtkLineSource",
    "vtkOutlineCornerFilter",
    "vtkParametricFunctionSource",
    "vtkPlaneSource",
    "vtkPointSource",
    "vtkProgrammableSource",
    "vtkSphereSource",
    "vtkTexturedSphereSource",
    "vtkTessellatedBoxSource",
]
for name in as_strings:
    location[name] = "vtkFiltersSources"


from vtkmodules.vtkFiltersTexture import vtkTextureMapToPlane
location["vtkTextureMapToPlane"] = "vtkFiltersTexture"

from vtkmodules.vtkFiltersVerdict import vtkMeshQuality, vtkCellSizeFilter
location["vtkMeshQuality"] = "vtkFiltersVerdict"
location["vtkCellSizeFilter"] = "vtkFiltersVerdict"

from vtkmodules.vtkImagingStencil import vtkPolyDataToImageStencil
location["vtkPolyDataToImageStencil"] = "vtkImagingStencil"

from vtkmodules.vtkIOExport import vtkX3DExporter
location["vtkX3DExporter"] = "vtkIOExport"

from vtkmodules.vtkIOExportGL2PS import vtkGL2PSExporter
location["vtkGL2PSExporter"] = "vtkIOExportGL2PS"

from vtkmodules.vtkIOGeometry import (
    vtkBYUReader,
    vtkFacetWriter,
    vtkOBJReader,
    vtkOpenFOAMReader,
    vtkParticleReader,
    vtkSTLReader,
    vtkSTLWriter,
)
as_strings = [
    "vtkBYUReader",
    "vtkFacetWriter",
    "vtkOBJReader",
    "vtkOpenFOAMReader",
    "vtkParticleReader",
    "vtkSTLReader",
    "vtkSTLWriter",
]
for name in as_strings:
    location[name] = "vtkIOGeometry"


from vtkmodules.vtkIOImage import (
    vtkBMPReader,
    vtkBMPWriter,
    vtkDEMReader,
    vtkDICOMImageReader,
    vtkHDRReader,
    vtkJPEGReader,
    vtkJPEGWriter,
    vtkMetaImageReader,
    vtkMetaImageWriter,
    vtkNIFTIImageReader,
    vtkNIFTIImageWriter,
    vtkNrrdReader,
    vtkPNGReader,
    vtkPNGWriter,
    vtkSLCReader,
    vtkTIFFReader,
    vtkTIFFWriter,
)
as_strings = [
    "vtkBMPReader",
    "vtkBMPWriter",
    "vtkDEMReader",
    "vtkDICOMImageReader",
    "vtkHDRReader",
    "vtkJPEGReader",
    "vtkJPEGWriter",
    "vtkMetaImageReader",
    "vtkMetaImageWriter",
    "vtkNIFTIImageReader",
    "vtkNIFTIImageWriter",
    "vtkNrrdReader",
    "vtkPNGReader",
    "vtkPNGWriter",
    "vtkSLCReader",
    "vtkTIFFReader",
    "vtkTIFFWriter",
]
for name in as_strings:
    location[name] = "vtkIOImage"

from vtkmodules.vtkIOImport import (
    vtk3DSImporter,
    vtkOBJImporter,
    vtkVRMLImporter,
)
location["vtk3DSImporter"] = "vtkIOImport"
location["vtkOBJImporter"] = "vtkIOImport"
location["vtkVRMLImporter"] = "vtkIOImport"


from vtkmodules.vtkIOLegacy import (
    vtkSimplePointsWriter,
    vtkStructuredGridReader,
    vtkStructuredPointsReader,
    vtkDataSetReader,
    vtkDataSetWriter,
    vtkPolyDataWriter,
    vtkRectilinearGridReader,
    vtkUnstructuredGridReader,
)
as_strings = [
    "vtkSimplePointsWriter",
    "vtkStructuredGridReader",
    "vtkStructuredPointsReader",
    "vtkDataSetReader",
    "vtkDataSetWriter",
    "vtkPolyDataWriter",
    "vtkRectilinearGridReader",
    "vtkUnstructuredGridReader",
]
for name in as_strings:
    location[name] = "vtkIOLegacy"


from vtkmodules.vtkIOPLY import vtkPLYReader, vtkPLYWriter
location["vtkPLYReader"] = "vtkIOPLY"
location["vtkPLYWriter"] = "vtkIOPLY"

from vtkmodules.vtkIOXML import (
    vtkXMLGenericDataObjectReader,
    vtkXMLImageDataReader,
    vtkXMLImageDataWriter,
    vtkXMLMultiBlockDataReader,
    vtkXMLMultiBlockDataWriter,
    vtkXMLPRectilinearGridReader,
    vtkXMLPUnstructuredGridReader,
    vtkXMLPolyDataReader,
    vtkXMLPolyDataWriter,
    vtkXMLRectilinearGridReader,
    vtkXMLStructuredGridReader,
    vtkXMLUnstructuredGridReader,
    vtkXMLUnstructuredGridWriter,
)
as_strings = [
    "vtkXMLGenericDataObjectReader",
    "vtkXMLImageDataReader",
    "vtkXMLImageDataWriter",
    "vtkXMLMultiBlockDataReader",
    "vtkXMLMultiBlockDataWriter",
    "vtkXMLPRectilinearGridReader",
    "vtkXMLPUnstructuredGridReader",
    "vtkXMLPolyDataReader",
    "vtkXMLPolyDataWriter",
    "vtkXMLRectilinearGridReader",
    "vtkXMLStructuredGridReader",
    "vtkXMLUnstructuredGridReader",
    "vtkXMLUnstructuredGridWriter",
]
for name in as_strings:
    location[name] = "vtkIOXML"


from vtkmodules.vtkImagingColor import (
    vtkImageLuminance,
    vtkImageMapToWindowLevelColors,
)
location["vtkImageLuminance"] = "vtkImagingColor"
location["vtkImageMapToWindowLevelColors"] = "vtkImagingColor"

from vtkmodules.vtkImagingCore import (
    vtkExtractVOI,
    vtkImageAppendComponents,
    vtkImageBlend,
    vtkImageCast,
    vtkImageConstantPad,
    vtkImageExtractComponents,
    vtkImageFlip,
    vtkImageMapToColors,
    vtkImageMirrorPad,
    vtkImagePermute,
    vtkImageResample,
    vtkImageResize,
    vtkImageReslice,
    vtkImageThreshold,
    vtkImageTranslateExtent,
)
as_strings = [
    "vtkImageAppendComponents",
    "vtkImageBlend",
    "vtkImageCast",
    "vtkImageConstantPad",
    "vtkImageExtractComponents",
    "vtkImageFlip",
    "vtkImageMapToColors",
    "vtkImageMirrorPad",
    "vtkImagePermute",
    "vtkImageResample",
    "vtkImageResize",
    "vtkImageReslice",
    "vtkImageThreshold",
    "vtkImageTranslateExtent",
]
for name in as_strings:
    location[name] = "vtkImagingCore"

from vtkmodules.vtkImagingFourier import (
    vtkImageButterworthHighPass,
    vtkImageButterworthLowPass,
    vtkImageFFT,
    vtkImageFourierCenter,
    vtkImageRFFT,
)
as_strings = [
    "vtkImageButterworthHighPass",
    "vtkImageButterworthLowPass",
    "vtkImageFFT",
    "vtkImageFourierCenter",
    "vtkImageRFFT",
]
for name in as_strings:
    location[name] = "vtkImagingFourier"

from vtkmodules.vtkImagingGeneral import (
    vtkImageCorrelation,
    vtkImageEuclideanDistance,
    vtkImageGaussianSmooth,
    vtkImageGradient,
    vtkImageHybridMedian2D,
    vtkImageLaplacian,
    vtkImageMedian3D,
    vtkImageNormalize,
)
as_strings = [
    "vtkImageCorrelation",
    "vtkImageEuclideanDistance",
    "vtkImageGaussianSmooth",
    "vtkImageGradient",
    "vtkImageHybridMedian2D",
    "vtkImageLaplacian",
    "vtkImageMedian3D",
    "vtkImageNormalize",
]
for name in as_strings:
    location[name] = "vtkImagingGeneral"

from vtkmodules.vtkImagingHybrid import vtkImageToPoints, vtkSampleFunction
from vtkmodules.vtkImagingMath import (
    vtkImageDivergence,
    vtkImageDotProduct,
    vtkImageLogarithmicScale,
    vtkImageMagnitude,
    vtkImageMathematics,
)
as_strings = [
    "vtkImageDivergence",
    "vtkImageDotProduct",
    "vtkImageLogarithmicScale",
    "vtkImageMagnitude",
    "vtkImageMathematics",
]
for name in as_strings:
    location[name] = "vtkImagingMath"

from vtkmodules.vtkImagingMorphological import (
    vtkImageContinuousDilate3D,
    vtkImageContinuousErode3D,
)
as_strings = [
    "vtkImageContinuousDilate3D",
    "vtkImageContinuousErode3D",
]
for name in as_strings:
    location[name] = "vtkImagingMorphological"

from vtkmodules.vtkImagingSources import vtkImageCanvasSource2D
location["vtkImageCanvasSource2D"] = "vtkImagingSources"

from vtkmodules.vtkImagingStencil import vtkImageStencil
location["vtkImageStencil"] = "vtkImagingStencil"

from vtkmodules.vtkInfovisLayout import (
    vtkCircularLayoutStrategy,
    vtkClustering2DLayoutStrategy,
    vtkConeLayoutStrategy,
    vtkFast2DLayoutStrategy,
    vtkForceDirectedLayoutStrategy,
    vtkGraphLayout,
    vtkSimple2DLayoutStrategy,
    vtkSimple3DCirclesStrategy,
    vtkSpanTreeLayoutStrategy,
)
as_strings = [
    "vtkCircularLayoutStrategy",
    "vtkClustering2DLayoutStrategy",
    "vtkConeLayoutStrategy",
    "vtkFast2DLayoutStrategy",
    "vtkForceDirectedLayoutStrategy",
    "vtkGraphLayout",
    "vtkSimple2DLayoutStrategy",
    "vtkSimple3DCirclesStrategy",
    "vtkSpanTreeLayoutStrategy",
]
for name in as_strings:
    location[name] = "vtkInfovisLayout"

from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleFlight,
    vtkInteractorStyleImage,
    vtkInteractorStyleJoystickActor,
    vtkInteractorStyleJoystickCamera,
    vtkInteractorStyleRubberBand2D,
    vtkInteractorStyleRubberBand3D,
    vtkInteractorStyleRubberBandZoom,
    vtkInteractorStyleTerrain,
    vtkInteractorStyleTrackballActor,
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleUnicam,
    vtkInteractorStyleUser,
)

as_strings = [
    "vtkInteractorStyleFlight",
    "vtkInteractorStyleImage",
    "vtkInteractorStyleJoystickActor",
    "vtkInteractorStyleJoystickCamera",
    "vtkInteractorStyleRubberBand2D",
    "vtkInteractorStyleRubberBand3D",
    "vtkInteractorStyleRubberBandZoom",
    "vtkInteractorStyleTerrain",
    "vtkInteractorStyleTrackballActor",
    "vtkInteractorStyleTrackballCamera",
    "vtkInteractorStyleUnicam",
    "vtkInteractorStyleUser",
]
for name in as_strings:
    location[name] = "vtkInteractionStyle"

from vtkmodules.vtkInteractionWidgets import (
    vtkBalloonRepresentation,
    vtkBalloonWidget,
    vtkBoxWidget,
    vtkContourWidget,
    vtkPlaneWidget,
    vtkFocalPlanePointPlacer,
    vtkImplicitPlaneWidget,
    vtkOrientationMarkerWidget,
    vtkOrientedGlyphContourRepresentation,
    vtkPolygonalSurfacePointPlacer,
    vtkSliderRepresentation2D,
    vtkSliderRepresentation3D,
    vtkSliderWidget,
    vtkSphereWidget,
)

as_strings = [
    "vtkBalloonRepresentation",
    "vtkBalloonWidget",
    "vtkBoxWidget",
    "vtkContourWidget",
    "vtkPlaneWidget",
    "vtkFocalPlanePointPlacer",
    "vtkImplicitPlaneWidget",
    "vtkOrientationMarkerWidget",
    "vtkOrientedGlyphContourRepresentation",
    "vtkPolygonalSurfacePointPlacer",
    "vtkSliderRepresentation2D",
    "vtkSliderRepresentation3D",
    "vtkSliderWidget",
    "vtkSphereWidget",
]
for name in as_strings:
    location[name] = "vtkInteractionWidgets"

try:
    from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
    location["vtkCameraOrientationWidget"] = "vtkInteractionWidgets"
except ImportError:
    pass

from vtkmodules.vtkRenderingAnnotation import (
    vtkAnnotatedCubeActor,
    vtkAxesActor,
    vtkAxisActor2D,
    vtkCaptionActor2D,
    vtkCornerAnnotation,
    vtkCubeAxesActor,
    vtkLegendBoxActor,
    vtkLegendScaleActor,
    vtkPolarAxesActor,
    vtkScalarBarActor,
    vtkXYPlotActor,
)
as_strings = [
    "vtkAnnotatedCubeActor",
    "vtkAxesActor",
    "vtkAxisActor2D",
    "vtkCaptionActor2D",
    "vtkCornerAnnotation",
    "vtkCubeAxesActor",
    "vtkLegendBoxActor",
    "vtkLegendScaleActor",
    "vtkPolarAxesActor",
    "vtkScalarBarActor",
    "vtkXYPlotActor",
]
for name in as_strings:
    location[name] = "vtkRenderingAnnotation"


from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActor2D,
    vtkAreaPicker,
    vtkAssembly,
    vtkBillboardTextActor3D,
    vtkCamera,
    vtkCameraInterpolator,
    vtkColorTransferFunction,
    vtkCoordinate,
    vtkDataSetMapper,
    vtkDistanceToCamera,
    vtkFlagpoleLabel,
    vtkFollower,
    vtkHierarchicalPolyDataMapper,
    vtkImageActor,
    vtkImageMapper,
    vtkImageProperty,
    vtkImageSlice,
    vtkInteractorEventRecorder,
    vtkInteractorObserver,
    vtkLight,
    vtkLogLookupTable,
    vtkMapper,
    vtkPointGaussianMapper,
    vtkPolyDataMapper,
    vtkPolyDataMapper2D,
    vtkProp,
    vtkProp3D,
    vtkPropAssembly,
    vtkPropCollection,
    vtkPropPicker,
    vtkProperty,
    vtkRenderWindow,
    vtkRenderer,
    vtkRenderWindowInteractor,
    vtkSelectVisiblePoints,
    vtkSkybox,
    vtkTextActor,
    vtkTextMapper,
    vtkTextProperty,
    vtkTextRenderer,
    vtkTexture,
    vtkViewport,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
)
as_strings = [
    "vtkActor",
    "vtkActor2D",
    "vtkAreaPicker",
    "vtkAssembly",
    "vtkBillboardTextActor3D",
    "vtkCamera",
    "vtkCameraInterpolator",
    "vtkColorTransferFunction",
    "vtkCoordinate",
    "vtkDataSetMapper",
    "vtkDistanceToCamera",
    "vtkFlagpoleLabel",
    "vtkFollower",
    "vtkHierarchicalPolyDataMapper",
    "vtkImageActor",
    "vtkImageMapper",
    "vtkImageProperty",
    "vtkImageSlice",
    "vtkInteractorEventRecorder",
    "vtkInteractorObserver",
    "vtkLight",
    "vtkLogLookupTable",
    "vtkMapper",
    "vtkPointGaussianMapper",
    "vtkPolyDataMapper",
    "vtkPolyDataMapper2D",
    "vtkProp",
    "vtkProp3D",
    "vtkPropAssembly",
    "vtkPropCollection",
    "vtkPropPicker",
    "vtkProperty",
    "vtkRenderWindow",
    "vtkRenderer",
    "vtkRenderWindowInteractor",
    "vtkSelectVisiblePoints",
    "vtkSkybox",
    "vtkTextActor",
    "vtkTextMapper",
    "vtkTextProperty",
    "vtkTextRenderer",
    "vtkTexture",
    "vtkViewport",
    "vtkVolume",
    "vtkVolumeProperty",
    "vtkWindowToImageFilter",
]
for name in as_strings:
    location[name] = "vtkRenderingCore"

from vtkmodules.vtkRenderingFreeType import vtkVectorText
location["vtkVectorText"] = "vtkRenderingFreeType"

from vtkmodules.vtkRenderingImage import vtkImageResliceMapper
location["vtkImageResliceMapper"] = "vtkRenderingImage"

from vtkmodules.vtkRenderingLabel import vtkLabeledDataMapper
location["vtkLabeledDataMapper"] = "vtkRenderingLabel"

from vtkmodules.vtkRenderingOpenGL2 import (
    vtkDepthOfFieldPass,
    vtkCameraPass,
    vtkDualDepthPeelingPass,
    vtkEquirectangularToCubeMapTexture,
    vtkLightsPass,
    vtkOpaquePass,
    vtkOverlayPass,
    vtkRenderPassCollection,
    vtkSSAOPass,
    vtkSequencePass,
    vtkShader,
    vtkShadowMapPass,
    vtkTranslucentPass,
    vtkVolumetricPass,
)
as_strings = [
    "vtkDepthOfFieldPass",
    "vtkCameraPass",
    "vtkDualDepthPeelingPass",
    "vtkEquirectangularToCubeMapTexture",
    "vtkLightsPass",
    "vtkOpaquePass",
    "vtkOverlayPass",
    "vtkRenderPassCollection",
    "vtkSSAOPass",
    "vtkSequencePass",
    "vtkShader",
    "vtkShadowMapPass",
    "vtkTranslucentPass",
    "vtkVolumetricPass",
]
for name in as_strings:
    location[name] = "vtkRenderingOpenGL2"

from vtkmodules.vtkRenderingVolume import (
    vtkFixedPointVolumeRayCastMapper,
    vtkGPUVolumeRayCastMapper,
    vtkProjectedTetrahedraMapper,
    vtkUnstructuredGridVolumeRayCastMapper,
    vtkUnstructuredGridVolumeZSweepMapper,
)
as_strings = [
    "vtkFixedPointVolumeRayCastMapper",
    "vtkGPUVolumeRayCastMapper",
    "vtkProjectedTetrahedraMapper",
    "vtkUnstructuredGridVolumeRayCastMapper",
    "vtkUnstructuredGridVolumeZSweepMapper",
]
for name in as_strings:
    location[name] = "vtkRenderingVolume"

from vtkmodules.vtkRenderingVolumeOpenGL2 import (
    vtkOpenGLGPUVolumeRayCastMapper,
    vtkSmartVolumeMapper,
)
as_strings = [
    "vtkOpenGLGPUVolumeRayCastMapper",
    "vtkSmartVolumeMapper",
]
for name in as_strings:
    location[name] = "vtkRenderingVolumeOpenGL2"

#########################################################
# print("successfully finished importing vtkmodules")
del as_strings
del name
