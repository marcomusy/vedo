#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset of vtk classes to be imported directly or lazily.
"""
from importlib import import_module


location = dict()
module_cache = {}


def get(cls_name="", module_name=""):
    """
    Get a vtk class from its name.
    
    Example:
    ```python
    from vedo import vtkclasses as vtk
    print(vtk.vtkActor)
    print(vtk.location["vtkActor"])
    print(vtk.get("vtkActor"))
    print(vtk.get("vtkActor", "vtkRenderingCore"))
    ```
    """
    if cls_name and not cls_name.startswith("vtk"):
        cls_name = "vtk" + cls_name
    if not module_name:
        module_name = location[cls_name]
    module_name = "vtkmodules." + module_name
    if module_name not in module_cache:
        module = import_module(module_name)
        module_cache[module_name] = module
    if cls_name:
        return getattr(module_cache[module_name], cls_name)
    else:
        return module_cache[module_name]

def dump_hierarchy_to_file(fname=""):
    """
    Print all available vtk classes.
    Dumps the list to a file named `vtkmodules_<version>_hierarchy.txt`
    """
    try:
        import pkgutil
        import vtkmodules
        from vtkmodules.all import vtkVersion
        ver = vtkVersion()
    except AttributeError:
        print("Unable to detect VTK version.")
        return
    major = ver.GetVTKMajorVersion()
    minor = ver.GetVTKMinorVersion()
    patch = ver.GetVTKBuildVersion()
    vtkvers = f"{major}.{minor}.{patch}"
    if not fname:
        fname = f"vtkmodules_{vtkvers}_hierarchy.txt"
    with open(fname,"w") as w:
        for pkg in pkgutil.walk_packages(
            vtkmodules.__path__, vtkmodules.__name__ + "."):
            try:
                module = import_module(pkg.name)
            except ImportError:
                continue
            for subitem in sorted(dir(module)):
                if "all" in module.__name__:
                    continue
                if ".web." in module.__name__:
                    continue
                if ".test." in module.__name__:
                    continue
                if ".tk." in module.__name__:
                    continue
                if "__" in module.__name__ or "__" in subitem:
                    continue
                w.write(f"{module.__name__}.{subitem}\n")

######################################################################
as_strings = [
    "vtkKochanekSpline",
    "vtkCardinalSpline",
    "vtkParametricSpline",
    "vtkParametricFunctionSource",
    "vtkParametricTorus",
    "vtkParametricBoy",
    "vtkParametricConicSpiral",
    "vtkParametricCrossCap",
    "vtkParametricDini",
    "vtkParametricEllipsoid",
    "vtkParametricEnneper",
    "vtkParametricFigure8Klein",
    "vtkParametricKlein",
    "vtkParametricMobius",
    "vtkParametricRandomHills",
    "vtkParametricRoman",
    "vtkParametricSuperEllipsoid",
    "vtkParametricSuperToroid",
    "vtkParametricBohemianDome",
    "vtkParametricBour",
    "vtkParametricCatalanMinimal",
    "vtkParametricHenneberg",
    "vtkParametricKuen",
    "vtkParametricPluckerConoid",
    "vtkParametricPseudosphere",
]
for name in as_strings:
    location[name] = "vtkCommonComputationalGeometry"


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
    vtkPolyData,
    vtkImageData,
    vtkUnstructuredGrid,
)

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

from vtkmodules.vtkCommonMath import vtkMatrix4x4
location["vtkMatrix4x4"] = "vtkCommonMath"
location["vtkQuaternion"] = "vtkCommonMath"

from vtkmodules.vtkCommonTransforms import (
    vtkHomogeneousTransform,
    vtkLandmarkTransform,
    vtkLinearTransform,
    vtkThinPlateSplineTransform,
    vtkTransform,
)
as_strings = [
    "vtkHomogeneousTransform",
    "vtkLandmarkTransform",
    "vtkLinearTransform",
    "vtkThinPlateSplineTransform",
    "vtkTransform",
]
for name in as_strings:
    location[name] = "vtkCommonTransforms"

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

location["vtkStaticCleanUnstructuredGrid"] = "vtkFiltersCore"
location["vtkPolyDataPlaneCutter"] = "vtkFiltersCore"

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
    vtkImplicitModeller,
    vtkPolyDataSilhouette,
    vtkProcrustesAlignmentFilter,
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

location["vtkTextureMapToPlane"] = "vtkFiltersTexture"

location["vtkMeshQuality"] = "vtkFiltersVerdict"
location["vtkCellSizeFilter"] = "vtkFiltersVerdict"

location["vtkPolyDataToImageStencil"] = "vtkImagingStencil"

location["vtkX3DExporter"] = "vtkIOExport"

location["vtkGL2PSExporter"] = "vtkIOExportGL2PS"

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

location["vtk3DSImporter"] = "vtkIOImport"
location["vtkOBJImporter"] = "vtkIOImport"
location["vtkVRMLImporter"] = "vtkIOImport"

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


location["vtkPLYReader"] = "vtkIOPLY"
location["vtkPLYWriter"] = "vtkIOPLY"

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
as_strings = ["vtkImageToPoints", "vtkSampleFunction"]
for name in as_strings:
    location[name] = "vtkImagingHybrid"

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

as_strings = [
    "vtkImageContinuousDilate3D",
    "vtkImageContinuousErode3D",
]
for name in as_strings:
    location[name] = "vtkImagingMorphological"

location["vtkImageCanvasSource2D"] = "vtkImagingSources"

location["vtkImageStencil"] = "vtkImagingStencil"

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

location["vtkCameraOrientationWidget"] = "vtkInteractionWidgets"


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

location["vtkVectorText"] = "vtkRenderingFreeType"

location["vtkImageResliceMapper"] = "vtkRenderingImage"

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
