#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset of vtk classes to be imported directly
"""

import vtkmodules.vtkCommonComputationalGeometry

from vtkmodules.vtkCommonColor import vtkNamedColors

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

from vtkmodules.vtkCommonDataModel import (
    VTK_TETRA,
    VTK_VOXEL,
    VTK_WEDGE,
    VTK_PYRAMID,
    VTK_HEXAGONAL_PRISM,
    VTK_PENTAGONAL_PRISM,
    vtkCell,
    vtkCellArray,
    vtkBox,
    vtkCellLocator,
    vtkCylinder,
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

from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm

from vtkmodules.vtkCommonMath import vtkMatrix4x4, vtkQuaternion

from vtkmodules.vtkCommonTransforms import (
    vtkLandmarkTransform,
    vtkThinPlateSplineTransform,
    vtkTransform
)

from vtkmodules.vtkFiltersCore import (
    VTK_BEST_FITTING_PLANE,
    vtkAppendPolyData,
    vtkCellCenters,
    vtkCellDataToPointData,
    vtkCenterOfMass,
    vtkCleanPolyData,
    vtkClipPolyData,
    vtkConnectivityFilter,
    vtkContourFilter,
    vtkContourGrid,
    vtkCutter,
    vtkDecimatePro,
    vtkDelaunay2D,
    vtkDelaunay3D,
    vtkElevationFilter,
    vtkFeatureEdges,
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

from vtkmodules.vtkFiltersExtraction import (
    vtkExtractCellsByType,
    vtkExtractGeometry,
    vtkExtractPolyDataGeometry,
    vtkExtractSelection,
)
try:
    from vtkmodules.vtkFiltersExtraction import vtkExtractEdges # vtk9.0
except ImportError:
    from vtkmodules.vtkFiltersCore import vtkExtractEdges # vtk9.2

from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer

from vtkmodules.vtkFiltersGeneral import (
    vtkBooleanOperationPolyDataFilter,
    vtkClipDataSet,
    vtkBoxClipDataSet,
    vtkContourTriangulator,
    vtkCurvatures,
    vtkDataSetTriangleFilter,
    vtkDistancePolyDataFilter,
    vtkGradientFilter,
    vtkIntersectionPolyDataFilter,
    vtkMultiBlockDataGroupFilter,
    vtkTransformPolyDataFilter,
    vtkOBBTree,
    vtkQuantizePolyDataPoints,
    vtkShrinkFilter,
    vtkShrinkPolyData,
    vtkRectilinearGridToTetrahedra,
    vtkVertexGlyphFilter,
)

try:
    from vtkmodules.vtkCommonDataModel import vtkCellTreeLocator
except ImportError:
    from vtkmodules.vtkFiltersGeneral import vtkCellTreeLocator

from vtkmodules.vtkFiltersGeometry import (
    vtkGeometryFilter,
    vtkDataSetSurfaceFilter,
    vtkImageDataGeometryFilter,
)

from vtkmodules.vtkFiltersHybrid import (
    vtkFacetReader,
    vtkImplicitModeller,
    vtkPolyDataSilhouette,
    vtkProcrustesAlignmentFilter,
    vtkRenderLargeImage,
)

from vtkmodules.vtkFiltersModeling import (
    vtkAdaptiveSubdivisionFilter,
    vtkBandedPolyDataContourFilter,
    vtkButterflySubdivisionFilter,
    vtkContourLoopExtraction,
    vtkDijkstraGraphGeodesicPath,
    vtkFillHolesFilter,
    vtkHausdorffDistancePointSetFilter,
    vtkLinearExtrusionFilter,
    vtkLinearSubdivisionFilter,
    vtkLoopSubdivisionFilter,
    vtkRibbonFilter,
    vtkRotationalExtrusionFilter,
    vtkRuledSurfaceFilter,
    vtkSelectEnclosedPoints,
    vtkSelectPolyData,
    vtkSubdivideTetra,
)

try:
    from vtkmodules.vtkFiltersModeling import vtkImprintFilter
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

from vtkmodules.vtkFiltersSources import (
    vtkArcSource,
    vtkArrowSource,
    vtkConeSource,
    vtkCubeSource,
    vtkCylinderSource,
    vtkDiskSource,
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

from vtkmodules.vtkFiltersTexture import vtkTextureMapToPlane

from vtkmodules.vtkFiltersVerdict import vtkMeshQuality, vtkCellSizeFilter

from vtkmodules.vtkImagingStencil import vtkPolyDataToImageStencil

from vtkmodules.vtkIOExport import vtkX3DExporter

from vtkmodules.vtkIOExportGL2PS import vtkGL2PSExporter

from vtkmodules.vtkIOGeoJSON import vtkGeoJSONReader

from vtkmodules.vtkIOGeometry import (
    vtkBYUReader,
    vtkFacetWriter,
    vtkOBJReader,
    vtkOpenFOAMReader,
    vtkParticleReader,
    vtkSTLReader,
    vtkSTLWriter,
)

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

from vtkmodules.vtkIOImport import (
    vtk3DSImporter,
    vtkOBJImporter,
    vtkVRMLImporter,
)

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

from vtkmodules.vtkIOPLY import vtkPLYReader, vtkPLYWriter

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

from vtkmodules.vtkImagingColor import (
    vtkImageLuminance,
    vtkImageMapToWindowLevelColors,
)

from vtkmodules.vtkImagingCore import (
    vtkExtractVOI,
    vtkImageAppendComponents,
    vtkImageBlend,
    vtkImageCast,
    vtkImageConstantPad,
    vtkImageExtractComponents,
    vtkImageFlip,
    vtkImageMirrorPad,
    vtkImagePermute,
    vtkImageResample,
    vtkImageResize,
    vtkImageReslice,
    vtkImageThreshold,
)

from vtkmodules.vtkImagingFourier import (
    vtkImageButterworthHighPass,
    vtkImageButterworthLowPass,
    vtkImageFFT,
    vtkImageFourierCenter,
    vtkImageRFFT,
)

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

from vtkmodules.vtkImagingHybrid import vtkImageToPoints, vtkSampleFunction
from vtkmodules.vtkImagingMath import (
    vtkImageDivergence,
    vtkImageDotProduct,
    vtkImageLogarithmicScale,
    vtkImageMagnitude,
    vtkImageMathematics,
)

from vtkmodules.vtkImagingMorphological import (
    vtkImageContinuousDilate3D,
    vtkImageContinuousErode3D,
)

from vtkmodules.vtkImagingSources import vtkImageCanvasSource2D
from vtkmodules.vtkImagingStencil import vtkImageStencil
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

from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleImage,
    vtkInteractorStyleJoystickCamera,
    vtkInteractorStyleRubberBandZoom,
    vtkInteractorStyleTrackballActor,
    vtkInteractorStyleTrackballCamera,
)

from vtkmodules.vtkInteractionWidgets import (
    vtkBalloonRepresentation,
    vtkBalloonWidget,
    vtkBoxWidget,
    vtkContourWidget,
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

try:
    from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
except ImportError:
    pass

from vtkmodules.vtkRenderingAnnotation import (
    vtkAnnotatedCubeActor,
    vtkAxesActor,
    vtkCaptionActor2D,
    vtkCornerAnnotation,
    vtkCubeAxesActor,
    vtkLegendBoxActor,
    vtkLegendScaleActor,
    vtkPolarAxesActor,
    vtkScalarBarActor,
    vtkXYPlotActor,
)

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActor2D,
    vtkAssembly,
    vtkBillboardTextActor3D,
    vtkCamera,
    vtkColorTransferFunction,
    vtkCoordinate,
    vtkDataSetMapper,
    vtkFollower,
    vtkHierarchicalPolyDataMapper,
    vtkImageActor,
    vtkImageMapper,
    vtkImageProperty,
    vtkImageSlice,
    vtkInteractorEventRecorder,
    vtkLight,
    vtkLogLookupTable,
    vtkMapper,
    vtkPolyDataMapper,
    vtkPolyDataMapper2D,
    vtkProp,
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
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
)

from vtkmodules.vtkRenderingFreeType import vtkVectorText

from vtkmodules.vtkRenderingImage import vtkImageResliceMapper

from vtkmodules.vtkRenderingLabel import vtkLabeledDataMapper

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

from vtkmodules.vtkRenderingVolume import (
    vtkFixedPointVolumeRayCastMapper,
    vtkGPUVolumeRayCastMapper,
    vtkProjectedTetrahedraMapper,
    vtkUnstructuredGridVolumeRayCastMapper,
    vtkUnstructuredGridVolumeZSweepMapper,
)

from vtkmodules.vtkRenderingVolumeOpenGL2 import (
    vtkOpenGLGPUVolumeRayCastMapper,
    vtkSmartVolumeMapper,
)
