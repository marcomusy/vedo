from vedo import UnstructuredGrid, Points, show, settings, utils

#####################################
def makeTetrahedron():
    """A tetrahedron"""
    pts = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 1),
    ]
    cells = [[0, 1, 2, 3]]
    cellstypes = [10]
    ug = UnstructuredGrid([pts, cells, cellstypes])
    ug.c('w', 0.25).lw(2).lighting("off")
    return ug


#####################################
def makeHexahedron():
    """A regular hexagon (cube) with all faces square and three squares around
    each vertex is created below.
    Setup the coordinates of eight points (the two faces must be in
    counter clockwise order as viewed from the outside).
    """
    pts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    cells = [[0, 1, 2, 3, 4, 5, 6, 7]]
    cellstypes = [12]
    ug = UnstructuredGrid([pts, cells, cellstypes])
    ug.c('w', 0.25).lw(2).lighting("off")
    return ug


#####################################
def makePyramid():
    """Make a regular square pyramid"""
    pts = [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    cells = [[0, 1, 2, 3, 4]]
    cellstypes = [14]
    ug = UnstructuredGrid([pts, cells, cellstypes])
    ug.c('w', 0.25).lw(2).lighting("off")
    return ug


#####################################
def makeWedge():
    """A wedge consists of two triangular ends and three rectangular faces"""
    pts = [
        (0, 1, 0),
        (0, 0, 0),
        (0, 0.25, 0.25),
        (1, 1, 0),
        (1, 0.0, 0.0),
        (1, 0.25, 0.25),
    ]
    cells = [[0, 1, 2, 3, 4, 5]]
    cellstypes = [13]
    ug = UnstructuredGrid([pts, cells, cellstypes])
    ug.c('w', 0.25).lw(2).lighting("off")
    return ug


#####################################
def makeHexagonalPrism():
    """Hexagonal prism: a wedge with an hexagonal base.
    Be careful, the base face ordering is different from wedge.
    """
    pts = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.5, 0.5, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (-0.5, 0.5, 1.0),
        ####
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.5, 0.5, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (-0.5, 0.5, 0.0),
    ]
    cells = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    cellstypes = [16]
    ug = UnstructuredGrid([pts, cells, cellstypes])
    ug.c('w', 0.25).lw(2).lighting("off")
    return ug


#####################################
def makePentagonalPrism():
    """A 3D pentagonal prism: a wedge with an pentagonal base."""
    pts = [
        (11, 10, 10),
        (13, 10, 10),
        (14, 12, 10),
        (12, 14, 10),
        (10, 12, 10),
        (11, 10, 14),
        (13, 10, 14),
        (14, 12, 14),
        (12, 14, 14),
        (10, 12, 14),
    ]
    cells = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    cellstypes = [15]
    ug = UnstructuredGrid([pts, cells, cellstypes])
    ug.c('w', 0.25).lw(2).lighting("off")
    return ug


def makeQuadraticTetra():
    import vtk

    print(utils.vtkVersionIsAtLeast(9))
    aTetra = vtk.vtkQuadraticTetra()
    points = vtk.vtkPoints()

    pcoords = aTetra.GetParametricCoords()
    rng = vtk.vtkMinimalStandardRandomSequence()
    points.SetNumberOfPoints(aTetra.GetNumberOfPoints())
    rng.SetSeed(5070)  # for testing
    for i in range(0, aTetra.GetNumberOfPoints()):
        perturbation = [0.0] * 3
        for j in range(0, 3):
            rng.Next()
            perturbation[j] = rng.GetRangeValue(-0.1, 0.1)
        aTetra.GetPointIds().SetId(i, i)
        points.SetPoint(i, [pcoords[3 * i], pcoords[3 * i + 1], pcoords[3 * i + 2]])

    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(points)
    ug.InsertNextCell(aTetra.GetCellType(), aTetra.GetPointIds())
    return UnstructuredGrid(ug)


#####################################
tetr = makeTetrahedron()
hexa = makeHexahedron()
pirm = makePyramid()
wedg = makeWedge()
hexp = makeHexagonalPrism()
penp = makePentagonalPrism()
# qtetr= makeQuadraticTetra()

settings.immediate_rendering = False  # faster for multi-renderers

show(
    [
        ["Tetrahedron", tetr, Points(tetr.vertices, r=15, c="o")],
        ["Hexahedron", hexa, Points(hexa.vertices, r=15, c="o")],
        ["Pyramid", pirm, Points(pirm.vertices, r=15, c="o")],
        ["Wedge", wedg, Points(wedg.vertices, r=15, c="o")],
        ["HexagonalPrism", hexp, Points(hexp.vertices, r=15, c="o")],
        ["PentagonalPrism", penp, Points(penp.vertices, r=15, c="o")],
    ],
    N=6,
    bg="blue3",
    sharecam=False,
    axes=1,
).close()
