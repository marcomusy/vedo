vpath = 'https://github.com/marcomusy/vedo/tree/master/examples';

vedo_example_db =
    [
        {
            pyname: 'buildmesh',    // python script name
            categ: 'basic',        // category
            short: 'Build a Mesh',  // short description, as card footer
            long: 'Build a simple mesh from a set of points and faces.',
            imgsrc: 'images/basic/buildmesh.png',  //image path
        },

        {
            pyname: 'colorcubes',
            categ: 'basic',
            short: 'Color Schemes',
            long: 'Display a cube for each available color name. Multiple color schemes are supported, including Matplotlib, Bootstrap, and VTK.',
            imgsrc: 'images/basic/colorcubes.png',
        },

        {
            pyname: 'texturecubes',
            categ: 'basic',
            short: 'Mesh Textures',
            long: 'Display a cube for each available texture. Any JPG file can be used as a texture.',
            imgsrc: 'images/basic/texturecubes.png',
        },

        {
            pyname: 'colormap_list',
            categ: 'basic',
            short: 'Colormap List',
            long: 'Display all colormaps available in vedo.',
            imgsrc: 'images/basic/colormap_list.png',
        },

        {
            pyname: 'colormaps',
            categ: 'basic',
            short: 'Discrete Colormap',
            long: 'Assign colors to mesh vertices using a discretized Matplotlib colormap.',
            imgsrc: 'images/basic/colormaps.png',
        },

        {
            pyname: 'mesh_custom',
            categ: 'basic',
            short: 'Colorize a Mesh',
            long: 'Control a mesh\'s color and transparency using different colormap definitions.',
            imgsrc: 'images/basic/mesh_custom.png',
        },

        {
            pyname: 'color_mesh_cells1',
            categ: 'basic',
            short: 'Face Colors',
            long: 'Color mesh faces by passing a one-to-one list of colors and, optionally, transparencies.',
            imgsrc: 'images/basic/colorMeshCells.png',
        },

        {
            pyname: 'color_mesh_cells2',
            categ: 'basic',
            short: 'Face Colors',
            long: 'Color mesh faces by passing a one-to-one list of colors and, optionally, transparencies.',
            imgsrc: 'images/basic/color_mesh_cells2.png',
        },

        {
            pyname: 'mesh_lut',
            categ: 'basic',
            short: 'Custom Colormap',
            long: 'Build a custom colormap, including colors for out-of-range values, NaNs, and labels.',
            imgsrc: 'images/basic/mesh_lut.png',
        },

        {
            pyname: 'multirenderers',
            categ: 'basic',
            short: 'Multiple Renderers',
            long: 'Manually define the number, shape, and position of renderers inside the rendering window.',
            imgsrc: 'images/basic/multirenderers.png',
        },

        {
            pyname: 'silhouette1',
            categ: 'basic',
            short: 'Mesh Silhouette',
            long: 'Generate the silhouette of a mesh as seen from a specified direction.',
            imgsrc: 'images/basic/silhouette1.png',
        },

        {
            pyname: 'silhouette2',
            categ: 'basic',
            short: 'Projected Silhouette',
            long: 'Generate the projected silhouette of a mesh as seen from a specified direction.',
            imgsrc: 'images/basic/silhouette2.png',
        },

        {
            pyname: 'cut_interactive',
            categ: 'basic',
            short: 'Interactive Cutter',
            long: 'Cut a mesh interactively and save the result to a file.',
            imgsrc: 'images/basic/cutter.gif',
        },

        {
            pyname: 'cut_freehand',
            categ: 'basic',
            short: 'Freehand Cutter',
            long: 'Cut a mesh interactively by drawing a freehand contour.',
            imgsrc: 'images/basic/cutFreeHand.gif',
        },

        {
            pyname: 'shrink',
            categ: 'basic',
            short: 'Shrink a Mesh',
            long: 'Shrink mesh polygons to reveal the interior.',
            imgsrc: 'images/basic/shrink.png',
        },

        {
            pyname: 'boundaries',
            categ: 'basic',
            short: 'Mesh Boundaries',
            long: 'Extract boundary points from a mesh and add labels to all vertices.',
            imgsrc: 'images/basic/boundaries.png',
        },

        {
            pyname: 'mesh_modify',
            categ: 'basic',
            short: 'Move Vertices',
            long: 'Modify the positions of mesh vertices.',
            imgsrc: 'images/basic/mesh_modify.png',
        },

        {
            pyname: 'connected_vtx',
            categ: 'basic',
            short: 'Connected Vertices',
            long: 'Find all vertices connected to a specific vertex in a mesh.',
            imgsrc: 'images/basic/connVtx.png',
        },

        {
            pyname: 'largestregion',
            categ: 'basic',
            short: 'Largest Connectd Region',
            long: 'Extract the connected mesh region with the largest surface area.',
            imgsrc: 'images/basic/largestregion.png',
        },

        {
            pyname: 'fillholes',
            categ: 'basic',
            short: 'Fill Mesh Holes',
            long: 'Fill holes in a mesh by locating boundary edges, linking them into loops, and triangulating the result.',
            imgsrc: 'images/basic/fillholes.png',
        },

        {
            pyname: 'sliders1',
            categ: 'basic',
            short: 'Slider Controls',
            long: 'Use two sliders to change a mesh\'s color and transparency.',
            imgsrc: 'images/basic/sliders1.png',
        },

        {
            pyname: 'boolean',
            categ: 'basic',
            short: 'Boolean Operations',
            long: 'Perform various Boolean operations on meshes.',
            imgsrc: 'images/basic/boolean.png',
        },

        {
            pyname: 'delaunay2d',
            categ: 'basic',
            short: '2D Delaunay',
            long: 'Perform 2D triangulation using the Delaunay algorithm.',
            imgsrc: 'images/basic/delaunay2d.png',
        },

        {
            pyname: 'voronoi1',
            categ: 'basic',
            short: 'Voronoi Tessellation',
            long: 'Perform a 2D Voronoi tessellation of a set of input points.',
            imgsrc: 'images/basic/voronoi1.png',
        },

        {
            pyname: 'flatarrow',
            categ: 'basic',
            short: 'Flat Arrows',
            long: 'Use two lines to define a flat arrow.',
            imgsrc: 'images/basic/flatarrow.png',
        },

        {
            pyname: 'shadow1',
            categ: 'basic',
            short: 'Simple Shadows',
            long: 'Project the shadow of two meshes onto the x, y, or z wall.',
            imgsrc: 'images/basic/shadow1.png',
        },

        {
            pyname: 'shadow2',
            categ: 'basic',
            short: 'Multiple Shadows',
            long: 'Project realistic shadows of two meshes onto the xy plane.',
            imgsrc: 'images/basic/shadow2.png',
        },

        {
            pyname: 'extrude',
            categ: 'basic',
            short: 'Polygon Extrusion',
            long: 'Extrude a 2D polygon along the vertical axis.',
            imgsrc: 'images/basic/extrude.png',
        },

        {
            pyname: 'align1',
            categ: 'basic',
            short: 'Shape Registration',
            long: 'Align the red line with the yellow shape using the ' + insertLink('ICP algorithm', 'en.wikipedia.org/wiki/Iterative_closest_point'),
            imgsrc: 'images/basic/align1.png',
        },

        {
            pyname: 'align2',
            categ: 'basic',
            short: 'Point-Cloud Registration',
            long: 'Generate two random sets of points and align them using the ' + insertLink('ICP algorithm', 'en.wikipedia.org/wiki/Iterative_closest_point'),
            imgsrc: 'images/basic/align2.png',
        },

        {
            pyname: 'align4',
            categ: 'basic',
            short: 'Procrustes Registration',
            long: 'Align a set of curves in space using the ' + insertLink('Procrustes method', 'en.wikipedia.org/wiki/Procrustes_analysis'),
            imgsrc: 'images/basic/align4.png',
        },

        {
            pyname: 'align5',
            categ: 'basic',
            short: 'Landmark Registration',
            long: 'Transform a mesh by specifying how a chosen set of points (landmarks) should move.',
            imgsrc: 'images/basic/align5.png',
        },

        {
            pyname: 'buttons1',
            categ: 'basic',
            short: 'Window Buttons',
            long: 'Add a button with N possible states to the rendering window and trigger an external function.',
            imgsrc: 'images/basic/buttons.png',
        },

        {
            pyname: 'cells_within_bounds',
            categ: 'basic',
            short: 'Cells in Bounds',
            long: 'Find cells within specified bounds along x, y, and/or z.',
            imgsrc: 'images/basic/cellsWithinBounds.png',
        },

        {
            pyname: 'clustering',
            categ: 'basic',
            short: 'Clustering and Outliers',
            long: 'Automatically cluster point clouds and remove outliers.',
            imgsrc: 'images/basic/clustering.png',
        },

        {
            pyname: 'pca_ellipsoid',
            categ: 'basic',
            short: 'PCA Ellipsoid',
            long: 'Fit an ellipsoid to a point cloud using PCA (Principal Component Analysis).',
            imgsrc: 'images/basic/pca.png',
        },

        {
            pyname: 'manypoints',
            categ: 'basic',
            short: 'Million Points',
            long: 'Draw a very large number (1M) of points with different colors and transparencies.',
            imgsrc: 'images/basic/manypoints.jpg',
        },

        {
            pyname: 'manyspheres',
            categ: 'basic',
            short: '50K Spheres',
            long: 'Draw a very large number (50k) of spheres or points with different colors and radii.',
            imgsrc: 'images/basic/manyspheres.jpg',
        },

        {
            pyname: 'colorlines',
            categ: 'basic',
            short: 'Color by Scalar',
            long: 'Color line cells using a scalar array and a Matplotlib colormap.',
            imgsrc: 'images/basic/colorlines.png',
        },

        {
            pyname: 'ribbon',
            categ: 'basic',
            short: 'Ribbon Surface',
            long: 'Create a ribbon-like surface by joining two lines, or by sweeping a single line along its tangent.',
            imgsrc: 'images/basic/ribbon.png',
        },

        {
            pyname: 'mirror',
            categ: 'basic',
            short: 'Mirror Mesh',
            long: 'Mirror a mesh along one of the Cartesian axes. Hover to distinguish the original from the mirrored copy.',
            imgsrc: 'images/basic/mirror.png',
        },

        {
            pyname: 'delete_mesh_pts',
            categ: 'basic',
            short: 'Delete Elements',
            long: 'Remove the points and cells of a mesh that are closest to a specified point.',
            imgsrc: 'images/basic/deleteMeshPoints.png',
        },

        {
            pyname: 'mousehighlight',
            categ: 'basic',
            short: 'Highlight Mesh',
            long: 'Click an object to select and highlight it.',
            imgsrc: 'images/basic/mousehighlight.png',
        },

        {
            pyname: 'mousehover1',
            categ: 'basic',
            short: 'Hover Values',
            long: 'Interactively visualize scalar values by hovering over a mesh.',
            imgsrc: 'images/basic/mousehover1.gif',
        },

        {
            pyname: 'mousehover2',
            categ: 'basic',
            short: 'Hover to Fit',
            long: 'Interactively fit a sphere to a region of a mesh by hovering over it.',
            imgsrc: 'images/basic/mousehover2.gif',
        },

        {
            pyname: 'mousehover3',
            categ: 'basic',
            short: 'World Coordinates',
            long: 'Compute 3D world coordinates from 2D screen pixel coordinates while hovering the mouse.',
            imgsrc: 'images/basic/mousehover3.jpg',
        },

        {
            pyname: 'spline_tool',
            categ: 'basic',
            short: 'Interactive Spline Tool',
            long: 'Modify a spline interactively by clicking and dragging with the mouse.',
            imgsrc: 'images/basic/spline_tool.png',
        },

        {
            pyname: 'distance2mesh',
            categ: 'basic',
            short: 'Signed Distance',
            long: 'Compute the signed distance from one mesh to another and store the result as an array on the mesh.',
            imgsrc: 'images/basic/distance2mesh.png',
        },

        {
            pyname: 'glyphs1',
            categ: 'basic',
            short: 'Mesh Glyphs',
            long: 'Attach another mesh to each vertex of a source mesh (for example, a sphere), with different orientation options.',
            imgsrc: 'images/basic/glyphs.png',
        },

        {
            pyname: 'glyphs3',
            categ: 'basic',
            short: 'Glyph Symbols',
            long: 'Attach an oriented mesh (here a cone) to each 3D point and color it by vector magnitude.',
            imgsrc: 'images/pyplot/glyphs3.png',
        },

        {
            pyname: 'lightings',
            categ: 'basic',
            short: 'Lighting Modes',
            long: 'Adjust a mesh\'s lighting properties to change its appearance.',
            imgsrc: 'images/basic/lightings.png',
        },

        {
            pyname: 'cartoony',
            categ: 'basic',
            short: 'Cartoon Shading',
            long: 'Give a 3D polygonal mesh a cartoon-like appearance.',
            imgsrc: 'images/basic/cartoony.png',
        },

        {
            pyname: 'ssao',
            categ: 'basic',
            short: 'Ambient Occlusion',
            long: 'Render a scene using Screen Space Ambient Occlusion (SSAO).',
            imgsrc: 'images/basic/ssao.jpg',
        },

        {
            pyname: 'surf_intersect',
            categ: 'basic',
            short: 'Mesh Intersection',
            long: 'Find the intersection curve of two polygonal meshes.',
            imgsrc: 'images/basic/surfIntersect.png',
        },

        {
            pyname: 'lin_interpolate',
            categ: 'basic',
            short: 'Vector Interpolation',
            long: 'Linearly interpolate vectors defined at specific points in space.',
            imgsrc: 'images/basic/linInterpolate.png',
        },

        {
            pyname: 'mesh_map2cell',
            categ: 'basic',
            short: 'Map to Cells',
            long: 'Map an array defined on mesh vertices to its cells.',
            imgsrc: 'images/basic/mesh_map2cell.png',
        },

        {
            pyname: 'tube_radii',
            categ: 'basic',
            short: 'Variable Tube',
            long: 'Use an array to vary the radius and color of a line represented as a tube.',
            imgsrc: 'images/basic/tube.png',
        },

        {
            pyname: 'rotate_image',
            categ: 'basic',
            short: 'Rotate an Image',
            long: 'Load regular JPG/PNG images, then crop, rotate, and position them anywhere in a 3D scene.',
            imgsrc: 'images/basic/rotateImage.png',
        },

        {
            pyname: 'background_image',
            categ: 'basic',
            short: 'Background Image',
            long: 'Set a JPEG background image on a separate rendering layer.',
            imgsrc: 'images/basic/bgImage.png',
        },

        {
            pyname: 'skybox',
            categ: 'basic',
            short: 'Skybox Scene',
            long: 'Place a mesh inside a skybox environment with physically based rendering (PBR) lighting.',
            imgsrc: 'images/basic/skybox.jpg',
        },

        {
            pyname: 'align3',
            categ: 'basic',
            short: 'Procrustes Alignment',
            long: 'Align three random point sets with Procrustes analysis. Uses multiple renderers to compare views in parallel.',
            imgsrc: 'images/basic/align3.png',
        },

        {
            pyname: 'align6',
            categ: 'basic',
            short: 'Bounding-Box Alignment',
            long: 'Align objects based on their bounding boxes.',
            imgsrc: 'images/basic/align6.png',
        },

        {
            pyname: 'buttons2',
            categ: 'basic',
            short: 'Visibility Toggles',
            long: 'Create three checkbox buttons to toggle objects on and off.',
            imgsrc: 'images/basic/buttons2.png',
        },

        {
            pyname: 'buttons3',
            categ: 'basic',
            short: 'Icon Button',
            long: 'Create a button that uses an image icon to show its state.',
            imgsrc: 'images/basic/buttons3.png',
        },

        {
            pyname: 'extrude1',
            categ: 'basic',
            short: 'Polygon Extrusion',
            long: 'Extrude a polygon along the z-axis.',
            imgsrc: 'images/basic/extrude1.png',
        },

        {
            pyname: 'extrude2',
            categ: 'basic',
            short: 'Linear Extrusion',
            long: 'Perform a linear extrusion.',
            imgsrc: 'images/basic/extrude2.png',
        },

        {
            pyname: 'glyphs2',
            categ: 'basic',
            short: 'Arrow Glyphs',
            long: 'Draw colored arrow glyphs.',
            imgsrc: 'images/basic/glyphs2.png',
        },

        {
            pyname: 'hover_legend',
            categ: 'basic',
            short: 'Hover Legend',
            long: 'Hover over a mesh to display object details.',
            imgsrc: 'images/basic/hover_legend.png',
        },

        {
            pyname: 'input_box',
            categ: 'basic',
            short: 'Input Box',
            long: 'Type a color name to update a mesh\'s appearance in real time.',
            imgsrc: 'images/basic/input_box.png',
        },

        {
            pyname: 'interaction_modes2',
            categ: 'basic',
            short: 'Mesh Selection',
            long: 'Use the mouse to select objects and vertices in a mesh.',
            imgsrc: 'images/basic/interaction_modes2.png',
        },

        {
            pyname: 'interaction_modes3',
            categ: 'basic',
            short: 'Surface Flyover',
            long: 'Use an interaction mode designed for flying over a surface.',
            imgsrc: 'images/basic/interaction_modes3.png',
        },

        {
            pyname: 'interaction_modes4',
            categ: 'basic',
            short: 'Panel Focus Toggle',
            long: 'Press TAB to toggle the active panel and freeze the other. Uses multiple renderers to compare views in parallel.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'keypress',
            categ: 'basic',
            short: 'Key Press',
            long: 'Implement a custom function triggered by a keyboard key while the rendering window is in interactive mode.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'legendbox',
            categ: 'basic',
            short: 'Legend Box',
            long: 'Customize a legend box.',
            imgsrc: 'images/basic/legendbox.png',
        },

        {
            pyname: 'mesh_alphas',
            categ: 'basic',
            short: 'Mesh Alphas',
            long: 'Create a set of transparency values that can be passed to method cmap().',
            imgsrc: 'images/basic/mesh_alphas.png',
        },

        {
            pyname: 'mesh_coloring',
            categ: 'basic',
            short: 'Mesh Coloring',
            long: 'Specify color mapping for the cells and points of a mesh. Uses multiple renderers to compare views in parallel.',
            imgsrc: 'images/basic/mesh_coloring.png',
        },

        {
            pyname: 'mesh_sharemap',
            categ: 'basic',
            short: 'Shared Mesh Map',
            long: 'Share the same color map across different meshes.',
            imgsrc: 'images/basic/mesh_sharemap.png',
        },

        {
            pyname: 'mesh_threshold',
            categ: 'basic',
            short: 'Mesh Threshold',
            long: 'Extract mesh cells that satisfy a threshold criterion for a scalar field defined on the mesh.',
            imgsrc: 'images/basic/mesh_threshold.png',
        },

        {
            pyname: 'mouseclick1',
            categ: 'basic',
            short: 'Click Callbacks',
            long: 'Trigger a custom function in response to mouse clicks and other events.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'mouseclick2',
            categ: 'basic',
            short: 'Object Observers',
            long: 'Add an observer to specific objects in a scene.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'mousehover0',
            categ: 'basic',
            short: 'Hover Flagposts',
            long: 'Use a flagpost object to visualize properties interactively.',
            imgsrc: 'images/basic/mousehover0.png',
        },

        {
            pyname: 'multiwindows2',
            categ: 'basic',
            short: 'Linked Windows',
            long: 'Create synchronized Plotter windows.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'multiwindows1',
            categ: 'basic',
            short: 'Multi-Window Layout',
            long: 'Draw objects in different windows and/or subwindows within the same window. Uses multiple renderers to compare views in parallel.',
            imgsrc: 'images/basic/multiwindows1.png',
        },

        {
            pyname: 'multi_viewer2',
            categ: 'advanced',
            short: 'Multi-Window Viewer',
            long: 'Create two windows that can interact and share functionality.',
            imgsrc: 'images/advanced/multi_viewer.png',
        },

        {
            pyname: 'pca_ellipse',
            categ: 'basic',
            short: 'PCA Ellipse',
            long: 'Draw the ellipse (dark) and the ellipsoid (light) that each contain 50% of a point cloud, then check how many points lie inside both objects.',
            imgsrc: 'images/basic/pca_ellipse.png',
        },

        {
            pyname: 'record_play',
            categ: 'basic',
            short: 'Record and Play',
            long: 'Record and play back camera movements and other events.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'scalarbars',
            categ: 'basic',
            short: 'Scalar Bars',
            long: 'Insert 2D and 3D scalar bars into the rendering scene.',
            imgsrc: 'images/basic/scalarbars.png',
        },

        {
            pyname: 'shadow3',
            categ: 'basic',
            short: 'Directional Shadow',
            long: 'Project the shadow of a mesh in a specified direction.',
            imgsrc: 'images/basic/shadow3.png',
        },

        {
            pyname: 'silhouette3',
            categ: 'basic',
            short: 'Camera Silhouette',
            long: 'Make an object silhouette move together with the camera position.',
            imgsrc: 'images/basic/silhouette3.png',
        },

        {
            pyname: 'slider_browser',
            categ: 'basic',
            short: 'Slider Browser',
            long: 'Explore mouse hind limb growth from day 10 9h to day 15 21h using a slider.',
            imgsrc: 'images/basic/slider_browser.png',
        },

        {
            pyname: 'sliders2',
            categ: 'basic',
            short: 'Slider Controls',
            long: 'Use sliders and buttons to control scene objects in real time, wiring UI widgets directly to 3D scene properties.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'sliders3d',
            categ: 'basic',
            short: '3D Sliders',
            long: 'Use a 3D slider to move a mesh interactively.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'sliders_hsv',
            categ: 'basic',
            short: 'HSV Sliders',
            long: 'Explore RGB and HSV color spaces by adjusting hue, saturation, and value sliders to visualize color model transitions.',
            imgsrc: 'images/basic/sliders_hsv.png',
        },

        {
            pyname: 'sliders_range',
            categ: 'basic',
            short: 'Sliders Range',
            long: 'Create a double-handle range slider that independently scales two spheres, demonstrating constrained multi-value input.',
            imgsrc: 'images/basic/sliders_range.png',
        },

        {
            pyname: 'specular',
            categ: 'basic',
            short: 'Specular Lighting',
            long: 'Set illumination properties such as ambient, diffuse, specular power, and color.',
            imgsrc: 'images/basic/specular.png',
        },

        {
            pyname: 'light_sources',
            categ: 'basic',
            short: 'Custom Lights',
            long: 'Set custom lights in a 3D scene by specifying direction, position, intensity, and color.',
            imgsrc: 'images/basic/lights.png',
        },

        {
            pyname: 'texture_coords',
            categ: 'basic',
            short: 'Texture Coordinates',
            long: 'Assign texture coordinates to a polygon.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'geological_model',
            categ: 'advanced',
            short: 'Geological Model',
            long: 'Recreate a complex 3D model of a geothermal reservoir in Utah (USA) and export it to a ' + insertLink('web page.', 'vedo.embl.es/examples/geo_scene.html'),
            imgsrc: 'images/advanced/geological_model.jpg',
        },

        {
            pyname: 'geodesic',
            categ: 'advanced',
            short: 'Geodesic Paths',
            long: 'Use Dijkstra algorithm to compute geodesics, the shortest paths between two points on a surface.',
            imgsrc: 'images/advanced/geodesic.png',
        },

        {
            pyname: 'moving_least_squares1D',
            categ: 'advanced',
            short: 'MLS Curve',
            long: 'Use the ' + insertLink('Moving Least Squares', 'en.wikipedia.org/wiki/Moving_least_squares') + ' algorithm to project a point cloud onto a smooth curve.',
            imgsrc: 'images/advanced/moving_least_squares1D.png',
        },

        {
            pyname: 'moving_least_squares2D',
            categ: 'advanced',
            short: 'MLS Surface',
            long: 'Use the ' + insertLink('Moving Least Squares', 'en.wikipedia.org/wiki/Moving_least_squares') + ' algorithm to project a point cloud onto a smooth surface.',
            imgsrc: 'images/advanced/least_squares2D.png',
        },

        {
            pyname: 'recosurface',
            categ: 'advanced',
            short: 'Point Cloud to Mesh',
            long: 'Reconstruct a triangular mesh from a noisy point cloud.',
            imgsrc: 'images/advanced/recosurface.png',
        },

        {
            pyname: 'line2mesh_tri',
            categ: 'advanced',
            short: 'Tri Mesh',
            long: 'Generate a triangular mesh from a 2D line contour.',
            imgsrc: 'images/advanced/line2mesh_tri.jpg',
        },

        {
            pyname: 'line2mesh_quads',
            categ: 'advanced',
            short: 'Quad Mesh',
            long: 'Generate a quad mesh from a 2D line contour.',
            imgsrc: 'images/advanced/line2mesh_quads.png',
        },

        {
            pyname: 'voronoi2',
            categ: 'advanced',
            short: 'Voronoi Tessellation',
            long: 'Perform a 2D Voronoi tessellation of a set of input points on a grid.',
            imgsrc: 'images/advanced/voronoi2.png',
        },

        {
            pyname: 'meshquality',
            categ: 'advanced',
            short: 'Mesh Quality',
            long: 'Visualize various quality metrics for the cells of a triangular mesh.',
            imgsrc: 'images/advanced/meshquality.png',
        },

        {
            pyname: 'mesh_smoother2',
            categ: 'advanced',
            short: 'Mesh Smoothing',
            long: 'Smooth a mesh using different combinations of algorithms and parameters.',
            imgsrc: 'images/advanced/mesh_smoother2.png',
        },

        {
            pyname: 'warp1',
            categ: 'advanced',
            short: 'TPS Warp',
            long: 'Use Thin Plate Spline transformations to describe a nonlinear warp defined by source and target points.',
            imgsrc: 'images/advanced/warp1.png',
        },

        {
            pyname: 'warp2',
            categ: 'advanced',
            short: 'TPS Warp 3D',
            long: 'Warp part of a mesh using Thin Plate Splines. Red points remain fixed while one point in space moves along the arrow.',
            imgsrc: 'images/advanced/warp2.png',
        },

        {
            pyname: 'warp3',
            categ: 'advanced',
            short: 'Warp Fit 2D',
            long: 'Use two sets of landmark points to define a displacement field with thin plate splines.',
            imgsrc: 'images/advanced/warp3.png',
        },

        {
            pyname: 'warp4a',
            categ: 'advanced',
            short: 'Morph 2D',
            long: 'Morph or warp a 2D shape by manually setting displacement arrows.',
            imgsrc: 'images/advanced/warp4.png',
        },

        {
            pyname: 'warp4b',
            categ: 'advanced',
            short: 'Morph 3D',
            long: 'Morph or warp a 3D shape by manually assigning a set of corresponding landmarks.',
            imgsrc: 'images/advanced/warp4b.jpg',
        },

        {
            pyname: 'warp5',
            categ: 'advanced',
            short: 'Quadratic Morphing',
            long: 'Morph a source mesh onto a target mesh by fitting the 18 parameters of a quadratic transformation.',
            imgsrc: 'images/advanced/warp5.png',
        },

        {
            pyname: 'splitmesh',
            categ: 'advanced',
            short: 'Mesh Connectivity',
            long: 'Split a mesh by connectivity and order the pieces by surface area.',
            imgsrc: 'images/advanced/splitmesh.png',
        },

        {
            pyname: 'fitline',
            categ: 'advanced',
            short: 'Fit Lines and Planes',
            long: 'Fit a line and a plane to a 3D point cloud.',
            imgsrc: 'images/advanced/fitline.png',
        },

        {
            pyname: 'fitspheres1',
            categ: 'advanced',
            short: 'Fit a Sphere',
            long: 'Fit spheres to a surface region defined by the n points closest to a given point.',
            imgsrc: 'images/advanced/fitspheres1.jpg',
        },

        {
            pyname: 'convex_hull',
            categ: 'advanced',
            short: 'Convex Hull',
            long: 'Create the convex hull of a mesh or a set of input points.',
            imgsrc: 'images/advanced/convexHull.png',
        },

        {
            pyname: 'contours2mesh',
            categ: 'advanced',
            short: 'Contours to Mesh',
            long: 'Generate a surface mesh by joining a set of nearby contour lines.',
            imgsrc: 'images/advanced/contours2mesh.png',
        },

        {
            pyname: 'interpolate_field',
            categ: 'advanced',
            short: 'Field Interpolation',
            long: 'Interpolate a vector field with Thin Plate Splines or radial basis functions. Share the camera between different windows.',
            imgsrc: 'images/advanced/interpolateField.png',
        },

        {
            pyname: 'interpolate_scalar1',
            categ: 'advanced',
            short: 'Transfer Arrays',
            long: 'Interpolate scalar values from one mesh or point cloud object onto another.',
            imgsrc: 'images/advanced/interpolateScalar1.png',
        },

        {
            pyname: 'interpolate_scalar2',
            categ: 'advanced',
            short: 'RBF Interpolation',
            long: 'Use SciPy radial basis functions to interpolate scalar values known at a set of points on a mesh.',
            imgsrc: 'images/advanced/interpolateScalar2.png',
        },

        {
            pyname: 'interpolate_scalar3',
            categ: 'advanced',
            short: 'Closest-Point Transfer',
            long: 'Interpolate arrays from a source mesh onto another mesh (the ellipsoid) by averaging the values of the closest points.',
            imgsrc: 'images/advanced/interpolateScalar3.png',
        },

        {
            pyname: 'interpolate_scalar4',
            categ: 'advanced',
            short: 'Cell Interpolation',
            long: 'Interpolate cell values from a quad mesh to a tri mesh of different resolution.',
            imgsrc: 'images/advanced/interpolateScalar4.png',
        },

        {
            pyname: 'diffuse_data',
            categ: 'advanced',
            short: 'Smooth Arrays',
            long: 'Smooth or diffuse a scalar array on a mesh.',
            imgsrc: 'images/advanced/diffuse_data.png',
        },

        {
            pyname: 'cut_with_mesh1',
            categ: 'advanced',
            short: 'Mesh-by-Mesh Cut',
            long: 'Cut a mesh with another mesh.',
            imgsrc: 'images/advanced/cutWithMesh1.jpg',
        },

        {
            pyname: 'cut_with_points1',
            categ: 'advanced',
            short: 'Cut with Points',
            long: 'Define a loop of points on a mesh to cut it or select a region.',
            imgsrc: 'images/advanced/cutWithPoints1.png',
        },

        {
            pyname: 'cut_with_points2',
            categ: 'advanced',
            short: 'Cut with a Loop',
            long: 'Define a loop of points on a mesh to cut it or select the cells inside.',
            imgsrc: 'images/advanced/cutWithPoints2.png',
        },

        {
            pyname: 'cut_and_cap',
            categ: 'advanced',
            short: 'Cut and Cap',
            long: 'Cut a mesh with another mesh and cap the resulting holes.',
            imgsrc: 'images/advanced/cutAndCap.png',
        },

        {
            pyname: 'gyroid',
            categ: 'advanced',
            short: 'Textured Gyroid',
            long: 'A textured gyroid shape cut by a sphere. Any image texture can be downloaded on the fly.',
            imgsrc: 'images/advanced/gyroid.png',
        },

        {
            pyname: 'timer_callback2',
            categ: 'advanced',
            short: 'Timer App',
            long: 'Create a simple application controlled by a timer callback.',
            imgsrc: 'images/advanced/timer_callback1.jpg',
        },

        {
            pyname: 'spline_draw1',
            categ: 'advanced',
            short: 'Draw a Spline',
            long: 'Draw a spline interactively on a Picture.',
            imgsrc: 'images/advanced/spline_draw.png',
        },

        {
            pyname: 'capping_mesh',
            categ: 'advanced',
            short: 'Capping Mesh',
            long: 'Show manual capping of a mesh. Uses local geometric fitting on mesh neighborhoods.',
            imgsrc: 'images/advanced/capping_mesh.png',
        },

        {
            pyname: 'fitplanes',
            categ: 'advanced',
            short: 'Fit Planes',
            long: 'Fit a plane to regions of a surface defined by the N points closest to a given point on the surface. Uses local geometric fitting on mesh neighborhoods.',
            imgsrc: 'images/advanced/fitplanes.png',
        },

        {
            pyname: 'fitspheres2',
            categ: 'advanced',
            short: 'Local Sphere Fit',
            long: 'For each point, find the 12 closest points and fit a sphere.',
            imgsrc: 'images/advanced/fitspheres2.png',
        },

        {
            pyname: 'geodesic_curve',
            categ: 'advanced',
            short: 'Geodesic Curve',
            long: 'Use Dijkstra\'s algorithm to compute the graph geodesic.',
            imgsrc: 'images/advanced/geodesic_curve.png',
        },

        {
            pyname: 'interpolate_scalar5',
            categ: 'advanced',
            short: 'Surface Interpolation',
            long: 'Interpolate a 2D surface through a set of points.',
            imgsrc: 'images/advanced/interpolate_scalar5.png',
        },

        {
            pyname: 'measure_curvature1',
            categ: 'advanced',
            short: 'Sphere-Fit Curvature',
            long: 'Calculate the surface curvature of an object by fitting a sphere to each vertex. Uses multiple renderers to compare views in parallel.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'measure_curvature2',
            categ: 'advanced',
            short: 'Quadratic Curvature',
            long: 'Estimate Gaussian and mean curvature by local quadratic fitting. Uses multiple renderers to compare views in parallel.',
            imgsrc: 'images/volumetric/measure_curvature2.png',
        },

        {
            pyname: 'mesh_smoother1',
            categ: 'advanced',
            short: 'Mesh Smoothing',
            long: 'Compare a mesh before and after iterative smoothing.',
            imgsrc: 'images/advanced/mesh_smoother1.png',
        },

        {
            pyname: 'skeletonize',
            categ: 'advanced',
            short: 'Skeletonize',
            long: 'Use 1D Moving Least Squares to skeletonize a surface.',
            imgsrc: 'images/advanced/skeletonize.png',
        },

        {
            pyname: 'spline_draw2',
            categ: 'advanced',
            short: 'Image Spline Drawing',
            long: 'Draw a continuous line on an image with the DrawingWidget.',
            imgsrc: 'images/advanced/spline_draw2.png',
        },

        {
            pyname: 'timer_callback0',
            categ: 'advanced',
            short: 'Repeating Timer',
            long: 'Use a repeating timer callback to animate an actor.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'timer_callback1',
            categ: 'advanced',
            short: 'Play/Pause Timer',
            long: 'Create a simple play/pause app with a timer event. You can interact with the scene during the loop.',
            imgsrc: 'images/advanced/timer_callback1.png',
        },

        {
            pyname: 'timer_callback3',
            categ: 'advanced',
            short: 'Dual Timers',
            long: 'Create two independent timer callbacks.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'warp6',
            categ: 'advanced',
            short: 'Hover Warp',
            long: 'Press c while hovering to warp one Mesh onto another Mesh.',
            imgsrc: 'images/advanced/warp6.png',
        },

        {
            pyname: 'numpy2volume2',
            categ: 'volumetric',
            short: 'NumPy Volume',
            long: 'Create a Volume dataset from a ' + insertLink('NumPy', 'numpy.org') + ' array.',
            imgsrc: 'images/volumetric/numpy2volume2.png',
        },

        {
            pyname: 'numpy2volume1',
            categ: 'volumetric',
            short: 'mgrid Volume',
            long: 'Create a Volume dataset from a ' + insertLink('numpy.mgrid', 'numpy.org/doc/stable/reference/generated/numpy.mgrid.html') + ' object.',
            imgsrc: 'images/volumetric/numpy2volume1.png',
        },

        {
            pyname: 'app_isobrowser',
            categ: 'volumetric',
            short: 'Iso Browser',
            long: 'Peel isosurfaces from an input Volume using a slider.',
            imgsrc: 'images/advanced/app_isobrowser.gif',
        },

        {
            pyname: 'app_raycaster',
            categ: 'volumetric',
            short: 'Ray Casting',
            long: 'Visualize an input Volume using ray casting in different modes.',
            imgsrc: 'images/advanced/app_raycaster.gif',
        },

        {
            pyname: 'slicer1',
            categ: 'volumetric',
            short: 'Volume Slicer',
            long: 'Use sliders to control slicing planes through an input volume, and add a button to change the colormap.',
            imgsrc: 'images/volumetric/slicer1.jpg',
        },

        {
            pyname: 'read_volume3',
            categ: 'volumetric',
            short: 'Slice Viewer',
            long: 'Inspect a volumetric dataset interactively by slicing 2D planes with the mouse.',
            imgsrc: 'images/volumetric/read_volume3.jpg',
        },

        {
            pyname: 'isosurfaces1',
            categ: 'volumetric',
            short: 'Isosurfaces',
            long: 'Generate the isosurfaces corresponding to a set of thresholds. These surfaces form a single object.',
            imgsrc: 'images/volumetric/isosurfaces.png',
        },

        {
            pyname: 'read_volume1',
            categ: 'volumetric',
            short: 'Volume Transfer',
            long: 'Load a 3D volume and set voxel color and visibility by defining transfer functions.',
            imgsrc: 'images/volumetric/read_volume1.png',
        },

        {
            pyname: 'read_volume2',
            categ: 'volumetric',
            short: 'Volume Modes',
            long: 'Load a 3D volume and visualize it with either "composite" or "maximum-projection" rendering.',
            imgsrc: 'images/volumetric/read_volume2.png',
        },

        {
            pyname: 'interpolate_volume',
            categ: 'volumetric',
            short: 'Volume Interpolation',
            long: 'Generate a volume by interpolating a scalar field known only at a scattered set of points.',
            imgsrc: 'images/volumetric/59095175-1ec5a300-8918-11e9-8bc0-fd35c8981e2b.jpg',
        },

        {
            pyname: 'densifycloud',
            categ: 'volumetric',
            short: 'Point-Cloud Densify',
            long: 'Add new points to a point cloud so that neighboring points stay within a target distance of one another.',
            imgsrc: 'images/volumetric/densifycloud.png',
        },

        {
            pyname: 'legosurface',
            categ: 'volumetric',
            short: 'Lego Voxels',
            long: "Represent a volume as Lego blocks (voxels). Colors correspond to the volume's scalar values.",
            imgsrc: 'images/volumetric/56820682-da40e500-684c-11e9-8ea3-91cbcba24b3a.png',
        },

        {
            pyname: 'streamlines4',
            categ: 'volumetric',
            short: '2D Streamlines',
            long: 'Draw the streamlines of a 2D vector field.',
            imgsrc: 'images/volumetric/81459343-b9210d00-919f-11ea-846c-152d62cba06e.png',
        },

        {
            pyname: 'streamlines2',
            categ: 'volumetric',
            short: '3D Streamlines',
            long: 'Load an existing structured grid and draw the streamlines of a velocity field.',
            imgsrc: 'images/volumetric/streamlines2.png',
        },

        {
            pyname: 'streamlines2_linewidget',
            categ: 'volumetric',
            short: '3D Streamlines Widget',
            long: 'Load an existing structured grid and draw the streamlines of a velocity field. Use a line widget to interactively change the seed points.',
            imgsrc: 'images/volumetric/streamlines2_linewidget.png',
        },

        {
            pyname: 'office',
            categ: 'volumetric',
            short: 'Airflow Tubes',
            long: 'Show airflow stream tubes in an office with ventilation and a burning cigarette.',
            imgsrc: 'images/volumetric/56964003-9145a500-6b5a-11e9-9d9e-9736d90e1900.png',
        },

        // {
        //     pyname: 'streamlines3',
        //     categ: 'volumetric',
        //     short: 'Streamlines in Cavity',
        //     long: 'Draw streamlines for the cavity case from the ' + insertLink('OpenFOAM tutorial', 'cfd.direct/openfoam/user-guide/v6-cavity'),
        //     imgsrc: 'images/volumetric/streamlines3.png',
        // },

        {
            pyname: 'tensors',
            categ: 'volumetric',
            short: 'Tensors',
            long: 'Visualize stress tensors as oriented ellipsoids.',
            imgsrc: 'images/volumetric/tensors.png',
        },

        {
            pyname: 'multiscalars',
            categ: 'volumetric',
            short: 'Scalar Channels',
            long: 'Extract one scalar channel from a volumetric dataset with multiple scalars associated with each voxel.',
            imgsrc: 'images/volumetric/multiscalars.png',
        },

        {
            pyname: 'lowpassfilter',
            categ: 'volumetric',
            short: 'Low-Pass Filter',
            long: 'Cut off high frequencies in the Fourier transform of a volumetric dataset.',
            imgsrc: 'images/volumetric/lowpassfilter.png',
        },

        {
            pyname: 'erode_dilate',
            categ: 'volumetric',
            short: 'Erode and Dilate',
            long: 'Erode or dilate a Volume by replacing a voxel with the max/min over an ellipsoidal neighborhood.',
            imgsrc: 'images/volumetric/erode_dilate.png',
        },

        {
            pyname: 'mesh2volume',
            categ: 'volumetric',
            short: 'Volume Binarization',
            long: 'Build a volume from a mesh, setting inside voxels to 1 and outside voxels to 0.',
            imgsrc: 'images/volumetric/mesh2volume.png',
        },

        {
            pyname: 'probe_points',
            categ: 'volumetric',
            short: 'Point Probe',
            long: 'Probe a volumetric dataset with a point cloud and plot the intensity values.',
            imgsrc: 'images/volumetric/probePoints.png',
        },

        {
            pyname: 'probe_line2',
            categ: 'volumetric',
            short: 'Line Probe',
            long: 'Probe a volumetric dataset with a line and plot the intensity values.',
            imgsrc: 'images/volumetric/probeLine2.png',
        },

        {
            pyname: 'probe_line1',
            categ: 'volumetric',
            short: 'Line Probes',
            long: 'Probe a volumetric dataset with lines and color-code them.',
            imgsrc: 'images/volumetric/probeLine1.png',
        },

        {
            pyname: 'slice_plane1',
            categ: 'volumetric',
            short: 'Plane Probe',
            long: 'Slice/probe a Volume with a simple oriented plane.',
            imgsrc: 'images/volumetric/slicePlane1.gif',
        },

        {
            pyname: 'slice_plane2',
            categ: 'volumetric',
            short: 'Plane Probes',
            long: 'Slice/probe a Volume with multiple planes. Make low scalar values completely transparent.',
            imgsrc: 'images/volumetric/slicePlane2.png',
        },

        {
            pyname: 'slice_mesh',
            categ: 'volumetric',
            short: 'Surface Probe',
            long: 'Slice/probe a Volume with a polygonal mesh.',
            imgsrc: 'images/volumetric/sliceMesh.png',
        },

        {
            pyname: 'slice_plane3',
            categ: 'volumetric',
            short: 'Interactive Probing',
            long: 'Slice/probe a Volume interactively.',
            imgsrc: 'images/volumetric/slicePlane3.jpg',
        },

        {
            pyname: 'slab_vol',
            categ: 'volumetric',
            short: 'Slice a Slab',
            long: 'Average the intensity over a thick "slab" of a Volume.',
            imgsrc: 'images/volumetric/slab_vol.jpg',
        },

        {
            pyname: 'delaunay3d',
            categ: 'volumetric',
            short: '3D Delaunay',
            long: 'Use the Delaunay algorithm to generate a tetrahedral mesh from a convex surface.',
            imgsrc: 'images/volumetric/delaunay3d.png',
        },

        {
            pyname: 'tetralize_surface',
            categ: 'volumetric',
            short: 'Tetralize Surface',
            long: 'Generate a tetrahedral mesh from an arbitrary closed polygonal surface.',
            imgsrc: 'images/volumetric/tetralize_surface.jpg',
        },

        {
            pyname: 'tet_threshold',
            categ: 'volumetric',
            short: 'Tetmesh Thresholding',
            long: 'Threshold a tetrahedral mesh using a scalar array.',
            imgsrc: 'images/volumetric/82767103-2500a800-9e25-11ea-8506-e583e8ec4b01.jpg',
        },

        {
            pyname: 'tet_cut1',
            categ: 'volumetric',
            short: 'Tetmesh Cutting',
            long: 'Cut a tetrahedral mesh with an arbitrary polygonal mesh.',
            imgsrc: 'images/volumetric/82767107-2631d500-9e25-11ea-967c-42558f98f721.jpg',
        },

        {
            pyname: 'tet_isos_slice',
            categ: 'volumetric',
            short: 'Tetmesh Slicing',
            long: 'Slice a tetrahedral mesh with a plane.',
            imgsrc: 'images/volumetric/tet_isos_slice.png',
        },

        {
            pyname: 'earth_model',
            categ: 'volumetric',
            short: 'Earth Model',
            long: 'Create a customized representation of a tetrahedral Earth model.',
            imgsrc: 'images/volumetric/earth_model.jpg',
        },

        {
            pyname: 'ugrid2',
            categ: 'volumetric',
            short: 'Unstructured Grids',
            long: 'Cut an unstructured grid with a plane.',
            imgsrc: 'images/volumetric/ugrid2.png',
        },

        {
            pyname: 'colorize_volume',
            categ: 'volumetric',
            short: 'Colorize Volume',
            long: 'Define custom color and transparency maps for Volumes.',
            imgsrc: 'images/volumetric/colorize_volume.png',
        },

        {
            pyname: 'euclidian_dist',
            categ: 'volumetric',
            short: 'Euclidean Distance',
            long: 'Compute the Euclidean distance transform using the Saito algorithm.',
            imgsrc: 'images/volumetric/euclidian_dist.png',
        },

        {
            pyname: 'isosurfaces2',
            categ: 'volumetric',
            short: 'Isosurfaces',
            long: 'Extract isosurfaces from a volume dataset with discrete values (labels).',
            imgsrc: 'images/volumetric/isosurfaces2.png',
        },

        {
            pyname: 'numpy2volume0',
            categ: 'volumetric',
            short: 'NumPy to Volume',
            long: 'Modify a Volume in place from a NumPy array.',
            imgsrc: 'images/volumetric/numpy2volume0.png',
        },

        {
            pyname: 'numpy_imread',
            categ: 'volumetric',
            short: 'NumPy Image Read',
            long: 'Create a Volume from a NumPy object using imread.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'rectl_grid1',
            categ: 'volumetric',
            short: 'Rectilinear Grid',
            long: 'A RectilinearGrid is a dataset whose edges are parallel to the coordinate axes.',
            imgsrc: 'images/volumetric/rectl_grid1.png',
        },

        {
            pyname: 'slicer2',
            categ: 'volumetric',
            short: 'Volume Slicer',
            long: 'Slice multiple volumetric datasets simultaneously, navigating cross-sections with interactive slice-position controls.',
            imgsrc: 'images/volumetric/slicer2.png',
        },

        {
            pyname: 'slicer_set_volume',
            categ: 'volumetric',
            short: 'Volume Slicer',
            long: 'Swap the input of Slicer3DPlotter without recreating the window.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'streamlines1',
            categ: 'volumetric',
            short: 'Streamlines',
            long: 'Generate streamlines from a set of seed points in space within a vector field defined on a small set of points. Computes streamlines from a volumetric vector field.',
            imgsrc: 'images/volumetric/streamlines1.png',
        },

        {
            pyname: 'struc_grid1',
            categ: 'volumetric',
            short: 'Structured Grid',
            long: 'Create, cut, and visualize a noisy structured grid dataset.',
            imgsrc: 'images/volumetric/struc_grid1.png',
        },

        {
            pyname: 'tet_astyle',
            categ: 'volumetric',
            short: 'Tet Mesh Styles',
            long: 'Load a tetrahedral mesh and display it in different styles.',
            imgsrc: 'images/volumetric/tet_astyle.png',
        },

        {
            pyname: 'tet_build',
            categ: 'volumetric',
            short: 'Build Tetrahedral Mesh',
            long: 'Build a tetrahedral mesh by manually defining vertices and cells.',
            imgsrc: 'images/volumetric/tet_build.png',
        },

        {
            pyname: 'tet_cut2',
            categ: 'volumetric',
            short: 'Cut Tetrahedral Mesh',
            long: 'Cut a tetrahedral mesh with a Mesh to generate an UnstructuredGrid.',
            imgsrc: 'images/volumetric/tet_cut2.png',
        },

        {
            pyname: 'tet_explode',
            categ: 'volumetric',
            short: 'Explode Tetrahedrl Mesh',
            long: 'Segment a tetrahedral mesh with a custom scalar and show an exploded view of it.',
            imgsrc: 'images/volumetric/tet_explode.png',
        },

        {
            pyname: 'ugrid1',
            categ: 'volumetric',
            short: 'Unstructured Grid',
            long: 'Cut an unstructured grid with a mesh.',
            imgsrc: 'images/volumetric/ugrid1.png',
        },

        {
            pyname: 'vol2points',
            categ: 'volumetric',
            short: 'Volume to Points',
            long: 'Extract all image voxels as points. Converts voxels into a point-cloud representation.',
            imgsrc: 'images/volumetric/vol2points.png',
        },

        {
            pyname: 'volume_from_mesh',
            categ: 'volumetric',
            short: 'Volume From Mesh',
            long: 'Generate a Volume from the signed distance to a Mesh, then extract the isosurface at distance -0.5.',
            imgsrc: 'images/volumetric/volume_from_mesh.png',
        },

        {
            pyname: 'volume_operations',
            categ: 'volumetric',
            short: 'Volume Operations',
            long: 'Perform simple mathematical operations between 3D Volumes. Uses multiple renderers to compare views in parallel.',
            imgsrc: 'images/volumetric/volume_operations.png',
        },

        {
            pyname: 'volume_sharemap',
            categ: 'volumetric',
            short: 'Shared Volume ColorMap',
            long: 'Share the same color and transparency mapping across different volumes.',
            imgsrc: 'images/volumetric/volume_sharemap.png',
        },

        {
            pyname: 'warp_scalars',
            categ: 'volumetric',
            short: 'Warp Scalars',
            long: 'Warp scalars inside a volumetric dataset.',
            imgsrc: 'images/volumetric/warp_scalars.png',
        },

        {
            pyname: 'image_rgba',
            categ: 'other',
            short: 'NumPy Image',
            long: 'Create an image from a NumPy array containing an alpha channel for opacity.',
            imgsrc: 'images/extras/image_rgba.png',
        },

        {
            pyname: 'image_false_colors',
            categ: 'other',
            short: 'False Colors',
            long: 'Generate the Mandelbrot set as a color-mapped Picture object.',
            imgsrc: 'images/extras/image_false_colors.png',
        },

        {
            pyname: 'image_to_mesh',
            categ: 'other',
            short: 'Image to Mesh',
            long: 'Transform a standard JPG/PNG image into a polygonal mesh, or threshold it.',
            imgsrc: 'images/extras/image_to_mesh.jpg',
        },

        {
            pyname: 'image_probe',
            categ: 'other',
            short: 'Image Probe',
            long: 'Probe image intensities along a set of lines.',
            imgsrc: 'images/extras/image_probe.jpg',
        },

        {
            pyname: 'image_fft',
            categ: 'other',
            short: '2D Fourier Transform',
            long: 'Perform a 2D Fast Fourier Transform on an image.',
            imgsrc: 'images/extras/image_fft.png',
        },


        {
            pyname: 'spline_ease',
            categ: 'simulations',
            short: 'Eased Spline',
            long: 'Spline a set of points into a line of a given resolution. Control point density to create an ' + insertLink('easing', 'easings.net') + ' effect.',
            imgsrc: 'images/simulations/spline_ease.gif',
        },

        {
            pyname: 'trail',
            categ: 'simulations',
            short: 'Motion Trail',
            long: 'Add a trailing line to a moving object.',
            imgsrc: 'images/simulations/trail.gif',
        },

        {
            pyname: 'airplane1',
            categ: 'simulations',
            short: 'One Airplane',
            long: 'Draw the shadow and trailing line of a moving object.',
            imgsrc: 'images/simulations/airplane1.png',
        },

        {
            pyname: 'airplane2',
            categ: 'simulations',
            short: 'Airplane Trails',
            long: 'Draw shadows and trailing lines for two moving objects.',
            imgsrc: 'images/simulations/57341963-b8910900-713c-11e9-898a-84b6d3712bce.gif',
        },

        {
            pyname: 'aspring1',
            categ: 'simulations',
            short: 'Spring Damping',
            long: 'Simulate a block connected to a spring in a viscous medium.',
            imgsrc: 'images/simulations/50738955-7e891800-11d9-11e9-85cd-02bd4f3f13ea.gif',
        },

        {
            pyname: 'mag_field1',
            categ: 'simulations',
            short: 'Magnetic Field',
            long: 'Drag points to compute and visualize the magnetic field generated by a wire.',
            imgsrc: 'images/simulations/mag_field.png',
        },

        {
            pyname: 'grayscott',
            categ: 'simulations',
            short: 'Reaction-Diffusion',
            long: 'A Turing reaction-diffusion system between two molecules: the ' + insertLink('Gray-Scott', 'mrob.com/pub/comp/xmorphia/index.html') + ' model.',
            imgsrc: 'images/simulations/grayscott.gif',
        },

        {
            pyname: 'doubleslit',
            categ: 'simulations',
            short: 'Double Slit',
            long: 'Simulate the double-slit experiment with any number of slits and arbitrary slit geometry.',
            imgsrc: 'images/simulations/96374703-86c70300-1174-11eb-9bfb-431a1ae5346d.png',
        },

        {
            pyname: 'tunnelling1',
            categ: 'simulations',
            short: 'Tunneling',
            long: 'Simulate quantum tunneling with a fourth-order Runge-Kutta method and an arbitrary potential shape.',
            imgsrc: 'images/simulations/96375030-e0c8c800-1176-11eb-8fde-83a65de41330.gif',
        },

        {
            pyname: 'tunnelling2',
            categ: 'simulations',
            short: 'Quantum Grid',
            long: 'Show the evolution of a particle in a box hitting a sinusoidal potential barrier.',
            imgsrc: 'images/simulations/tunneling2.gif',
        },

        {
            pyname: 'particle_simulator',
            categ: 'simulations',
            short: 'Rutherford Scattering',
            long: 'Simulate Rutherford scattering of interacting charged particles in 3D space.',
            imgsrc: 'images/simulations/50738891-db380300-11d8-11e9-84c2-0f55be7228f1.gif',
        },

        {
            pyname: 'lorenz',
            categ: 'simulations',
            short: 'Lorenz Attractor',
            long: 'Show the classic ' + insertLink('Lorenz attractor', 'en.wikipedia.org/wiki/Lorenz_system'),
            imgsrc: 'images/simulations/lorenz.png',
        },

        {
            pyname: 'fourier_epicycles',
            categ: 'simulations',
            short: 'Fourier Epicycles',
            long: 'Reconstruct a 2D shape with Fourier epicycles, showing its ' + insertLink('epicycle components', 'thecodingtrain.com/CodingChallenges/130.2-fourier-transform-drawing.html'),
            imgsrc: 'images/simulations/fourier_epicycles.gif',
        },

        {
            pyname: 'pendulum_ode',
            categ: 'simulations',
            short: '2D Pendulum',
            long: 'Simulate a composite pendulum by solving the corresponding system of ODEs.',
            imgsrc: 'images/simulations/pendulum_ode.gif',
        },

        {
            pyname: 'pendulum_3d',
            categ: 'simulations',
            short: '3D Pendulum',
            long: 'Simulate a ' + insertLink('composite pendulum', 'www.youtube.com/watch?v=MtG9cueB548') + ' with Lagrangian mechanics in 3D.',
            imgsrc: 'images/simulations/pendulum_3d.gif',
        },

        {
            pyname: 'multiple_pendulum',
            categ: 'simulations',
            short: 'N-Pendulum',
            long: 'Simulate a multiple pendulum using simple Euler integration.',
            imgsrc: 'images/simulations/multiple_pendulum.gif',
        },

        {
            pyname: 'gyroscope1',
            categ: 'simulations',
            short: 'Hanging Gyroscope',
            long: 'Simulate a gyroscope hanging from a spring.',
            imgsrc: 'images/simulations/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif',
        },

        {
            pyname: 'wave_equation1d',
            categ: 'simulations',
            short: 'Oscillator Chain',
            long: 'Simulate a set of coupled oscillators to compare two integration schemes: Euler vs. Runge-Kutta4.',
            imgsrc: 'images/simulations/39360796-ea5f9ef0-4a1f-11e8-85cb-f3e21072c7d5.gif',
        },

        {
            pyname: 'wave_equation2d',
            categ: 'simulations',
            short: 'Wave Equation',
            long: 'Solve the 2D wave equation using finite differences and the forward Euler method.',
            imgsrc: 'images/simulations/wave2d.gif',
        },

        {
            pyname: 'brownian2d',
            categ: 'simulations',
            short: 'Brownian Swarm',
            long: 'Show the motion of a large Brownian particle in a swarm of small particles in 2D.',
            imgsrc: 'images/simulations/50738948-73ce8300-11d9-11e9-8ef6-fc4f64c4a9ce.gif',
        },

        {
            pyname: 'gas',
            categ: 'simulations',
            short: 'Gas in a Toroidal Tank',
            long: 'Model an ideal gas with hard-sphere collisions.',
            imgsrc: 'images/simulations//50738954-7e891800-11d9-11e9-95aa-67c92ca6476b.gif',
        },

        {
            pyname: 'volterra',
            categ: 'simulations',
            short: 'Lotka-Volterra Model',
            long: 'Show the Lotka-Volterra model, where x is the number of prey and y is the number of predators.',
            imgsrc: 'images/simulations/volterra.png',
        },

        {
            pyname: 'drag_chain',
            categ: 'simulations',
            short: 'Forward Kinematics',
            long: 'Move the mouse over a 3D surface to drag the chain of rigid segments',
            imgsrc: 'images/simulations/drag_chain.gif',
        },
        {
            pyname: 'optics_main2',
            categ: 'simulations',
            short: 'Optical System',
            long: 'Simulate an optical system with lenses of arbitrary shapes and orientations.',
            imgsrc: 'images/simulations/optics_main2.png',
        },

        {
            pyname: 'optics_main3',
            categ: 'simulations',
            short: 'Butterfly Effect',
            long: 'Show the ' + insertLink('butterfly effect', 'www.youtube.com/watch?v=kBow0kTVn3s') + ' with cylindrical mirrors, a laser, and a photon detector.',
            imgsrc: 'images/simulations/optics_main3.gif',
        },

        {
            pyname: 'self_org_maps2d',
            categ: 'simulations',
            short: 'Self-Organizing Maps',
            long: 'Show self-organizing maps ' + insertLink('(SOM)', 'en.wikipedia.org/wiki/Self-organizing_map') + ', a type of artificial neural network trained by unsupervised learning.',
            imgsrc: 'images/simulations/self_org_maps2d.gif',
        },

        {
            pyname: 'value_iteration',
            categ: 'simulations',
            short: 'Maze Solver',
            long: 'Solve a random maze with a Markov Decision Process ' + insertLink('(MDP)', 'en.wikipedia.org/wiki/Markov_decision_process'),
            imgsrc: 'images/simulations/value_iteration.png',
        },

        {
            pyname: 'aizawa_attractor',
            categ: 'simulations',
            short: 'Aizawa Attractor',
            long: 'Show an interactive particle swarm on the Aizawa strange attractor.',
            imgsrc: 'images/simulations/aizawa_attractor.png',
        },

        {
            pyname: 'aspring2_player',
            categ: 'simulations',
            short: 'Viewer Play and Pause',
            long: 'Animate a block attached to a spring.',
            imgsrc: 'images/simulations/aspring2_player.png',
        },

        {
            pyname: 'koch_fractal',
            categ: 'simulations',
            short: 'Koch Fractal',
            long: 'Show the Koch snowflake fractal.',
            imgsrc: 'images/simulations/koch_fractal.png',
        },

        {
            pyname: 'optics_main1',
            categ: 'simulations',
            short: 'Optics',
            long: 'Show several optical scenarios using lenses, mirrors, and ray tracing.',
            imgsrc: 'images/simulations/optics_main1.png',
        },

        {
            pyname: 'springs_fem',
            categ: 'simulations',
            short: 'Springs FEM',
            long: 'Solve a system of springs using the finite element method.',
            imgsrc: 'images/simulations/springs_fem.png',
        },

        {
            pyname: 'earthquake_browser',
            categ: 'plotting',
            short: "Earthquake Feed",
            long: 'Visualize magnitude 2.5+ earthquakes from the past 30 days with a slider. Marker areas are proportional to energy release.',
            imgsrc: 'images/pyplot/earthquake_browser.jpg',
        },

        {
            pyname: 'fonts3d',
            categ: 'plotting',
            short: '3D Fonts',
            long: 'Visualize all available 2D and 3D polygonal fonts (check for more ' + insertLink('here', 'vedo.embl.es/fonts') + ')',
            imgsrc: 'images/pyplot/fonts3d.png',
        },

        {
            pyname: 'latex',
            categ: 'plotting',
            short: 'LaTeX Formulas',
            long: 'Generate an expression image from standard LaTeX syntax.',
            imgsrc: 'images/pyplot/latex.png',
        },

        {
            pyname: 'custom_axes1',
            categ: 'plotting',
            short: 'Custom Axes',
            long: 'Create customized axes with more than 40 parameter options.',
            imgsrc: 'images/pyplot/customAxes1.png',
        },

        {
            pyname: 'custom_axes2',
            categ: 'plotting',
            short: 'Inverted Axes',
            long: 'Shift and invert axis directions and labels.',
            imgsrc: 'images/pyplot/customAxes2.png',
        },

        {
            pyname: 'custom_axes3',
            categ: 'plotting',
            short: 'Shifted Planes',
            long: 'Displace Cartesian planes from their default lower-range positions.',
            imgsrc: 'images/pyplot/customAxes3.png',
        },

        {
            pyname: 'custom_axes4',
            categ: 'plotting',
            short: 'Local Axes',
            long: 'Create individual axes for each object in a scene. Access any element to change its size and color.',
            imgsrc: 'images/pyplot/customIndividualAxes.png',
        },

        {
            pyname: 'markpoint',
            categ: 'plotting',
            short: 'Follow the Camera',
            long: 'Lock an object\'s orientation so it constantly faces the scene camera.',
            imgsrc: 'images/pyplot/markpoint.jpg',
        },

        {
            pyname: 'scatter2',
            categ: 'plotting',
            short: 'Variable Marker Sizes',
            long: 'Create a scatter plot with marker size proportional to sin(2x) and red level proportional to cos(2x).',
            imgsrc: 'images/pyplot/scatter2.png',
        },

        {
            pyname: 'scatter3',
            categ: 'plotting',
            short: 'Scatter Plot',
            long: 'Create a scatter plot that overlays three different point distributions.',
            imgsrc: 'images/pyplot/scatter3.png',
        },

        {
            pyname: 'plot_errbars',
            categ: 'plotting',
            short: 'Plot Styles',
            long: 'Overlay 1D plots with different line and marker styles.',
            imgsrc: 'images/pyplot/plot_errbars.png',
        },

        {
            pyname: 'plot_pip',
            categ: 'plotting',
            short: 'Picture-in-Picture',
            long: 'Create a picture-in-picture plot.',
            imgsrc: 'images/pyplot/plot_pip.png',
        },

        {
            pyname: 'fit_polynomial1',
            categ: 'plotting',
            short: 'Linear Fit',
            long: 'Perform linear fitting and use a Monte Carlo + bootstrap technique to obtain reliable errors and error bands.',
            imgsrc: 'images/pyplot/fitPolynomial1.png',
        },

        {
            pyname: 'fit_polynomial2',
            categ: 'plotting',
            short: 'Polynomial Fit',
            long: 'Perform polynomial fitting and use a Monte Carlo + bootstrap technique to obtain reliable errors and error bands.',
            imgsrc: 'images/pyplot/fitPolynomial2.png',
        },

        {
            pyname: 'fit_erf',
            categ: 'plotting',
            short: 'Custom Fit',
            long: 'Fit data with error bars to a custom function. Add labels to the figure.',
            imgsrc: 'images/pyplot/fit_erf.png',
        },

        {
            pyname: 'fit_curve1',
            categ: 'plotting',
            short: 'Curve Fit',
            long: 'Fit a curve to a dataset and add a legend to the figure.',
            imgsrc: 'images/pyplot/fit_curve.png',
        },

        {
            pyname: 'plot_errband',
            categ: 'plotting',
            short: 'Line with Error Bands',
            long: 'Plot continuous functions with known error bands.',
            imgsrc: 'images/pyplot/plot_errband.png',
        },

        {
            pyname: 'plot_extra_yaxis',
            categ: 'plotting',
            short: 'Extra Y-Axis',
            long: 'Add a secondary y-axis for unit conversion and embed it in the 3D world-coordinate system.',
            imgsrc: 'images/pyplot/plot_extra_yaxis.png',
        },

        {
            pyname: 'fit_circle',
            categ: 'plotting',
            short: 'Fit Circles in 3D',
            long: 'Perform fast, analytic circle fitting in 3D. Compute the signed curvature of a curve in space.',
            imgsrc: 'images/pyplot/fitCircle.png',
        },

        {
            pyname: 'lines_intersect',
            categ: 'plotting',
            short: 'Coplanar Intersections',
            long: 'Find the intersection points of two coplanar lines.',
            imgsrc: 'images/pyplot/lines_intersect.png',
        },

        {
            pyname: 'intersect2d',
            categ: 'plotting',
            short: 'Intersect Triangles',
            long: 'Find the overlapping area of two triangles.',
            imgsrc: 'images/pyplot/intersect2d.png',
        },

        {
            pyname: 'explore5d',
            categ: 'plotting',
            short: 'Point-Cloud Analysis',
            long: 'Read data from an ASCII file and perform a simple analysis by visualizing 3 of the dataset\'s 5 dimensions.',
            imgsrc: 'images/pyplot/explore5d.png',
        },

        {
            pyname: 'plot_density2d',
            categ: 'plotting',
            short: '2D Density',
            long: 'Create a density plot from a distribution of points in 2D.',
            imgsrc: 'images/pyplot/plot_density2d.png',
        },

        {
            pyname: 'plot_density3d',
            categ: 'plotting',
            short: '3D Density',
            long: 'Create a volumetric density plot from a distribution of points in 3D.',
            imgsrc: 'images/pyplot/plot_density3d.png',
        },

        {
            pyname: 'plot_density4d',
            categ: 'plotting',
            short: '4D Density',
            long: 'Plot the time evolution of a density field in space.',
            imgsrc: 'images/pyplot/plot_density4d.gif',
        },

        {
            pyname: 'goniometer',
            categ: 'plotting',
            short: 'Goniometer',
            long: 'Create a 3D ruler-style axis, a vignette, and a goniometer.',
            imgsrc: 'images/pyplot/goniometer.png',
        },

        {
            pyname: 'graph_network',
            categ: 'plotting',
            short: 'Graph Network',
            long: 'Optimize and visualize a 2D/3D network with its properties.',
            imgsrc: 'images/pyplot/graph_network.png',
        },

        {
            pyname: 'graph_lineage',
            categ: 'plotting',
            short: 'Lineage Graph',
            long: 'Generate a lineage graph of cell divisions.',
            imgsrc: 'images/pyplot/graph_lineage.png',
        },

        {
            pyname: 'plot_fxy0',
            categ: 'plotting',
            short: 'Plot 1D Function f(x)',
            long: 'Create an interactive 1D function viewer where a parameter slider updates the plotted curve instantly.',
            imgsrc: 'images/pyplot/plot_fxy0.png',
        },

        {
            pyname: 'plot_fxy1',
            categ: 'plotting',
            short: 'Plot 2D Function f(x, y)',
            long: 'Draw a z = f(x,y) surface specified as a string or as a reference to an external function.',
            imgsrc: 'images/pyplot/plot_fxy.png',
        },

        {
            pyname: 'plot_fxy2',
            categ: 'plotting',
            short: 'Plot Imaginary Function',
            long: 'Draw a z = f(x,y) surface specified as a string or as a reference to an external function.',
            imgsrc: 'images/pyplot/plot_fxy2.jpg',
        },

        {
            pyname: 'isolines',
            categ: 'plotting',
            short: 'Isolines and Gradients',
            long: 'Draw the isolines and isobands of a scalar field on a surface. Compute the gradient of the field.',
            imgsrc: 'images/pyplot/isolines.png',
        },

        {
            pyname: 'histo_1d_b',
            categ: 'plotting',
            short: '1D Histogram',
            long: 'Create and overlay a simple 1D histogram with error bars.',
            imgsrc: 'images/pyplot/histo_1D.png',
        },

        {
            pyname: 'histo_gauss',
            categ: 'plotting',
            short: 'Histograms and Curves',
            long: 'Create and overlay a simple 1D histogram with fitted curves.',
            imgsrc: 'images/pyplot/histo_gauss.png',
        },

        {
            pyname: 'histo_pca',
            categ: 'plotting',
            short: 'Axis Histogram',
            long: 'Create a 1D histogram of a distribution along a PCA axis.',
            imgsrc: 'images/pyplot/histo_pca.png',
        },

        {
            pyname: 'plot_bars',
            categ: 'plotting',
            short: 'Bar Plot',
            long: 'Create a bar-style plot. Useful for plotting categories.',
            imgsrc: 'images/pyplot/plot_bars.png',
        },

        {
            pyname: 'histo_2d_a',
            categ: 'plotting',
            short: 'Histogram in 2D',
            long: 'Create a histogram of two independent variables.',
            imgsrc: 'images/pyplot/histo_2D.png',
        },

        {
            pyname: 'np_matrix',
            categ: 'plotting',
            short: 'Matrix View',
            long: 'Visualize a NumPy array or a categorical 2D scalar field.',
            imgsrc: 'images/pyplot/np_matrix.png',
        },

        {
            pyname: 'plot_hexcells',
            categ: 'plotting',
            short: 'Hex Bar Plot',
            long: 'Plot two independent variables as hexagonal bars.',
            imgsrc: 'images/pyplot/plot_hexcells.png',
        },

        {
            pyname: 'histo_hexagonal',
            categ: 'plotting',
            short: 'Hex Histogram',
            long: 'Create a histogram of two independent variables using hexagonal bins.',
            imgsrc: 'images/pyplot/histo_hexagonal.png',
        },

        {
            pyname: 'histo_3d',
            categ: 'plotting',
            short: '3D Histogram',
            long: 'Create a histogram of three independent variables.',
            imgsrc: 'images/pyplot/histo_3D.png',
        },

        {
            pyname: 'plot_spheric',
            categ: 'plotting',
            short: 'Spherical Surface',
            long: 'Create a surface plot in spherical coordinates. The spherical harmonic function is Y(l=2, m=0).',
            imgsrc: 'images/pyplot/plot_spheric.png',
        },

        {
            pyname: 'quiver',
            categ: 'plotting',
            short: 'Quiver Plot',
            long: 'Create a simple quiver-style plot.',
            imgsrc: 'images/pyplot/quiver.png',
        },

        {
            pyname: 'plot_stream',
            categ: 'plotting',
            short: 'Streamlines',
            long: 'Plot streamlines of a 2D field starting from a given set of seed points.',
            imgsrc: 'images/pyplot/plot_stream.png',
        },

        {
            pyname: 'histo_violin',
            categ: 'plotting',
            short: 'Violin Plot',
            long: 'Create a violin-style plot of a few well-known statistical distributions.',
            imgsrc: 'images/pyplot/histo_violin.png',
        },

        {
            pyname: 'whiskers',
            categ: 'plotting',
            short: 'Whisker Plot',
            long: 'Create a whisker-style plot with quantile indications (the horizontal line shows the mean value).',
            imgsrc: 'images/pyplot/whiskers.png',
        },

        {
            pyname: 'anim_lines',
            categ: 'plotting',
            short: 'Time-Series Plot',
            long: 'Create an animated plot showing the evolution of multiple temporal datasets.',
            imgsrc: 'images/pyplot/anim_lines.gif',
        },

        {
            pyname: 'donut',
            categ: 'plotting',
            short: 'Donut Plot',
            long: 'Create a donut-style plot with labels.',
            imgsrc: 'images/pyplot/donut.png',
        },

        {
            pyname: 'plot_polar',
            categ: 'plotting',
            short: 'Splined Polar Plot',
            long: 'Create a polar function plot with optional coordinate splining.',
            imgsrc: 'images/pyplot/plot_polar.png',
        },

        {
            pyname: 'histo_polar',
            categ: 'plotting',
            short: 'Polar Histogram',
            long: 'Create a polar histogram with error bars and/or color mapping.',
            imgsrc: 'images/pyplot/histo_polar.png',
        },

        {
            pyname: 'histo_spheric',
            categ: 'plotting',
            short: 'Spherical Histogram',
            long: 'Create a spherical histogram with elevation and/or color mapping.',
            imgsrc: 'images/pyplot/histo_spheric.png',
        },

        {
            pyname: 'triangulate2d',
            categ: 'plotting',
            short: 'Triangulate Areas',
            long: 'Triangulate arbitrary line contours in 2D. The contours may be concave and may even contain holes.',
            imgsrc: 'images/pyplot/triangulate2d.png',
        },

        {
            pyname: 'andrews_cluster',
            categ: 'plotting',
            short: 'Andrews Cluster',
            long: 'Show Andrews curves for the Iris dataset.',
            imgsrc: 'images/pyplot/andrews_cluster.png',
        },

        {
            pyname: 'embed_matplotlib2',
            categ: 'plotting',
            short: 'Matplotlib + Vedo',
            long: 'Combine vedo plots and ' + insertLink('Matplotlib', 'matplotlib.org') + ' charts in one scene, with a shared slider that updates both views simultaneously.',
            imgsrc: 'images/pyplot/embed_matplotlib2.png',
        },

        {
            pyname: 'embed_matplotlib1',
            categ: 'plotting',
            short: 'Matplotlib Backgrounds',
            long: 'Place charts generated with ' + insertLink('Matplotlib', 'matplotlib.org') + ' into a vedo scene as image overlays.',
            imgsrc: 'images/pyplot/embed_matplotlib1.png',
        },

        {
            pyname: 'fill_gap',
            categ: 'plotting',
            short: 'Fill the Gap',
            long: 'Interpolate the gap between two functions.',
            imgsrc: 'images/pyplot/fill_gap.png',
        },

        {
            pyname: 'fit_curve2',
            categ: 'plotting',
            short: 'Slider Curve Fit',
            long: 'Fit a curve to data and sweep the k parameter with a slider, watching the fitted curve respond live.',
            imgsrc: 'images/pyplot/fit_curve2.png',
        },

        {
            pyname: 'histo_1d_a',
            categ: 'plotting',
            short: 'Basic Histogram',
            long: 'Show a minimal 1D histogram example.',
            imgsrc: 'images/pyplot/histo_1d_a.png',
        },

        {
            pyname: 'histo_1d_c',
            categ: 'plotting',
            short: 'Weighted Histogram',
            long: 'Show a uniform distribution weighted by sin^2 12x + :onehalf.',
            imgsrc: 'images/pyplot/histo_1d_c.png',
        },

        {
            pyname: 'histo_1d_d',
            categ: 'plotting',
            short: 'Nested Histogram',
            long: 'Insert one Figure into another (note that the x-axes stay aligned).',
            imgsrc: 'images/pyplot/histo_1d_d.png',
        },

        {
            pyname: 'histo_1d_e',
            categ: 'plotting',
            short: 'Distance Histogram',
            long: 'Plot a histogram of the distance from each point on a sphere to the oceans mesh.',
            imgsrc: 'images/pyplot/histo_1d_e.png',
        },

        {
            pyname: 'histo_2d_b',
            categ: 'plotting',
            short: '3D Bar Histogram',
            long: 'Create a histogram of two variables as 3D bars.',
            imgsrc: 'images/pyplot/histo_2d_b.png',
        },

        {
            pyname: 'histo_manual',
            categ: 'plotting',
            short: 'Manual Histogram',
            long: 'Show categories and repeats.',
            imgsrc: 'images/pyplot/histo_manual.png',
        },

        {
            pyname: 'markers',
            categ: 'plotting',
            short: 'Markers',
            long: 'Show a set of markers, analogous to Matplotlib.',
            imgsrc: 'images/pyplot/markers.png',
        },

        {
            pyname: 'pie_chart',
            categ: 'plotting',
            short: 'Pie Chart',
            long: 'Build a pie chart and display it as a 2D overlay.',
            imgsrc: 'images/pyplot/pie_chart.png',
        },

        {
            pyname: 'plot_empty',
            categ: 'plotting',
            short: 'Empty Plot',
            long: 'Create an empty Figure to be filled in a loop. Any 3D Mesh object can be added to the figure.',
            imgsrc: 'images/pyplot/plot_empty.png',
        },

        {
            pyname: 'scatter1',
            categ: 'plotting',
            short: 'Basic Scatter',
            long: 'Create a simple scatter plot.',
            imgsrc: 'images/pyplot/scatter1.png',
        },

        {
            pyname: 'scatter_large',
            categ: 'plotting',
            short: 'Large Scatter',
            long: 'Create a scatter plot of 1M points with assigned colors and transparencies.',
            imgsrc: 'images/pyplot/scatter_large.png',
        },

        {
            pyname: 'caption',
            categ: 'plotting',
            short: '2D Captions',
            long: 'Attach a 2D caption to an object and use Chinese, Japanese, and Russian fonts.',
            imgsrc: 'images/pyplot/caption.png',
        },

        {
            pyname: 'make_video',
            categ: 'other',
            short: 'Make a Video',
            long: 'Make a video by setting a sequence of camera positions or by adding individual frames.',
            imgsrc: 'images/extras/makeVideo.gif',
        },

        {
            pyname: 'clone2d',
            categ: 'other',
            short: '2D Clones',
            long: 'Make a static 2D copy of a 3D mesh and place it anywhere in the rendering window.',
            imgsrc: 'images/extras/clone2d.png',
        },

        {
            pyname: 'inset',
            categ: 'other',
            short: 'Inset Rendering',
            long: 'Render meshes and other custom objects into inset frames, which can optionally be dragged.',
            imgsrc: 'images/extras/inset.png',
        },

        {
            pyname: 'flag_labels1',
            categ: 'other',
            short: 'Flag Labels',
            long: 'Add a flag-style label and/or a flagpole indicator that can follow the camera.',
            imgsrc: 'images/extras/flag_labels.png',
        },

        {
            pyname: 'flag_labels2',
            categ: 'other',
            short: 'Flag Indicators',
            long: 'Add a flagpost-style indicator that can follow the camera.',
            imgsrc: 'images/extras/flag_labels2.png',
        },

        {
            pyname: 'qt_window2',
            categ: 'other',
            short: 'Qt Embed',
            long: 'A minimal example of how to embed a rendering window into a ' + insertLink('Qt', 'www.qt.io/') + ' application.',
            imgsrc: 'images/extras/qt_window2.png',
        },

        {
            pyname: 'spherical_harmonics1',
            categ: 'other',
            short: 'Spherical Harmonics',
            long: 'Expand and reconstruct any surface (here a simple box) into ' + insertLink('spherical harmonics', 'en.wikipedia.org/wiki/Spherical_harmonics') + ' with ' + insertLink('SHTOOLS', 'shtools.oca.eu/shtools/public/index.html'),
            imgsrc: 'images/extras/spherical_harmonics1.png',
        },

        {
            pyname: 'ellipt_fourier_desc',
            categ: 'other',
            short: 'Fourier Descriptors',
            long: 'Reconstruct a line with ' + insertLink('Elliptic Fourier Descriptors', 'github.com/hbldh/pyefd'),
            imgsrc: 'images/extras/ellipt_fourier_desc.png',
        },

        {
            pyname: 'nevergrad_opt',
            categ: 'other',
            short: 'Nevergrad',
            long: 'Visualize a 2D minimization problem solved by ' + insertLink('nevergrad', 'github.com/facebookresearch/nevergrad'),
            imgsrc: 'images/extras/nevergrad_opt.png',
        },

        {
            pyname: 'iminuit1',
            categ: 'other',
            short: 'iminuit Minimizer',
            long: 'Visualize a 3D minimization problem solved by ' + insertLink('iminuit', 'github.com/scikit-hep/iminuit'),
            imgsrc: 'images/extras/iminuit1.jpg',
        },

        {
            pyname: 'iminuit2',
            categ: 'other',
            short: 'IMinuit Surface Fit',
            long: 'Fit a 3D polynomial surface to noisy data with ' + insertLink('iminuit', 'scikit-hep.org/iminuit') + '.',
            imgsrc: 'images/extras/iminuit2.png',
        },

        {
            pyname: 'meshio_read',
            categ: 'other',
            short: 'meshio',
            long: 'Interface vedo with the ' + insertLink('meshio library', 'github.com/nschloe/meshio'),
            imgsrc: 'images/extras/meshio_read.png',
        },

        {
            pyname: 'pymeshlab1',
            categ: 'other',
            short: 'PyMeshLab',
            long: 'Use vedo with the ' + insertLink('pymeshlab library', 'github.com/cnr-isti-vclab/PyMeshLab'),
            imgsrc: 'images/extras/pymeshlab1.jpg',
        },

        {
            pyname: 'madcad1',
            categ: 'other',
            short: 'pymadcad',
            long: 'Use vedo with the ' + insertLink('madcad library', 'pymadcad.readthedocs.io/en/latest/index.html'),
            imgsrc: 'images/extras/madcad1.png',
        },

        {
            pyname: 'pygeodesic1',
            categ: 'other',
            short: 'pygeodesic',
            long: 'Compute geodesic distances between points on a surface with the ' + insertLink('pygeodesic library', 'github.com/mhogg/pygeodesic'),
            imgsrc: 'images/extras/pygeodesic1.jpg',
        },

        {
            pyname: 'pygmsh_cut',
            categ: 'other',
            short: 'pygmsh',
            long: 'Use vedo with the ' + insertLink('pygmsh library', 'github.com/nschloe/pygmsh'),
            imgsrc: 'images/extras/pygmsh_cut.png',
        },

        {
            pyname: 'tetgen1',
            categ: 'other',
            short: 'tetgenpy',
            long: 'Interface vedo with ' + insertLink('tetgenpy', 'github.com/tataratat/tetgenpy') + ' to create tetrahedral meshes.',
            imgsrc: 'images/extras/tetgen1.png',
        },

        {
            pyname: 'remesh_ACVD',
            categ: 'other',
            short: 'PyACVD',
            long: 'Interface vedo with ' + insertLink('pyvista', 'github.com/pyvista/pyvista') + ' and ' + insertLink('pyacvd', 'github.com/akaszynski/pyacvd') + ' libraries.',
            imgsrc: 'images/extras/remesh_ACVD.png',
        },

        {
            pyname: 'fast_simpl',
            categ: 'other',
            short: 'Fast Decimation',
            long: 'Use the ' + insertLink('fast-simplification', 'github.com/pyvista/fast-simplification') + ' library to decimate a mesh and transfer data defined on the original vertices.',
            imgsrc: 'images/extras/fast_decim.jpg',
        },

        {
            pyname: 'napari1',
            categ: 'other',
            short: 'Napari Viewer',
            long: 'Visualize a vedo mesh in the ' + insertLink('napari', 'napari.org/') + ' image viewer. Check out also the ' + insertLink('napari-vedo plugin', 'github.com/jo-mueller/napari-vedo-bridge') + ' for napari.',
            imgsrc: 'images/extras/napari1.png',
        },

        {
            pyname: 'magic-class1',
            categ: 'other',
            short: 'magic-class Library',
            long: 'Visualize objects using the ' + insertLink('magic-class', 'github.com/hanjinliu/magic-class') + ' library.',
            imgsrc: 'images/extras/magic-class1.png',
        },

        {
            pyname: 'chemistry1',
            categ: 'other',
            short: 'Molecule Viewer',
            long: 'Chemistry rendering and interaction example. Uses chemistry-specific helpers for molecular visualization.',
            imgsrc: 'images/extras/chemistry1.png',
        },

        {
            pyname: 'chemistry2',
            categ: 'other',
            short: 'Molecular Streamlines',
            long: 'Draw molecular streamlines from a Gaussian cube file. Computes streamlines from a volumetric vector field.',
            imgsrc: 'images/extras/chemistry2.png',
        },

        {
            pyname: 'export_numpy',
            categ: 'other',
            short: 'Export NumPy',
            long: 'External-tools interoperability example.',
            imgsrc: 'images/extras/export_numpy.png',
        },

        {
            pyname: 'export_x3d',
            categ: 'other',
            short: 'Export X3D',
            long: 'Embed a 3D scene in a webpage with ' + insertLink('X3D', 'www.web3d.org/x3d/what-x3d') + '.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'madcad2',
            categ: 'other',
            short: 'Madcad Mesh Exchange',
            long: 'Convert a vedo mesh to ' + insertLink('madcad', 'pymadcad.readthedocs.io/en/latest/index.html') + ' and back again.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'meshlib1',
            categ: 'other',
            short: 'MeshLib Relaxation',
            long: 'Process and relax a surface with ' + insertLink('MeshLib', 'meshlib.io') + '.',
            imgsrc: 'images/placeholders/gears.png',
        },

        // {
        //     pyname: 'morphomatics_riemann',
        //     categ: 'other',
        //     short: 'Morphomatics Riemann',
        //     long: 'Compute the Riemannian mean of Bézier splines on a sphere with ' + insertLink('Morphomatics', 'morphomatics.github.io') + '.',
        //     imgsrc: 'images/extras/morphomatics_riemann.png',
        // },

        {
            pyname: 'morphomatics_tube',
            categ: 'other',
            short: 'Morphomatics Warping',
            long: 'Visualize a tube workflow built with ' + insertLink('Morphomatics', 'morphomatics.github.io') + '.',
            imgsrc: 'images/extras/morphomatics_tube.png',
        },

        {
            pyname: 'nelder-mead',
            categ: 'other',
            short: 'Nelder-Mead',
            long: 'Show the Nelder-Mead minimization algorithm for a 4D function.',
            imgsrc: 'images/extras/nelder-mead.png',
        },

        {
            pyname: 'pymeshlab2',
            categ: 'other',
            short: 'Ball-Pivot Mesh Reco',
            long: 'Reconstruct a surface by ball pivoting with ' + insertLink('PyMeshLab', 'pymeshlab.readthedocs.io/en/latest/') + '.',
            imgsrc: 'images/extras/pymeshlab2.png',
        },

        {
            pyname: 'pysr_regression',
            categ: 'other',
            short: 'PySR Regression',
            long: 'Fit a symbolic regression model to a 2D dataset with ' + insertLink('PySR', 'github.com/MilesCranmer/PySR') + '.',
            imgsrc: 'images/extras/pysr_regression.png',
        },

        {
            pyname: 'qt_cutter',
            categ: 'other',
            short: 'Qt Cutter',
            long: 'Embed an interactive cutter inside a ' + insertLink('Qt', 'www.qt.io/') + ' interface.',
            imgsrc: 'images/extras/qt_cutter.png',
        },

        {
            pyname: 'qt_tabs',
            categ: 'other',
            short: 'Qt Tabs',
            long: 'Organize vedo views inside a tabbed ' + insertLink('Qt', 'www.qt.io/') + ' application.',
            imgsrc: 'images/extras/qt_tabs.png',
        },

        {
            pyname: 'qt_window1',
            categ: 'other',
            short: 'Qt Embedded Plotter',
            long: 'Embed a vedo render window in a ' + insertLink('Qt', 'www.qt.io/') + ' application.',
            imgsrc: 'images/extras/qt_window1.png',
        },

        {
            pyname: 'qt_window3',
            categ: 'other',
            short: 'Qt Dual View',
            long: 'Drive a dual-view vedo layout from a ' + insertLink('Qt', 'www.qt.io/') + ' interface. Uses multiple renderers to compare views in parallel.',
            imgsrc: 'images/placeholders/gears.png',
        },

        {
            pyname: 'quaternions',
            categ: 'other',
            short: 'Quaternions',
            long: 'A quaternion is a compact way to represent a 3D rotation. Uses multiple renderers to compare views in parallel.',
            imgsrc: 'images/extras/quaternions.png',
        },

        {
            pyname: 'remesh_meshfix',
            categ: 'other',
            short: 'Remesh with MeshFix',
            long: 'Repair and remesh a surface with ' + insertLink('MeshFix', 'github.com/MarcoAttene/MeshFix-V2.1') + '. Uses multiple renderers to compare views in parallel.',
            imgsrc: 'images/extras/remesh_meshfix.png',
        },

        {
            pyname: 'tensor_grid1',
            categ: 'other',
            short: 'Tensor Field Grid',
            long: 'External-tools interoperability example.',
            imgsrc: 'images/extras/tensor_grid1.png',
        },

        {
            pyname: 'trimesh_nearest',
            categ: 'other',
            short: 'Trimesh: Nearest Points',
            long: 'Find nearest points on a surface with the ' + insertLink('trimesh', 'trimesh.org') + ' library.',
            imgsrc: 'images/extras/trimesh_nearest.png',
        },

        {
            pyname: 'trimesh_ray',
            categ: 'other',
            short: 'Trimesh: Ray Casting',
            long: 'Cast rays against a mesh with the ' + insertLink('trimesh', 'trimesh.org') + ' library.',
            imgsrc: 'images/extras/trimesh_ray.png',
        },

        {
            pyname: 'trimesh_section',
            categ: 'other',
            short: 'Trimesh: Mesh Section',
            long: 'Compute mesh cross-sections with the ' + insertLink('trimesh', 'trimesh.org') + ' library.',
            imgsrc: 'images/extras/trimesh_section.png',
        },

        {
            pyname: 'trimesh_shortest',
            categ: 'other',
            short: 'Trimesh: Shortest Path',
            long: 'Compute shortest paths on a surface with the ' + insertLink('trimesh', 'trimesh.org') + ' library.',
            imgsrc: 'images/extras/trimesh_shortest.png',
        },

        {
            pyname: 'icon',
            categ: 'other',
            short: 'Icons and Logos',
            long: 'Create an orientation icon and place it in one of the four corners of the same renderer.',
            imgsrc: 'images/extras/icon.png',
        }

    ];
