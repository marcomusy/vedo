#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Settings:
    """
    General settings to modify the global behavior

    Usage Example
    -------------
    .. code-block:: python

        from vedo import settings, Cube
        settings.use_parallel_projection = True
        Cube().color('g').show()

    Parameters
    ----------

        # Set a default for the font to be used for axes, comments etc.
        default_font = 'Normografo' # check font options in shapes.Text

        # Palette number when using an integer to choose a color
        palette = 0

        # Scale magnification of the screenshot (must be an integer)
        screeshot_scale = 1
        screenshot_transparent_background = False
        screeshot_large_image = False # Sometimes setting this to True gives better results

        # Allow to continuously interact with scene during interactive() execution
        allow_interaction = False

        # Set up default mouse and keyboard functionalities
        enable_default_mouse_callbacks = True
        enable_default_keyboard_callbacks = True

        # If False, when multiple renderers are present do not render each one for separate
        #  but do it just once at the end (when interactive() is called)
        immediate_rendering = True

        # Show a gray frame margin in multirendering windows
        renderer_frame_color = None
        renderer_frame_alpha = 0.5
        renderer_frame_width = 0.5
        renderer_frame_padding = 0.0001

        # In multirendering mode set the position of the horizontal of vertical splitting [0,1]
        window_splitting_position = None

        # Enable / disable color printing by printc()
        enable_print_color = True

        # Wrap lines in tubes
        render_lines_as_tubes = False

        # Smoothing options
        point_smoothing = False
        line_smoothing = False
        polygon_smoothing = False

        # Remove hidden lines when in wireframe mode
        hidden_line_removal = False

        # Turn on/off the automatic repositioning of lights as the camera moves.
        light_follows_camera = False
        two_sided_lighting = True

        # Turn on/off rendering of translucent material with depth peeling technique.
        use_depth_peeling = False
        alpha_bit_planes  = True   # options only active if useDepthPeeling=True
        multi_samples     = 8      # force to not pick a framebuffer with a multisample buffer
        max_number_of_peels= 4     # maximum number of rendering passes
        occlusion_ratio   = 0.0    # occlusion ratio, 0 = exact image.

        # Turn on/off nvidia FXAA post-process anti-aliasing, if supported.
        use_fxaa = False           # either True or False

        # By default, the depth buffer is reset for each renderer. If True, use the existing depth buffer
        preserve_depth_buffer = False

        # Use a polygon/edges offset to possibly resolve conflicts in rendering
        use_polygon_offset    = False
        polygon_offset_factor = 0.1
        polygon_offset_units  = 0.1

        # Interpolate scalars to render them smoothly
        interpolate_scalars_before_mapping = True

        # Set parallel projection On or Off (place camera to infinity, no perspective effects)
        use_parallel_projection = False

        # Set orientation type when reading TIFF files (volumes):
        # TOPLEFT  1 (row 0 top, col 0 lhs)    TOPRIGHT 2 (row 0 top, col 0 rhs)
        # BOTRIGHT 3 (row 0 bottom, col 0 rhs) BOTLEFT  4 (row 0 bottom, col 0 lhs)
        # LEFTTOP  5 (row 0 lhs, col 0 top)    RIGHTTOP 6 (row 0 rhs, col 0 top)
        # RIGHTBOT 7 (row 0 rhs, col 0 bottom) LEFTBOT  8 (row 0 lhs, col 0 bottom)
        tiff_orientation_type = 1

        # Annotated cube axis type nr. 5 options:
        annotated_cube_color      = (0.75, 0.75, 0.75)
        annotated_cube_text_color = None # use default, otherwise specify a single color
        annotated_cube_text_scale = 0.2
        annotated_cube_texts      = ["right","left ", "front","back ", " top ", "bttom"]

        # k3d settings for jupyter notebooks
        k3d_menu_visibility = True
        k3d_plot_height = 512
        k3d_antialias   = True
        k3d_lighting    = 1.2
        k3d_camera_autofit = True
        k3d_grid_autofit= True
        k3d_axes_helper = True    # size of the small triad of axes on the bottom right
        k3d_point_shader= "mesh"  # others are '3d', '3dSpecular', 'dot', 'flat'
        k3d_line_shader = "thick" # others are 'flat', 'mesh'
    """
    def __init__(self, level=0):

        self.level = level

        self.default_font = "Normografo"

        # Palette number when using an integer to choose a color
        self.palette = 0

        self.remember_last_figure_format = False

        # Scale magnification of the screenshot (must be an integer)
        self.screeshot_scale = 1
        self.screenshot_transparent_background = False
        self.screeshot_large_image = False

        # Allow to continuously interact with scene during interactor.Start() execution
        self.allow_interaction = False

        # BUG in vtk9.0 (if true close works but sometimes vtk crashes, if false doesnt crash but cannot close)
        # see plotter.py line 555
        self.hack_call_screen_size = True

        # Set up default mouse and keyboard functionalities
        self.enable_default_mouse_callbacks = True
        self.enable_default_keyboard_callbacks = True

        # When multiple renderers are present do not render each one for separate.
        # but do it just once at the end (when interactive() is called)
        self.immediate_rendering = True

        # Show a gray frame margin in multirendering windows
        self.renderer_frame_color = None
        self.renderer_frame_alpha = 0.5
        self.renderer_frame_width = 0.5
        self.renderer_frame_padding = 0.0001

        # Wrap lines in tubes
        self.render_lines_as_tubes = False

        # Remove hidden lines when in wireframe mode
        self.hidden_line_removal = False

        # Smoothing options
        self.point_smoothing = False
        self.line_smoothing = False
        self.polygon_smoothing = False

        # For Structured and RectilinearGrid: show internal edges not only outline
        self.visible_grid_edges = False

        # Turn on/off the automatic repositioning of lights as the camera moves.
        self.light_follows_camera = False
        self.two_sided_lighting = True

        # Turn on/off rendering of translucent material with depth peeling technique.
        # print("vtk_version sys_platform", vtk_version, sys_platform)
        self.use_depth_peeling = False
        self.multi_samples = 8
        # if vtk_version[0] >= 9: # moved to __init__
        #    if "Windows" in sys_platform:
        #        useDepthPeeling = True
        # only relevant if depthpeeling is on
        self.alpha_bit_planes = 1
        self.max_number_of_peels = 4
        self.occlusion_ratio = 0.1

        # Turn on/off nvidia FXAA anti-aliasing, if supported.
        self.use_fxaa = False  # either True or False

        # By default, the depth buffer is reset for each renderer. If true, use the existing depth buffer
        self.preserve_depth_buffer = False

        # Use a polygon/edges offset to possibly resolve conflicts in rendering
        self.use_polygon_offset = True
        self.polygon_offset_factor = 0.1
        self.polygon_offset_units  = 0.1

        # Interpolate scalars to render them smoothly
        self.interpolate_scalars_before_mapping = True

        # Set parallel projection On or Off (place camera to infinity, no perspective effects)
        self.use_parallel_projection = False

        # In multirendering mode set the position of the horizontal of vertical splitting [0,1]
        self.window_splitting_position = None

        # Set orientation type when reading TIFF files (volumes):
        # TOPLEFT  1 (row 0 top, col 0 lhs)    TOPRIGHT 2 (row 0 top, col 0 rhs)
        # BOTRIGHT 3 (row 0 bottom, col 0 rhs) BOTLEFT  4 (row 0 bottom, col 0 lhs)
        # LEFTTOP  5 (row 0 lhs, col 0 top)    RIGHTTOP 6 (row 0 rhs, col 0 top)
        # RIGHTBOT 7 (row 0 rhs, col 0 bottom) LEFTBOT  8 (row 0 lhs, col 0 bottom)
        self.tiff_orientation_type = 1

        # AnnotatedCube axis (type 5) customization:
        self.annotated_cube_color = (0.75, 0.75, 0.75)
        self.annotated_cube_text_color = None # use default, otherwise specify a single color
        self.annotated_cube_text_scale = 0.2
        self.annotated_cube_texts = ["right","left ", "front","back ", " top ", "bttom"]

        # enable / disable color printing
        self.enable_print_color = True

        ####################################################################################
        # k3d settings for jupyter notebooks
        self.k3d_menu_visibility = True
        self.k3d_plot_height = 512
        self.k3d_antialias  = True
        self.k3d_lighting   = 1.2
        self.k3d_camera_autofit = True
        self.k3d_grid_autofit= True
        self.k3d_axes_helper = True    # size of the small triad of axes on the bottom right
        self.k3d_point_shader= "mesh"  # others are '3d', '3dSpecular', 'dot', 'flat'
        self.k3d_line_shader = "thick" # others are 'flat', 'mesh'

        ####################################################################################
        self.flag_delay = 150  # values will be superseded
        self.flag_font = "Courier"
        self.flag_font_size = 18
        self.flag_justification = 0
        self.flag_angle = 0
        self.flag_bold = False
        self.flag_italic = False
        self.flag_shadow = False
        self.flag_color = "k"
        self.flag_background_color = "w"

        ####################################################################################
        ####################################################################################
        # mono       # means that all letters occupy the same space slot horizontally
        # hspacing   # an horizontal stretching factor (affects both letters and words)
        # lspacing   # horizontal spacing inbetween letters (not words)
        # islocal    # is locally stored in /fonts, otherwise it's on vedo.embl.es/fonts
        #
        self.font_parameters = dict(
            Normografo=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~·",
                islocal=True,
            ),
            Bongas=dict(
                mono=False,
                fscale=0.875,
                hspacing=0.52,
                lspacing=0.25,
                dotsep="·",
                islocal=True,
            ),
            Calco=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="·",
                islocal=True,
            ),
            Comae=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~·",
                islocal=True,
            ),
            ComicMono=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="·",
                islocal=False,
            ),
            Glasgo=dict(
                mono=True,
                fscale=0.75,
                lspacing=0.1,
                hspacing=1,
                dotsep="·",
                islocal=True,
            ),
            Kanopus=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.15,
                hspacing=0.75,
                dotsep="~·",
                islocal=True,
            ),
            LogoType=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="·~~",
                islocal=False,
            ),
            STFANGSO=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~·",
                islocal=False,
            ),
            Quikhand=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.6,
                lspacing=0.15,
                dotsep="~~·~",
                islocal=True,
            ),
            SmartCouric=dict(
                mono=True,
                fscale=0.8,
                hspacing=1.05,
                lspacing=0.1,
                dotsep="·",
                islocal=True,
            ),
            Spears=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.5,
                lspacing=0.2,
                dotsep="·",
                islocal=False,
            ),
            Theemim=dict(
                mono=False,
                fscale=0.825,
                hspacing=0.52,
                lspacing=0.3,
                dotsep="~·",
                islocal=True,
            ),
            VictorMono=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="·",
                islocal=True,
            ),
            Justino1=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="·",
                islocal=False,
            ),
            Justino2=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="·",
                islocal=False,
            ),
            Justino3=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="·",
                islocal=False,
            ),
            Capsmall=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.75,
                lspacing=0.15,
                dotsep="·",
                islocal=False,
            ),
            Cartoons123=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.75,
                lspacing=0.15,
                dotsep="·",
                islocal=False,
            ),
            Vega=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.75,
                lspacing=0.15,
                dotsep="·",
                islocal=False,
            ),
            Meson=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.9,
                lspacing=0.225,
                dotsep="~^.~ ",
                islocal=False,
            ),
            Komika=dict(
                mono=False,
                fscale=0.7,
                hspacing=0.75,
                lspacing=0.225,
                dotsep="~^.~ ",
                islocal=False,
            ),
            Vogue=dict(
                mono=False,
                fscale=0.7,
                hspacing=0.75,
                lspacing=0.225,
                dotsep="~^.~ ",
                islocal=False,
            ),
            Brachium=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="·",
                islocal=False,
            ),
            Dalim=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~·",
                islocal=False,
            ),
            Miro=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~·",
                islocal=False,
            ),
            Ubuntu=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~·",
                islocal=False,
            ),
        )

    ####################################################################################
    def warn(self, key):
        if self.level == 0:
            print(f'\x1b[1m\x1b[33;20m Warning! Please use "settings.{key}" instead!\x1b[0m')


    ####################################################################################
    # Deprecations
    ####################################################################################
    @property
    def defaultFont(self):
        self.warn("default_font")
        return self.default_font
    @defaultFont.setter
    def defaultFont(self, value):
        self.warn("default_font")
        self.default_font = value
    ##################################
    @property
    def rememberLastFigureFormat(self):
        self.warn("remember_last_figure_format")
        return self.remember_last_figure_format
    @rememberLastFigureFormat.setter
    def rememberLastFigureFormat(self, value):
        self.warn("remember_last_figure_format")
        self.remember_last_figure_format = value
    ##################################
    @property
    def screeshotScale(self):
        self.warn("screeshot_scale")
        return self.screeshot_scale
    @screeshotScale.setter
    def screeshotScale(self, value):
        self.warn("screeshot_scale")
        self.screeshot_scale = value
    ##################################
    @property
    def screenshotTransparentBackground(self):
        self.warn("screenshot_transparent_background")
        return self.NAME_SNAKE
    @screenshotTransparentBackground.setter
    def screenshotTransparentBackground(self, value):
        self.warn("screenshot_transparent_background")
        self.screenshot_transparent_background = value
    ##################################
    @property
    def screeshotLargeImage(self):
        self.warn("screeshot_large_image")
        return self.screeshot_large_image
    @screeshotLargeImage.setter
    def screeshotLargeImage(self, value):
        self.warn("screeshot_large_image")
        self.screeshot_large_image = value
    ##################################
    @property
    def allowInteraction(self):
        self.warn("allow_interaction")
        return self.allow_interaction
    @allowInteraction.setter
    def allowInteraction(self, value):
        self.warn("allow_interaction")
        self.allow_interaction = value
    ##################################
    @property
    def hackCallScreenSize(self):
        self.warn("hack_call_screen_size")
        return self.hack_call_screen_size
    @hackCallScreenSize.setter
    def hackCallScreenSize(self, value):
        self.warn("hack_call_screen_size")
        self.hack_call_screen_size = value
    ##################################
    @property
    def enableDefaultMouseCallbacks(self):
        self.warn("enable_default_mouse_callbacks")
        return self.enable_default_mouse_callbacks
    @enableDefaultMouseCallbacks.setter
    def enableDefaultMouseCallbacks(self, value):
        self.warn("enable_default_mouse_callbacks")
        self.enable_default_mouse_callbacks = value
    ##################################
    @property
    def enableDefaultKeyboardCallbacks(self):
        self.warn("enable_default_keyboard_callbacks")
        return self.enable_default_keyboard_callbacks
    @enableDefaultKeyboardCallbacks.setter
    def enableDefaultKeyboardCallbacks(self, value):
        self.warn("enable_default_keyboard_callbacks")
        self.enable_default_keyboard_callbacks = value
    ##################################
    @property
    def immediateRendering(self):
        self.warn("immediate_rendering")
        return self.immediate_rendering
    @immediateRendering.setter
    def immediateRendering(self, value):
        self.warn("immediate_rendering")
        self.immediate_rendering = value
    ##################################
    @property
    def rendererFrameColor(self):
        self.warn("renderer_frame_color")
        return self.renderer_frame_color
    @rendererFrameColor.setter
    def rendererFrameColor(self, value):
        self.warn("renderer_frame_color")
        self.renderer_frame_color = value
    ##################################
    @property
    def rendererFrameAlpha(self):
        self.warn("renderer_frame_alpha")
        return self.renderer_frame_alpha
    @rendererFrameAlpha.setter
    def rendererFrameAlpha(self, value):
        self.warn("renderer_frame_alpha")
        self.renderer_frame_alpha = value
    ##################################
    @property
    def rendererFrameWidth(self):
        self.warn("renderer_frame_width")
        return self.renderer_frame_width
    @rendererFrameWidth.setter
    def rendererFrameWidth(self, value):
        self.warn("renderer_frame_width")
        self.renderer_frame_width = value
    ##################################
    @property
    def rendererFramePadding(self):
        self.warn("renderer_frame_padding")
        return self.renderer_frame_padding
    @rendererFramePadding.setter
    def rendererFramePadding(self, value):
        self.warn("renderer_frame_padding")
        self.renderer_frame_padding = value
    ##################################
    @property
    def renderLinesAsTubes(self):
        self.warn("render_lines_as_tubes")
        return self.render_lines_as_tubes
    @renderLinesAsTubes.setter
    def renderLinesAsTubes(self, value):
        self.warn("render_lines_as_tubes")
        self.render_lines_as_tubes = value
    ##################################
    @property
    def hiddenLineRemoval(self):
        self.warn("hidden_line_removal")
        return self.hidden_line_removal
    @hiddenLineRemoval.setter
    def hiddenLineRemoval(self, value):
        self.warn("hidden_line_removal")
        self.hidden_line_removal = value
    ##################################
    @property
    def pointSmoothing(self):
        self.warn("point_smoothing")
        return self.point_smoothing
    @pointSmoothing.setter
    def pointSmoothing(self, value):
        self.warn("point_smoothing")
        self.point_smoothing = value
    ##################################
    @property
    def lineSmoothing(self):
        self.warn("line_smoothing")
        return self.line_smoothing
    @lineSmoothing.setter
    def lineSmoothing(self, value):
        self.warn("line_smoothing")
        self.line_smoothing = value
    ##################################
    @property
    def polygonSmoothing(self):
        self.warn("polygon_smoothing")
        return self.polygon_smoothing
    @polygonSmoothing.setter
    def polygonSmoothing(self, value):
        self.warn("polygon_smoothing")
        self.polygon_smoothing = value
    ##################################
    @property
    def visibleGridEdges(self):
        self.warn("visible_grid_edges")
        return self.visible_grid_edges
    @visibleGridEdges.setter
    def visibleGridEdges(self, value):
        self.warn("visible_grid_edges")
        self.visible_grid_edges = value
    ##################################
    @property
    def lightFollowsCamera(self):
        self.warn("light_follows_camera")
        return self.light_follows_camera
    @lightFollowsCamera.setter
    def lightFollowsCamera(self, value):
        self.warn("light_follows_camera")
        self.light_follows_camera = value
    ##################################
    @property
    def twoSidedLighting(self):
        self.warn("two_sided_lighting")
        return self.two_sided_lighting
    @twoSidedLighting.setter
    def twoSidedLighting(self, value):
        self.warn("two_sided_lighting")
        self.two_sided_lighting = value
    ##################################
    @property
    def useDepthPeeling(self):
        self.warn("use_depth_peeling")
        return self.use_depth_peeling
    @useDepthPeeling.setter
    def useDepthPeeling(self, value):
        self.warn("use_depth_peeling")
        self.use_depth_peeling = value
    ##################################
    @property
    def multiSamples(self):
        self.warn("multi_samples")
        return self.multi_samples
    @multiSamples.setter
    def multiSamples(self, value):
        self.warn("multi_samples")
        self.multi_samples = value
    ##################################
    @property
    def alphaBitPlanes(self):
        self.warn("alpha_bit_planes")
        return self.alpha_bit_planes
    @alphaBitPlanes.setter
    def alphaBitPlanes(self, value):
        self.warn("alpha_bit_planes")
        self.alpha_bit_planes = value
    ##################################
    @property
    def maxNumberOfPeels(self):
        self.warn("max_number_of_peels")
        return self.max_number_of_peels
    @maxNumberOfPeels.setter
    def maxNumberOfPeels(self, value):
        self.warn("max_number_of_peels")
        self.max_number_of_peels = value
    ##################################
    @property
    def occlusionRatio(self):
        self.warn("occlusion_ratio")
        return self.occlusion_ratio
    @occlusionRatio.setter
    def occlusionRatio(self, value):
        self.warn("occlusion_ratio")
        self.occlusion_ratio = value
    ##################################
    @property
    def useFXAA(self):
        self.warn("use_fxaa")
        return self.use_fxaa
    @useFXAA.setter
    def useFXAA(self, value):
        self.warn("use_fxaa")
        self.use_fxaa = value
    ##################################
    @property
    def preserveDepthBuffer(self):
        self.warn("preserve_depth_buffer")
        return self.preserve_depth_buffer
    @preserveDepthBuffer.setter
    def preserveDepthBuffer(self, value):
        self.warn("preserve_depth_buffer")
        self.preserve_depth_buffer = value
    ##################################
    @property
    def usePolygonOffset(self):
        self.warn("use_polygon_offset")
        return self.use_polygon_offset
    @usePolygonOffset.setter
    def usePolygonOffset(self, value):
        self.warn("use_polygon_offset")
        self.use_polygon_offset = value
    ##################################
    @property
    def polygonOffsetFactor(self):
        self.warn("polygon_offset_factor")
        return self.polygon_offset_factor
    @polygonOffsetFactor.setter
    def polygonOffsetFactor(self, value):
        self.warn("polygon_offset_factor")
        self.polygon_offset_factor = value
    ##################################
    @property
    def polygonOffsetUnits(self):
        self.warn("polygon_offset_units")
        return self.polygon_offset_units
    @polygonOffsetUnits.setter
    def polygonOffsetUnits(self, value):
        self.warn("polygon_offset_units")
        self.polygon_offset_units = value
    ##################################
    @property
    def interpolateScalarsBeforeMapping(self):
        self.warn("interpolate_scalars_before_mapping")
        return self.interpolate_scalars_before_mapping
    @interpolateScalarsBeforeMapping.setter
    def interpolateScalarsBeforeMapping(self, value):
        self.warn("interpolate_scalars_before_mapping")
        self.interpolate_scalars_before_mapping = value
    ##################################
    @property
    def useParallelProjection(self):
        self.warn("use_parallel_projection")
        return self.use_parallel_projection
    @useParallelProjection.setter
    def useParallelProjection(self, value):
        self.warn("use_parallel_projection")
        self.use_parallel_projection = value
    ##################################
    @property
    def windowSplittingPosition(self):
        self.warn("window_splitting_position")
        return self.window_splitting_position
    @windowSplittingPosition.setter
    def windowSplittingPosition(self, value):
        self.warn("window_splitting_position")
        self.window_splitting_position = value
    ##################################
    @property
    def tiffOrientationType(self):
        self.warn("tiff_orientation_type")
        return self.tiff_orientation_type
    @tiffOrientationType.setter
    def tiffOrientationType(self, value):
        self.warn("tiff_orientation_type")
        self.tiff_orientation_type = value
    ##################################
    @property
    def enablePrintColor(self):
        self.warn("enable_print_color")
        return self.enable_print_color
    @enablePrintColor.setter
    def enablePrintColor(self, value):
        self.warn("enable_print_color")
        self.enable_print_color = value

