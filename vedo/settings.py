#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

__docformat__ = "google"


class Settings:
    """
    General settings to modify the global behavior

    Usage Example:
        ```python
        from vedo import settings, Cube
        settings.use_parallel_projection = True
        Cube().color('g').show().close()
        ```

    List of available properties:

    ```python
    # Set a default for the font to be used for axes, comments etc.
    default_font = 'Normografo' # check font options in shapes.Text

    # Palette number when using an integer to choose a color
    palette = 0

    screenshot_transparent_background = False
    screeshot_large_image = False # Sometimes setting this to True gives better results

    # [DISABLED] Allow to continuously interact with scene during interactive() execution
    allow_interaction = True

    # Enable tracking pipeline functionality: 
    #  allows to show a graph with the pipeline of action which let to a final object
    #  this is achieved by calling "myobj.pipeline.show()" (a new window will pop up)
    self.enable_pipeline = True

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

    # By default, the depth buffer is reset for each renderer.
    #  If True, use the existing depth buffer
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

    # Automatically close the Plotter instance after show() in jupyter sessions
    # setting it to False will keep the current Plotter instance active
    backend_autoclose = True

    # k3d settings for jupyter notebooks
    k3d_menu_visibility = True
    k3d_plot_height = 512
    k3d_antialias   = True
    k3d_lighting    = 1.5
    k3d_camera_autofit = True
    k3d_grid_autofit= True
    k3d_axes_color  = "gray4"
    k3d_axes_helper = 1.0     # size of the small triad of axes on the bottom right
    k3d_point_shader= "mesh"  # others are '3d', '3dSpecular', 'dot', 'flat'
    k3d_line_shader = "thick" # others are 'flat', 'mesh'
    ```
    """

    # Restrict the attributes so accidental typos will generate an AttributeError exception
    __slots__ = [
        "_level",
        "default_font",
        "default_backend",
        "palette",
        "remember_last_figure_format",
        "screenshot_transparent_background",
        "screeshot_large_image",
        "allow_interaction",
        "hack_call_screen_size",
        "enable_default_mouse_callbacks",
        "enable_default_keyboard_callbacks",
        "enable_pipeline",
        "immediate_rendering",
        "renderer_frame_color",
        "renderer_frame_alpha",
        "renderer_frame_width",
        "renderer_frame_padding",
        "render_lines_as_tubes",
        "hidden_line_removal",
        "point_smoothing",
        "line_smoothing",
        "polygon_smoothing",
        "visible_grid_edges",
        "light_follows_camera",
        "two_sided_lighting",
        "use_depth_peeling",
        "multi_samples",
        "alpha_bit_planes",
        "max_number_of_peels",
        "occlusion_ratio",
        "use_fxaa",
        "preserve_depth_buffer",
        "use_polygon_offset",
        "polygon_offset_factor",
        "polygon_offset_units",
        "interpolate_scalars_before_mapping",
        "use_parallel_projection",
        "window_splitting_position",
        "tiff_orientation_type",
        "annotated_cube_color",
        "annotated_cube_text_color",
        "annotated_cube_text_scale",
        "annotated_cube_texts",
        "enable_print_color",
        "backend_autoclose",
        "k3d_menu_visibility",
        "k3d_plot_height",
        "k3d_antialias",
        "k3d_lighting",
        "k3d_camera_autofit",
        "k3d_grid_autofit",
        "k3d_axes_color",
        "k3d_axes_helper",
        "k3d_point_shader",
        "k3d_line_shader",
        "font_parameters",
    ]

    def __init__(self, level=0):

        self._level = level

        # Default font
        self.default_font = "Normografo"

        # Default backend engine in jupyter notebooks
        self.default_backend = "vtk"

        # enable tracking pipeline functionality
        self.enable_pipeline = True

        if any(["SPYDER" in name for name in os.environ]):
            self.default_backend = "vtk"
        else:
            try:
                get_ipython()
                self.default_backend = "2d"
            except NameError:
                pass

        # Palette number when using an integer to choose a color
        self.palette = 0

        self.remember_last_figure_format = False

        self.screenshot_transparent_background = False
        self.screeshot_large_image = False

        # [DISABLED] Allow to continuously interact with scene during interactor.Start() execution
        self.allow_interaction = True

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
        self.use_depth_peeling = False
        self.multi_samples = 8
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
        self.annotated_cube_text_color = None  # use default, otherwise specify a single color
        self.annotated_cube_text_scale = 0.2
        self.annotated_cube_texts = ["right", "left ", "front", "back ", " top ", "bttom"]

        # enable / disable color printing
        self.enable_print_color = True

        ####################################################################################
        # Automatically close the Plotter instance after show() in jupyter sessions,
        #  setting it to False will keep the current Plotter instance active
        self.backend_autoclose = True

        # k3d settings for jupyter notebooks
        self.k3d_menu_visibility = True
        self.k3d_plot_height = 512
        self.k3d_antialias  = True
        self.k3d_lighting   = 1.5
        self.k3d_camera_autofit = True
        self.k3d_grid_autofit= True
        self.k3d_axes_color  = "k4"
        self.k3d_axes_helper = 1.0     # size of the small triad of axes on the bottom right
        self.k3d_point_shader= "mesh"  # others are '3d', '3dSpecular', 'dot', 'flat'
        self.k3d_line_shader = "thick" # others are 'flat', 'mesh'

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
                dotsep="~×",
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
                dotsep="×",
                islocal=True,
            ),
            Comae=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=True,
            ),
            ComicMono=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="x",
                islocal=False,
            ),
            Edo=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~x ",
                islocal=False,
            ),
            FiraMonoMedium=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            FiraMonoBold=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            Glasgo=dict(
                mono=True,
                fscale=0.75,
                lspacing=0.1,
                hspacing=1,
                dotsep="~×",
                islocal=True,
            ),
            Kanopus=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.15,
                hspacing=0.75,
                dotsep="~×",
                islocal=True,
            ),
            LogoType=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~×",
                islocal=False,
            ),
            Quikhand=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.6,
                lspacing=0.15,
                dotsep="~~×~",
                islocal=True,
            ),
            SmartCouric=dict(
                mono=True,
                fscale=0.8,
                hspacing=1.05,
                lspacing=0.1,
                dotsep="×",
                islocal=True,
            ),
            Spears=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.5,
                lspacing=0.2,
                dotsep="~×",
                islocal=False,
            ),
            Theemim=dict(
                mono=False,
                fscale=0.825,
                hspacing=0.52,
                lspacing=0.3,
                dotsep="~~×",
                islocal=True,
            ),
            VictorMono=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=True,
            ),
            Justino1=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            Justino2=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            Justino3=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            Calibri=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~×",
                islocal=False,
            ),
            Capsmall=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.75,
                lspacing=0.15,
                dotsep="~×",
                islocal=False,
            ),
            Cartoons123=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.75,
                lspacing=0.15,
                dotsep="x",
                islocal=False,
            ),
            Vega=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.75,
                lspacing=0.15,
                dotsep="×",
                islocal=False,
            ),
            Meson=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.9,
                lspacing=0.225,
                dotsep="~×",
                islocal=False,
            ),
            Komika=dict(
                mono=False,
                fscale=0.7,
                hspacing=0.75,
                lspacing=0.225,
                dotsep="~×",
                islocal=False,
            ),
            Vogue=dict(
                mono=False,
                fscale=0.7,
                hspacing=0.75,
                lspacing=0.225,
                dotsep="~x",
                islocal=False,
            ),
            Brachium=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="x",
                islocal=False,
            ),
            Dalim=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            Miro=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            Ubuntu=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            Mizar=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=0.75,
                dotsep="~×",
                islocal=False,
            ),
            LiberationSans=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            DejavuSansMono=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="~×",
                islocal=False,
            ),
            SunflowerHighway=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            Swansea=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            Housekeeper=dict(  # support chinese glyphs
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~×",
                islocal=False,
            ),
            Wananti=dict(  # support chinese glyphs
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~x",
                islocal=False,
            ),
            AnimeAce=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~x",
                islocal=False,
            ),
            AnimeAceBold=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~x",
                islocal=False,
            ),
        )

    ####################################################################################
    def reset(self):
        """Reset all settings to their default status."""
        self.__init__()

    def print(self):
        """Print function."""
        print(" " + "-" * 80)
        s = Settings.__doc__.replace("   ", "")
        s = s.replace(".. code-block:: python\n", "")
        try:
            from pygments import highlight
            from pygments.lexers import Python3Lexer
            from pygments.formatters import Terminal256Formatter

            s = highlight(s, Python3Lexer(), Terminal256Formatter(style="zenburn"))
            print(s, end="")

        except ModuleNotFoundError:
            print("\x1b[33;1m" + s + "\x1b[0m")

    def _warn(self, key):
        if self._level == 0:
            print(f'\x1b[1m\x1b[33;20m Warning! Please use "settings.{key}" instead!\x1b[0m')

    def __getitem__(self, key):
        """Make the class work like a dictionary too"""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Make the class work like a dictionary too"""
        setattr(self, key, value)

    ####################################################################################
    # Deprecations
    ####################################################################################
    @property
    def defaultFont(self):
        self._warn("default_font")
        return self.default_font
    @defaultFont.setter
    def defaultFont(self, value):
        self._warn("default_font")
        self.default_font = value
    ##################################
    @property
    def screenshotTransparentBackground(self):
        self._warn("screenshot_transparent_background")
        return self.NAME_SNAKE
    @screenshotTransparentBackground.setter
    def screenshotTransparentBackground(self, value):
        self._warn("screenshot_transparent_background")
        self.screenshot_transparent_background = value
    ##################################
    @property
    def screeshotLargeImage(self):
        self._warn("screeshot_large_image")
        return self.screeshot_large_image
    @screeshotLargeImage.setter
    def screeshotLargeImage(self, value):
        self._warn("screeshot_large_image")
        self.screeshot_large_image = value
    ##################################
    @property
    def allowInteraction(self):
        self._warn("allow_interaction")
        return self.allow_interaction
    @allowInteraction.setter
    def allowInteraction(self, value):
        self._warn("allow_interaction")
        self.allow_interaction = value
    ##################################
    @property
    def enableDefaultMouseCallbacks(self):
        self._warn("enable_default_mouse_callbacks")
        return self.enable_default_mouse_callbacks
    @enableDefaultMouseCallbacks.setter
    def enableDefaultMouseCallbacks(self, value):
        self._warn("enable_default_mouse_callbacks")
        self.enable_default_mouse_callbacks = value
    ##################################
    @property
    def enableDefaultKeyboardCallbacks(self):
        self._warn("enable_default_keyboard_callbacks")
        return self.enable_default_keyboard_callbacks
    @enableDefaultKeyboardCallbacks.setter
    def enableDefaultKeyboardCallbacks(self, value):
        self._warn("enable_default_keyboard_callbacks")
        self.enable_default_keyboard_callbacks = value
    ##################################
    @property
    def useDepthPeeling(self):
        self._warn("use_depth_peeling")
        return self.use_depth_peeling
    @useDepthPeeling.setter
    def useDepthPeeling(self, value):
        self._warn("use_depth_peeling")
        self.use_depth_peeling = value
    ##################################
    @property
    def multiSamples(self):
        self._warn("multi_samples")
        return self.multi_samples
    @multiSamples.setter
    def multiSamples(self, value):
        self._warn("multi_samples")
        self.multi_samples = value
    ##################################
    @property
    def maxNumberOfPeels(self):
        self._warn("max_number_of_peels")
        return self.max_number_of_peels
    @maxNumberOfPeels.setter
    def maxNumberOfPeels(self, value):
        self._warn("max_number_of_peels")
        self.max_number_of_peels = value
    ##################################
    @property
    def interpolateScalarsBeforeMapping(self):
        self._warn("interpolate_scalars_before_mapping")
        return self.interpolate_scalars_before_mapping
    @interpolateScalarsBeforeMapping.setter
    def interpolateScalarsBeforeMapping(self, value):
        self._warn("interpolate_scalars_before_mapping")
        self.interpolate_scalars_before_mapping = value
    ##################################
    @property
    def useParallelProjection(self):
        self._warn("use_parallel_projection")
        return self.use_parallel_projection
    @useParallelProjection.setter
    def useParallelProjection(self, value):
        self._warn("use_parallel_projection")
        self.use_parallel_projection = value
    ##################################
    @property
    def tiffOrientationType(self):
        self._warn("tiff_orientation_type")
        return self.tiff_orientation_type
    @tiffOrientationType.setter
    def tiffOrientationType(self, value):
        self._warn("tiff_orientation_type")
        self.tiff_orientation_type = value
    ##################################
    @property
    def enablePrintColor(self):
        self._warn("enable_print_color")
        return self.enable_print_color
    @enablePrintColor.setter
    def enablePrintColor(self, value):
        self._warn("enable_print_color")
        self.enable_print_color = value

