from __future__ import annotations
"""Video generation utilities from rendered scenes."""

import os
import shutil
from tempfile import TemporaryDirectory

import vedo
from vedo import colors, utils

from .scene import ask, screenshot

__docformat__ = "google"

class Video:
    """
    Generate a video from a rendering window.
    """

    def __init__(self, name="movie.mp4", duration=None, fps=24, scale=1, backend="imageio"):
        """
        Class to generate a video from the specified rendering window.
        Program `ffmpeg` is used to create video from each generated frame.

        Arguments:
            name : (str | os.PathLike)
                name of the output file.
            duration : (float)
                set the total `duration` of the video and recalculates `fps` accordingly.
            fps : (int)
                set the number of frames per second.
            scale : (int)
                set the image magnification as an integer multiplicative factor.
            backend : (str)
                the backend engine to be used `['imageio', 'ffmpeg', 'cv']`

        Examples:
            - [make_video.py](https://github.com/marcomusy/vedo/tree/master/examples/other/make_video.py)

            ![](https://user-images.githubusercontent.com/32848391/50739007-2bfc2b80-11da-11e9-97e6-620a3541a6fa.jpg)
        """
        self.name = str(name)
        self.duration = duration
        self.backend = backend
        self.fps = float(fps)
        self.command = "ffmpeg -loglevel panic -y -r"
        self.options = "-b:v 8000k"
        self.scale = scale

        self.frames = []
        self.tmp_dir = TemporaryDirectory()
        self.get_filename = lambda x: os.path.join(self.tmp_dir.name, x)
        colors.printc(":video:  Video file", self.name, "is open... ", c="m", end="")

    def add_frame(self) -> Video:
        """Add frame to current video."""
        fr = self.get_filename(str(len(self.frames)) + ".png")
        screenshot(fr, scale=self.scale)
        self.frames.append(fr)
        return self

    def pause(self, pause=0) -> Video:
        """Insert a `pause`, in seconds."""
        fr = self.frames[-1]
        n = int(self.fps * pause)
        for _ in range(n):
            fr2 = self.get_filename(str(len(self.frames)) + ".png")
            self.frames.append(fr2)
            shutil.copyfile(fr, fr2)
        return self

    def action(self, elevation=(0, 80), azimuth=(0, 359), cameras=(), resetcam=False) -> Video:
        """
        Automatic shooting of a static scene by specifying rotation and elevation ranges.

        Arguments:
            elevation : list
                initial and final elevation angles
            azimuth_range : list
                initial and final azimuth angles
            cameras : list
                list of cameras to go through, each camera can be dictionary or a vtkCamera
        """
        if not self.duration:
            self.duration = 5

        plt = vedo.current_plotter()
        if not plt:
            vedo.logger.error("No vedo plotter found, cannot make video.")
            return self
        n = int(self.fps * self.duration)

        cams = []
        for cm in cameras:
            cams.append(utils.camera_from_dict(cm))
        nc = len(cams)

        plt.show(resetcam=resetcam, interactive=False)

        if nc:
            for i in range(n):
                plt.move_camera(cams, i / (n-1))
                plt.render()
                self.add_frame()

        else:  ########################################

            for i in range(n):
                plt.camera.Elevation((elevation[1] - elevation[0]) / n)
                plt.camera.Azimuth((azimuth[1] - azimuth[0]) / n)
                plt.render()
                self.add_frame()

        return self

    def close(self) -> None:
        """
        Render the video and write it to file.
        """
        if self.duration:
            self.fps = int(len(self.frames) / float(self.duration) + 0.5)
            colors.printc("recalculated fps:", self.fps, c="m", end="")
        else:
            self.fps = int(self.fps)

        ########################################
        if self.backend == "ffmpeg":
            out = os.system(
                self.command
                + " "
                + str(self.fps)
                + " -i "
                + f"'{self.tmp_dir.name}'"
                + os.sep
                + "%01d.png "
                + self.options
                + " "
                + f"'{self.name}'"
            )
            if out:
                vedo.logger.error(f":noentry: backend {self.backend} returning error: {out}")
            else:
                colors.printc(f":save: saved to {self.name}", c="m")

        ########################################
        elif "cv" in self.backend:
            try:
                import cv2  # type: ignore
            except ImportError:
                vedo.logger.error("opencv is not installed")
                return

            cap = cv2.VideoCapture(os.path.join(self.tmp_dir.name, "%1d.png"))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            plt = vedo.current_plotter()
            if plt:
                w, h = plt.window.GetSize()
                writer = cv2.VideoWriter(self.name, fourcc, self.fps, (w, h), True)
            else:
                vedo.logger.error("No vedo plotter found, cannot make video.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            cap.release()
            writer.release()

        ########################################
        elif "imageio" in self.backend:
            try:
                import imageio
            except ImportError:
                vedo.logger.error("Please install imageio with:\n pip install imageio[ffmpeg]")
                return

            if self.name.endswith(".mp4"):
                writer = imageio.get_writer(self.name, fps=self.fps)
            elif self.name.endswith(".gif"):
                writer = imageio.get_writer(self.name, mode="I", duration=1 / self.fps)
            elif self.name.endswith(".webm"):
                writer = imageio.get_writer(self.name, format="webm", fps=self.fps)
            else:
                vedo.logger.error(f"Unknown format of {self.name}.")
                return

            for f in utils.humansort(self.frames):
                image = imageio.v3.imread(f)
                try:
                    writer.append_data(image)
                except TypeError:
                    vedo.logger.error(f"Could not append data to video {self.name}")
                    vedo.logger.error("Please install imageio with: pip install imageio[ffmpeg]")
                    break
            try:
                writer.close()
                colors.printc(f"... saved as {self.name}", c="m")
            except:
                colors.printc(f":noentry: Could not save video {self.name}", c="r")

        # finalize cleanup
        self.tmp_dir.cleanup()

    def split_frames(self, output_dir="video_frames", prefix="frame_", file_format="png") -> None:
        """Split an existing video file into frames."""
        try:
            import imageio
        except ImportError:
            vedo.logger.error("\nPlease install imageio with:\n pip install imageio")
            return

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a reader object to read the video
        reader = imageio.get_reader(self.name)

        # Loop through each frame of the video and save it as image
        print()
        for i, frame in utils.progressbar(
            enumerate(reader), title=f"writing {file_format} frames", c="m", width=20
        ):
            output_file = os.path.join(output_dir, f"{prefix}{str(i).zfill(5)}.{file_format}")
            imageio.imwrite(output_file, frame, format=file_format)

__all__ = ["Video"]
