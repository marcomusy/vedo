import logging
from typing import Callable

import vedo
from vedo.addons import SliderWidget
from vedo.plotter import Event

logger = logging.getLogger("main." + __name__)

# pylint: disable=too-many-instance-attributes
class PlayerAnimation:
    PLAY_SYMBOL = "\u23F5 Play  "
    PAUSE_SYMBOL = "\u23F8 Pause"
    ONE_BACK_SYMBOL = "\u29CF Step"
    ONE_FORWARD_SYMBOL = "\u29D0 Step"

    def __init__(
        self,
        func: Callable,
        min_val: int = 0,
        max_val: int = 100,
        dt: float = 0.1,
        **kwargs,
    ):
        self._func = func
        self.val = min_val - 1
        self.min_val = min_val
        self.max_val = max_val
        self.dt = dt
        self.is_playing = False
        self.timer_id = None
        self.plotter = vedo.Plotter(**kwargs)
        self.timer_callback = self.plotter.add_callback("timer", self.handle_timer)

        self.play_pause_button = self.plotter.add_button(
            self.play_pause,
            pos=(0.5, 0.13),  # x,y fraction from bottom left corner
            states=[PlayerAnimation.PLAY_SYMBOL, PlayerAnimation.PAUSE_SYMBOL],
            font="Kanopus",
            size=32,
        )
        self.button_oneback = self.plotter.add_button(
            self.onebackward,
            pos=(0.35, 0.13),  # x,y fraction from bottom left corner
            states=[self.ONE_BACK_SYMBOL],
            font="Kanopus",
            size=32,
        )
        self.button_oneforward = self.plotter.add_button(
            self.oneforward,
            pos=(0.65, 0.13),  # x,y fraction from bottom left corner
            states=[self.ONE_FORWARD_SYMBOL],
            font="Kanopus",
            size=32,
        )
        self.slider: SliderWidget = self.plotter.add_slider(
            self.slider_callback,
            self.min_val,
            self.max_val,
            self.min_val,
            pos=5,
        )

    def set_play_pause_button(self, new_status: str) -> None:
        if self.play_pause_button.status() is new_status:
            return
        self.play_pause_button.status(new_status)
        self.plotter.render()

    def pause(self) -> None:
        logger.info("pause")
        self.is_playing = False
        if self.timer_id is not None:
            self.plotter.timer_callback("destroy", self.timer_id)
        self.set_play_pause_button(self.PLAY_SYMBOL)

    def resume(self) -> None:
        logger.info("resume")
        if self.timer_id is not None:
            self.plotter.timer_callback("destroy", self.timer_id)
        self.timer_id = self.plotter.timer_callback("create", dt=round(self.dt * 1000))
        self.is_playing = True
        self.set_play_pause_button(self.PAUSE_SYMBOL)

    def play_pause(self) -> None:
        if not self.is_playing:
            self.resume()
        else:
            self.pause()

    def oneforward(self) -> None:
        self.pause()
        self.set_val(self.val + 1)

    def onebackward(self) -> None:
        self.pause()
        self.set_val(self.val - 1)

    def set_val(self, next_val: int) -> None:
        if next_val == self.val:
            return
        if next_val < self.min_val or next_val > self.max_val:
            self.pause()
            return
        self.val = next_val
        self.slider.value = self.val
        self._func(self.val)
        self.plotter.render()

    def slider_callback(self, widget: SliderWidget, _: str) -> None:
        self.pause()
        self.set_val(int(round(widget.value)))

    def handle_timer(self, _: Event = None) -> None:
        self.set_val(self.val + 1)
