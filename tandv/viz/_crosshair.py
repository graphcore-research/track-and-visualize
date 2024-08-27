# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import time
from typing import Dict, List, Union

import matplotlib
import numpy as np

NoneType = type(None)


class SnappingCrossHair:
    """
    A vertical line cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    Args:
        ax (matplotlib.axes.Axes): This is the axis of the figure for \
            which events are being listened to
        other_ax (List[matplotlib.axes.Axes]): This is the list of other axes


    """

    def __init__(self, ax: matplotlib.axes.Axes,
                 other_ax: List[matplotlib.axes.Axes],
                 lines: Dict[str, matplotlib.lines.Line2D],
                 sensitivity: int = 10):  # type: ignore
        self.ax = ax
        self.other_ax = other_ax
        self.vertical_line = ax.axvline(color="k",
                                        lw=0.8,
                                        ls="--")
        self._creating_background = False
        if not isinstance(other_ax, NoneType):
            self.other_vlines = []
            for oax in self.other_ax:
                self.other_vlines.append(oax.axvline(color="k",
                                                     lw=0.8,
                                                     ls="--"))
        self.lines_data = dict()

        for k, v in lines.items():
            x, y = v.get_data()
            self.lines_data[k] = {
                "x": x,
                "y": y,
            }
        self._last_index = None
        self._last_key = None
        # text location in axes coords
        self.text = ax.text(0.25, 0.75, "", transform=ax.transAxes)
        self.time_out = sensitivity * 1e-6
        self.last_draw_time = time.time()

        # ax.figure.canvas.mpl_connect('draw_event', self.on_draw)

    def get_step_in_crosshairs(self) -> Union[int, None]:
        if not isinstance(self._last_index, NoneType):
            self.ax.figure.canvas.draw()
            return int(self.lines_data[self._last_key]["x"][self._last_index])

    def set_cross_hair_visible(self, visible):
        need_redraw = self.vertical_line.get_visible() != visible
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        if not isinstance(self.other_ax, NoneType):
            others_need_redraw = any(
                [vl.get_visible() != visible for vl in self.other_vlines]
            )
            [vl.set_visible(visible) != visible for vl in self.other_vlines]
            return need_redraw or others_need_redraw
        return need_redraw

    def on_mouse_move(self,
                      event:
                      matplotlib.backend_bases.MouseEvent):  # type: ignore
        ctime = time.time()
        if not (event.inaxes is self.ax):
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
                if not isinstance(self.other_ax, NoneType):
                    [_ax.figure.canvas.draw() for _ax in self.other_ax[:1]]

        elif ctime - self.last_draw_time > self.time_out:
            self.set_cross_hair_visible(True)
            x = event.xdata
            # print(len(self.x) - 1)

            for k, v in self.lines_data.items():
                index = min(np.searchsorted(v["x"], x), len(v["x"]) - 1)
                key = k
            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            self._last_key = key
            x = self.lines_data[key]["x"][index]

            # y = self.y[index]
            # update the line positions
            self.vertical_line.set_xdata([x])
            if not isinstance(self.other_ax, NoneType):
                [vl.set_xdata([x]) for vl in self.other_vlines]

            def _format_text(ind: int) -> str:
                return "\n".join(
                    [f'{k}={v["y"][ind]:1.2f}'
                     for k, v in self.lines_data.items()]
                )

            self.text.set_text(
                f"step={int(x)}\n{_format_text(self._last_index)}")
            self.ax.figure.canvas.draw()
            if not isinstance(self.other_ax, NoneType):
                [_ax.figure.canvas.draw() for _ax in self.other_ax[:1]]
            self.last_draw_time = time.time()
