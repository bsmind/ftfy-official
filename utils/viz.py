import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

import numpy as np

def merge_image(image_list, is_horizontal=True):
    if is_horizontal:
        return np.hstack(image_list)
    return np.vstack(image_list)

def get_rect(xy, w, h, color):
    return Rectangle(
        xy=xy, width=w, height=h,
        facecolor='none', edgecolor=color, linewidth=1
    )



class ScalarPlot(object):
    def __init__(self, n_plots, n_lines:list, legends:list):
        assert n_plots == len(n_lines), \
            'Unmatched n_lines ({:d}) and n_plots ({:d})'.format(len(n_lines), n_plots)
        assert n_plots == len(legends), \
            'Unmatched n_lines ({:d}) and n_legends ({:d})'.format(len(n_lines), len(legends))

        fig, ax = plt.subplots(1, n_plots)
        if n_plots == 1:
            ax = [ax]

        line_handlers = []
        for idx in range(n_plots):
            _ax = ax[idx]
            _n_lines = n_lines[idx]
            _legends = legends[idx]

            h = []
            for i in range(_n_lines):
                _h, = _ax.plot([], [], label=_legends[i])
                h.append(_h)
            line_handlers.append(h)

            _ax.set_autoscale_on(True)
            _ax.autoscale_view(True, True, True)
            _ax.legend()

        self.fig = fig
        self.ax = ax
        self.line_handlers = line_handlers

        plt.ion()
        plt.show()

    def update(self, idx, values):
        line_handlers = self.line_handlers[idx]
        ax = self.ax[idx]
        if len(values) != len(line_handlers):
            raise ValueError("values must be length of {:d}.".format(len(line_handlers)))

        for line, v in zip(line_handlers, values):
            data = np.append(line.get_ydata(), v)
            line.set_xdata(np.arange(len(data)))
            line.set_ydata(data)

        ax.relim()
        ax.autoscale_view(True, True, True)

        #plt.draw()
        plt.pause(0.001)

    def hold(self):
        plt.ioff()
        plt.show()



