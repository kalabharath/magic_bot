#!/usr/bin/python3
import numpy as np


class ComplexRadar():
    """
    Radarplot code stolen from
    "https://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart"
    and modified accordingly.
    """

    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):

        angles = np.arange(0, 360, 360. / len(variables))

        axes = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True,
                             label="axes{}".format(i))
                for i in range(len(variables))]

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)

            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1]  # hack to invert grid
            ax.set_rgrids(grid)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = data
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = data
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)