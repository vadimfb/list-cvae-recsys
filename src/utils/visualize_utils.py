from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np


def set_figure_default_params():
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['figure.figsize'] = (16.0, 8.0)
    plt.rcParams['font.size'] = 15


def scatter_plot(x: Iterable, y: Iterable, x_label: str, y_label: str, legend: str, path:str):

    plt.style.use('seaborn')
    set_figure_default_params()
    plt.plot(np.array(x), np.array(y), 'bo--', label=legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(path)
    plt.close()