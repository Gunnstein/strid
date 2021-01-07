# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from .utils import modal_assurance_criterion

__all__ = ["StabilizationDiagram", ]


class ModePicker(object):
    """Class to pick modes (poles) from stabilization diagrams.

    The class alters the marker size of the picked artist and changes the
    edgecolor of a picked artist in a stabilization diagram. The `on_pick` and
    `on_remove` methods are also called after picking or removing a mode from
    the stabilization diagram.


    The `picked` attribute contains a list of picked modes from the stabilization
    diagram for further analysis.

    Use class instance as callback to a `pick_event` in a matplotlib figure.

    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import strid
    >>> picker = strid.stabdiag.ModePicker()
    >>> fig = plt.figure()
    >>> fig.canvas.mpl_connect('pick_event', picker)

    """
    picked_edgecolor = 'k'
    picked_markersize_factor = 2

    def __init__(self):
        self._picked = []

    @property
    def picked(self):
        return sorted(self._picked, key=lambda x: x.f)

    def append(self, v):
        self._picked.append(v)

    def remove(self, v):
        self._picked.remove(v)

    def on_pick(self, event):
        """This method is called when a pole is picked

        Mode associated with the picked data point can
        be accessed with the event.artist.mode property.
        """
        mode = event.artist.mode
        print(f"Picked mode: f={mode.f:.1f} Hz, xi={mode.xi:.2%}, MPC={mode.mpc:.0%}")

    def on_remove(self, event):
        """This method is called when a pole is picked

        Mode associated with the picked data point can
        be accessed with the event.artist.mode property.
        """
        mode = event.artist.mode
        print(f"Removed mode: f={mode.f:.1f} Hz, xi={mode.xi:.2%}, MPC={mode.mpc:.0%}")

    def __call__(self, event):
        artist = event.artist
        ms = artist.get_markersize()
        mode = artist.mode
        try:
            oms = artist._original_markersize
        except AttributeError:
            oms = ms
            artist._original_markersize = ms
        if ms == oms:
            artist.set_markersize(ms*self.picked_markersize_factor)
            artist.set_mec(self.picked_edgecolor)
            self.append(mode)
            self.on_pick(event)
        else:
            artist.set_markersize(ms/self.picked_markersize_factor)
            artist.set_mec(artist.get_mfc())
            self.remove(mode)
            self.on_remove(event)


class TableModePicker(ModePicker):
    def __init__(self, mpl_axes):
        """Display a table of the picked modes on a matplotlib axes

        Arguments
        ---------
        mpl_axes : matplotlib.Axes
            Axes to add the table of picked modes to
        """
        super(TableModePicker, self).__init__()
        self.mpl_axes = mpl_axes

    def on_pick(self, event):
        self.write_table()

    def on_remove(self, event):
        self.write_table()

    def write_table(self):
        self.mpl_axes.clear()
        self.mpl_axes.axis('off')
        if len(self.picked) > 0:
            sorted_modes = self.picked
            cell_text = [
                (f"{mode.f:.2f}", f"{mode.xi:.2%}", f"{mode.mpc:.1%}")
                for j, mode in enumerate(sorted_modes)]
            self.mpl_axes.table(
                cell_text,
                colLabels=("Frequency (Hz)", "Damping", "MPC"),
                rowLabels=[f"{j+1:n}" for j in range(len(sorted_modes))],
                loc='upper left')


class StabilizationDiagram:
    """Stabilization diagram for selecting physical modes

    One fundamental issue in system identification is that the model
    order is an unknown parameter. The solution is to overestimate the
    model order and then separate physical modes from the spurious
    numerical ones. There exists several approaches to accomplish this,
    but a common approach is to use a _stabilization diagram_ which
    plots the system poles of increasing order and identifies a pole
    as stable if it is repeated (within tolerance) for increasing
    model order.
    """
    stable_color = (0, 1, 0, .3)
    unstable_color = (1, 0, 0, .3)
    markersize = 3.0
    marker = '.'
    pickradius = 2.0
    dpi = 144
    figsize = np.array([1, 1]) / 2.5 * 10
    picked_edgecolor = 'k'
    picked_markersize_factor = 2

    def __init__(self):
        """Stabilization diagram with a table of picked modes

        Stabilization diagram with plot for the modes and a table
        with the modes which are picked from the modes.

        Modes are defined stable or unstable by the `check_stability`
        method.

        After picking the modes, you can access the picked modes with
        the `picked_modes` property

        Plot modes with the `plot` method.
        """
        self.figure = plt.figure("Stabilization diagram", figsize=self.figsize,
                                 dpi=self.dpi)
        self.gridspec = self.figure.add_gridspec(nrows=2, ncols=1, wspace=0.15,
                                                 hspace=0.3)

        self.axes_plot = self.figure.add_subplot(self.gridspec[0])
        self.axes_plot.set(xlabel='Frequency (Hz)', ylabel='Model Order',
                           title='Stabilization Diagram')

        self.axes_table = self.figure.add_subplot(self.gridspec[1])
        self.axes_table.axis('off')

        self.picker = TableModePicker(self.axes_table)
        self.picker.picked_edgecolor = self.picked_edgecolor
        self.picker.picked_markersize_factor = self.picked_markersize_factor
        self.figure.canvas.mpl_connect('pick_event', self.picker)

    @property
    def picked_modes(self):
        return self.picker.picked

    def plot(self, modes):
        """Plot modes in the stabilization diagram

        This method takes in a dict where the key is the
        order of the model and the value is a list of modes
        pertaining to the model order. The `check_stability`
        method of the mode is used to determine whether a mode
        is considered to be _stable_ or not.

        Arguments
        ---------
        modes : dict
            Dictionary where the key is the model order and
            the value is a list of strid.Mode instances.
        """
        prev_modes = None
        orders = sorted([*modes.keys()])
        for order in orders:
            order_modes = modes[order]
            if prev_modes is None:
                prev_modes = order_modes
            for mode in order_modes:
                mode_is_stable = any(
                    [self.check_mode_stability(mode, other)
                     for other in prev_modes])
                if mode_is_stable:
                    color = self.stable_color
                else:
                    color = self.unstable_color
                lines = self.axes_plot.plot(
                    mode.f, order,
                    self.marker,
                    color=color,
                    ms=self.markersize,
                    picker=True,
                    pickradius=self.pickradius)
                lines[0].mode = mode
            prev_modes = order_modes

    def check_mode_stability(self, mode, other):
        """Check if a mode is stable in comparsion to other mode

        The mode is considered stable in comparison to another mode
        if the frequency, damping and modeshape is the same within a
        tolerance.

        In the default state, the following stability criteria are
        considered:

            * Frequency       : |f_mode-f_other|/f_other < 1%
            * Damping ratio   : |xi_mode-xi_other|/xi_other < 5%
            * Mode shape (MAC): 1 - MAC(q_mode, q_other) < 2%

        Subclass and redefine the method to consider other stability
        criterions.

        Arguments
        ---------
        mode, other : strid.Mode
            Mode and other mode check similarity between.

        Returns
        -------
        bool
            True if mode is stable according to stability criteria
        """
        tol = {"dfreq": .01, "dxi": .05, "dmac": .02}
        mac = modal_assurance_criterion(mode.v, other.v)
        return ((np.abs(mode.f-other.f) / other.f < tol["dfreq"])
                and (np.abs(mode.xi-other.xi) / other.xi < tol["dxi"])
                and (1-mac < tol["dmac"]))
