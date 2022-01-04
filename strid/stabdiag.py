# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from .utils import modal_assurance_criterion

__all__ = ["StabilizationDiagram", ]


class ModePicker(object):
    """Class to pick modes (poles) from stabilization diagrams in matplotlib.

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
        figure = artist.get_figure()
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
        figure.canvas.draw_idle()
        figure.canvas.flush_events()


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
                (f"{mode.f:.2f}Hz",
                 f"{mode.xi:.2%}",
                 f"{mode.mpc:.1%}",
                 f"{mode.mp/np.pi*180:.1f}°")
                for j, mode in enumerate(sorted_modes)]
            self.mpl_axes.table(
                cell_text,
                colLabels=("Frequency", "Damping", "MPC", "MP"),
                rowLabels=[f"{j:n}" for j in range(len(sorted_modes))],
                loc='upper left')


class StabilizationDiagram:
    """Stabilization diagram for selecting physical modes

    One fundamental issue in system identification is that the model
    order is an unknown parameter. The solution is to overestimate the
    model order and then separate physical modes from the spurious
    numerical ones. There exists several approaches to accomplish
    this, but a common approach is to use a _stabilization diagram_
    which plots the system poles of increasing order and identifies a
    pole as stable if it is repeated (within tolerance) for increasing
    model order.

    """
    stable_color = (0, 1, 0, .3)
    unstable_color = (1, 0, 0, .3)
    markersize = 3.0
    marker = '.'
    pickradius = 2.0
    dpi = 144
    figsize = np.array([1, 1]) / 2.5 * 16
    picked_edgecolor = 'k'
    picked_markersize_factor = 2

    def __init__(self, figure_name="Stabilization Diagram"):
        """Stabilization diagram with a table of picked modes

        Stabilization diagram with plot for the modes and a table
        with the modes which are picked from the modes.

        Modes are defined stable or unstable by the `check_stability`
        method.

        After picking the modes, you can access the picked modes with
        the `picked_modes` property

        Plot modes with the `plot` method.

        The PSD can be superimposed on the stabilization diagram by
        accessing/using the attribute `axes_psd` which is a matplotlib.Axes
        instance.
        """
        self.figure = plt.figure(figure_name, figsize=self.figsize,
                                 dpi=self.dpi)
        self.gridspec = self.figure.add_gridspec(nrows=2, ncols=1, wspace=0.15,
                                                 hspace=0.3)

        self.axes_psd = self.figure.add_subplot(self.gridspec[0])
        self.axes_psd.set(xlabel='Frequency (Hz)', ylabel='PSD',
                           title='Stabilization Diagram')
        self.axes_psd.yaxis.set_visible(False)

        self.axes_plot = self.axes_psd.twinx()
        self.axes_plot.set(ylabel="Model order")
        self.axes_plot.yaxis.tick_left()
        self.axes_plot.yaxis.set_label_position("left")

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

        See Also
        --------
        filter_modes
            Method to filter out modes not relevant for further analysis and
            thus not plotted in stabilization diagram.
        find_stable_modes
            Method to classify stable modes.
        """
        filtered_modes = self.filter_modes(modes)
        stable_modes = self.find_stable_modes(filtered_modes)
        orders = sorted([*modes.keys()])
        for order in orders:
            for mode in modes[order]:
                if mode not in filtered_modes[order]:
                    continue
                if mode in stable_modes[order]:
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

    def filter_modes(self, modes):
        """Filter out modes not relevant for further analysis.

        This method is called at the beginning of plot method to
        filter out modes that are not relevant for further analysis and
        which should be dropped in the stabilization diagram.

        In the default implementation, the following modes are filtered
        out prior to plotting the stabilization diagram:

            * Unstable modes, i.e. modes with positive damping / poles
              with positive real part.
            * Modes with negative frequency, i.e. complex conjugate of
              mode / poles with negative imaginary part.

        Arguments
        ---------
        modes : dict
            Dictionary where the key is the model order and
            the value is a list of strid.Mode instances.

        Return
        ------
        dict
            Dictionary where the key is the model order and
            the value is a list of filtered strid.Mode instances.
        """
        filtered_modes = {}
        for order, ms in modes.items():
            filtered_modes[order] = [mode for mode in ms
                                     if ((mode.eigenvalue.real < 0.)
                                         and mode.eigenvalue.imag > 0.)]
        return filtered_modes

    def find_stable_modes(self, modes):
        """Find all stable modes.

        A mode is considered stable if the frequency and damping
        is the same (within a tolerance) to a mode from the previous order.

        In the default state, the following stability criteria are
        considered:

            * Frequency       : |f_mode-f_prev|/f_prev < 1%
            * Damping ratio   : |xi_mode-xi_prev|/xi_prev < 5%
            * Mode shape (MAC): 1 - MAC(q_mode, q_other) < 2%

        Subclass and redefine the method to define other stability
        criterions.

        Arguments
        ---------
        modes : dict
            Dictionary where the key (int) is the model order and
            the value is a list of strid.Mode instances.

        Returns
        -------
        dict :
            Dictionary where the key (int) is the model order and
            the value is a list of stable strid.Mode instances.
        """
        orders = sorted([*modes.keys()])
        stable_modes = {}
        stable_modes[orders[0]] = []
        tol = {"f": .01, "xi": .05, "mac": .02}
        for order_prev, order_current in zip(orders[:-1], orders[1:]):
            stable_modes[order_current] = [mode for mode in modes[order_current]
                                   if any([
                                       (abs(mode.f-mode_prev.f)/mode_prev.f < tol["f"])
                                       and (abs(mode.xi-mode_prev.xi)/mode_prev.xi < tol["xi"])
                                       and (1-modal_assurance_criterion(mode.v, mode_prev.v) < tol["mac"])
                                       for mode_prev in modes[order_prev]])]
        return stable_modes
