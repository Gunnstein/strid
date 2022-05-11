.. image:: https://zenodo.org/badge/242476045.svg
   :target: https://zenodo.org/badge/latestdoi/242476045

=====
strid
=====

Python package for system identification of linear time-invariant systems.

The package is written from a structural engineering perspective.
However, the identification methods are applicable to linear
time-invariant systems, and the implementation may also be used in
other domains.

The primary focus of this package is to provide funtionality to
identify models of linear time-invariant systems.

Additionally, the package has a class for stabilization diagrams for
selecting modes interactively, a `spatial` module for establishing a
spatial model of a structure and converting a mode shape vector to a
deformation array for a higher order 3D mesh and a class for plotting
and animating mode shapes on spatial models with matplotlib. Below is
an animation of the first horizontal bending mode of the railway
bridge at Hell, identified and animated by `strid` alone:

|hell_animation|



Installation
------------

Either download the repository to your computer and install, e.g. by **pip**

::

   pip install .


or install directly from github

::

   pip install git+https://github.com/gunnstein/strid.git


or install directly from the python package index

::

   pip install strid


Usage
-----

The code example below shows how the modes of a combined
deterministic-stochastic system can be obtained from measurements of
the input `u` and the output `y`.


.. code:: python

   # ..
   # Assume that the measured input u and output y and sampling rate is available
   #
   # First, import the strid module
   import strid

   # Then instanciate the appropriate subspace identification (SID) object
   csid = strid.CombinedDeterministicStochasticSID(u, y, fs)

   # If we know the model order we can now perform the SID and obtain the
   # state space system matrices. For instance, we can  20 block rows and
   # a model order of 100 as shown below
   A, B, C, D = csid.perform(100, 20, estimate_B_and_D=True)

   # Most often, we do not know the model order, and instead we overestimate
   # model order and pick the physical modes with the help of a stabilization
   # diagram. Strid also includes a stabilization diagram and functionality to
   # pick modes directly from the plot.
   # First, we must estimate modes for a range of different model orders
   modes = dict()
   for order in range(5, 150, 1):
       A, C = csid.perform(order, 20)
       modes[order] = strid.Mode.find_modes_from_ss(A, C, csid.fs)

   # Then we can create and plot a stabilization diagram (see image below)
   stabdiag = strid.StabilizationDiagram()
   stabdiag.plot(modes)

   # And we can use the mouse to pick the stable poles from
   # the diagram and then access the picked modes with the
   # `picked_modes` property of the StabilizationDiagram instance.
   picked_modes = stabdiag.picked_modes

|stab_plot|

Additional examples are found in the `examples folder <https://github.com/Gunnstein/strid/tree/master/examples>`_.



Support
-------

Please `open an issue <https://github.com/Gunnstein/strid/issues/new>`_
for support.


Contributing
------------

Please contribute using `Github Flow
<https://guides.github.com/introduction/flow/>`_.
Create a branch, add commits, and
`open a pull request <https://github.com/Gunnstein/strid/compare/>`_.


.. |stab_plot| image:: https://github.com/Gunnstein/strid/blob/master/example.png
.. |hell_animation| image:: https://github.com/Gunnstein/strid/blob/master/hell.gif
