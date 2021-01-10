=====
strid
=====

Python package for system identification of structural dynamic systems.

The identification methods are applicable to general linear systems,
and the implementation may also be used in other domains.



Installation
------------

Either download the repository to your computer and install, e.g. by **pip**

::

   pip install .


or install directly from github.

::

   pip install git+https://github.com/gunnstein/strid.git

Usage
-----

The package provides funtionality to identify structural systems through
subspace identification methodology. Utility functionality associated with
characterizing modal properties and behaviour is also included.

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
   for order in range(5, 150, 5):
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
