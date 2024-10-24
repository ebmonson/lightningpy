Plotting
========

We have implemented a number of plotting functions in lightning to make visualizing results easy. These are typically
implemented such that they take a ``Lightning`` object as their first argument, followed by an MCMC chain (and the chain
of log-probability values, where necessary). Most plotting functions can accept dictionaries of keyword arguments
describing the colors and styles of lines, markers, etc. These are passed through to the relevant ``matplotlib`` functions.
All the plotting functions can also plot into existing axes by supplying the ``ax`` keyword (for ``corner_plot``, set the
``fig`` keyword instead to a figure containing the appropriate number of axes).

Two convenience items are also implemented in this module: the ``ModelBand`` class adapted from ``Ultranest``, and the
``step_curve`` function designed to make plotting step functions with shaded uncertainties easier. Both are mostly meant
for internal use, but are documented below for completeness.

We also provide three ``matplotlib`` style sheets (mainly for consistency across the examples) which can be loaded
using ``matplotlib.pyplot.style.use(name)`` where ``name`` is one of
``[lightning-serif, lightning-sans, lightning-dark]``.

.. automodule:: lightning.plots
    :members:
    :show-inheritance:
