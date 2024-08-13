Prior Distributions
===================

Priors in ``Lightning`` are built around two base classes: ``AnalyticPrior`` for
priors with a functional form and ``TabulatedPrior`` for empirically derived
priors. New user-specific priors can be added by defining classes that inherit
from one of these base classes and define the ``evaluate``, ``quantile``, and ``sample``
methods.

In the course of normal use, the ``sample`` method is the only one likely to be used regularly,
in initializing the mcmc.

.. autoclass:: lightning.priors.base.AnalyticPrior
    :members: evaluate, quantile, sample

.. autoclass:: lightning.priors.base.TabulatedPrior
    :members: evaluate, quantile, sample

The ``UniformPrior`` and ``NormalPrior`` classes inherent from ``AnalyticPrior`` and
define the uniform and normal distributions, respectively.

.. autoclass:: lightning.priors.UniformPrior
    :show-inheritance:

.. autoclass:: lightning.priors.NormalPrior
    :show-inheritance:

Constant parameters in Lightning can be fixed to a single value using the ``ConstantPrior``,
which implements a delta-function-like prior. In practice this isn't a true prior, but it
tells ``Lightning`` not to bother sampling the given parameter and what its value should be.

.. autoclass:: lightning.priors.ConstantPrior
    :show-inheritance:
