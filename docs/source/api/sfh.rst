SFH Models
==========

SFH modeling in ``Lightning`` is biased toward the use of the "non-parametric" ``PiecewiseConstSFH`` as in every previous
``Lightning`` publication.

.. autoclass:: lightning.sfh.PiecewiseConstSFH
    :show-inheritance:
    :members: evaluate, multiply, sum

Several parametric SFH models are also included in the code for implementing toy models,
simulating populations, reproducing literature results, etc.

.. autoclass:: lightning.sfh.DelayedExponentialSFH
    :show-inheritance:
    :members: evaluate, multiply, integrate

.. autoclass:: lightning.sfh.SingleExponentialSFH
    :show-inheritance:
    :members: evaluate, multiply, integrate

New user-specific parametric SFHs can be added by defining classes that extend
the ``FunctionalSFH`` class and define the ``evaluate`` method.

.. autoclass:: lightning.sfh.base.FunctionalSFH
    :members: evaluate, multiply, integrate
