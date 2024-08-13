Attenuation Curves
==================

The default attenuation model in lightning is the "modified Calzetti" curve from Noll+(2009) with a 2175 Angstrom
bump and a variable UV slope linked to the bump strength. This model is implemented in the ``ModifiedCalzettiAtten``
class.

.. autoclass:: lightning.attenuation.ModifiedCalzettiAtten
    :show-inheritance:
    :members:

For toy models and instances where there is insufficient data to constrain the variable UV slope of the
``ModifiedCalzettiAtten`` model, we also implement the pure Calzetti featureless attenuation curve in the
``CalzettiAtten`` class.

.. autoclass:: lightning.attenuation.CalzettiAtten
    :show-inheritance:
    :members:

The SMC extinction curve is also implemented in the ``SMC`` class for internal use by the polar dust attenuation model,
as in X-Cigale.

.. autoclass:: lightning.attenuation.SMC
    :show-inheritance:
    :members:

The above attenuation and extinction models extend the ``AnalyticAtten`` and ``TabulatedAtten`` classes.

.. autoclass:: lightning.attenuation.base.AnalyticAtten

.. autoclass:: lightning.attenuation.base.TabulatedAtten
