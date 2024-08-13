Dust Model
==========

Dust emission powered by the stellar radiation field is modeled in lightning using the Draine & Li (2007) models. Setting
``dust_emission=True`` when initializing ``Lightning`` automatically selects this model; we do not currently provide
an alternative.

.. autoclass:: lightning.dust.DL07Dust
    :members:
    :show-inheritance:

The ``GrayBody`` class is used internally to add an additional cold dust component to the AGN emission spectra, balanced
with the attenuated power of the "polar dust" component. It cannot currently be selected to replace the Draine & Li model
to model the dust emission powered by attenuated stars. It is documented here for completeness; note that it can be
initialized on its own and used to build your own custom models.

.. autoclass:: lightning.dust.Graybody
    :members:
    :show-inheritance:
