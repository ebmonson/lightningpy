X-ray Models
============

X-ray Emission
--------------

The X-ray fitting module implements two simple power law models for XRB populations and AGN, as well as a version
of the ``QSOSED`` model of Kubota & Done (2018).

One noteworthy point about the ``StellarPlaw`` and ``AGNPlaw`` models is that they both rely on knowledge of the
corresponding UV-IR models to set their normalization. You'll thus see that the ``StellarPlaw`` ``get_*`` functions all
require a stellar model and its parameters as input, while their equivalents for the ``AGNPlaw`` model require the
AGN model and its parameters.

.. autoclass:: lightning.xray.StellarPlaw
    :members:
    :show-inheritance:

.. autoclass:: lightning.xray.AGNPlaw
    :members:
    :show-inheritance:

The ``QSOSED`` model, however, doesn't need to know about the UV-IR AGN model, because the opposite is true: the UV-IR
AGN component is normalized by the ``QSOSED`` model, allowing the black hole mass and Eddington ratio to set the overall
luminosity of the entire AGN model.

.. autoclass:: lightning.xray.Qsosed
    :members:
    :show-inheritance:

The emission models above extend the ``XrayPlaw`` and ``XrayEmissionModel`` classes documented at the bottom of this
page for the sake of completeness.

X-ray Absorption
----------------

Two absorption models are implemented: the Tubingen-Boulder absorption model (``tbabs``) which includes more atomic
physics and extends to the UV, and the photoelectric absorption model from XSpec (``phabs``) which covers only the
X-rays.

.. autoclass:: lightning.xray.Tbabs
    :members:
    :show-inheritance:

.. autoclass:: lightning.xray.Phabs
    :members:
    :show-inheritance:

X-ray Base Classes
------------------

.. autoclass:: lightning.xray.XrayPlawExpcut
    :members:
    :show-inheritance:

.. autoclass:: lightning.xray.base.XrayEmissionModel
    :members:
    :show-inheritance:
