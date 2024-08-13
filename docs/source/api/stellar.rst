Stellar Model
=============

Available stellar models in Lightning are based on the PEGASE and BPASS spectral population synthesis codes;
models with the A24 suffix include a nebular component generated from custom Cloudy simulations run by Amirnezam Amiri
in 2024 (hence, A24). The ``PEGASEModel`` class is the same set of stellar population models used in IDL ``Lightning``,
with a Kroupa IMF and Z = 0.001-0.1 

.. autoclass:: lightning.stellar.PEGASEModel
    :members:
    :show-inheritance:

.. autoclass:: lightning.stellar.PEGASEModelA24
    :members:
    :show-inheritance:

.. autoclass:: lightning.stellar.PEGASEBurstA24
    :members:
    :show-inheritance:

.. autoclass:: lightning.stellar.BPASSModel
    :members:
    :show-inheritance:

.. autoclass:: lightning.stellar.BPASSModelA24
    :members:
    :show-inheritance:

.. autoclass:: lightning.stellar.BPASSBurstA24
    :members:
    :show-inheritance:
