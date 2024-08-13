Choosing a Model
================

Stellar Model and Star Formation History
----------------------------------------
Modeling in ``lightning.py`` is biased toward the use of piecewise-constant (sometimes called "non-parametric")
star formation histories (SFH). These are adopted as the default choice for fitting galaxy SEDs. Several other
functional SFH forms are provided for toy models and simulations.

PEGASE
^^^^^^
The default stellar population model remains the ``PEGASE`` models used in IDL ``Lightning``, with a slightly expanded
range of metallicity. These population models include an analytic prescription for nebular emission. The IMF is
locked to the Kroupa et al. (2001) IMF.

PEGASE-A24
^^^^^^^^^^
This release includes a new set of ``PEGASE`` models including a nebular emission model derived from
single-cloud ``Cloudy`` simulations run by Amirnezam Amiri (in 2024, hence 'A24'). See `Nebular Emission Recipe <nebular_emission_recipe.html>`_
for a description of the procedure. The IMF remains Kroupa et al. (2001).

BPASS
^^^^^
The ``BPASS`` v2.1 release, including *the BPASS collaboration's* ``Cloudy`` *simulations*. The IMF is the ``imf_135_300``
option from ``BPASS``.

BPASS-A24
^^^^^^^^^
``BPASS`` v2.1 models including a nebular emission model derived from
single-cloud ``Cloudy`` simulations run by Amirnezam Amiri (in 2024, hence 'A24'). See `Nebular Emission Recipe <nebular_emission_recipe.html>`_
for a description of the procedure. The IMF is Chabrier et al. (2003).

BPASS-ULX-G24
^^^^^^^^^^^^^
``BPASS`` v2.1 models including a nebular emission model derived from
single-cloud ``Cloudy`` simulations run by Amirnezam Amiri following the same recipe as above. However, the source SEDs
are drawn from the Garofali et al. (2024, hence 'G24') "Simple X-ray Population" models, which match the ``BPASS`` UV-IR
spectra with a ULX model.

Attenuation Models
------------------
The default attenuation curve in ``Lightning`` is the Noll et al. (2009) modification of the Calzetti curve, which adds
a Drude profile at 2175 Å, and a parameter :math:`\delta` which controls the deviation from the Calzetti slope
in the UV. Note that the IDL ``Lightning`` implementation of birth cloud attenuation is no longer included -- the
``tauV_BC`` parameter is nonfunctional and should be left constant at 0 in any fits. The option to include dust grains
in the HII regions adds some element of extra attenuation for young stars, though it shouldn't be applied incautiously,
especially at low metallicity.

The featureless Calzetti curve is also preserved as an option.

Dust Emission Model
-------------------
The only option for dust emission is the Draine and Li (2007) model, with 5 possible free parameters. **Energy balance
between the dust emission and attenuation models is always enforced.**

UV-IR AGN Model
---------------
We implement the SKIRTOR model grid from Stalevski et al. (2016), with 3-4 free parameters, fixing the opening angle
of the torus at 40 degrees. There are several quirks to our implementation:

- The UV-IR AGN model sets the normalization of the X-ray model *unless using the* ``QSOSED`` *X-ray AGN model* in which
  case the situation is reversed: the X-ray AGN model sets the normalization of the UV-IR model. For practical purposes
  this means the UV-IR normalization should be held constant when fitting the ``QSOSED`` model.
- When the polar dust model is selected the attenuation of the AGN and stellar population are decoupled - the accretion
  disk is attenuated only by the torus and polar dust. When the polar dust model *isn't* used, the accretion disk
  is subject to the same attenuation as the stellar population.


X-ray Models
------------
TKTKTK
