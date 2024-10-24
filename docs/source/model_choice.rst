Choosing a Model
================

.. note::

    For a visual guide to the included models, see the :ref:`modellib-link`.

Stellar Model and Star Formation History
----------------------------------------
Modeling in ``lightning.py`` is biased toward the use of piecewise-constant (sometimes called "non-parametric")
star formation histories (SFH). These are adopted as the default choice for fitting galaxy SEDs. Several other
functional SFH forms are provided for toy models and simulations.

.. note::

    The only stellar population models included in the github distribution are the "legacy"
    PEGASE models used in IDL Lightning. Due to restrictions on file size, the recently developed BPASS + Cloudy
    and PEGASE + Cloudy models used in Lehmer+(2024) must be `downloaded <https://www.dropbox.com/scl/fo/is74ra0tc1t0jdo4dsntm/ADDNjrtxro2euqCWmYrCO0Y?rlkey=9v113nb8rqgl5zul6xawuwdde&st=kzgq6kxr&dl=0>`_ before they
    can be used with Lightning.

PEGASE
^^^^^^
The default stellar population model remains the ``PEGASE`` models used in IDL ``Lightning``, with a slightly expanded
range of metallicity. These population models include an analytic prescription for nebular emission. The IMF is
locked to the `Kroupa et al. (2001)`_ IMF.

PEGASE-A24
^^^^^^^^^^
This release includes a new set of ``PEGASE`` models including a nebular emission model derived from
single-cloud ``Cloudy`` simulations run by Amirnezam Amiri (in 2024, hence 'A24'). See `Nebular Emission Recipe <nebular_emission_recipe.html>`_
for a description of the procedure. The IMF remains `Kroupa et al. (2001)`_.

BPASS
^^^^^
The ``BPASS`` v2.1 release, including *the BPASS collaboration's* ``Cloudy`` *simulations*. The IMF is the ``imf_135_300``
option from ``BPASS``.

BPASS-A24
^^^^^^^^^
``BPASS`` v2.1 models including a nebular emission model derived from
single-cloud ``Cloudy`` simulations run by Amirnezam Amiri (in 2024, hence 'A24'). See `Nebular Emission Recipe <nebular_emission_recipe.html>`_
for a description of the procedure. The IMF is `Chabrier et al. (2003)`_.

BPASS-ULX-G24
^^^^^^^^^^^^^
``BPASS`` v2.1 models including a nebular emission model derived from
single-cloud ``Cloudy`` simulations run by Amirnezam Amiri following the same recipe as above. However, the source SEDs
are drawn from the `Garofali et al. (2024)`_ (hence 'G24') "Simple X-ray Population" models, which match the ``BPASS`` UV-IR
spectra with a ULX model. The IMF is the ``imf_135_100`` option from ``BPASS``. We note that our implementation of
these models do not extend directly to the X-rays in a self-consistent way. A self-consistent (between X-rays and
emission lines) treatment of ULXs is an area of active development. We provide these models mainly as a means for
simulating the emission line ratios of composite populations containing ULXs rather than as a component for SED fitting.

Attenuation Models
------------------
The default attenuation curve in ``Lightning`` is the `Noll et al. (2009)`_ modification of the `Calzetti et al. (2000)`_ curve, which adds
a Drude profile at 2175 Å, and a parameter :math:`\delta` which controls the deviation from the Calzetti slope
in the UV. Note that the IDL ``Lightning`` implementation of birth cloud attenuation is no longer included -- the
``tauV_BC`` parameter is nonfunctional and should be left constant at 0 in any fits. The option to include dust grains
in the HII regions adds some element of extra attenuation for young stars, though it shouldn't be applied incautiously,
especially at low metallicity.

The featureless Calzetti curve is also preserved as an option.

Dust Emission Model
-------------------
The only option for dust emission is the `Draine and Li (2007)`_ model, with 5 possible free parameters. **Energy balance
between the dust emission and attenuation models is always enforced.**

UV-IR AGN Model
---------------
We implement the SKIRTOR model grid from `Stalevski et al. (2016)`_, with 3-4 free parameters, fixing the opening angle
of the torus at 40 degrees. There are several quirks to our implementation:

- The UV-IR AGN model sets the normalization of the X-ray model *unless using the* ``QSOSED`` *X-ray AGN model* in which
  case the situation is reversed: the X-ray AGN model sets the normalization of the UV-IR model. For practical purposes
  this means the UV-IR normalization should be held constant when fitting the ``QSOSED`` model.
- When the polar dust model is selected the attenuation of the AGN and stellar population are decoupled - the accretion
  disk is attenuated only by the torus and polar dust. When the polar dust model *isn't* used, the accretion disk
  is subject to the same attenuation as the stellar population.


X-ray Models
------------

Stars
^^^^^
X-ray emission from compact object binaries is modeled as a power law with a high energy exponential cutoff at 100 keV.
:math:`\Gamma` is available as a free parameter, though in practice we typically leave it fixed. The normalization of
the X-ray stellar population model is set using the empirical :math:`L_X - \log(t)` relationship derived by
`Gilbertson et al. (2022)`_ from normal galaxies in the Chandra Deep Fields. The hot gas spectral shape is not explicitly included,
though the modeling by Gilbertson et al. is such that the *luminosity* of the hot gas is included. This is a weakness of
our current implementation, and improvement of the X-ray stellar population models is an active area of development.

We expect that most users of the X-ray fitting capability of ``Lightning`` will be interested in using it to constrain
the overall luminosity of an AGN component. We advise such users that they should still consider the contribution
from X-ray binaries by enabling the X-ray stellar population model (perhaps with fixed :math:`\Gamma`), especially for
fainter AGN.

AGN
^^^

Power-Law
#########
We provide a simple power law model with a high-energy exponential cutoff at 300 keV. This model is linked to the
luminosity of the UV-IR AGN model at 2500 :math:`\rm \mathring{A}` using the `Lusso and Risaliti (2017)`_
:math:`L_{\rm 2~keV}-L_{2500~\rm \mathring{A}}` relationship, where we allow for a deviation :math:`\delta` from the
relationship, representing the scatter in the population. The two sample-able parameters are thus :math:`\Gamma`,
the power law index, and :math:`\delta`.

QSOSED
######
We also provide an implementation of the `Kubota and Done (2018)`_ QSOSED model family. These physically-motivated models
are constructed to reproduce the soft X-ray excess observed in AGN with an accretion disk and two comptonizing
components. In our implementation, the parameters are the black hole mass :math:`M_{\rm SMBH}` and the Eddington
ratio :math:`\dot m = \dot M / \dot M_{\rm Edd}`. As noted above, when this model is selected, the normalization of the
entire X-ray-to-IR AGN model is set by the combination of :math:`M_{\rm SMBH}` and :math:`\dot m` by linking the
accretion disk luminosities of the model components at 2500 :math:`\rm \mathring{A}`.

The range of Eddington ratios available in this implementation are limited, and thus so is the flexibility of the model
for fitting diverse populations of AGN. We've had some success in applying it to relatively obscured, distant AGN. In
the future, we may attempt to make this model more flexible by falling back on the AGNSED model family, which allows a
greater range of Eddington ratio, and incorporating further physically-motivated AGN models.

Absorption
^^^^^^^^^^
We provide two X-ray absorption models, familiar to users of ``Sherpa`` and ``Xspec``. We generated the curves on a
log-spaced energy grid from 0.01 to 20 keV using ``Sherpa``, with default abundances. Both models have a single
parameter, the hydrogen column density :math:`N_H` in units of :math:`10^{20}~{\rm cm^{-2}}`.

phabs
#####
Photoelectric absorption.

tbabs
#####
Tubingen-Boulder absorption model from `Wilms et al. (2000)`_, including more metal edges than ``phabs``.


.. links
.. _Kroupa et al. (2001): https://doi.org/10.1046/j.1365-8711.2001.04022.x
.. _Chabrier et al. (2003): https://doi.org/10.1086/376392
.. _Garofali et al. (2024): https://doi.org/10.3847/1538-4357/ad0a6a
.. _Noll et al. (2009): https://doi.org/10.1051/0004-6361/200912497
.. _Calzetti et al. (2000): https://doi.org/10.1086/308692
.. _Draine and Li (2007): https://doi.org/10.1086/511055
.. _Stalevski et al. (2016): https://doi.org/10.1093/mnras/stw444
.. _Gilbertson et al. (2022): https://doi.org/10.3847/1538-4357/ac4049
.. _Lusso and Risaliti (2017): https://doi.org/10.1051/0004-6361/201630079
.. _Kubota and Done (2018): https://doi.org/10.1093/mnras/sty1890
.. _Wilms et al. (2000): https://doi.org/10.1086/317016
