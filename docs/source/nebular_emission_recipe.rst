Nebular Emission Recipe
=======================

Our custom nebular emission models are single cloud models, using ``Cloudy`` simulations assuming:

- An open geometry
- :math:`\log U \in [-4, -1.5]`
- Gas-phase metallicity equal to the stellar metallicity (i.e. the metallicity of the youngest stars)
- Two densities -- :math:`\log n_H = 2` and :math:`\log n_H = 3.5`.
- Constant pressure.

For each spectral template, :math:`Q_0` is effectively fixed, and thus varying :math:`\log U` is equivalent to varying
:math:`R`. We convert the line and continuum intensities produced by ``Cloudy`` back into luminosities by solving
for the appropriate :math:`R` given :math:`Q_0`, :math:`\log U`, and :math:`\log n_H`.

We include the option to add dust grains inside the H II region, which we note produces the effect of extra attenuation
and warm dust emission in the emitted continuum in addition to affecting the emitted line ratios. 
