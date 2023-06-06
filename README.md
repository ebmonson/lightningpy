# plightning

python prototype of the Lightning SED fitting code.

Keep in mind that everything here is not necessarily *right* or produced according to best practices. Yet.

## Current Caveats/Notes/Planned Changes
----------
### General
- Constant dimensions must be set by making a boolean mask with the same number of entries as the model has parameters.
- Constant dimensions are *excluded* from the sampler returned by `emcee`; you may have to add them back in to produce any post-processed products (see all the functions in `lightning.plots`). This will be automated in the future.
- Parameters are currently handled throughout as arrays rather than e.g. a dictionary keyed by the parameter names.
- The same is true for priors; the sampling functions expect a list of `lightning.priors` objects with the same number of entries as the model has parameters. For constant parameters use `None`.

### X-ray Model
- The rest-frame X-ray wavelength grid should be produced in a very specific way to ensure that your bandpasses are covered,
  especially at high-z. I currently recommend something like:

  ```python

  import astropy.constants as const
  import astropy.units as u

  hc = (const.c * const.h).to(u.micron * u.keV).value
  E_lo = 0.5
  E_hi = 7.0 # Or whatever is appropriate for you
  xray_wave_grid = np.logspace(np.log10(hc / E_hi),
                               np.log10(hc / E_lo),
                               200)
  xray_wave_grid /= (1 + redshift)

  ```

  Note that the wavelength *must be monotonically increasing*. This construction will probably be done by default in the future; I'll also probably change the specification to energy rather than wavelength just to make it more sensible.
