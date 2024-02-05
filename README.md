# plightning

> [!Note]
>
> The full set of BPASS+Cloudy models is not included in this repo at the moment;
> they're still ~1.6 GB because I haven't decided how to cut down the wavelength grid yet.
> Message me and I'll send you a tarball.

python prototype of the Lightning SED fitting code.

Keep in mind that everything here is not necessarily *right* or produced according to best practices. Yet.

## Requirements
------
- numpy
- scipy
- astropy
- tqdm
- emcee (for `Lightning.fit(method='emcee')`)
- h5py (for `lightning.postprocessing.postprocess_catalog`)
- matplotlib (for everything in `lightning.plots`)
- corner (for `lightning.plots.corner_plot`)

You probably already have most of these; if you don't they're all easily retrievable with `conda`.

## Current Caveats/Notes/Planned Changes
----------
### General
- The implementation currently uses (abuses?) default `numpy` behavior that creates `NaN` as the result of `0.0 / 0.0` and `-inf` as a result of `np.log(0.0)`. So when you fit your galaxies you may see a divide by zero warning from `get_filters` whenever the model is initialized (if any of your filters don't fully overlap with the observed-frame wavelength grid) and a second divide by zero warning the first time the sampler encounters a region with 0 prior probability. Later on I'll make the first warning more useful and silence the second.
- Constant dimensions must be set by making a boolean mask with the same number of entries as the model has parameters.
- ~~Constant dimensions are *excluded* from the sampler returned by `emcee`; you may have to add them back in to produce any post-processed products (see all the functions in `lightning.plots`). This will be automated in the future.~~
- Constant dimensions are *excluded* from the sampler returned by `emcee` and thus by `Lightning.fit(method='emcee')`, but they can be added back in using the `Lightning.get_mcmc_chains` method using the `const_dims` and `const_vals` keywords, which specify which model parameters are constant and what their values are, respectively. See documentation for `Lightning.get_mcmc_chains`.
- Parameters are currently handled throughout as arrays rather than e.g. a dictionary keyed by the parameter names.
    - *Until* you run `lightning.postprocessing.postprocess_catalog`, which creates an HDF5 file where parameters can be referenced by name. This procedure is still sort of a prototype though, so it may change. See documentation.
- Priors are also handled as an array rather than a dict: the sampling functions expect a list of `lightning.priors` objects with the same number of entries as the model has parameters. For constant parameters use `None`.
- *Model serialization:* being able to load or recreate your Lightning models after the fact is really useful for making plots and doing analysis (suppose e.g. that I fit a bunch of galaxies and then later on for a proposal or some analysis I want to make synthetic observations in a different bandpass). There are now two ways to do this:
    1. `json`: use the `Lightning.save_json` and `Lightning.from_json` functions. This does not save the entire model, just the minimal configuration needed to recreate the model. It takes less disk space, but it may be slower for complex models and it may result in a loss of precision in input fluxes and wavelength grids (all arrays are demoted to builtin python lists for json serialization).
    2. `pickle`: use the `Lightning.save_pickle` and builtin `pickle.load` functions. This saves the entire model and all of its components, so the files can be big for complicated models. It is however fast and shouldn't result in a loss of precision. All the normal caveats of pickles apply.

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
