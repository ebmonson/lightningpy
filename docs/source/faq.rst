FAQ / Planned Changes
=====================

X-ray Model
-----------

- The rest-frame X-ray wavelength grid should be produced in a very specific way to ensure that your bandpasses are covered,
  especially at high-z. I currently recommend something like::

    import astropy.constants as const
    import astropy.units as u

    hc = (const.c * const.h).to(u.micron * u.keV).value
    E_lo = 0.5
    E_hi = 7.0 # Or whatever is appropriate for you
    xray_wave_grid = np.logspace(np.log10(hc / E_hi),
                               np.log10(hc / E_lo),
                               200)
    xray_wave_grid /= (1 + redshift)

  Note that the wavelength *must be monotonically increasing*. This construction will probably be done by default in
  the future; I'll also probably change the specification to energy rather than wavelength just to make it more sensible.

Upper Limits and Missing Bands
------------------------------

- Upper limits should be specified by setting their flux to 0 and the corresponding uncertainty to the :math:`1\sigma`-equivalent
  limiting flux.

  - The ``'exact'`` upper limit handling should be approached with caution - it needs further testing on real-world data.
  - The ``'approx'`` handling generally suffices to make upper limits behave like upper limits.

- Missing bands, on the other hand, should be specified by setting the flux to ``NaN`` and corresponding uncertainty to
  0. Take care when using masked arrays to fill in masked bands - in particular, ``astropy`` methods for reading fits
  tables may convert your data to masked arrays without your knowledge, producing unintended results.

Tests
-----

- Multiple of the tests which require model files to be installed are skipped automatically when we upload a new build. You can
  (and should) run these tests with ``pytest`` after installing the code and model libraries locally to make sure you've got everything in the
  right place.
