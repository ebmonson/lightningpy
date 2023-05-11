#!/usr/bin/env python

from pathlib import Path
import astropy.units as u
import astropy.constants as const
import numpy as np
import json
from scipy.interpolate import interp1d
from scipy.integrate import trapz

__all__ = ['get_filters']

def _parse_tophat_str(s):
    '''
    Turn the strings for ad-hoc filter labels
    into wavelength limits.
    '''

    comps = s.split('_')

    assert (len(comps) == 4), "tophat filter label '%s' is not formatted correctly." % (s)
    assert (comps[0] == 'XRAY'), "tophat filter label '%s' is not formatted correctly." % (s)

    try:
        lo = float(comps[1])
        hi = float(comps[2])
    except TypeError as e:
        print("Unable to convert limits in tophat filter label '%s' to numbers." % (s))
        raise e

    try:
        unit = u.Unit(comps[3])
    except ValueError as e:
        print("Unit string '%s' not understood." % (comps[3]))
        raise e

    # attach units
    lo = lo << unit
    hi = hi << unit

    # Take the provided units and convert them to wavelength.
    # Is it energy?
    if (unit.is_equivalent(u.erg)):
        hc = const.h * const.c
        tmp = hc / hi
        hi = hc / lo
        lo = tmp
    # Frequency?
    elif (unit.is_equivalent(u.Hz)):
        tmp = const.c / hi
        hi = const.c / lo
        lo = tmp
    # or wavelength?
    elif (unit.is_equivalent(u.cm)):
        hi = hi
        lo = lo
    else:
        raise ValueError("Unit '%s' is not understood as wavelength/frequency/energy." % (comps[3]))

    return lo, hi

def get_filters(filter_labels, wave_grid, path_to_filters=None):
    '''Look up and load filters from files.

    Filters are interpolated to a common wavelength grid. Completely
    non-overlapping filters are set to NaN. Note that filters have all
    been previously re-normalized to an equal-energy response.

    TODO:
        - Set option to disallow partial overlap?

    Parameters
    ----------
    filter_labels : array-like, str
        List of labels for filters.
    wave_grid : array-like, float
        Wavelength grid in microns to interpolate filters on. Filters will be
        normalized on this grid, so if you want to integrate them against nu later you'll need to
        re-normalize.

    Returns
    -------
    filters : dict
        with keys pulled from ``filter_labels`` and values equal to the normalized transmission of the
        filter.
    '''

    if (path_to_filters is None):
        path_to_filters = str(Path(__file__).parent.resolve()) + '/filters/'
    else:
        if(path_to_filters[-1] != '/'): path_to_filters = path_to_filters + '/'

    #if (path_to_filters[-1] != '/'): path_to_filters = path_to_filters + '/'

    # Load the JSON blob containing the filter names.
    with open(path_to_filters + 'filters.json') as f:
        filter_paths = json.load(f)

    # Initialize a dict for the filters. Might change this to a structured numpy array.
    filters = dict()
    for label in filter_labels:
        filters[label] = np.zeros(len(wave_grid), dtype='float')

    for i, label in enumerate(filter_labels):

        # X-ray filters are ad-hoc.
        # Really no reason to limit this to X-rays either;
        # I can imagine situations where having a narrow top hat
        # at the location of a given spectral feature would have good
        # predictive power.
        if ('XRAY' in label):

            lo, hi = _parse_tophat_str(label)
            lo_um = lo.to(u.micron).value
            hi_um = hi.to(u.micron).value
            transm = np.zeros_like(wave_grid)
            # What's the best way to handle this, interpolating a tophat
            # function to an arbitrary grid? We want to keep the sharp sides,
            # and we want the normalization to be set by the full width
            if (np.all(wave_grid > hi_um) or np.all(wave_grid < lo_um)):
                transm[:] = np.nan # replicating the behavior in the UV-IR
            else:
                transm[(wave_grid >= lo_um) & (wave_grid <= hi_um)] = 1

            transm_norm = transm / (hi_um - lo_um)
            filters[label] = transm_norm

        else:

            path_to_this_filter = path_to_filters + filter_paths[label]

            filt_arr = np.loadtxt(path_to_this_filter, dtype=[('wave', 'float'), ('transm', 'float')])

            # Interpolate the filter onto the internal wavelength grid
            f = interp1d(filt_arr['wave'], filt_arr['transm'], bounds_error=False, fill_value=0.0)
            transm_interp = f(wave_grid)

            # And normalize the filter
            # NOTE: right now this sets *completely* non-overlapping filters to NaN
            #       but partially-overlapping filters are still finite, and normalized
            #       to the region of overlap rather than their native area. This is
            #       probably not the ideal behavior.
            transm_norm_interp = transm_interp / trapz(transm_interp, wave_grid)
            filters[label] = transm_norm_interp

    return filters

def main():

    print(get_filters.__doc__)

# end if

if (__name__ == '__main__'):

    main()

# end if
