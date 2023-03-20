#!/usr/bin/env python

'''
    get_filters.py

    Define a function for loading the bandpasses.
'''

from pathlib import Path
import numpy as np
import json
from scipy.interpolate import interp1d
from scipy.integrate import trapz

__all__ = ['get_filters']

def get_filters(filter_labels, wave_grid, path_to_filters=None):
    '''
        get_filters

        Input:
            filter_labels - array-like, str, List of labels for filters.
            wave_grid - array-like, float, Wavelength grid in microns to interpolate filters on. Filters will be
                        normalized on this grid, so if you want to integrate them against nu later you'll need to
                        re-normalize.
        Output:
            filters - dict with keys pulled from filter_labels and values equal to the normalized transmission of the
                      filter.
    '''

    if (path_to_filters is None):
        path_to_filters = str(Path(__file__).parent.resolve()) + '/filters/'
    else:
        if(path_to_filters[-1] != '/'): path_to_filters = path_to_filters + '/'
    # end if

    #if (path_to_filters[-1] != '/'): path_to_filters = path_to_filters + '/'

    # Load the JSON blob containing the filter names.
    with open(path_to_filters + 'filters.json') as f:
        filter_paths = json.load(f)
    # end with

    # Initialize a dict for the filters. Might change this to a structured numpy array.
    filters = dict()
    for label in filter_labels:
        filters[label] = np.zeros(len(wave_grid), dtype='float')
    # end loop

    for i, label in enumerate(filter_labels):

        path_to_this_filter = path_to_filters + filter_paths[label]

        filt_arr = np.loadtxt(path_to_this_filter, dtype=[('wave', 'float'), ('transm', 'float')])

        # Interpolate the filter onto the internal wavelength grid
        f = interp1d(filt_arr['wave'], filt_arr['transm'], bounds_error=False, fill_value=0.0)
        transm_interp = f(wave_grid)

        # And normalize the filter
        transm_norm_interp = transm_interp / trapz(transm_interp, wave_grid)
        filters[label] = transm_norm_interp
    # end loop

    return filters
# end definition

def main():

    print(get_filters.__doc__)

# end if

if (__name__ == '__main__'):

    main()

# end if
