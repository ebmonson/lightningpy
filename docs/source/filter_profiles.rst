Filter Profiles
===============

A full list of available filters in Lightning can be found [below](filter_table). All filters are in terms of the
spectral response per energy (i.e., `equal-energy <https://en.wikipedia.org/wiki/AB_magnitude#Definition>`_).

Directory Structure
-------------------

The ``filters/`` directory contains all the current filter profiles available in Lightning. The structure of the
directory follows the pattern ``filters/observatory/instrument/instrument_band.txt``. In the case of survey
specific filters (e.g., 2MASS, SDSS, etc.), the pattern is changed to ``filters/survey/survey_band.txt``. Examples
for HST WFC3/F125W and 2MASS J band would be ``filters/HST/WFC3/WFC3_F125W.txt`` and ``filters/2MASS/2MASS_J.txt``,
respectively.

File Format
-----------

Each filter profile file in Lightning has been edited to a common format.
The original source files can be found in the linked URLs in the table [below](filter_table).
Each formatted filter profile file has two, space delimited columns giving the wavelength
(in microns) and throughput function (normalized to the maximum value). The values in both columns are
formatted using the FORTRAN (C) formatting code ``E13.7`` (``%13.7e``).

For example, the first several lines for the 2MASS J band are::

    # wave[microns]    norm_trans
      1.0620000E+00    0.0000000E+00
      1.0660000E+00    4.0706800E-04
      1.0700000E+00    1.5429300E-03
      1.0750000E+00    2.6701300E-03
      1.0780000E+00    5.5064300E-03


Addition of New Filters
-----------------------

New filters can be added manually, by formatting them as above and placing them in a subdirectory of the ``filters/``
directory. Filters must then be added to ``filters/filters.json``, where the key is the 'filter_label' which will be
used to specify the filter in lightning, and the value is the relative path to the filter.

.. note::
    Adding new filters will make your local git repository out of sync with the remote.
    It is currently recommended to make a ``filters/user/`` directory, which you back up before
    updating the code and replace after updating. Note that paths to your filters need not follow the exact formula
    as the built-in filters, as long as the path in ``filters/filters.json`` is correct.

List of Filters in Lightning
----------------------------

The below table gives a list of all the filters currently available in Lightning, with links to the source and the
corresponding filter labels.

.. note::

    Filter labels in the below table may not render correctly in the PDF version of this manual.
    Check ``filters/filters.json`` for consistency.


.. csv-table:: Filters available
    :header-rows: 1
    :widths: 20 20 30 30
    :file: filter_table.csv
