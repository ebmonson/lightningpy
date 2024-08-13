Installation
============

To install Lightning, all you need to do is clone the git repository
to a location of your choosing and then add that location to your
``PYTHONPATH`` system variable.

Dependencies must be handled manually because Erik has never looked up
how ``setuptools`` works::

    - numpy
    - scipy
    - emcee
    - corner
    - h5py
    - astropy

That said, you probably have compatible versions of all of the above already,
and if you don't they can be installed through ``conda`` very easily.
