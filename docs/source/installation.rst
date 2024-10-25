Installation
============

Currently, ``lightning`` is not available on `PyPI <https://pypi.org/>`_ or `conda-forge <https://conda-forge.org/>`_.
Until it is uploaded, ``lightning`` and its dependencies can be installed by cloning the repo, creating a conda
environment with the required dependencies, and then installing the package locally:

.. code-block:: bash

    git clone https://github.com/ebmonson/lightningpy.git
    cd lightningpy
    conda env create -f environment.yml
    conda activate lightning
    pip install .  --no-deps

Model files will then need to be `downloaded from Dropbox`_ and dropped in the appropriate directories of the
installed package. The folders have readme files to guide you on which files belong where; note that you only need to
download a model set if you plan on using it. If you're following the instructions above, the root directory for the models will be

.. code-block:: bash

    path/to/conda/envs/lightning/lib/python3.XX/site-packages/lightning/data/models/

where ``3.XX`` should be replaced with the version of python used by your environment. You can locate your lightning install by opening a python terminal and doing

>>> import lightning
>>> print(lightning.__file__)

which will show you the root of the lightning install.

.. _downloaded from Dropbox: https://www.dropbox.com/scl/fo/is74ra0tc1t0jdo4dsntm/ADDNjrtxro2euqCWmYrCO0Y?rlkey=9v113nb8rqgl5zul6xawuwdde&st=kzgq6kxr&dl=0
