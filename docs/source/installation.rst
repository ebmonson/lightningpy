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
