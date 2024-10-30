#!/bin/bash

echo "Simulating data..." 2> tests.err 1> tests.out
python simulate.py 2>> tests.err 1>> tests.out
echo "Testing normal galaxy fitting..." 2>> tests.err 1>> tests.out
python fit_PEGASE.py 2>> tests.err 1>> tests.out
python fit_BPASS.py 2>> tests.err 1>> tests.out
python fit_burst.py 2>> tests.err 1>> tests.out
echo "Testing AGN fitting..." 2>> tests.err 1>> tests.out
python fit_AGN.py 2>> tests.err 1>> tests.out
echo "Testing catalog postprocessing..." 2>> tests.err 1>> tests.out
python postprocess.py 2>> tests.err 1>> tests.out
echo "Done with tests. Inspect tests.out and tests.err."
echo "Warnings like log(0) are ok, and the AGN model should have"
echo "warned that one of the filters (GALEX FUV) is outside the"
echo "wavelength range."
echo ""
echo "Note that the test fits probably aren't great, we didn't run them"
echo "very long."
