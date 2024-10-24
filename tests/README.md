# Tests
-------

Run the tests in this folder with

```bash
time ./run_tests.sh
```

At the moment these tests are intended to test that Lightning will run
correctly in a variety of common situations. Fits produced in the course
of these tests are not expected to be exactly correct, since we run the MCMC
for a very short length, but if the MCMC posterior median is very distant
from the true solution, this could indicate a bug. The above script will produce two files, `test.err` and `test.out`, containing the contents of the `STDERR` and `STDOUT` streams. If you see any exceptions in `test.err`, `Lightning` may not be correctly configured (i.e. you may have missing or out-of-date model files).
