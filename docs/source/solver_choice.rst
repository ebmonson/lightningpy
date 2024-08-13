Choosing a Solver
=================

Fitting in ``Lightning`` is biased toward the Bayesian exploration of parameter space rather than maximum likelihood
estimation. As such the majority of our plotting and post-processing functions assume that you've used ``emcee`` for
fitting.

We do however include an option for fitting the SED model with the L-BFGS-B minimization algorithm, results from which
are compatible with the ``lightning_postprocess`` function. There is additionally an option to solve the problem with
L-BFGS-B and then, if it converges, perform a brief exploration of parameter around the best fitting solution
(converting any preexisting bounds on the parameters to uniform priors). This method is potentially much faster than a
brute force fit with ``emcee``, though it risks getting stuck in a local minimum and is not possible
when the solver fails to converge. The results from this MCMC followup are naturally compatible with the MCMC plotting functions.
