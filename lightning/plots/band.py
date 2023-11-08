import numpy as np
import matplotlib.pyplot as plt

class ModelBand:
    '''
    A class interface for storing a bunch of model
    realizations and plotting them, as shaded bands,
    quantiles, or individual realizations.

    This is very much taken from UltraNest's ``PredictionBand``
    class with some tweaks. Ultranest is (c) 2019 by Johannes Buchner,
    and is available under GPL v3. Find it here:
    <https://johannesbuchner.github.io/UltraNest/>_

    All plotting functions (``shade``, ``line``, ``realizations``) draw into the
    current axes unless the ``ax`` keyword is set. The plotting functions also
    all pass through keyword arguments, allowing you to modify color, alpha, etc.
    '''

    def __init__(self, x, seed=None):

        self.x = x
        self.y = []
        self.Nreal = 0
        self.rng = np.random.default_rng(seed)

    def add(self, y):
        '''
        Add a realization to the band. Currently limited
        to one at a time.
        '''

        assert len(self.x) == len(y), 'Model realization should have the same number of points (%d) as independent variable points.' % (len(self.x))
        self.y.append(y)
        self.Nreal += 1

    def shade(self, q=(0.16, 0.84), ax=None, **kwargs):
        '''
        Draw a shaded region between the quantiles specified
        by ``q``. Defaults to the 68% interval.
        '''

        assert len(q) == 2, 'Specify a quantile interval by its lower and upper quantiles.'

        lo, hi = np.quantile(np.array(self.y), axis=0, q=q)

        if ax is None:
            return plt.fill_between(self.x, lo, hi, **kwargs)
        else:
            return ax.fill_between(self.x, lo, hi, **kwargs)

    def line(self, q=0.50, ax=None, **kwargs):
        '''
        Draw a line at the specified quantile ``q``.
        Defaults to the median.
        '''

        l = np.quantile(np.array(self.y), axis=0, q=q)

        if ax is None:
            return plt.plot(self.x, l, **kwargs)
        else:
            return ax.plot(self.x, l, **kwargs)

    def realizations(self, num=1, replace=False, ax=None, **kwargs):
        '''
        Plot ``num`` randomly chosen model realizations. If ``replace`` is set,
        the same realization can be chosen multiple times.
        '''

        idcs = self.rng.choice(Nreal, size=num, replace=replace)

        real = np.array(self.y)[idcs,:]

        lines = []
        for l in real:
            if ax is None:
                line, = plt.plot(self.x, l, **kwargs)
            else:
                line, = ax.plot(self.x, l, **kwargs)
            lines.append(line)
        return lines
