import pytest
from lightning.sfh import PiecewiseConstSFH, DelayedExponentialSFH, SingleExponentialSFH
import numpy as np
from scipy.integrate import trapezoid

class TestSFH:

    def test_piecewise(self):

        age_bins = [0.0, 1e7, 1e8, 1e9, 5e9, 13.4e9]
        pw = PiecewiseConstSFH(age_bins)

        sfh = pw.evaluate(np.array([1,1,1,1,1]))
        assert np.all(sfh == np.array([1,1,1,1,1]))

    def test_delayed_exponential(self):

        ages = np.logspace(6, np.log10(13.4e9), 100)
        de = DelayedExponentialSFH(ages)

        sfh = de.evaluate(np.array([1, 5e7]))
        assert sfh.size == 100
        # Not really equal because the age grid starts at 1 Myr instead
        # of 0.0. I could simply calculate the missing portion. But...
        assert trapezoid(sfh, ages) == pytest.approx(1 * 5e7, rel=0.05)

    def test_single_exponential(self):

        ages = np.logspace(6, np.log10(13.4e9), 100)
        se = SingleExponentialSFH(ages)

        sfh = se.evaluate(np.array([1, 5e7, 5e7]))
        assert sfh.size == 100
        # Not really equal because the age grid starts at 1 Myr instead
        # of 0.0
        assert trapezoid(sfh, ages) == pytest.approx(1, rel=0.05)

