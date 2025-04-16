from lightning.attenuation import SMC, CalzettiAtten, ModifiedCalzettiAtten
from lightning.xray.absorption import Phabs, Tbabs
import numpy as np

class TestAtten:
    '''Note that the X-ray absorption subclasses
    UV-IR attenuation, so PHABS and TBABS are 
    covered here.
    '''

    wave_grid = np.logspace(-2, 1.3, 100)
    xray_wave_grid = np.logspace(-6,-2,100)

    def test_SMC(self):

        s = SMC(self.wave_grid)

        emtau = s.evaluate(np.array([0.1]))
        assert emtau.size == self.wave_grid.size

    def test_calzetti(self):

        c = CalzettiAtten(self.wave_grid)

        emtau = c.evaluate(np.array([0.1]))
        assert emtau.size == self.wave_grid.size

        av = c.get_AV(np.array([0.1]))
        assert av == 2.5 * 0.1 / np.log(10)

    def test_mod_calzetti(self):

        mc = ModifiedCalzettiAtten(self.wave_grid)

        emtau = mc.evaluate(np.array([0.1, -0.3, 0.0]))
        
        assert emtau.size == self.wave_grid.size
        av = mc.get_AV(np.array([0.1]))
        assert av == 2.5 * 0.1 / np.log(10)

    def test_phabs(self):

        ph = Phabs(self.xray_wave_grid)

        emtau = ph.evaluate(np.array([100]))
        assert emtau.size == self.xray_wave_grid.size

    def test_Tbabs(self):

        tb = Tbabs(self.xray_wave_grid)

        emtau = tb.evaluate(np.array([100]))
        assert emtau.size == self.xray_wave_grid.size

