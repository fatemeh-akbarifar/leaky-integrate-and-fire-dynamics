import unittest
from dataclasses import replace
import numpy as np
from model import Parameters, simulate, analytic_rate, measured_rate

class NeuronChecks(unittest.TestCase):
    def test_rest_and_threshold_boundary(self):
        t,v,s = simulate(0.)
        np.testing.assert_array_equal(v, -65.*np.ones_like(v))
        self.assertEqual(len(s), 0)
        self.assertEqual(len(simulate(.15)[2]), 0)
    def test_subthreshold_analytic_solution(self):
        t,v,s = simulate(.1)
        exact = -55 - 10*np.exp(-t/20)
        self.assertLess(float(np.max(abs(v-exact))), .01)
        self.assertEqual(len(s),0)
    def test_rates_and_refinement(self):
        for ref in (1.,2.,5.):
            p=replace(Parameters(),refractory_ms=ref)
            for current in (.175,.2,.4,.5):
                spikes=simulate(current,p)[2]
                self.assertTrue((np.diff(spikes)>ref).all())
                rate=measured_rate(spikes)
                fine=measured_rate(simulate(current,p,dt_ms=.05)[2])
                self.assertLessEqual(abs(rate-fine), max(2,.02*fine))
                self.assertLessEqual(abs(rate-analytic_rate(current,p)),2.)
    def test_reset_hold_and_invalid_inputs(self):
        t,v,s=simulate(.4)
        k=int(round(s[0]/.1))
        np.testing.assert_array_equal(v[k:k+21],np.full(21,-65.))
        for kwargs in ({'dt_ms':0},{'dt_ms':.3},{'current_na':np.nan}):
            with self.assertRaises(ValueError):
                simulate(**{'current_na':.2,**kwargs})

if __name__=='__main__': unittest.main()
