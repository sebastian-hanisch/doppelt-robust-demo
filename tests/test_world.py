"""Vehikel: Zuteilung, Ergebnis, wahre Störfunktionen und Stichprobe."""

import math

import numpy as np
import pytest

import dml_constants as C
import dml_world as W
from kausal_generator import SplitMix64, normal_cdf

X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])


def test_score_and_baseline_by_hand():
    w = W.World(1.0, 1.0)
    assert W.score(X, w)[0] == pytest.approx(0.7 + 0.5 * 3.0 + 0.5 * 20.0 + 0.5 * math.sin(6.0))
    assert W.baseline(X, w)[0] == pytest.approx(0.5 + 0.8 * 3.0 + 0.8 * 20.0 + 0.6 * math.sin(6.0) + 2.5)
    flat = W.World(0.0, 1.0)
    assert W.score(X, flat)[0] == pytest.approx(0.7) and W.baseline(X, flat)[0] == pytest.approx(0.5 + 2.5)


def test_propensity_is_a_probit_of_the_scaled_index():
    w = W.World(0.5, 0.9)
    x = np.zeros((1, 5))
    assert W.propensity(x, w)[0] == pytest.approx(float(normal_cdf(0.9 * (-0.5 * 0.5))))
    assert W.propensity(x, W.World(0.0, 0.9))[0] == pytest.approx(0.5)


def test_true_nuisances_differ_by_the_effect():
    w = W.World(0.8, 0.9)
    x = np.random.default_rng(0).normal(size=(50, 5))
    m0, m1, e = W.true_nuisances(x, w)
    assert (m1 - m0 == pytest.approx(C.TAU)) and (e > 0).all() and (e < 1).all()
    assert m0 == pytest.approx(W.baseline(x, w)) and e == pytest.approx(W.propensity(x, w))


def test_simulate_shapes_streams_and_reproducibility():
    w = W.World(0.4, 0.9)
    a = W.simulate(w, 30, SplitMix64(5), reps=4)
    b = W.simulate(w, 30, SplitMix64(5), reps=4)
    assert a[0].shape == (4, 30, 5) and a[1].shape == a[2].shape == (4, 30)
    assert all((u == v).all() for u, v in zip(a, b))
    assert set(np.unique(a[1])) <= {0.0, 1.0} and not (W.simulate(w, 30, SplitMix64(6), reps=4)[0] == a[0]).all()


def test_simulation_matches_the_model():
    w = W.World(0.8, 0.9)
    x, t, y = (a[0] for a in W.simulate(w, 200000, SplitMix64(7)))
    assert t.mean() == pytest.approx(W.propensity(x, w).mean(), abs=0.004)
    assert (y - C.TAU * t - W.baseline(x, w)).std() == pytest.approx(1.0, abs=0.01)
    assert (y - C.TAU * t - W.baseline(x, w)).mean() == pytest.approx(0.0, abs=0.01)
    assert y[t == 1].mean() - y[t == 0].mean() > C.TAU + 0.3                             # konfundiert: der naive Vergleich liegt weit daneben


def test_nonlinearity_confounds_only_through_both_functions():
    x, t, y = (a[0] for a in W.simulate(W.World(0.0, 0.9), 100000, SplitMix64(8)))
    lin = np.column_stack([np.ones(len(y)), t, x])
    assert np.linalg.lstsq(lin, y, rcond=None)[0][1] == pytest.approx(C.TAU, abs=0.03)       # ohne Nichtlinearität trifft die lineare Regression
    x, t, y = (a[0] for a in W.simulate(W.World(1.0, 0.9), 100000, SplitMix64(8)))
    lin = np.column_stack([np.ones(len(y)), t, x])
    assert abs(np.linalg.lstsq(lin, y, rcond=None)[0][1] - C.TAU) > 0.5                      # mit ihr nicht mehr


def test_effective_sample_size_by_hand():
    t = np.array([1.0, 1.0, 0.0, 0.0])
    e = np.array([0.5, 0.5, 0.75, 0.75])                                                 # Gewichte 2, 2, 4, 4: Summe 12, Quadrate 40
    assert W.effective_sample_size(t, e) == pytest.approx(144.0 / 40.0)
    assert W.effective_sample_size(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(2.0)
    assert math.isfinite(W.effective_sample_size(np.array([1.0, 0.0]), np.array([1.0, 0.0])))   # geklemmt: keine Division durch null
