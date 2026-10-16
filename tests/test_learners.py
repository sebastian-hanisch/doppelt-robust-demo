"""Lernverfahren: Merkmale, Probit, parametrische Modelle und Zufallswald."""

import math

import numpy as np
import pytest

import dml_constants as C
import dml_learners as L
import dml_world as W
from kausal_generator import SplitMix64, normal_cdf


def test_feature_maps_by_hand():
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert L.lin_features(x).tolist() == [[1.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    assert L.outcome_basis(x)[0] == pytest.approx([1.0, 1.0, 3.0, 20.0, math.sin(6.0), 5.0])
    assert L.ps_basis(x)[0] == pytest.approx([1.0, 1.0, 3.0, 20.0, math.sin(6.0)])


def test_probit_recovers_the_coefficients_and_solves_the_score_equations():
    rng = SplitMix64(3)
    n = 40000
    d = np.column_stack([np.ones(n), rng.normal(n), rng.normal(n)])
    true = np.array([0.2, 0.8, -0.5])
    t = (rng.uniform(n) < normal_cdf(d @ true)).astype(float)
    b = L.probit_fit(d, t)
    assert b == pytest.approx(true, abs=0.05)
    eta = d @ b
    p = normal_cdf(eta)
    phi = np.exp(-0.5 * eta ** 2) / math.sqrt(2 * math.pi)
    score = d.T @ (phi * (t - p) / (p * (1 - p)))
    assert np.abs(score).max() < 1e-3 * n ** 0.5 + 0.5


def test_clip_probability_bounds():
    out = L.clip_probability(np.array([0.0, 0.001, 0.5, 0.999, 1.0]))
    assert out.tolist() == [C.CLIP, C.CLIP, 0.5, 1 - C.CLIP, 1 - C.CLIP]


def test_parametric_learner_recovers_noise_free_linear_functions():
    rng = SplitMix64(4)
    n = 600
    x = rng.normal(n * C.P).reshape(n, C.P)
    t = (rng.uniform(n) < 0.5).astype(float)
    y = 1.0 + 2.0 * x[:, 0] + 3.0 * t + 0.5 * x[:, 1] * t
    pred = L.ParametricLearner(L.lin_features, L.lin_features).fit(x, t, y)
    m0, m1, e = pred(x)
    f = L.lin_features(x)
    # je Gruppe exakt anpassen, soweit das Modell reicht: Behandelte (y = 4 + 2 x₁ + 0,5 x₂), Kontrollen (y = 1 + 2 x₁)
    assert m0[t == 0] == pytest.approx(1.0 + 2.0 * x[t == 0, 0], abs=1e-9)
    assert m1[t == 1] == pytest.approx(4.0 + 2.0 * x[t == 1, 0] + 0.5 * x[t == 1, 1], abs=1e-9)
    assert (e >= C.CLIP).all() and (e <= 1 - C.CLIP).all() and f.shape[1] == 6


def test_basis_learner_has_the_right_propensity_in_a_large_sample():
    w = W.World(1.0, 0.9)
    x, t, y = (a[0] for a in W.simulate(w, 30000, SplitMix64(5)))
    m0, m1, e = L.get_learner("Richtige Basis").fit(x, t, y)(x)
    _, _, et = W.true_nuisances(x, w)
    assert np.sqrt(np.mean((e - np.clip(et, C.CLIP, 1 - C.CLIP)) ** 2)) < 0.01
    assert np.sqrt(np.mean((m1 - m0 - C.TAU) ** 2)) < 0.05
    lin = L.get_learner("Linear").fit(x, t, y)(x)
    assert np.sqrt(np.mean((lin[2] - et) ** 2)) > 0.05                                   # das lineare Modell trifft die Zuteilung nicht


def test_fit_regression_of_the_parametric_learner():
    rng = SplitMix64(6)
    n = 300
    x = rng.normal(n * C.P).reshape(n, C.P)
    z = 1.0 + 2.0 * x[:, 2]
    assert L.ParametricLearner(L.lin_features, L.lin_features).fit_regression(x, z, "y")(x) == pytest.approx(z, abs=1e-9)
    assert L.ParametricLearner(L.outcome_basis, L.ps_basis).fit_regression(x, 3.0 + x[:, 0], "t")(x) == pytest.approx(3.0 + x[:, 0], abs=1e-9)
    assert L.ParametricLearner(L.outcome_basis, L.ps_basis).fit_regression(x, 3.0 + x[:, 4], "y")(x) == pytest.approx(3.0 + x[:, 4], abs=1e-9)


def test_forest_learner_shapes_determinism_and_range():
    w = W.World(0.4, 0.9)
    x, t, y = (a[0] for a in W.simulate(w, 300, SplitMix64(7)))
    pred = L.ForestLearner().fit(x, t, y)
    m0, m1, e = pred(x[:50])
    assert m0.shape == m1.shape == e.shape == (50,)
    assert (e >= C.CLIP).all() and (e <= 1 - C.CLIP).all()
    assert m0.min() >= y[t == 0].min() - 1e-9 and m1.max() <= y[t == 1].max() + 1e-9
    again = L.ForestLearner().fit(x, t, y)(x[:50])
    assert all((a == b).all() for a, b in zip((m0, m1, e), again))


def test_overfitting_forest_memorises_the_training_data():
    w = W.World(0.4, 0.9)
    x, t, y = (a[0] for a in W.simulate(w, 400, SplitMix64(8)))
    err = {}
    for leaf in (C.LEAF, C.OVERFIT_LEAF):
        m0, m1, _ = L.ForestLearner(C.TREES, leaf).fit(x, t, y)(x)
        err[leaf] = np.sqrt(np.mean(((np.where(t == 1, m1, m0)) - y) ** 2))
    assert err[C.OVERFIT_LEAF] < 0.7 * err[C.LEAF]


def test_forest_fit_regression_and_names():
    rng = SplitMix64(9)
    n = 300
    x = rng.normal(n * C.P).reshape(n, C.P)
    z = x[:, 0] ** 2
    f = L.ForestLearner().fit_regression(x, z, "y")
    assert np.corrcoef(f(x), z)[0, 1] > 0.8
    assert L.get_learner("Zufallswald").name == "Zufallswald" and L.get_learner("Linear").name == "Linear" and L.get_learner("Richtige Basis").name == "Richtige Basis"
    with pytest.raises(ValueError):
        L.get_learner("Unbekannt")
