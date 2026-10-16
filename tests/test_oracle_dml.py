"""Orakel: unabhängige Rechenwege (Schleifen, Normalgleichungen, scipy, direkter Aufruf von scikit-learn)."""

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

import dml_constants as C
import dml_estimators as E
import dml_evaluation as ev
import dml_learners as L
import dml_world as W
from kausal_generator import SplitMix64


def sample(n=300, seed=3, w=W.World(0.8, 0.9)):
    x, t, y = W.simulate(w, n, SplitMix64(seed))
    return x[0], t[0], y[0]


def test_aipw_against_a_loop():
    x, t, y = sample()
    m0, m1, e = L.get_learner("Richtige Basis").fit(x, t, y)(x)
    psi = [m1[i] - m0[i] + (t[i] * (y[i] - m1[i]) / e[i] if t[i] == 1 else 0.0) - ((y[i] - m0[i]) / (1 - e[i]) if t[i] == 0 else 0.0) for i in range(len(y))]
    est, se = E.aipw(m0, m1, e, t, y)
    assert est == pytest.approx(np.mean(psi)) and se == pytest.approx(np.std(psi, ddof=1) / np.sqrt(len(y)))


def test_cross_fit_against_explicit_train_test_splits():
    x, t, y = sample(240, 5)
    k = 4
    labels = E.fold_labels(len(y), k, SplitMix64(13))
    learner = L.get_learner("Richtige Basis")
    m0, m1, e = E.cross_fit(learner, x, t, y, k, SplitMix64(13))
    for f in range(k):
        train, test = labels != f, labels == f
        p0, p1, pe = learner.fit(x[train], t[train], y[train])(x[test])
        assert m0[test] == pytest.approx(p0) and m1[test] == pytest.approx(p1) and e[test] == pytest.approx(pe)


def test_partially_linear_against_the_residual_formula():
    x, t, y = sample(400, 6)
    k = 5
    labels = E.fold_labels(len(y), k, SplitMix64(17))
    learner = L.get_learner("Linear")
    ry, rt = np.empty(len(y)), np.empty(len(y))
    for f in range(k):
        train, test = labels != f, labels == f
        for z, res, kind in ((y, ry, "y"), (t, rt, "t")):
            d = L.lin_features(x[train])
            coef = np.linalg.solve(d.T @ d, d.T @ z[train])
            res[test] = z[test] - L.lin_features(x[test]) @ coef
    theta = np.sum(rt * ry) / np.sum(rt * rt)
    got, se = E.partially_linear(learner, x, t, y, k, SplitMix64(17))
    assert got == pytest.approx(theta, rel=1e-8)
    psi = rt * (ry - theta * rt)
    assert se == pytest.approx(np.sqrt(np.mean(psi ** 2) / np.mean(rt * rt) ** 2 / len(y)), rel=1e-8)


def test_probit_against_scipy_maximum_likelihood():
    x, t, y = sample(800, 7)
    d = L.ps_basis(x)

    def nll(b):
        eta = d @ b
        return -np.sum(t * norm.logcdf(eta) + (1 - t) * norm.logcdf(-eta))

    ref = minimize(nll, np.zeros(d.shape[1]), method="BFGS", options={"gtol": 1e-9}).x
    assert L.probit_fit(d, t) == pytest.approx(ref, abs=1e-4)


def test_forest_learner_against_direct_scikit_learn():
    x, t, y = sample(250, 8)
    m0, m1, e = L.ForestLearner().fit(x, t, y)(x)
    rf = RandomForestRegressor(C.TREES, min_samples_leaf=C.LEAF, max_features=C.MAX_FEATURES, random_state=0).fit(x[t == 1], y[t == 1])
    assert m1 == pytest.approx(rf.predict(x))
    pf = RandomForestRegressor(C.TREES, min_samples_leaf=C.LEAF, max_features=C.MAX_FEATURES, random_state=0).fit(x, t)
    assert e == pytest.approx(np.clip(pf.predict(x), C.CLIP, 1 - C.CLIP))


def test_orthogonality_second_order_scaling():
    """Doppelte Störung in beiden Störfunktionen: der AIPW-Zuwachs wächst um etwa den Faktor vier, das Plug-in um zwei."""
    res = ev.orthogonality_experiment(n=120000, grid=((0.2, 0.2), (0.4, 0.4), (0.4, 0.0)))
    rows = {(r["b1"], r["b2"]): r for r in res["rows"]}
    base = rows[(0.0, 0.0)]["aipw"]
    d1, d2 = rows[(0.2, 0.2)]["aipw"] - base, rows[(0.4, 0.4)]["aipw"] - base
    assert d1 > 0 and 3.0 < d2 / d1 < 6.0
    assert rows[(0.4, 0.4)]["plugin"] == pytest.approx(2 * rows[(0.2, 0.2)]["plugin"], rel=0.02)
    assert abs(rows[(0.4, 0.0)]["aipw"] - base) < 0.15 * d2                                     # eine Störung allein: weit unter dem Produktterm


def test_double_robustness_in_a_large_sample_for_both_one_sided_cases():
    w = W.World(1.0, 0.9)
    x, t, y = (a[0] for a in W.simulate(w, 60000, SplitMix64(9)))
    right_out_wrong_ps = L.ParametricLearner(L.outcome_basis, L.lin_features).fit(x, t, y)(x)
    wrong_out_right_ps = L.ParametricLearner(L.lin_features, L.ps_basis).fit(x, t, y)(x)
    both_wrong = L.get_learner("Linear").fit(x, t, y)(x)
    assert E.aipw(*right_out_wrong_ps, t, y)[0] == pytest.approx(C.TAU, abs=0.03)
    assert E.aipw(*wrong_out_right_ps, t, y)[0] == pytest.approx(C.TAU, abs=0.12)           # Gewichte sind unruhig: weiter Toleranzrahmen
    assert abs(E.aipw(*both_wrong, t, y)[0] - C.TAU) > 0.5
