"""Schätzer: jede Einheit an den Handrechnungen aus conftest.py und an kleinen Beispielen."""

import numpy as np
import pytest

import dml_constants as C
import dml_estimators as E
import dml_world as W
from conftest import MeanLearner, OracleLearner
from kausal_generator import SplitMix64


def test_naive_by_hand(four_units):
    t, y, *_ = four_units
    est, se = E.naive(t, y)
    assert est == pytest.approx(1.5) and se == pytest.approx(np.sqrt(1.25))


def test_plugin_and_hajek_by_hand(four_units):
    t, y, m0, m1, e = four_units
    assert E.or_plugin(m0, m1) == pytest.approx(1.0)
    assert E.hajek(t, y, e) == pytest.approx(1.5)


def test_aipw_by_hand(four_units):
    t, y, m0, m1, e = four_units
    assert E.psi_aipw(m0, m1, e, t, y).tolist() == pytest.approx([3.0, -1.0, 1.0, 3.0])
    est, se = E.aipw(m0, m1, e, t, y)
    assert est == pytest.approx(1.5) and se == pytest.approx(np.sqrt(11.0 / 3.0) / 2.0)


def test_aipw_with_the_true_outcome_model_ignores_the_propensity_model():
    """Doppelte Robustheit in einer Zeile: Mit wahren m₀, m₁ ist der Score unabhängig von e im Mittel; mit falschem e bleibt die Schätzung nahe der Wirkung."""
    w = W.World(0.4, 0.9)
    x, t, y = (a[0] for a in W.simulate(w, 40000, SplitMix64(3)))
    m0, m1, e = W.true_nuisances(x, w)
    wrong_e = np.clip(0.5 + 0.0 * e, C.CLIP, 1 - C.CLIP)
    assert E.aipw(m0, m1, wrong_e, t, y)[0] == pytest.approx(C.TAU, abs=0.03)
    assert E.hajek(t, y, wrong_e) != pytest.approx(C.TAU, abs=0.2)                     # das Gewichten allein trifft dann nicht


def test_aipw_with_the_true_propensity_ignores_the_outcome_model():
    w = W.World(0.4, 0.9)
    x, t, y = (a[0] for a in W.simulate(w, 60000, SplitMix64(4)))
    _, _, e = W.true_nuisances(x, w)
    zeros = np.zeros(len(y))
    assert E.aipw(zeros, zeros, e, t, y)[0] == pytest.approx(C.TAU, abs=0.06)         # m ≡ 0: Gewichten mit der wahren Zuteilung
    assert E.or_plugin(zeros, zeros) == 0.0


def test_fold_labels_are_a_balanced_deterministic_partition():
    a = E.fold_labels(10, 3, SplitMix64(1))
    assert sorted(np.bincount(a)) == [3, 3, 4] and set(a) == {0, 1, 2}
    assert (a == E.fold_labels(10, 3, SplitMix64(1))).all()
    assert not (a == E.fold_labels(10, 3, SplitMix64(2))).all()
    assert (np.bincount(E.fold_labels(100, 5, SplitMix64(3))) == 20).all()


def test_cross_fit_uses_models_that_did_not_see_the_observation():
    rng = SplitMix64(5)
    n, k = 30, 3
    x = rng.normal(n * C.P).reshape(n, C.P)
    t = (np.arange(n) % 2).astype(float)
    y = np.arange(n, dtype=float)
    learner = MeanLearner()
    m0, m1, e = E.cross_fit(learner, x, t, y, k, SplitMix64(9))
    labels = E.fold_labels(n, k, SplitMix64(9))
    for i in range(n):
        keep = labels != labels[i]
        assert m0[i] == pytest.approx(y[keep & (t == 0)].mean()) and m1[i] == pytest.approx(y[keep & (t == 1)].mean())
    assert learner.calls == [20, 20, 20]


def test_in_sample_fits_once_on_everything():
    learner = MeanLearner()
    x = np.zeros((6, C.P))
    t = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    y = np.array([4.0, 1.0, 6.0, 3.0, 8.0, 2.0])
    m0, m1, e = E.in_sample(learner, x, t, y)
    assert learner.calls == [6] and m0.tolist() == [2.0] * 6 and m1.tolist() == [6.0] * 6 and e.tolist() == [0.5] * 6


def test_partially_linear_with_the_true_residuals_recovers_the_effect():
    w = W.World(0.4, 0.9)
    x, t, y = (a[0] for a in W.simulate(w, 30000, SplitMix64(6)))
    theta, se = E.partially_linear(OracleLearner(w, C.TAU), x, t, y, 4, SplitMix64(7))
    assert theta == pytest.approx(C.TAU, abs=0.03) and 0 < se < 0.03


def test_estimate_all_names_and_structure():
    w = W.World(0.4, 0.9)
    x, t, y = (a[0] for a in W.simulate(w, 400, SplitMix64(8)))
    res = E.estimate_all(OracleLearner(w, C.TAU), x, t, y, 3, SplitMix64(9))
    assert tuple(res["estimates"]) == E.NAMES
    assert res["estimates"]["Plug-in (Ergebnismodell)"][1] is None and res["estimates"]["Gewichten (normiert)"][1] is None
    assert all(res["estimates"][name][1] > 0 for name in E.NAMES if name not in ("Plug-in (Ergebnismodell)", "Gewichten (normiert)"))
    assert all(len(v) == 400 for trio in (res["in_sample"], res["cross_fit"]) for v in trio)
