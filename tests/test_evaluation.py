"""Auswertung: Einstellungen, Analyse einer Stichprobe, Experimente (feste Ströme), Formatierung."""

import numpy as np
import pytest

import dml_constants as C
import dml_estimators as E
import dml_evaluation as ev
import dml_world as W


def test_number_format():
    assert ev.de(1.5, 1) == "1,5" and ev.de(-0.25) == "−0,25" and ev.signed(0.3) == "+0,30" and ev.signed(-0.3) == "−0,30" and ev.signed(0.0) == "+0,00"


def test_settings_world():
    s = ev.Settings(amp=0.7, gamma=1.1)
    assert s.world == W.World(0.7, 1.1) and ev.Settings().world == W.World(C.DEFAULT_AMP, C.DEFAULT_GAMMA)


def test_analysis_structure_and_consistency():
    a = ev.analyse(ev.Settings(n=300, k=3))
    assert [r.name for r in a.rows] == list(E.NAMES) and a.truth == C.TAU
    for r in a.rows:
        assert r.error == pytest.approx(r.estimate - C.TAU)
        if r.se is None:
            assert r.lo is None and r.hi is None and r.covers is None
        else:
            assert r.lo == pytest.approx(r.estimate - C.Z_CRIT * r.se) and r.hi == pytest.approx(r.estimate + C.Z_CRIT * r.se)
            assert r.covers == (r.lo <= C.TAU <= r.hi)
    assert [r.se is None for r in a.rows] == [False, True, True, False, False, False]
    assert a.treated_n == int(a.t.sum()) and len(a.e_cross) == 300 and a.ess > 0 and 0 <= a.outside <= 1
    assert set(a.err_ps) == set(a.err_out) == {"in_sample", "cross_fit"} and all(v > 0 for v in list(a.err_ps.values()) + list(a.err_out.values()))


def test_analysis_is_deterministic_and_the_seed_matters():
    a, b = ev.analyse(ev.Settings(n=250, k=3, learner="Linear")), ev.analyse(ev.Settings(n=250, k=3, learner="Linear"))
    assert [r.estimate for r in a.rows] == [r.estimate for r in b.rows]
    assert ev.analyse(ev.Settings(n=250, k=3, learner="Linear", seed=36)).rows[0].estimate != a.rows[0].estimate


def test_basis_learner_trifft_und_lineares_nicht():
    basis = ev.analyse(ev.Settings(amp=1.0, n=3000, learner="Richtige Basis", k=4))
    lin = ev.analyse(ev.Settings(amp=1.0, n=3000, learner="Linear", k=4))
    by_b, by_l = {r.name: r for r in basis.rows}, {r.name: r for r in lin.rows}
    assert abs(by_b["AIPW ohne Kreuzanpassung"].error) < 0.1 and by_b["AIPW ohne Kreuzanpassung"].covers
    assert abs(by_l["AIPW ohne Kreuzanpassung"].error) > 0.5 and not by_l["AIPW ohne Kreuzanpassung"].covers
    assert basis.err_ps["in_sample"] < 0.03 < lin.err_ps["in_sample"]


def test_analysis_at_the_slider_edges_does_not_raise():
    for kw in (dict(n=C.N_MIN), dict(amp=C.AMP_MAX, gamma=C.GAMMA_MAX), dict(amp=C.AMP_MIN, gamma=C.GAMMA_MIN), dict(k=C.K_MAX, n=C.N_MIN * 2), dict(k=C.K_MIN)):
        a = ev.analyse(ev.Settings(**{"learner": "Linear", **kw}))
        assert np.isfinite([r.estimate for r in a.rows]).all()


def test_summarize():
    s = ev.summarize([0.4, 0.6, 0.5, 0.5], [True, False, True, True])
    assert s["n"] == 4 and s["mean"] == pytest.approx(0.5) and s["bias"] == pytest.approx(0.0, abs=1e-12) and s["sd"] == pytest.approx(np.sqrt(0.005)) and s["cover"] == 0.75
    assert s["rmse"] == pytest.approx(np.sqrt(0.005)) and "cover" not in ev.summarize([1.0])


def test_double_robust_experiment_small():
    res = ev.double_robust_experiment(n=300, reps=24)
    assert list(res["combos"]) == [c[0] for c in ev.DR_COMBOS] and res["n"] == 300 and res["reps"] == 24 and res["naive"]["bias"] > 0.8 and res["ess"] > 50
    c = res["combos"]
    assert abs(c["Ergebnis richtig, Zuteilung falsch"]["aipw"]["bias"]) < 0.25 < abs(c["Ergebnis falsch, Zuteilung falsch"]["aipw"]["bias"])
    assert c["Ergebnis richtig, Zuteilung falsch"]["plugin"]["bias"] == pytest.approx(c["Ergebnis richtig, Zuteilung richtig"]["plugin"]["bias"])      # Plug-in ignoriert die Zuteilung
    assert c["Ergebnis falsch, Zuteilung richtig"]["ipw"]["bias"] == pytest.approx(c["Ergebnis richtig, Zuteilung richtig"]["ipw"]["bias"])            # Gewichten ignoriert das Ergebnismodell
    assert all("cover" in v["aipw"] and "cover" not in v["plugin"] for v in c.values())
    assert all(v[m]["n"] == 24 for v in c.values() for m in ("plugin", "ipw", "aipw"))
    assert c["Ergebnis richtig, Zuteilung richtig"]["aipw"]["cover"] > 0.8 and c["Ergebnis falsch, Zuteilung falsch"]["aipw"]["cover"] < 0.2


def test_orthogonality_experiment_small():
    res = ev.orthogonality_experiment(n=40000)
    rows = {(r["b1"], r["b2"]): r for r in res["rows"]}
    assert len(res["rows"]) == 10 and rows[(0.0, 0.0)]["plugin"] == 0.0
    assert rows[(0.4, 0.0)]["plugin"] == pytest.approx(0.4, abs=0.01) and rows[(0.0, 0.4)]["plugin"] == 0.0
    base = rows[(0.0, 0.0)]["aipw"]
    assert abs(rows[(0.4, 0.0)]["aipw"] - base) < 0.02 and abs(rows[(0.0, 0.4)]["aipw"] - base) < 0.02
    assert rows[(0.4, 0.4)]["aipw"] - base > 0.01
    assert abs(base) < 0.05 and abs(rows[(0.0, 0.0)]["ipw"]) < 0.3                           # absolute Lage: die Abweichung vom ATE, nicht von seinem Doppelten
    assert rows[(0.0, 0.4)]["ipw"] != rows[(0.0, 0.0)]["ipw"] and rows[(0.4, 0.0)]["ipw"] == rows[(0.0, 0.0)]["ipw"]


def test_run_learner_structure():
    from dml_learners import get_learner

    res = ev.run_learner(get_learner("Linear"), W.World(0.4, 0.9), 300, 6, 11, which=("plugin", "aipw_plain", "aipw_cf_3", "plm"))
    assert set(res) == {"plugin", "aipw_plain", "aipw_cf_3", "plm"} and "cover" not in res["plugin"] and all("cover" in res[k] for k in ("aipw_plain", "aipw_cf_3", "plm"))
    assert res == ev.run_learner(get_learner("Linear"), W.World(0.4, 0.9), 300, 6, 11, which=("plugin", "aipw_plain", "aipw_cf_3", "plm"))
    assert all(v["n"] == 6 for v in res.values())
    other = ev.run_learner(get_learner("Linear"), W.World(0.4, 0.9), 300, 6, 11, which=("aipw_cf_5",))["aipw_cf_5"]
    assert other != res["aipw_cf_3"] and other["mean"] != res["aipw_cf_3"]["mean"]                  # die Faltenzahl steckt im Namen


def test_forest_run_is_cached_and_deterministic():
    a = ev.forest_run(C.LEAF, 0.4, 200, 2)
    assert a is ev.forest_run(C.LEAF, 0.4, 200, 2) and set(a) == set(ev.FOREST_WHICH) and "cover" in a["aipw_cf_5"]
    assert ev.forest_run.cache_info().hits >= 1


def test_crossfit_and_rate_experiments_small():
    cf = ev.crossfit_experiment(reps=2)
    assert set(cf["runs"]) == {(C.LEAF, 500), (C.LEAF, 1000), (C.OVERFIT_LEAF, 500)} and cf["reps"] == 2 and cf["amp"] == C.EXP_CF_AMP
    rt = ev.rate_experiment(reps=2)
    assert [(r["amp"], r["n"]) for r in rt["rows"]] == list(C.EXP_RATE_RUNS)
    for r in rt["rows"]:
        assert list(r["results"]) == ["Plug-in (Wald)", "AIPW ohne Kreuzanpassung (Wald)", "AIPW mit Kreuzanpassung (Wald)", "AIPW (linear)", "AIPW (richtige Basis)"]
        assert all(v["rmse_sqrt_n"] == pytest.approx(v["rmse"] * np.sqrt(r["n"])) for v in r["results"].values())
    shared = [r for r in rt["rows"] if (r["amp"], r["n"]) == (0.4, 500)][0]["results"]["AIPW mit Kreuzanpassung (Wald)"]
    assert {k: v for k, v in shared.items() if k != "rmse_sqrt_n"} == cf["runs"][(C.LEAF, 500)]["aipw_cf_5"]                                # Experiment 3 und 4 teilen den Lauf


def test_run_learner_coverage_with_the_right_basis():
    from dml_learners import get_learner

    res = ev.run_learner(get_learner("Richtige Basis"), W.World(0.4, 0.9), 800, 12, 21, which=("aipw_plain", "aipw_cf_3", "plm"))
    assert all(res[k]["cover"] >= 0.6 for k in res) and all(abs(res[k]["bias"]) < 0.1 for k in res)


def test_run_learner_uses_the_fold_count_from_the_name_and_the_default_for_the_partially_linear_model():
    from conftest import MeanLearner

    learner = MeanLearner()
    ev.run_learner(learner, W.World(0.4, 0.9), 60, 2, 1, which=("aipw_cf_3",))
    assert learner.calls == [40] * 3 * 2
    learner = MeanLearner()
    ev.run_learner(learner, W.World(0.4, 0.9), 60, 2, 1, which=("aipw_cf_4",))
    assert learner.calls == [45] * 4 * 2
    learner = MeanLearner()
    ev.run_learner(learner, W.World(0.4, 0.9), 60, 1, 1, which=("plm",), k=6)
    assert learner.calls == [50] * 6 * 2                                                      # je Falte zwei Regressionen (Y und T)
