"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil); Wald-Anpassungen und Fließkomma-Reihenfolgen können je Bibliotheksversion die letzte
Stelle ändern, deshalb Toleranzen von einer halben Einheit der gedruckten Stelle. Die Wald-Läufe teilen sich den Zwischenspeicher von `forest_run`.
"""

import pytest

import dml_constants as C
import dml_evaluation as ev


@pytest.fixture(scope="module")
def dr():
    return ev.double_robust_experiment()


@pytest.fixture(scope="module")
def ortho():
    return ev.orthogonality_experiment()


@pytest.fixture(scope="module")
def cf():
    return ev.crossfit_experiment()


@pytest.fixture(scope="module")
def rate():
    return {(r["amp"], r["n"]): r["results"] for r in ev.rate_experiment()["rows"]}


def test_default_sample():
    a = ev.analyse(ev.Settings())
    by = {r.name: r for r in a.rows}
    assert by["Naiver Vergleich"].estimate == pytest.approx(1.28, abs=0.005) and by["Plug-in (Ergebnismodell)"].estimate == pytest.approx(0.85, abs=0.005)
    assert by["Gewichten (normiert)"].estimate == pytest.approx(1.03, abs=0.005) and by["AIPW ohne Kreuzanpassung"].estimate == pytest.approx(0.81, abs=0.005)
    assert by["AIPW mit Kreuzanpassung"].estimate == pytest.approx(0.60, abs=0.005) and by["Partiell lineares DML"].estimate == pytest.approx(0.70, abs=0.005)
    assert by["AIPW ohne Kreuzanpassung"].covers is False and by["AIPW mit Kreuzanpassung"].covers is True
    assert a.err_ps["in_sample"] == pytest.approx(0.16, abs=0.005) and a.err_ps["cross_fit"] == pytest.approx(0.13, abs=0.005)
    assert a.ess == pytest.approx(554, abs=1) and ev.verdict_instance(a)[0] == "success"


def test_experiment_1_double_robustness(dr):
    c = dr["combos"]
    assert dr["n"] == 500 and dr["reps"] == 500 and dr["amp"] == 1.0 and dr["naive"]["bias"] == pytest.approx(1.55, abs=0.005)
    both_wrong = c["Ergebnis falsch, Zuteilung falsch"]["aipw"]
    assert both_wrong["bias"] == pytest.approx(1.39, abs=0.005) and both_wrong["cover"] == 0.0
    out_right = c["Ergebnis richtig, Zuteilung falsch"]
    assert out_right["aipw"]["bias"] == pytest.approx(0.01, abs=0.005) and out_right["aipw"]["cover"] == pytest.approx(0.90, abs=0.005)
    assert out_right["ipw"]["bias"] == pytest.approx(1.38, abs=0.005)
    ps_right = c["Ergebnis falsch, Zuteilung richtig"]["aipw"]
    assert ps_right["bias"] == pytest.approx(0.19, abs=0.005) and ps_right["cover"] == pytest.approx(0.85, abs=0.005)
    both_right = c["Ergebnis richtig, Zuteilung richtig"]["aipw"]
    assert both_right["bias"] == pytest.approx(0.01, abs=0.005)


def test_experiment_1_price_of_the_insurance(dr):
    c = dr["combos"]
    assert c["Ergebnis richtig, Zuteilung richtig"]["aipw"]["sd"] == pytest.approx(0.151, abs=0.0006)
    assert c["Ergebnis richtig, Zuteilung richtig"]["plugin"]["sd"] == pytest.approx(0.115, abs=0.0006)
    assert dr["ess"] == pytest.approx(245, abs=1)
    assert c["Ergebnis falsch, Zuteilung richtig"]["aipw"]["sd"] == pytest.approx(0.286, abs=0.0006)
    assert c["Ergebnis falsch, Zuteilung richtig"]["aipw"]["sd"] > c["Ergebnis richtig, Zuteilung falsch"]["plugin"]["sd"]


def test_experiment_2_orthogonality(ortho):
    rows = {(r["b1"], r["b2"]): r for r in ortho["rows"]}
    base = rows[(0.0, 0.0)]["aipw"]
    assert ortho["n"] == 200000
    assert rows[(0.4, 0.0)]["plugin"] == pytest.approx(0.40, abs=0.005) and rows[(0.4, 0.0)]["aipw"] - base == pytest.approx(-0.002, abs=0.0005)
    d2, d4 = rows[(0.2, 0.2)]["aipw"] - base, rows[(0.4, 0.4)]["aipw"] - base
    assert d2 == pytest.approx(0.0045, abs=0.00005) and d4 == pytest.approx(0.0194, abs=0.00005) and d4 / d2 == pytest.approx(4.3, abs=0.05)
    assert rows[(0.4, 0.4)]["plugin"] == pytest.approx(2 * rows[(0.2, 0.2)]["plugin"], rel=0.001)


def test_experiment_3_cross_fitting(cf):
    r5, r10, over = cf["runs"][(C.LEAF, 500)], cf["runs"][(C.LEAF, 1000)], cf["runs"][(C.OVERFIT_LEAF, 500)]
    assert cf["reps"] == 16 and cf["amp"] == 0.4
    assert r5["aipw_plain"]["bias"] == pytest.approx(0.25, abs=0.005) and r5["aipw_plain"]["cover"] == pytest.approx(0.19, abs=0.005)
    assert r5["aipw_cf_5"]["bias"] == pytest.approx(0.13, abs=0.005) and r5["aipw_cf_5"]["cover"] == pytest.approx(0.94, abs=0.005)
    assert r10["aipw_plain"]["bias"] == pytest.approx(0.21, abs=0.005) and r10["aipw_plain"]["cover"] == 0.0
    assert r10["aipw_cf_5"]["bias"] == pytest.approx(0.05, abs=0.005) and r10["aipw_cf_5"]["cover"] == pytest.approx(0.94, abs=0.005)
    assert over["aipw_plain"]["bias"] == pytest.approx(0.25, abs=0.005) and over["aipw_plain"]["cover"] == pytest.approx(0.06, abs=0.005)
    assert over["aipw_cf_5"]["bias"] == pytest.approx(-0.34, abs=0.005) and over["aipw_cf_5"]["sd"] == pytest.approx(0.68, abs=0.005)


def test_experiment_4_mild_and_hard(rate):
    cf5, pi, lin, basis = "AIPW mit Kreuzanpassung (Wald)", "Plug-in (Wald)", "AIPW (linear)", "AIPW (richtige Basis)"
    assert [rate[(0.4, n)][cf5]["bias"] for n in (500, 1000, 2000)] == pytest.approx([0.13, 0.05, -0.01], abs=0.005)
    assert [rate[(0.4, n)][cf5]["rmse_sqrt_n"] for n in (500, 1000, 2000)] == pytest.approx([3.89, 3.85, 4.10], abs=0.005)
    assert [rate[(0.4, n)][pi]["bias"] for n in (500, 2000)] == pytest.approx([0.29, 0.17], abs=0.005)
    assert [rate[(0.4, n)][pi]["rmse_sqrt_n"] for n in (500, 1000, 2000)] == pytest.approx([6.85, 7.73, 7.73], abs=0.005)
    assert [rate[(0.4, n)][lin]["bias"] for n in (500, 1000, 2000)] == pytest.approx([0.31, 0.30, 0.28], abs=0.005)
    assert [rate[(0.4, n)][lin]["rmse_sqrt_n"] for n in (500, 2000)] == pytest.approx([7.28, 12.47], abs=0.005)
    assert [rate[(0.4, n)][basis]["bias"] for n in (500, 1000, 2000)] == pytest.approx([0.01, 0.02, -0.01], abs=0.005)
    assert [rate[(0.4, n)][basis]["rmse_sqrt_n"] for n in (500, 1000, 2000)] == pytest.approx([2.68, 2.99, 2.13], abs=0.005)
    assert [rate[(1.0, n)][cf5]["bias"] for n in (500, 1000)] == pytest.approx([0.51, 0.39], abs=0.005)
    assert [rate[(1.0, n)][cf5]["cover"] for n in (500, 1000)] == pytest.approx([0.25, 0.31], abs=0.005)
    assert [rate[(1.0, n)][pi]["bias"] for n in (500, 1000)] == pytest.approx([0.93, 0.78], abs=0.005)
    assert [rate[(1.0, n)][lin]["bias"] for n in (500, 1000)] == pytest.approx([1.41, 1.35], abs=0.005)
    assert [rate[(1.0, n)][basis]["bias"] for n in (500, 1000)] == pytest.approx([0.01, 0.01], abs=0.005)
