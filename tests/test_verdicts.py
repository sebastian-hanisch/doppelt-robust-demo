"""Verdikte an handgesetzten Zeilen: jede Schwelle und jede Zahl im Text."""

import numpy as np
import pytest

import dml_constants as C
import dml_estimators as E
import dml_evaluation as ev


def analysis(errors: dict, ess=600.0, n=1000, covers=True):
    rows = []
    for name in E.NAMES:
        err = errors.get(name, 0.0)
        has_se = name not in ("Plug-in (Ergebnismodell)", "Gewichten (normiert)")
        est = C.TAU + err
        rows.append(ev.Row(name, est, 0.1 if has_se else None, est - 0.2 if has_se else None, est + 0.2 if has_se else None, err, covers if has_se else None))
    return ev.Analysis(ev.Settings(n=n, k=5), rows, C.TAU, ess, 0.03, n // 2, np.zeros(n), np.full(n, 0.5), {"in_sample": 0.1, "cross_fit": 0.12}, {"in_sample": 0.3, "cross_fit": 0.35})


def test_verdict_instance_text_and_good_case():
    a = analysis({"Naiver Vergleich": 0.86, "AIPW ohne Kreuzanpassung": 0.51, "AIPW mit Kreuzanpassung": 0.05})
    kind, text = ev.verdict_instance(a)
    assert kind == "success"
    assert text == ("Der naive Vergleich liegt um +0,86 daneben, der AIPW ohne Kreuzanpassung um +0,51, mit Kreuzanpassung (K = 5) um +0,05; das Intervall mit Kreuzanpassung "
                    "deckt die Wahrheit 0,50.")


def test_verdict_instance_thresholds_are_strict():
    assert ev.verdict_instance(analysis({"AIPW mit Kreuzanpassung": 0.2}))[0] == "success"
    kind, text = ev.verdict_instance(analysis({"AIPW mit Kreuzanpassung": 0.2001}))
    assert kind == "warning" and "Lernverfahren trifft die Störfunktionen hier nicht gut genug" in text
    assert ev.verdict_instance(analysis({"AIPW mit Kreuzanpassung": -0.25}))[0] == "warning"
    assert ev.verdict_instance(analysis({}, ess=400.0))[0] == "success"                      # genau 0,4 n: noch kein Hinweis
    kind, text = ev.verdict_instance(analysis({}, ess=399.0))
    assert kind == "warning" and "knapp: effektive Stichprobe 399 von 1000" in text
    assert "verfehlt die Wahrheit" in ev.verdict_instance(analysis({}, covers=False))[1]


def test_verdict_double_robust_numbers():
    def cell(bias, cover=0.9):
        return {"bias": bias, "cover": cover}

    res = {"n": 500, "combos": {"Ergebnis falsch, Zuteilung falsch": {"aipw": cell(1.387, 0.0)}, "Ergebnis richtig, Zuteilung falsch": {"aipw": cell(0.009)},
                                "Ergebnis falsch, Zuteilung richtig": {"aipw": cell(0.186)}}}
    kind, text = ev.verdict_double_robust(res)
    assert kind == "info" and "um +1,39 daneben (Deckung 0 %)" in text and "trifft er (+0,009)" in text and "bei n = 500 eine Restverzerrung von +0,19" in text


def test_verdict_orthogonality_numbers():
    rows = [{"b1": 0.0, "b2": 0.0, "plugin": 0.0, "ipw": 0.0, "aipw": -0.004}, {"b1": 0.4, "b2": 0.0, "plugin": 0.4012, "ipw": 0.0, "aipw": -0.006},
            {"b1": 0.4, "b2": 0.4, "plugin": 0.4012, "ipw": 0.0, "aipw": 0.0154}]
    kind, text = ev.verdict_orthogonality({"rows": rows})
    assert kind == "info" and "verschiebt sich das Plug-in um +0,40, der AIPW um −0,0020." in text and "bewegt er sich (+0,0194)" in text


def test_verdict_crossfit_numbers():
    def run(bias_p, cover_p, bias_c, cover_c, sd_p=0.05, sd_c=0.2):
        return {"aipw_plain": {"bias": bias_p, "cover": cover_p, "sd": sd_p}, "aipw_cf_5": {"bias": bias_c, "cover": cover_c, "sd": sd_c}}

    res = {"runs": {(C.LEAF, 500): run(0.28, 0.02, 0.16, 0.88), (C.LEAF, 1000): run(0.2, 0.12, 0.02, 0.92), (C.OVERFIT_LEAF, 500): run(0.77, 0.0, 0.05, 0.76, 0.14, 0.63)}}
    kind, text = ev.verdict_crossfit(res)
    assert kind == "info" and "(n = 1000) deckt das Intervall ohne Kreuzanpassung in 12 % der Wiederholungen, mit K = 5 in 92 %" in text
    assert "von +0,20 auf +0,02" in text and "von +0,77 auf +0,05, die Streuung steigt von 0,14 auf 0,63" in text


def test_verdict_rate_numbers():
    def row(amp, n, bias):
        return {"amp": amp, "n": n, "results": {"AIPW mit Kreuzanpassung (Wald)": {"bias": bias}}}

    res = {"rows": [row(0.4, 500, 0.16), row(0.4, 2000, 0.02), row(1.0, 300, 0.5), row(1.0, 4000, 0.3), row(1.0, 1000, 0.45)]}
    kind, text = ev.verdict_rate(res)
    assert kind == "info"
    assert text == ("Im milden Vehikel (Nichtlinearität 0,4) geht die Verzerrung des AIPW mit Kreuzanpassung von +0,16 (n = 500) auf +0,02 (n = 2000); im schweren (1,0) bleibt sie "
                    "bei +0,30 (n = 4000).")
