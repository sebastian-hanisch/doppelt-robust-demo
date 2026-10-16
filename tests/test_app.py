"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt die Regler an die Grenzen, startet die schweren Experimente auf Knopfdruck und
prüft, dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import dml_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 900


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_the_truth():
    at = _fresh_app()
    m = _metrics(at)
    assert m["Wahre Wirkung"] == "0,50" and m["Naiver Vergleich"].startswith("+1,")
    assert at.slider(key="amp_slider").value == C.DEFAULT_AMP and at.selectbox(key="learner_select").value == C.DEFAULT_LEARNER and at.slider(key="k_slider").value == C.DEFAULT_K
    assert any("Der naive Vergleich liegt um" in s.value for s in list(at.success) + list(at.warning))


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if label.startswith("▶"):
            continue                                                         # die schweren Experimente laufen im eigenen Test
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_sliders_at_min_and_max_do_not_raise():
    for key in ("amp_slider", "gamma_slider", "k_slider", "n_slider"):
        for edge in ("min", "max"):
            at = _fresh_app()
            at.selectbox(key="learner_select").set_value("Linear").run(timeout=TIMEOUT)
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="amp_slider").value == preset["amp"] and at.slider(key="gamma_slider").value == preset["gamma"]
        assert at.selectbox(key="learner_select").value == preset["learner"] and at.slider(key="n_slider").value == preset["n"]


def test_the_basis_learner_hits_the_truth_and_the_linear_one_does_not():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Richtige Basis")].click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("deckt die Wahrheit" in s.value for s in list(at.success) + list(at.warning))
    at.button[[b.label for b in at.button].index("Linear (beide Modelle falsch)")].click().run(timeout=TIMEOUT)
    assert any("trifft die Störfunktionen hier nicht gut genug" in w.value for w in at.warning)


def test_the_heavy_experiments_run_on_click():
    at = _fresh_app()
    assert "crossfit_ready" not in at.session_state and "rate_ready" not in at.session_state
    at.button(key="crossfit_run").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Beim Wald (n = 1000) deckt das Intervall ohne Kreuzanpassung" in i.value for i in at.info)
    at.button(key="rate_run").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Im milden Vehikel" in i.value for i in at.info)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["a"] = "9"
    at.query_params["g"] = "0.04"
    at.query_params["n"] = "50"
    at.query_params["k"] = "99"
    at.query_params["l"] = "Linear"
    at.query_params["seed"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="amp_slider").value == C.AMP_MAX and at.slider(key="gamma_slider").value == C.GAMMA_MIN
    assert at.slider(key="n_slider").value == C.N_MIN and at.slider(key="k_slider").value == C.K_MAX
    assert at.selectbox(key="learner_select").value == "Linear" and at.number_input(key="seed_input").value == C.DEFAULT_SEED


def test_unknown_learner_in_the_permalink_is_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["l"] = "Nirgendwo"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="learner_select").value == C.DEFAULT_LEARNER


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.slider(key="amp_slider").set_value(0.7).run(timeout=TIMEOUT)
    assert at.query_params["a"] in ("0.7", ["0.7"])
