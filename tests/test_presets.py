"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import dml_constants as C
import dml_evaluation as ev
import dml_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            v = p[key]
            if spec.choices is not None:
                assert v in spec.choices, (name, key)
                continue
            assert spec.lo <= v <= spec.hi, (name, key)
            assert P.snap(v, spec) == pytest.approx(v), (name, key)


def test_first_preset_is_the_default_setting():
    p = C.PRESETS["Milde Nichtlinearität (Voreinstellung)"]
    assert ev.Settings() == ev.Settings(p["amp"], p["gamma"], p["n"], p["learner"], p["k"], p["seed"])
    assert all(P.SETTING_SPECS[sk].default == p[k] for k, sk in P.PRESET_KEYS.items())


def test_presets_differ_in_their_settings():
    seen = {tuple(sorted(p.items())) for p in C.PRESETS.values()}
    assert len(seen) == len(C.PRESETS)
    assert C.PRESETS["Richtige Basis"]["learner"] == "Richtige Basis" and C.PRESETS["Linear (beide Modelle falsch)"]["learner"] == "Linear"
    assert C.PRESETS["Schwere Nichtlinearität"]["amp"] > C.DEFAULT_AMP and C.PRESETS["Knappe Überlappung"]["gamma"] == C.GAMMA_MAX


def test_snap_clamps_and_rounds_to_the_grid():
    spec = P.SETTING_SPECS["amp_slider"]
    assert P.snap(5.0, spec) == C.AMP_MAX and P.snap(-1.0, spec) == C.AMP_MIN and P.snap(0.43, spec) == pytest.approx(0.4)
    gamma = P.SETTING_SPECS["gamma_slider"]
    assert P.snap(0.33, gamma) == pytest.approx(0.3) and P.snap(0.36, gamma) == pytest.approx(0.4)
    n = P.SETTING_SPECS["n_slider"]
    assert P.snap(50, n) == C.N_MIN and P.snap(10 ** 6, n) == C.N_MAX and P.snap(1049, n) == 1000 and P.snap(1051, n) == 1100
    assert P.snap(999999, P.SETTING_SPECS["seed_input"]) == 999999 and P.snap(7, P.SETTING_SPECS["k_slider"]) == 7


def test_bounds_defaults_and_params():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("k_slider") == (C.K_MIN, C.K_MAX)
    assert C.DEFAULT_N in range(C.N_MIN, C.N_MAX + 1, C.N_STEP) and C.K_MIN <= C.DEFAULT_K <= C.K_MAX and C.DEFAULT_LEARNER in C.LEARNERS
    assert {s.url_param for s in P.SETTING_SPECS.values()} == {"a", "g", "n", "l", "k", "seed"}


def test_experiment_constants_are_consistent():
    assert C.EXP_DR_REPS >= 200 and C.EXP_ORTHO_N >= 100000 and C.EXP_FOREST_REPS >= 12 and C.CLIP < 0.01 and C.OVERFIT_LEAF < C.LEAF
    assert set(C.EXP_RATE_RUNS) >= {(C.EXP_CF_AMP, n) for n in C.EXP_CF_NS}                      # Experiment 3 und 4 teilen die Läufe des milden Vehikels
    assert all(a in (C.EXP_CF_AMP, C.EXP_DR_AMP) for a, _ in C.EXP_RATE_RUNS) and C.EXP_CF_AMP < C.EXP_DR_AMP
