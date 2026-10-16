"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio). Alle Regler sind in jeder Einstellung wirksam und werden immer
gezeichnet; es gibt weder KEPT-Regler noch berechnete Grenzen."""

import random
from dataclasses import dataclass

import streamlit as st

import dml_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "amp_slider": SettingSpec("a", float, C.DEFAULT_AMP, C.AMP_MIN, C.AMP_MAX, C.AMP_STEP),
    "gamma_slider": SettingSpec("g", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX, C.GAMMA_STEP),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX, C.N_STEP),
    "learner_select": SettingSpec("l", str, C.DEFAULT_LEARNER, choices=C.LEARNERS),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX, 1),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, 1),
}
PRESET_KEYS = {"amp": "amp_slider", "gamma": "gamma_slider", "n": "n_slider", "learner": "learner_select", "k": "k_slider", "seed": "seed_input"}


def snap(value, spec: SettingSpec):
    """Auf das Raster des Reglers legen und in die Grenzen klemmen."""
    value = max(spec.lo, min(spec.hi, value))
    if spec.step and spec.caster is float:
        value = round(round((value - spec.lo) / spec.step) * spec.step + spec.lo, 6)
    elif spec.step and spec.caster is int and spec.step > 1:
        value = int(round((value - spec.lo) / spec.step) * spec.step + spec.lo)
    return value


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    init_session_state_defaults()
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            if spec.choices is not None:
                value = qp[spec.url_param]
                if value not in spec.choices:
                    continue
            else:
                value = snap(spec.caster(float(qp[spec.url_param])) if spec.caster is int else spec.caster(qp[spec.url_param]), spec)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values: dict):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
