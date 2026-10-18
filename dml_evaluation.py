"""Auswertung: Einstellungen, eine Stichprobe mit allen Schätzern, die vier Experimente über feste Ströme und die Verdikte. Kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import dml_constants as C
import dml_estimators as E
import dml_learners as L
import dml_world as W
from kausal_generator import SplitMix64


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


@dataclass(frozen=True)
class Settings:
    amp: float = C.DEFAULT_AMP
    gamma: float = C.DEFAULT_GAMMA
    n: int = C.DEFAULT_N
    learner: str = C.DEFAULT_LEARNER
    k: int = C.DEFAULT_K
    seed: int = C.DEFAULT_SEED

    @property
    def world(self) -> W.World:
        return W.World(self.amp, self.gamma)


@dataclass
class Row:
    name: str
    estimate: float
    se: float | None
    lo: float | None
    hi: float | None
    error: float
    covers: bool | None


@dataclass
class Analysis:
    settings: Settings
    rows: list
    truth: float
    ess: float
    outside: float
    treated_n: int
    t: np.ndarray
    e_cross: np.ndarray        # geschätzte Zuteilung aus der Kreuzanpassung
    err_ps: dict               # Wurzel des mittleren quadratischen Fehlers der Zuteilung gegen die wahre: in der Stichprobe und kreuzangepasst
    err_out: dict              # dasselbe für die Ergebnisfunktionen (m₀ und m₁ zusammen)


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def analyse(s: Settings) -> Analysis:
    w = s.world
    x, t, y = (a[0] for a in W.simulate(w, s.n, SplitMix64(s.seed)))
    res = E.estimate_all(L.get_learner(s.learner), x, t, y, s.k, SplitMix64(s.seed + 1))
    rows = []
    for name in E.NAMES:
        est, se = res["estimates"][name]
        if se is None:
            rows.append(Row(name, est, None, None, None, est - C.TAU, None))
        else:
            lo, hi = est - C.Z_CRIT * se, est + C.Z_CRIT * se
            rows.append(Row(name, est, se, lo, hi, est - C.TAU, bool(lo <= C.TAU <= hi)))
    m0, m1, e = W.true_nuisances(x, w)
    (i0, i1, ie), (c0, c1, ce) = res["in_sample"], res["cross_fit"]
    e = np.clip(e, C.CLIP, 1.0 - C.CLIP)
    return Analysis(s, rows, C.TAU, W.effective_sample_size(t, e), float(np.mean((e < 0.05) | (e > 0.95))), int(t.sum()), t, ce,
                    {"in_sample": _rmse(ie, e), "cross_fit": _rmse(ce, e)},
                    {"in_sample": _rmse(np.concatenate([i0, i1]), np.concatenate([m0, m1])), "cross_fit": _rmse(np.concatenate([c0, c1]), np.concatenate([m0, m1]))})


def summarize(values, covers=None) -> dict:
    v = np.asarray(values, dtype=float)
    out = {"n": len(v), "mean": float(v.mean()), "bias": float(v.mean() - C.TAU), "sd": float(v.std()), "rmse": float(np.sqrt(((v - C.TAU) ** 2).mean()))}
    if covers is not None:
        out["cover"] = float(np.mean(covers))
    return out


# ----------------------------------------------------------------------------- Experiment 1: doppelte Robustheit

DR_COMBOS = (
    ("Ergebnis falsch, Zuteilung falsch", L.lin_features, L.lin_features),
    ("Ergebnis richtig, Zuteilung falsch", L.outcome_basis, L.lin_features),
    ("Ergebnis falsch, Zuteilung richtig", L.lin_features, L.ps_basis),
    ("Ergebnis richtig, Zuteilung richtig", L.outcome_basis, L.ps_basis),
)


def double_robust_experiment(n: int = C.EXP_DR_N, reps: int = C.EXP_DR_REPS, amp: float = C.EXP_DR_AMP) -> dict:
    """2 × 2: Ergebnis- und Zuteilungsmodell je richtig oder falsch; Plug-in, Gewichten und AIPW (ohne Kreuzanpassung, parametrische Modelle)."""
    w = W.World(amp, C.DEFAULT_GAMMA)
    x, t, y = W.simulate(w, n, SplitMix64(C.SEED_BASE + 1), reps)
    naive_vals, ess = [], []
    store = {name: {"plugin": [], "ipw": [], "aipw": [], "cover": []} for name, _, _ in DR_COMBOS}
    for r in range(reps):
        naive_vals.append(E.naive(t[r], y[r])[0])
        ess.append(W.effective_sample_size(t[r], W.propensity(x[r], w)))
        for name, fo, fp in DR_COMBOS:
            m0, m1, e = in_sample = L.ParametricLearner(fo, fp).fit(x[r], t[r], y[r])(x[r])
            est, se = E.aipw(m0, m1, e, t[r], y[r])
            store[name]["plugin"].append(E.or_plugin(m0, m1))
            store[name]["ipw"].append(E.hajek(t[r], y[r], e))
            store[name]["aipw"].append(est)
            store[name]["cover"].append(abs(est - C.TAU) <= C.Z_CRIT * se)
    return {"n": n, "reps": reps, "amp": amp, "naive": summarize(naive_vals), "ess": float(np.mean(ess)),
            "combos": {name: {"plugin": summarize(v["plugin"]), "ipw": summarize(v["ipw"]), "aipw": summarize(v["aipw"], v["cover"])} for name, v in store.items()}}


# ----------------------------------------------------------------------------- Experiment 2: Orthogonalität

def orthogonality_experiment(n: int = C.EXP_ORTHO_N, grid=C.EXP_ORTHO_GRID, amp: float = C.EXP_DR_AMP) -> dict:
    """Population (große Stichprobe, wahre Störfunktionen gestört): m₁ + b₁·h mit h = 1 + X₁ und e + b₂·k·min(e, 1 − e) mit k = 0,5·tanh X₁.
    Gemessen wird die Abweichung vom ATE für Plug-in, Gewichten und AIPW."""
    w = W.World(amp, C.DEFAULT_GAMMA)
    x, t, y = (a[0] for a in W.simulate(w, n, SplitMix64(C.SEED_BASE + 2)))
    m0, m1, e = W.true_nuisances(x, w)
    h = 1.0 + x[:, 0]
    k = 0.5 * np.tanh(x[:, 0])
    rows = []
    for b1, b2 in ((0.0, 0.0),) + tuple(grid):
        eh = np.clip(e + b2 * k * np.minimum(e, 1.0 - e), 1e-6, 1.0 - 1e-6)
        m1h = m1 + b1 * h
        rows.append({"b1": b1, "b2": b2, "plugin": E.or_plugin(m0, m1h) - C.TAU, "ipw": E.hajek(t, y, eh) - C.TAU, "aipw": E.aipw(m0, m1h, eh, t, y)[0] - C.TAU})
    return {"n": n, "amp": amp, "rows": rows}


# ----------------------------------------------------------------------------- Läufe mit festen Strömen

def run_learner(learner, w: W.World, n: int, reps: int, seed: int, k: int = C.DEFAULT_K, which=("plugin", "aipw_plain", "aipw_cf", "plm")) -> dict:
    """`reps` feste Stichproben: Plug-in, AIPW ohne und mit Kreuzanpassung (k Falten) und partiell lineares DML; je Schätzer Verzerrung, Streuung, RMSE und Deckung."""
    x, t, y = W.simulate(w, n, SplitMix64(seed), reps)
    rng = SplitMix64(seed + 1)
    vals = {name: [] for name in which}
    cov = {name: [] for name in which if name != "plugin"}
    for r in range(reps):
        if "plugin" in which or "aipw_plain" in which:
            m0, m1, e = E.in_sample(learner, x[r], t[r], y[r])
        if "plugin" in which:
            vals["plugin"].append(E.or_plugin(m0, m1))
        if "aipw_plain" in which:
            est, se = E.aipw(m0, m1, e, t[r], y[r])
            vals["aipw_plain"].append(est)
            cov["aipw_plain"].append(abs(est - C.TAU) <= C.Z_CRIT * se)
        for name in which:
            if name.startswith("aipw_cf"):
                kk = int(name.split("_")[2]) if name.count("_") == 2 else k
                est, se = E.aipw(*E.cross_fit(learner, x[r], t[r], y[r], kk, rng), t[r], y[r])
                vals[name].append(est)
                cov[name].append(abs(est - C.TAU) <= C.Z_CRIT * se)
        if "plm" in which:
            est, se = E.partially_linear(learner, x[r], t[r], y[r], k, rng)
            vals["plm"].append(est)
            cov["plm"].append(abs(est - C.TAU) <= C.Z_CRIT * se)
    return {name: summarize(vals[name], cov.get(name)) for name in which}


# ----------------------------------------------------------------------------- Experiment 3 und 4: Wald mit und ohne Kreuzanpassung

FOREST_WHICH = ("plugin", "aipw_plain", "aipw_cf_5")
CF_LABELS = {C.LEAF: f"Zufallswald (Blattgröße {C.LEAF})", C.OVERFIT_LEAF: f"überanpassender Wald (Blattgröße {C.OVERFIT_LEAF})"}


@lru_cache(maxsize=None)
def forest_run(leaf: int, amp: float, n: int, reps: int = C.EXP_FOREST_REPS) -> dict:
    """Ein fester Wald-Lauf (30 Bäume, Blattgröße `leaf`, Gewichtung γ der Voreinstellung): Plug-in, AIPW ohne Kreuzanpassung und mit K = 5; von Experiment 3 und 4 geteilt."""
    seed = C.SEED_BASE + 1000 * int(round(10 * amp)) + 100 * leaf + n // 100
    return run_learner(L.ForestLearner(C.TREES, leaf), W.World(amp, C.DEFAULT_GAMMA), n, reps, seed, which=FOREST_WHICH)


def crossfit_experiment(reps: int = C.EXP_FOREST_REPS) -> dict:
    """AIPW ohne Kreuzanpassung gegen K = 5: Wald mit Blattgröße 5 bei n = 500 und 1 000 und überanpassender Wald (Blattgröße 1) bei n = 500, im milden Vehikel."""
    runs = {(C.LEAF, n): forest_run(C.LEAF, C.EXP_CF_AMP, n, reps) for n in C.EXP_CF_NS}
    runs[(C.OVERFIT_LEAF, C.EXP_CF_OVERFIT_N)] = forest_run(C.OVERFIT_LEAF, C.EXP_CF_AMP, C.EXP_CF_OVERFIT_N, reps)
    return {"amp": C.EXP_CF_AMP, "reps": reps, "runs": runs}


def rate_experiment(reps: int = C.EXP_FOREST_REPS) -> dict:
    """Wald (Plug-in, AIPW mit K = 5) über n im milden und im schweren Vehikel; dazu die parametrischen AIPW (linear, richtige Basis) als Maßstab. RMSE und RMSE·√n."""
    rows = []
    for amp, n in C.EXP_RATE_RUNS:
        forest = forest_run(C.LEAF, amp, n, reps)
        w = W.World(amp, C.DEFAULT_GAMMA)
        seed = C.SEED_BASE + 5000 + 1000 * int(round(10 * amp)) + n // 100
        res = {"Plug-in (Wald)": forest["plugin"], "AIPW ohne Kreuzanpassung (Wald)": forest["aipw_plain"], "AIPW mit Kreuzanpassung (Wald)": forest["aipw_cf_5"],
               "AIPW (linear)": run_learner(L.get_learner("Linear"), w, n, reps, seed, which=("aipw_plain",))["aipw_plain"],
               "AIPW (richtige Basis)": run_learner(L.get_learner("Richtige Basis"), w, n, reps, seed, which=("aipw_plain",))["aipw_plain"]}
        rows.append({"amp": amp, "n": n, "results": {k: dict(v, rmse_sqrt_n=v["rmse"] * math.sqrt(n)) for k, v in res.items()}})
    return {"reps": reps, "rows": rows}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    by = {r.name: r for r in a.rows}
    plain, cf, naive = by["AIPW ohne Kreuzanpassung"], by["AIPW mit Kreuzanpassung"], by["Naiver Vergleich"]
    text = (f"Der naive Vergleich liegt um {signed(naive.error)} daneben, der AIPW ohne Kreuzanpassung um {signed(plain.error)}, mit Kreuzanpassung (K = {a.settings.k}) um "
            f"{signed(cf.error)}; das Intervall mit Kreuzanpassung {'deckt' if cf.covers else 'verfehlt'} die Wahrheit {de(a.truth)}.")
    if abs(cf.error) > 0.2:
        return "warning", text + " Das Lernverfahren trifft die Störfunktionen hier nicht gut genug."
    if a.ess < 0.4 * a.settings.n:
        return "warning", text + f" Die Überlappung ist knapp: effektive Stichprobe {de(a.ess, 0)} von {a.settings.n}."
    return "success", text


def verdict_double_robust(res: dict) -> tuple[str, str]:
    c = res["combos"]
    both_wrong = c["Ergebnis falsch, Zuteilung falsch"]["aipw"]
    out_right = c["Ergebnis richtig, Zuteilung falsch"]["aipw"]
    ps_right = c["Ergebnis falsch, Zuteilung richtig"]["aipw"]
    return "info", (f"Sind beide Modelle falsch, liegt der AIPW um {signed(both_wrong['bias'])} daneben (Deckung {de(100 * both_wrong['cover'], 0)} %). Ist nur das Ergebnismodell "
                    f"richtig, trifft er ({signed(out_right['bias'], 3)}); ist nur die Zuteilung richtig, bleibt bei n = {res['n']} eine Restverzerrung von {signed(ps_right['bias'])}, vor allem durch die Kappung der Zuteilung auf [0,005; 0,995] (eine kleinere Schranke senkt die Verzerrung und erhöht die Streuung).")


def verdict_orthogonality(res: dict) -> tuple[str, str]:
    rows = {(r["b1"], r["b2"]): r for r in res["rows"]}
    a, b = rows[(0.4, 0.0)], rows[(0.4, 0.4)]
    base = rows[(0.0, 0.0)]["aipw"]
    return "info", (f"Bei einer Störung von 0,4 im Ergebnismodell verschiebt sich das Plug-in um {signed(a['plugin'])}, der AIPW um {signed(a['aipw'] - base, 4)}. Erst wenn auch die "
                    f"Zuteilung gestört ist, bewegt er sich ({signed(b['aipw'] - base, 4)}): Der Fehler geht als Produkt ein.")


def verdict_crossfit(res: dict) -> tuple[str, str]:
    n = max(C.EXP_CF_NS)
    plain, cf = res["runs"][(C.LEAF, n)]["aipw_plain"], res["runs"][(C.LEAF, n)]["aipw_cf_5"]
    over = res["runs"][(C.OVERFIT_LEAF, C.EXP_CF_OVERFIT_N)]
    return "info", (f"Beim Wald (n = {n}) deckt das Intervall ohne Kreuzanpassung in {de(100 * plain['cover'], 0)} % der Wiederholungen, mit K = 5 in {de(100 * cf['cover'], 0)} %; "
                    f"die Verzerrung geht von {signed(plain['bias'])} auf {signed(cf['bias'])}. Beim überanpassenden Wald (n = {C.EXP_CF_OVERFIT_N}) von "
                    f"{signed(over['aipw_plain']['bias'])} auf {signed(over['aipw_cf_5']['bias'])}, die Streuung steigt von {de(over['aipw_plain']['sd'], 2)} auf {de(over['aipw_cf_5']['sd'], 2)}.")


def verdict_rate(res: dict) -> tuple[str, str]:
    by = {(r["amp"], r["n"]): r["results"]["AIPW mit Kreuzanpassung (Wald)"] for r in res["rows"]}
    amps = sorted({a for a, _ in by})
    n_lo, n_hi = min(n for a, n in by if a == amps[0]), max(n for a, n in by if a == amps[0])
    n_hard = max(n for a, n in by if a == amps[-1])
    return "info", (f"Im milden Vehikel (Nichtlinearität {de(amps[0], 1)}) geht die Verzerrung des AIPW mit Kreuzanpassung von {signed(by[(amps[0], n_lo)]['bias'])} (n = {n_lo}) auf "
                    f"{signed(by[(amps[0], n_hi)]['bias'])} (n = {n_hi}); im schweren ({de(amps[-1], 1)}) bleibt sie bei {signed(by[(amps[-1], n_hard)]['bias'])} (n = {n_hard}).")
