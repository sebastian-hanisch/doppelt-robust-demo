"""Vehikel: ein Standort wählt die neue Regel B (T = 1) abhängig von fünf Störfaktoren; Ergebnis Servicegrad Y. Konstante Wirkung 0,5 (der ATE ist exakt bekannt).
Nur numpy und der Generator der Linie, kein Streamlit.

Modell: X ~ N(0, I₅); P(T = 1 | X) = Φ(γ s(X)); Y = m₀(X) + 0,5 T + ε.
  s(X)  = 0,7 X₁ + A (0,5 (X₂² − 1) + 0,5 X₄ X₅ + 0,5 sin 2X₃)
  m₀(X) = 0,5 X₁ + A (0,8 (X₂² − 1) + 0,8 X₄ X₅ + 0,6 sin 2X₃) + 0,5 X₅
A (Stärke der Nichtlinearität) ist der Schwierigkeitsregler; die nichtlinearen Terme stehen in beiden Funktionen und konfundieren daher."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dml_constants as C
from kausal_generator import SplitMix64, normal_cdf


@dataclass(frozen=True)
class World:
    amp: float = C.DEFAULT_AMP
    gamma: float = C.DEFAULT_GAMMA


def score(x: np.ndarray, w: World) -> np.ndarray:
    """Index der Zuteilung s(X); x hat die Form (..., 5)."""
    return 0.7 * x[..., 0] + w.amp * (0.5 * (x[..., 1] ** 2 - 1.0) + 0.5 * x[..., 3] * x[..., 4] + 0.5 * np.sin(2.0 * x[..., 2]))


def baseline(x: np.ndarray, w: World) -> np.ndarray:
    """m₀(X) = E[Y | X, T = 0]."""
    return (0.5 * x[..., 0] + w.amp * (0.8 * (x[..., 1] ** 2 - 1.0) + 0.8 * x[..., 3] * x[..., 4] + 0.6 * np.sin(2.0 * x[..., 2])) + 0.5 * x[..., 4])


def propensity(x: np.ndarray, w: World) -> np.ndarray:
    """Wahre Zuteilungswahrscheinlichkeit e(X) = Φ(γ s(X))."""
    return normal_cdf(w.gamma * score(x, w))


def true_nuisances(x: np.ndarray, w: World) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(m₀, m₁, e): die wahren Störfunktionen."""
    m0 = baseline(x, w)
    return m0, m0 + C.TAU, propensity(x, w)


def simulate(w: World, n: int, rng: SplitMix64, reps: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, t, y) mit Formen (reps, n, 5), (reps, n), (reps, n): erst alle Störfaktoren, dann die Zuteilungs-Gleichverteilten, dann das Rauschen aus einem Strom."""
    x = rng.normal(reps * n * C.P).reshape(reps, n, C.P)
    u = rng.uniform(reps * n).reshape(reps, n)
    eps = rng.normal(reps * n).reshape(reps, n)
    t = (u < propensity(x, w)).astype(float)
    y = baseline(x, w) + C.TAU * t + eps
    return x, t, y


def effective_sample_size(t: np.ndarray, e: np.ndarray) -> float:
    """(Σw)² / Σw² der Gewichte 1/e (Behandelte) und 1/(1 − e) (Kontrollen) mit der wahren Zuteilung."""
    e = np.clip(e, C.CLIP, 1.0 - C.CLIP)
    w = np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
    return float(w.sum() ** 2 / (w ** 2).sum())
