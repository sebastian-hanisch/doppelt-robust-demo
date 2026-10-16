"""Schätzer aus Störfunktionen: Plug-in, Gewichten, augmentiertes Gewichten (AIPW, doppelt robust) mit und ohne Kreuzanpassung, partiell lineares DML.
Nur numpy; die Lernverfahren kommen aus dml_learners. Jede Einheit eine reine Funktion."""

from __future__ import annotations

import numpy as np

from kausal_generator import SplitMix64

NAMES = (
    "Naiver Vergleich",
    "Plug-in (Ergebnismodell)",
    "Gewichten (normiert)",
    "AIPW ohne Kreuzanpassung",
    "AIPW mit Kreuzanpassung",
    "Partiell lineares DML",
)


def naive(t, y) -> tuple[float, float]:
    """Differenz der Gruppenmittel mit Welch-Standardfehler."""
    y1, y0 = y[t == 1], y[t == 0]
    return float(y1.mean() - y0.mean()), float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))


def or_plugin(m0, m1) -> float:
    """Plug-in: Mittel von m₁(X) − m₀(X)."""
    return float(np.mean(m1 - m0))


def hajek(t, y, e) -> float:
    """Normiertes Gewichten: gewichtete Gruppenmittel mit w = T/e bzw. (1 − T)/(1 − e)."""
    w1, w0 = t / e, (1.0 - t) / (1.0 - e)
    return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())


def psi_aipw(m0, m1, e, t, y) -> np.ndarray:
    """Einflusswert je Beobachtung: m₁ − m₀ + T (Y − m₁)/e − (1 − T)(Y − m₀)/(1 − e)."""
    return m1 - m0 + t * (y - m1) / e - (1.0 - t) * (y - m0) / (1.0 - e)


def aipw(m0, m1, e, t, y) -> tuple[float, float]:
    """AIPW-Schätzung (Mittel der Einflusswerte) und Einflussfunktions-Standardfehler (Streuung / √n)."""
    psi = psi_aipw(m0, m1, e, t, y)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))


def fold_labels(n: int, k: int, rng: SplitMix64) -> np.ndarray:
    """Zufällige Zerlegung in k gleich große Falten (±1): Falte je Beobachtung."""
    order = np.argsort(rng.uniform(n), kind="stable")
    labels = np.empty(n, dtype=int)
    labels[order] = np.arange(n) % k
    return labels


def cross_fit(learner, x, t, y, k: int, rng: SplitMix64):
    """k-fache Kreuzanpassung: Störfunktionen je Beobachtung aus einem Modell, das ihre Falte nicht gesehen hat."""
    labels = fold_labels(len(y), k, rng)
    m0, m1, e = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    for f in range(k):
        te = labels == f
        m0[te], m1[te], e[te] = learner.fit(x[~te], t[~te], y[~te])(x[te])
    return m0, m1, e


def in_sample(learner, x, t, y):
    """Störfunktionen aus einem Modell, das auf allen Beobachtungen angepasst und auf denselben ausgewertet wird."""
    return learner.fit(x, t, y)(x)


def partially_linear(learner, x, t, y, k: int, rng: SplitMix64) -> tuple[float, float]:
    """Partiell lineares DML (Robinson): Koeffizient der Regression der Y-Reste auf die T-Reste (Reste aus Kreuzanpassung) und Standardfehler aus dem orthogonalen Score."""
    n = len(y)
    labels = fold_labels(n, k, rng)
    ry, rt = np.empty(n), np.empty(n)
    for f in range(k):
        te = labels == f
        ry[te] = y[te] - learner.fit_regression(x[~te], y[~te], "y")(x[te])
        rt[te] = t[te] - learner.fit_regression(x[~te], t[~te], "t")(x[te])
    theta = float((rt * ry).sum() / (rt * rt).sum())
    psi = rt * (ry - theta * rt)
    return theta, float(np.sqrt((psi ** 2).mean() / (rt * rt).mean() ** 2 / n))


def estimate_all(learner, x, t, y, k: int, rng: SplitMix64) -> dict:
    """Alle Schätzer auf einer Stichprobe: Name -> (Schätzung, Standardfehler oder None); dazu die Störfunktionen für die Diagnose."""
    m0, m1, e = in_sample(learner, x, t, y)
    c0, c1, ce = cross_fit(learner, x, t, y, k, rng)
    out = {
        "Naiver Vergleich": naive(t, y),
        "Plug-in (Ergebnismodell)": (or_plugin(m0, m1), None),
        "Gewichten (normiert)": (hajek(t, y, e), None),
        "AIPW ohne Kreuzanpassung": aipw(m0, m1, e, t, y),
        "AIPW mit Kreuzanpassung": aipw(c0, c1, ce, t, y),
        "Partiell lineares DML": partially_linear(learner, x, t, y, k, rng),
    }
    return {"estimates": out, "in_sample": (m0, m1, e), "cross_fit": (c0, c1, ce)}
