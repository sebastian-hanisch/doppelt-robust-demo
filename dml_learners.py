"""Lernverfahren für die Störfunktionen m₀(X), m₁(X) (Ergebnis je Gruppe) und e(X) (Zuteilung): linear, „richtige Basis“ (die Merkmale der wahren Funktion) und
Zufallswald (scikit-learn). Jedes Lernverfahren hat zwei Schnittstellen: `fit(x, t, y)` liefert eine Vorhersage (m₀, m₁, e), `fit_regression(x, z, kind)` eine
Regression von z auf x (für das partiell lineare DML; kind „y“ oder „t“)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import dml_constants as C
from kausal_generator import normal_cdf


def lin_features(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def outcome_basis(x: np.ndarray) -> np.ndarray:
    """Genau die Merkmale von m₀(X): X₁, X₂² − 1, X₄X₅, sin 2X₃, X₅."""
    return np.column_stack([np.ones(len(x)), x[:, 0], x[:, 1] ** 2 - 1.0, x[:, 3] * x[:, 4], np.sin(2.0 * x[:, 2]), x[:, 4]])


def ps_basis(x: np.ndarray) -> np.ndarray:
    """Genau die Merkmale des Zuteilungsindex s(X): X₁, X₂² − 1, X₄X₅, sin 2X₃."""
    return np.column_stack([np.ones(len(x)), x[:, 0], x[:, 1] ** 2 - 1.0, x[:, 3] * x[:, 4], np.sin(2.0 * x[:, 2])])


def probit_fit(d: np.ndarray, t: np.ndarray, iters: int = 40) -> np.ndarray:
    """Probit-Regression von T auf die Merkmale d (Fisher-Scoring)."""
    b = np.zeros(d.shape[1])
    for _ in range(iters):
        eta = np.clip(d @ b, -8.0, 8.0)
        p = np.clip(normal_cdf(eta), 1e-8, 1.0 - 1e-8)
        phi = np.exp(-0.5 * eta ** 2) / np.sqrt(2.0 * np.pi)
        w = phi ** 2 / (p * (1.0 - p))
        z = eta + (t - p) / np.maximum(phi, 1e-12)
        step = np.linalg.solve(d.T @ (d * w[:, None]) + 1e-8 * np.eye(d.shape[1]), d.T @ (w * z)) - b
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def clip_probability(e: np.ndarray) -> np.ndarray:
    return np.clip(e, C.CLIP, 1.0 - C.CLIP)


class ParametricLearner:
    """Lineare Modelle auf festen Merkmalen: Ergebnis je Gruppe nach kleinsten Quadraten, Zuteilung als Probit."""

    def __init__(self, outcome_features, ps_features, name: str = "parametrisch"):
        self.outcome_features, self.ps_features, self.name = outcome_features, ps_features, name

    def fit(self, x, t, y):
        coefs = [np.linalg.lstsq(self.outcome_features(x[t == arm]), y[t == arm], rcond=None)[0] for arm in (0.0, 1.0)]
        bp = probit_fit(self.ps_features(x), t)

        def predict(xn):
            f = self.outcome_features(xn)
            return f @ coefs[0], f @ coefs[1], clip_probability(normal_cdf(self.ps_features(xn) @ bp))
        return predict

    def fit_regression(self, x, z, kind):
        feats = self.outcome_features if kind == "y" else self.ps_features
        coef = np.linalg.lstsq(feats(x), z, rcond=None)[0]
        return lambda xn: feats(xn) @ coef


class ForestLearner:
    """Zufallswald (scikit-learn, Regression) für beide Ergebnisfunktionen und die Zuteilung (Regression auf 0/1)."""

    def __init__(self, trees: int = C.TREES, leaf: int = C.LEAF, name: str = "Zufallswald"):
        self.trees, self.leaf, self.name = trees, leaf, name

    def _forest(self):
        return RandomForestRegressor(self.trees, min_samples_leaf=self.leaf, max_features=C.MAX_FEATURES, random_state=0)

    def fit(self, x, t, y):
        forests = []
        for arm in (0.0, 1.0):
            rf = self._forest()
            rf.fit(x[t == arm], y[t == arm])
            forests.append(rf)
        pf = self._forest()
        pf.fit(x, t)
        return lambda xn: (forests[0].predict(xn), forests[1].predict(xn), clip_probability(pf.predict(xn)))

    def fit_regression(self, x, z, kind):
        rf = self._forest()
        rf.fit(x, z)
        return rf.predict


def get_learner(name: str):
    if name == "Zufallswald":
        return ForestLearner()
    if name == "Linear":
        return ParametricLearner(lin_features, lin_features, "Linear")
    if name == "Richtige Basis":
        return ParametricLearner(outcome_basis, ps_basis, "Richtige Basis")
    raise ValueError(f"unbekanntes Lernverfahren: {name}")
