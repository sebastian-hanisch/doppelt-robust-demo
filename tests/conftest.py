"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def four_units():
    """Mini-Instanz 1: vier Einheiten, T = (1, 1, 0, 0), Y = (3, 1, 1, 0), konstante Störfunktionen m₁ = 2, m₀ = 1, e = 0,5.
    Handrechnung: naive Differenz 2 − 0,5 = 1,5, Welch-Standardfehler √(2/2 + 0,5/2) = √1,25; Plug-in 2 − 1 = 1;
    ψ = m₁ − m₀ + T (Y − m₁)/e − (1 − T)(Y − m₀)/(1 − e) = (1 + 2, 1 − 2, 1 − 0, 1 + 2) = (3, −1, 1, 3), Mittel 1,5 (Plug-in 1 plus Korrektur 0,5),
    Streuung (n − 1) 11/3, Standardfehler √(11/3)/2; normiertes Gewichten 1,5."""
    t = np.array([1.0, 1.0, 0.0, 0.0])
    y = np.array([3.0, 1.0, 1.0, 0.0])
    m1, m0, e = np.full(4, 2.0), np.full(4, 1.0), np.full(4, 0.5)
    return t, y, m0, m1, e


class OracleLearner:
    """Lernverfahren mit den wahren Störfunktionen (für Tests): kennt e(X) und E[Y | X] = g(X) + θ e(X)."""

    def __init__(self, world, theta):
        import dml_world as W

        self.world, self.theta, self.W = world, theta, W

    def fit(self, x, t, y):
        W = self.W
        return lambda xn: W.true_nuisances(xn, self.world)

    def fit_regression(self, x, z, kind):
        W = self.W
        if kind == "t":
            return lambda xn: W.propensity(xn, self.world)
        return lambda xn: W.baseline(xn, self.world) + self.theta * W.propensity(xn, self.world)


class MeanLearner:
    """Prüflernverfahren: sagt je Gruppe das Mittel der Trainingsdaten voraus und merkt sich die Größen der Trainingsmengen."""

    def __init__(self):
        self.calls = []

    def fit(self, x, t, y):
        self.calls.append(len(y))
        m0, m1 = y[t == 0].mean(), y[t == 1].mean()
        e = float(np.clip(t.mean(), 0.005, 0.995))
        return lambda xn: (np.full(len(xn), m0), np.full(len(xn), m1), np.full(len(xn), e))

    def fit_regression(self, x, z, kind):
        self.calls.append(len(z))
        mean = z.mean()
        return lambda xn: np.full(len(xn), mean)
