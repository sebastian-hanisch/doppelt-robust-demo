"""Gemeinsamer Strukturmodell-Generator der Linie "Kausale Diagramme und Adjustierung" (WORTGLEICHE Kopie in allen Repos der Linie, kein Modul-Präfix).

Version 1 (2026-10-14). Nur numpy und math, kein Streamlit, kein scipy. Änderungen an diesem Modul betreffen alle Repos der Linie: Schnittstelle,
Namen und Reihenfolge des Zufallsverbrauchs nur ändern, wenn alle Kopien nachgezogen werden (tools/portfolio_audit.py, SHARED_GROUPS).

Modell: ein lineares Strukturmodell über Knoten in topologischer Reihenfolge (Eltern haben einen kleineren Index). Jeder stetige Knoten ist
    wert = Achsenabschnitt + Summe(Koeffizient * Elternwert) + Streuung * Rauschen,
jeder binäre Knoten eine Probit-Schwelle:
    wert = 1  genau dann, wenn  Achsenabschnitt + Summe(Koeffizient * Elternwert) + Rauschen > 0   (Rauschen standardnormal).
Das Rauschen der stetigen Knoten ist `noise_kind` (gauss, laplace, uniform; alle mit Varianz 1). Die do-Wahrheit ist im linearen Fall
geschlossen (Summe der Pfadprodukte, `total_effect`); `sample(..., do={...})` rechnet dieselbe Intervention als Simulation, mit demselben
Zufallsstrom (gekoppelte Läufe), sodass beide Wege unabhängig voneinander prüfbar sind.

Zufall nur über einen übergebenen `SplitMix64`. Verbrauch je Aufruf in fester Reihenfolge: für jeden Knoten (in Index-Reihenfolge) ein Block
Rauschen der Länge reps * n, auch für einen Knoten, der per `do` festgehalten wird. Plattformhinweis: Normalverteilte laufen über `log` und `cos`
der C-Bibliothek und können sich zwischen Plattformen in der letzten Stelle unterscheiden; Tests dürfen Float-Werte daher nur mit Toleranz prüfen.
"""

import math
from dataclasses import dataclass

import numpy as np

GENERATOR_VERSION = 1
NOISE_KINDS = ("gauss", "laplace", "uniform")

_GAMMA = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_ERF = np.frompyfunc(math.erf, 1, 1)


class SplitMix64:
    """SplitMix64-Strom (Steele, Lea, Flood 2014). Ausgabe k hängt nur von (Seed, k) ab: plattform- und versionsunabhängig."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        """Die nächsten `size` 64-Bit-Werte; der Strom rückt um `size` Schritte weiter."""
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilte Werte in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormalverteilte Werte (Box-Muller, nur der Kosinus-Zweig); verbraucht 2 * size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)


# ----------------------------------------------------------------------------- Normalverteilung ohne scipy

def normal_cdf(z) -> np.ndarray:
    """Standardnormal-Verteilungsfunktion über math.erf."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.asarray(_ERF(z / math.sqrt(2.0)), dtype=float))


def normal_pdf(z) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)


def normal_ppf(p) -> np.ndarray:
    """Quantilfunktion der Standardnormalverteilung (Acklam-Näherung, ein Halley-Schritt über math.erf; p in (0, 1))."""
    p = np.asarray(p, dtype=float)
    x = np.zeros_like(p)
    low = p < 0.02425
    high = p > 1.0 - 0.02425
    mid = ~(low | high)
    q = np.sqrt(-2.0 * np.log(np.where(low, p, 1.0 - p)))
    tail = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    x = np.where(low, tail, np.where(high, -tail, x))
    r = (p - 0.5) ** 2
    q0 = p - 0.5
    centre = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q0 / (
        ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    x = np.where(mid, centre, x)
    e = normal_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * np.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


# ----------------------------------------------------------------------------- Strukturmodell

@dataclass(frozen=True)
class Scm:
    """Lineares Strukturmodell. `parents[i]` und `coefs[i]` gehören zusammen; Eltern haben einen kleineren Index als der Knoten."""

    names: tuple
    parents: tuple
    coefs: tuple
    intercepts: tuple
    noise_sd: tuple
    binary: tuple
    noise_kind: str = "gauss"

    def __post_init__(self):
        k = len(self.names)
        if not (len(self.parents) == len(self.coefs) == len(self.intercepts) == len(self.noise_sd) == len(self.binary) == k):
            raise ValueError("Längen von names, parents, coefs, intercepts, noise_sd und binary müssen übereinstimmen")
        if self.noise_kind not in NOISE_KINDS:
            raise ValueError(f"noise_kind muss eines von {NOISE_KINDS} sein")
        for i in range(k):
            if len(self.parents[i]) != len(self.coefs[i]):
                raise ValueError(f"Knoten {self.names[i]}: parents und coefs verschieden lang")
            if any(p >= i or p < 0 for p in self.parents[i]):
                raise ValueError(f"Knoten {self.names[i]}: Eltern müssen vor dem Knoten stehen (topologische Reihenfolge)")

    def index(self, name: str) -> int:
        return self.names.index(name)


def _noise_block(kind: str, rng: SplitMix64, size: int) -> np.ndarray:
    """Ein Block Rauschen mit Mittel 0 und Varianz 1."""
    if kind == "gauss":
        return rng.normal(size)
    u = rng.uniform(size)
    if kind == "uniform":
        return (u - 0.5) * math.sqrt(12.0)
    centred = u - 0.5                                      # Laplace mit Varianz 1: Skala b = 1 / sqrt(2)
    return -(1.0 / math.sqrt(2.0)) * np.sign(centred) * np.log1p(-2.0 * np.abs(centred))


def sample(scm: Scm, n: int, rng: SplitMix64, reps: int = 1, do: dict | None = None) -> dict:
    """Stichprobe: Dict Knotenname -> Array (reps, n). `do` hält Knoten auf einem festen Wert (Intervention); ihr Rauschblock wird dennoch gezogen,
    damit zwei Aufrufe mit gleichem Seed gekoppelte Läufe sind."""
    do = do or {}
    unknown = set(do) - set(scm.names)
    if unknown:
        raise ValueError(f"unbekannte Knoten in do: {sorted(unknown)}")
    out = np.empty((len(scm.names), reps, n))
    for i, name in enumerate(scm.names):
        kind = "gauss" if scm.binary[i] else scm.noise_kind
        noise = _noise_block(kind, rng, reps * n).reshape(reps, n)
        if name in do:
            out[i] = float(do[name])
            continue
        mean = scm.intercepts[i] + sum(c * out[p] for p, c in zip(scm.parents[i], scm.coefs[i]))
        out[i] = (mean + noise > 0.0).astype(float) if scm.binary[i] else mean + scm.noise_sd[i] * noise
    return {name: out[i] for i, name in enumerate(scm.names)}


def coefficient_matrix(scm: Scm) -> np.ndarray:
    """B[kind, eltern] = Koeffizient der Kante eltern -> kind."""
    k = len(scm.names)
    mat = np.zeros((k, k))
    for child in range(k):
        for parent, coef in zip(scm.parents[child], scm.coefs[child]):
            mat[child, parent] = coef
    return mat


def _descendants(scm: Scm, start: int) -> set:
    found, frontier = set(), {start}
    while frontier:
        nxt = {c for c in range(len(scm.names)) if any(p in frontier for p in scm.parents[c])} - found
        found |= nxt
        frontier = nxt
    return found


def total_effect(scm: Scm, treat: str, outcome: str) -> float:
    """Geschlossene do-Wirkung: Erhöhung von `treat` um 1 verändert den Erwartungswert von `outcome` um die Summe der Pfadprodukte.
    Gilt, wenn auf keinem Pfad von `treat` nach `outcome` ein anderer binärer Knoten liegt (Probit-Schwelle ist nicht linear); sonst ValueError."""
    i, j = scm.index(treat), scm.index(outcome)
    k = len(scm.names)
    between = {m for m in _descendants(scm, i) if j in _descendants(scm, m)}      # Zwischenknoten auf Pfaden von i nach j (ohne i und j selbst)
    if any(scm.binary[m] for m in between):
        raise ValueError("binärer Zwischenknoten auf dem Pfad: keine geschlossene Form")
    total = np.linalg.inv(np.eye(k) - coefficient_matrix(scm)) - np.eye(k)
    return float(total[j, i])


# ----------------------------------------------------------------------------- Vorlagen

STANDORTE = ("Schwankung", "Faktor", "Servicegrad")


def standorte_confounded(gamma: float, a: float, beta: float, tau: float, sigma: float, base: float = 80.0, coin_p: float | None = None) -> Scm:
    """Diagramm A: Schwankung -> Faktor, Schwankung -> Servicegrad, Faktor -> Servicegrad.

    Die Schwankung des Standorts ist standardnormal. Der hohe Faktor wird gewählt, wenn a + gamma * Schwankung + Rauschen > 0; der Servicegrad ist
    base - beta * Schwankung + tau * Faktor + sigma * Rauschen. Mit `coin_p` entscheidet stattdessen eine Münze mit dieser Wahrscheinlichkeit."""
    if coin_p is None:
        parents_x, coefs_x, a_x = (0,), (gamma,), a
    else:
        if not 0.0 < coin_p < 1.0:
            raise ValueError("coin_p muss in (0, 1) liegen")
        parents_x, coefs_x, a_x = (), (), float(normal_ppf(np.array([coin_p]))[0])
    return Scm(
        names=STANDORTE,
        parents=((), parents_x, (0, 1)),
        coefs=((), coefs_x, (-beta, tau)),
        intercepts=(0.0, a_x, base),
        noise_sd=(1.0, 1.0, sigma),
        binary=(False, True, False),
    )
