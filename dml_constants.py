"""Konstanten: Regler-Grenzen, Lernverfahren, feste Szenarien und Experimente, Farben und Presets."""

# Rechenaufwand am Maximum: n = 3 000, Zufallswald (30 Bäume) mit Kreuzanpassung K = 10: wenige Sekunden (zwischengespeichert).
N_MIN, N_MAX, N_STEP, DEFAULT_N = 200, 3000, 100, 1000
AMP_MIN, AMP_MAX, AMP_STEP, DEFAULT_AMP = 0.0, 1.2, 0.1, 0.4             # Stärke der Nichtlinearität (Schwierigkeit)
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.3, 1.5, 0.1, 0.9     # Zuteilungsstärke
K_MIN, K_MAX, DEFAULT_K = 2, 10, 5                                       # Falten der Kreuzanpassung
SEED_MAX, DEFAULT_SEED = 999999, 35
LEARNERS = ("Zufallswald", "Linear", "Richtige Basis")
DEFAULT_LEARNER = LEARNERS[0]
Z_CRIT = 1.959964

TAU = 0.5                                    # wahre (konstante) Wirkung
P = 5                                        # Anzahl der Störfaktoren
CLIP = 0.005                                 # Zuteilungswahrscheinlichkeit auf [CLIP, 1 - CLIP] begrenzt
TREES, LEAF, MAX_FEATURES = 30, 5, 0.7       # Zufallswald
OVERFIT_LEAF = 1                             # überanpassender Wald (Blattgröße 1, volle Tiefe)

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 970000
EXP_DR_N, EXP_DR_REPS, EXP_DR_AMP = 500, 500, 1.0                        # Experiment 1: 2 x 2 der doppelten Robustheit (parametrisch, schnell)
EXP_ORTHO_N = 200000                                                     # Experiment 2: Population, gestörte wahre Störfunktionen
EXP_ORTHO_GRID = ((0.1, 0.0), (0.2, 0.0), (0.4, 0.0), (0.0, 0.1), (0.0, 0.2), (0.0, 0.4), (0.1, 0.1), (0.2, 0.2), (0.4, 0.4))
EXP_FOREST_REPS = 16                                                     # Experiment 3 und 4: Wiederholungen je Wald-Lauf (je Lauf 18 Waldanpassungen)
EXP_CF_AMP, EXP_CF_NS, EXP_CF_OVERFIT_N = 0.4, (500, 1000), 500          # Experiment 3: Kreuzanpassung (Wald, milde Nichtlinearität)
EXP_RATE_RUNS = ((0.4, 500), (0.4, 1000), (0.4, 2000), (1.0, 500), (1.0, 1000))   # Experiment 4: (Nichtlinearität, n)

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_TREATED = "#4c78a8"
COLOR_CONTROL = "#8a8a8a"
COLOR_NEUTRAL = "#8a8a8a"


def preset(**kw) -> dict:
    base = dict(amp=DEFAULT_AMP, gamma=DEFAULT_GAMMA, n=DEFAULT_N, learner=DEFAULT_LEARNER, k=DEFAULT_K, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Milde Nichtlinearität (Voreinstellung)": preset(),
    "Schwere Nichtlinearität": preset(amp=1.0),
    "Richtige Basis": preset(amp=1.0, learner="Richtige Basis"),
    "Linear (beide Modelle falsch)": preset(amp=1.0, learner="Linear"),
    "Knappe Überlappung": preset(gamma=1.5),
}
PRESET_HELP = {
    "Milde Nichtlinearität (Voreinstellung)": "Schwache Nichtlinearität: Der Wald lernt die Störfunktionen gut genug, die Kreuzanpassung macht das Intervall brauchbar.",
    "Schwere Nichtlinearität": "Starke Nichtlinearität: Der Wald lernt die Störfunktionen schlecht, auch mit Kreuzanpassung bleibt eine Verzerrung.",
    "Richtige Basis": "Die Merkmale der Störfunktionen sind genau die richtigen: Beide Modelle stimmen, der doppelt robuste Schätzer trifft die Wirkung.",
    "Linear (beide Modelle falsch)": "Beide Modelle sind linear und falsch: Keine Konstruktion hilft, der Fehler bleibt bei jedem Schätzer.",
    "Knappe Überlappung": "Starke Zuteilung: Die Gewichte werden extrem, alle Schätzer mit Gewichten werden unruhig.",
}
