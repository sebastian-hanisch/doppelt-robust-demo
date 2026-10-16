"""Doppelt robust und Double ML: Störfunktionen lernen, ohne dass ihr Fehler durchschlägt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Linie Kausale Diagramme und Adjustierung. Fünf Störfaktoren, nichtlineare Zuteilung und nichtlineares Ergebnis, konstante Wirkung 0,5. Gezeigt: das
Plug-in, das Gewichten und der doppelt robuste Schätzer (AIPW) ohne und mit Kreuzanpassung, dazu das partiell lineare DML. Vehikel: dml_world.py, Lernverfahren: dml_learners.py.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import dml_constants as C
from dml_evaluation import (
    Settings,
    analyse,
    crossfit_experiment,
    de,
    double_robust_experiment,
    orthogonality_experiment,
    rate_experiment,
    signed,
    verdict_crossfit,
    verdict_double_robust,
    verdict_instance,
    verdict_orthogonality,
    verdict_rate,
)
from dml_pdf_export import build_pdf
from dml_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from dml_visualization import (
    build_crossfit_chart,
    build_crossfit_table,
    build_diagram,
    build_double_robust_chart,
    build_double_robust_table,
    build_estimates_chart,
    build_estimates_table,
    build_nuisance_table,
    build_orthogonality_chart,
    build_orthogonality_table,
    build_overlap_histogram,
    build_rate_chart,
    build_rate_table,
    style_estimates_table,
)

st.set_page_config(page_title="Doppelt robust und Double ML – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"
HEAVY = "Der Wald wird je Wiederholung mehrfach angepasst, das dauert etwa eine bis zwei Minuten."


@st.cache_data(show_spinner="Lernverfahren und Kreuzanpassung rechnen …")
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _double_robust():
    return double_robust_experiment()


@st.cache_data(show_spinner=SPINNER)
def _orthogonality():
    return orthogonality_experiment()


@st.cache_data(show_spinner=HEAVY)
def _crossfit():
    return crossfit_experiment()


@st.cache_data(show_spinner=HEAVY)
def _rate():
    return rate_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🧩 Doppelt robust und Double ML")
st.markdown(
    """
**Zwei Modelle, von denen eines stimmen darf, und eine Kreuzanpassung, die das Lernverfahren auf Abstand hält.** Dies ist das siebte Stück der Linie **Kausale Diagramme und
Adjustierung**. Das sechste Stück hat gezeigt, dass jeder Einzelschätzer ein zweites Modell braucht: Regression das für das Ergebnis, Gewichten das für die Zuteilung. Der
**doppelt robuste Schätzer** (AIPW) nimmt beide und ist richtig, wenn **eines** stimmt. **Double ML** erlaubt, dass beide Modelle von einem flexiblen Lernverfahren kommen, und
rechnet den Fehler an anderen Daten aus, als auf denen das Lernverfahren lernte (**Kreuzanpassung**). Fünf Störfaktoren, nichtlineare Zuteilung und nichtlineares Ergebnis, die
Wirkung ist 0,5.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So liest man die Schätzer", expanded=True):
    st.markdown(
        """
1. **Plug-in.** Das Ergebnismodell sagt für jeden Standort voraus, was mit und ohne Regel B herauskäme; man mittelt die Differenz. Jeder Fehler des Modells geht **ungedämpft** ein.
2. **Gewichten.** Jede Einheit zählt mit dem Kehrwert ihrer Zuteilungswahrscheinlichkeit. Jeder Fehler des Zuteilungsmodells geht ungedämpft ein.
3. **AIPW (doppelt robust).** Plug-in plus eine gewichtete Korrektur mit den Resten des Ergebnismodells. Die Fehler der beiden Modelle gehen als **Produkt** ein: Ist eines richtig,
   verschwindet der Fehler; sind beide klein, ist das Produkt sehr klein.
4. **Kreuzanpassung.** Die Störfunktionen jeder Beobachtung kommen aus einem Modell, das sie nicht gesehen hat. Ohne sie nutzt das Lernverfahren dieselben Daten zweimal und das
   Intervall lügt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Vehikel**")
    amp = st.slider("Stärke der Nichtlinearität", *bounds("amp_slider"), step=C.AMP_STEP, key="amp_slider",
                    help="Skaliert die nichtlinearen Terme in Zuteilung und Ergebnis. Bei 0 sind beide Funktionen linear in X₁ und X₅; hohe Werte machen sie für ein Lernverfahren schwer.")
    gamma = st.slider("Zuteilungsstärke γ", *bounds("gamma_slider"), step=C.GAMMA_STEP, key="gamma_slider",
                      help="Wie stark die Störfaktoren die Wahl der Regel B bestimmen. Hohe Werte heißen knappe Überlappung.")
    st.markdown("**Lernverfahren und Kreuzanpassung**")
    learner = st.selectbox("Lernverfahren für beide Störfunktionen", C.LEARNERS, key="learner_select",
                           help="Zufallswald: 30 Bäume, Blattgröße 5. Linear: lineare Modelle (beide falsch). Richtige Basis: lineare Modelle auf genau den Merkmalen der wahren Funktionen (beide richtig).")
    k = st.slider("Falten K der Kreuzanpassung", *bounds("k_slider"), step=1, key="k_slider", help="Je mehr Falten, desto mehr Daten für jedes Modell, desto mehr Rechenzeit.")
    st.markdown("**Daten**")
    n = st.slider("Beobachtungen n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Stichprobengröße der simulierten Daten.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"amp_slider": round(float(amp), 1), "gamma_slider": round(float(gamma), 1), "n_slider": int(n), "learner_select": learner, "k_slider": int(k), "seed_input": int(seed)})

settings = Settings(round(float(amp), 1), round(float(gamma), 1), int(n), learner, int(k), int(seed))
a = _analysis(settings)
by = {r.name: r for r in a.rows}

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
c1.metric("Wahre Wirkung", de(a.truth, 2), delta="konstant, aus dem Modell", delta_color="off")
c2.metric("Naiver Vergleich", signed(by["Naiver Vergleich"].estimate, 3), delta=f"Fehler {signed(by['Naiver Vergleich'].error, 2)}", delta_color="off",
          help="Mittel der Behandelten minus Mittel der Kontrollen, ohne Adjustierung.")
cf = by["AIPW mit Kreuzanpassung"]
c3.metric("AIPW mit Kreuzanpassung", signed(cf.estimate, 3), delta=f"Fehler {signed(cf.error, 2)}, Intervall {de(cf.lo, 2)} bis {de(cf.hi, 2)}", delta_color="off",
          help="Doppelt robuster Schätzer mit den Störfunktionen aus der Kreuzanpassung (K Falten); Intervall aus der Einflussfunktion.")
c4.metric("Effektive Stichprobe", de(a.ess, 0), delta=f"von n = {settings.n}, {de(100 * a.outside, 0)} % mit e < 0,05 oder > 0,95", delta_color="off",
          help="(Σw)² / Σw² der wahren Gewichte: Je kleiner, desto weniger tragen die Daten für alle Schätzer mit Gewichten.")

left, right = st.columns([1, 1.25])
with left:
    st.plotly_chart(build_diagram(), width="stretch", key="diagram_chart")
    st.plotly_chart(build_overlap_histogram(a), width="stretch", key="overlap_chart")
with right:
    st.plotly_chart(build_estimates_chart(a), width="stretch", key="estimates_chart")
    st.caption("Punkt mit Intervall: grün, wenn es die Wahrheit deckt, rot, wenn nicht. Raute: Plug-in und Gewichten haben kein Intervall. Orange Linie: die Wahrheit.")
verdict(verdict_instance(a))
st.dataframe(style_estimates_table(build_estimates_table(a)), hide_index=True, width="stretch")
st.markdown("**Fehler der Störfunktionen gegen die wahren** (Wurzel des mittleren quadratischen Fehlers)")
st.dataframe(build_nuisance_table(a), hide_index=True, width="stretch")
st.caption("Der AIPW-Fehler hängt am **Produkt** der beiden Fehler, nicht an jedem einzeln. Kreuzangepasst sind die Fehler meist etwas größer, weil jedes Modell weniger Daten sieht; "
           "dafür sind sie ehrlich.")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="doppelt-robust.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Ein richtiges Modell von zwei genügt")
res1 = _double_robust()
st.caption(f"n = {res1['n']}, {res1['reps']} feste Wiederholungen, Nichtlinearität {de(res1['amp'], 1)}. Ergebnis- und Zuteilungsmodell sind je „richtig“ (lineare Modelle auf genau den Merkmalen "
           f"der wahren Funktion) oder „falsch“ (linear in X). Naive Verzerrung {signed(res1['naive']['bias'])}; effektive Stichprobe der wahren Gewichte {de(res1['ess'], 0)} von {res1['n']}.")
st.plotly_chart(build_double_robust_chart(res1), width="stretch", key="dr_chart")
verdict(verdict_double_robust(res1))
st.dataframe(build_double_robust_table(res1), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Warum der Fehler als Produkt eingeht")
res2 = _orthogonality()
n_pop = f"{res2['n']:,}".replace(",", ".")
st.caption(f"Population (n = {n_pop}), die wahren Störfunktionen werden gestört: Das Ergebnismodell der Behandelten um b₁ · (1 + X₁), die Zuteilung um b₂ · 0,5 · tanh(X₁) · "
           "min(e, 1 − e). Gezeigt ist, wie weit sich die Schätzung bewegt.")
st.plotly_chart(build_orthogonality_chart(res2), width="stretch", key="ortho_chart")
verdict(verdict_orthogonality(res2))
st.dataframe(build_orthogonality_table(res2), hide_index=True, width="stretch")
st.caption("Plug-in und Gewichten bewegen sich **linear** mit der Störung. Der AIPW bleibt bei einer Störung stehen und bewegt sich bei beiden **quadratisch** (Verdopplung der Störung, "
           "etwa vierfacher Fehler): Das ist die Neyman-Orthogonalität.")

st.subheader("🔬 Experiment 3: Was die Kreuzanpassung ausmacht")
st.caption(f"Zufallswald, milde Nichtlinearität ({de(C.EXP_CF_AMP, 1)}), {C.EXP_FOREST_REPS} feste Wiederholungen je Zeile. Dasselbe Lernverfahren, einmal auf denselben Daten ausgewertet, "
           "einmal mit K = 5. Der überanpassende Wald (Blattgröße 1) merkt sich die Daten. " + HEAVY)
if st.button("▶ Kreuzanpassung berechnen", key="crossfit_run"):
    st.session_state["crossfit_ready"] = True
if st.session_state.get("crossfit_ready"):
    res3 = _crossfit()
    st.plotly_chart(build_crossfit_chart(res3), width="stretch", key="crossfit_chart")
    verdict(verdict_crossfit(res3))
    st.dataframe(build_crossfit_table(res3), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Reicht das Lernverfahren?")
st.caption(f"Plug-in und AIPW mit dem Wald gegen die parametrischen AIPW über n, im milden und im schweren Vehikel, {C.EXP_FOREST_REPS} feste Wiederholungen je Zeile. RMSE·√n bleibt flach, "
           "wenn der Schätzer mit der Rate 1/√n konvergiert. " + HEAVY)
if st.button("▶ Lernverfahren und Rate berechnen", key="rate_run"):
    st.session_state["rate_ready"] = True
if st.session_state.get("rate_ready"):
    res4 = _rate()
    st.plotly_chart(build_rate_chart(res4), width="stretch", key="rate_chart")
    verdict(verdict_rate(res4))
    st.dataframe(build_rate_table(res4), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Alle Störfaktoren sind beobachtet** | Ein verborgener Störfaktor verzerrt auch den doppelt robusten Schätzer; Kreuzanpassung ändert daran nichts. | Frontdoor, Instrumente, Sensitivität und Falsifikation |
| **Eines der beiden Modelle stimmt oder beide sind gut genug** | Sind beide falsch oder zu schlecht gelernt, bleibt eine Restverzerrung (Experiment 1, Experiment 4 im schweren Vehikel). | Wie falsch darf das Diagramm sein? |
| **Überlappung: die Gewichte bleiben zahm** | Bei knapper Überlappung wird die Korrektur unruhig, und der doppelt robuste Schätzer ist ungenauer als das richtige Ergebnismodell allein (Experiment 1). | Schätzen (Stück 6) |
| **Das Lernverfahren ist gut genug** | Die Theorie verlangt, dass das Produkt der Fehler schneller als 1/√n fällt; ein Wald mit 30 Bäumen im schweren Vehikel erfüllt das nicht (Experiment 4). | nicht gebaut (Abstimmung der Lernverfahren) |
| **Konstante Wirkung (partiell lineares DML)** | Das Robinson-Verfahren schätzt dann eine gewichtete Wirkung; die Demo hat nur eine konstante. | Heterogene Wirkungen sind nicht gebaut |
| **n ist groß genug für die Falten** | Bei kleinem n sieht jedes Modell zu wenige Daten; die Intervalle decken zu selten. | Welche Menge? (Stück 5) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $X \in \mathbb R^5$ normalverteilt; $P(T=1\mid X) = e(X) = \Phi\big(\gamma\, s(X)\big)$; $Y = m_0(X) + \tau T + \varepsilon$ mit konstanter Wirkung $\tau = 0{,}5$. Die Störfunktionen
sind $m_t(X) = \mathbb E[Y \mid X, T=t]$ und $e(X)$; $s$ und $m_0$ enthalten nichtlineare Terme der Stärke $A$.

**Schätzer.** Mit Schätzungen $\hat m_0, \hat m_1, \hat e$:
$$\hat\tau_{\mathrm{PI}} = \tfrac1n \sum_i \big(\hat m_1(X_i) - \hat m_0(X_i)\big), \qquad
\hat\tau_{\mathrm{AIPW}} = \tfrac1n \sum_i \psi_i, \quad
\psi_i = \hat m_1 - \hat m_0 + \frac{T_i\,(Y_i - \hat m_1)}{\hat e} - \frac{(1-T_i)\,(Y_i - \hat m_0)}{1-\hat e}.$$

**Doppelte Robustheit.** $\mathbb E[\psi] - \tau$ ist ein Produkt aus dem Fehler von $\hat e$ und dem von $\hat m$: Er verschwindet, wenn eines von beiden richtig ist.

**Neyman-Orthogonalität.** Die Ableitung des Scores nach den Störfunktionen ist an den wahren Werten null; ein Fehler $\delta$ in den Störfunktionen geht mit $\delta^2$ ein (beim Plug-in mit $\delta$).

**Kreuzanpassung.** Zerlegung in $K$ Falten; die Störfunktionen der Beobachtungen in Falte $k$ stammen aus Modellen, die auf den übrigen Falten angepasst wurden. Ohne sie hängen
$\hat m,\hat e$ von $Y_i$ ab, und der Score hat einen Überanpassungsterm.

**Intervall.** $\hat\tau \pm 1{,}96\,\hat\sigma_\psi/\sqrt n$ mit der Streuung der Einflusswerte $\psi_i$.

**Partiell lineares DML (Robinson).** $\hat\theta = \sum_i \tilde T_i \tilde Y_i / \sum_i \tilde T_i^2$ mit den kreuzangepassten Resten $\tilde Y = Y - \hat{\mathbb E}[Y\mid X]$ und
$\tilde T = T - \hat e(X)$.

**Literatur.** J. Robins, A. Rotnitzky, L. Zhao, *Estimation of regression coefficients when some regressors are not always observed*, JASA 89, 1994; D. Scharfstein, A. Rotnitzky,
J. Robins, *Adjusting for nonignorable drop-out using semiparametric nonresponse models*, JASA 94, 1999; P. Robinson, *Root-N-consistent semiparametric regression*, Econometrica 56,
1988; V. Chernozhukov u. a., *Double/debiased machine learning for treatment and structural parameters*, Econometrics Journal 21, 2018 (die ersten drei aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
