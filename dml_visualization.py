"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import dml_constants as C
import dml_estimators as E
from dml_evaluation import CF_LABELS, DR_COMBOS, Analysis, de, signed

SHORT = {
    "Naiver Vergleich": "naiv", "Plug-in (Ergebnismodell)": "Plug-in", "Gewichten (normiert)": "Gewichten", "AIPW ohne Kreuzanpassung": "AIPW ohne KA",
    "AIPW mit Kreuzanpassung": "AIPW mit KA", "Partiell lineares DML": "DML partiell linear",
}
PALETTE = ("#8a8a8a", "#4c78a8", "#D68A2E", "#c0392b", "#2F6B65", "#7b5ea7")


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_diagram() -> go.Figure:
    """Das feste Diagramm: fünf Störfaktoren X wirken auf die Regelwahl T und auf den Servicegrad Y, T wirkt auf Y (Wirkung 0,5)."""
    pos = {"X": (0.5, 1.0), "T": (0.0, 0.0), "Y": (1.0, 0.0)}
    names = {"X": "Störfaktoren<br>X₁ bis X₅", "T": "Regel B<br>(Behandlung T)", "Y": "Servicegrad<br>(Ergebnis Y)"}
    fig = go.Figure()
    for a, b, label, color, width in (("X", "T", "e(X), nichtlinear", C.COLOR_NEUTRAL, 2), ("X", "Y", "m₀(X), nichtlinear", C.COLOR_NEUTRAL, 2), ("T", "Y", "τ = 0,5", C.COLOR_GOOD, 4)):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        dx, dy = x1 - x0, y1 - y0
        ln = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / ln, dy / ln
        fig.add_annotation(x=x1 - ux * 0.17, y=y1 - uy * 0.17, ax=x0 + ux * 0.17, ay=y0 + uy * 0.17, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                           arrowsize=1.3, arrowwidth=width, arrowcolor=color, text="")
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.08, text=label, showarrow=False, font=dict(size=11, color="#555555"), bgcolor="rgba(255,255,255,0.75)")
    fig.add_trace(go.Scatter(x=[pos[k][0] for k in pos], y=[pos[k][1] for k in pos], mode="markers+text", text=[names[k] for k in pos], textposition="middle center",
                             textfont=dict(size=10), marker=dict(size=70, color="rgba(47,107,101,0.12)", line=dict(width=[2, 4, 4], color=["#2F6B65", "#1b1b1b", "#1b1b1b"])),
                             hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.3, 1.3])
    fig.update_yaxes(visible=False, range=[-0.35, 1.3])
    return _fixed(fig, 280, showlegend=False)


def build_estimates_chart(a: Analysis) -> go.Figure:
    """Schätzung je Schätzer; mit Einflussfunktions-Intervall, wo es eines gibt (grün: deckt die Wahrheit, rot: deckt nicht)."""
    fig = go.Figure()
    for r in a.rows:
        if r.se is None:
            fig.add_trace(go.Scatter(x=[r.estimate], y=[SHORT[r.name]], mode="markers", marker=dict(color=C.COLOR_NEUTRAL, size=10, symbol="diamond"), showlegend=False,
                                     hovertemplate=f"{r.name}: {de(r.estimate, 3)} (kein Intervall)<extra></extra>"))
        else:
            color = C.COLOR_GOOD if r.covers else C.COLOR_BAD
            fig.add_trace(go.Scatter(x=[r.estimate], y=[SHORT[r.name]], mode="markers", marker=dict(color=color, size=10), showlegend=False,
                                     error_x=dict(type="data", array=[r.hi - r.estimate], arrayminus=[r.estimate - r.lo], color=color, thickness=2),
                                     hovertemplate=f"{r.name}: {de(r.estimate, 3)} [{de(r.lo, 2)}; {de(r.hi, 2)}]<extra></extra>"))
    fig.add_vline(x=a.truth, line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"Wahrheit {de(a.truth, 2)}", annotation_position="top")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=[SHORT[n] for n in E.NAMES])
    fig.update_xaxes(title="geschätzte Wirkung der Regel B")
    return _fixed(fig, 380, showlegend=False)


def build_overlap_histogram(a: Analysis) -> go.Figure:
    bins = dict(start=0.0, end=1.0, size=0.05)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=a.e_cross[a.t == 0], xbins=bins, name="Kontrollen (Regel A)", marker_color=C.COLOR_CONTROL, opacity=0.7))
    fig.add_trace(go.Histogram(x=a.e_cross[a.t == 1], xbins=bins, name="Behandelte (Regel B)", marker_color=C.COLOR_TREATED, opacity=0.7))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="geschätzte Zuteilung e(X) (kreuzangepasst)", range=[0, 1])
    fig.update_yaxes(title="Anzahl Standorte")
    return _fixed(fig, 300)


def build_nuisance_table(a: Analysis) -> pd.DataFrame:
    """Fehler der Störfunktionen (Wurzel des mittleren quadratischen Fehlers) gegen die wahren: in der Stichprobe und kreuzangepasst."""
    return pd.DataFrame([
        {"Störfunktion": "Zuteilung e(X)", "in der Stichprobe": de(a.err_ps["in_sample"], 3), "kreuzangepasst": de(a.err_ps["cross_fit"], 3)},
        {"Störfunktion": "Ergebnis m₀(X), m₁(X)", "in der Stichprobe": de(a.err_out["in_sample"], 3), "kreuzangepasst": de(a.err_out["cross_fit"], 3)},
        {"Störfunktion": "Produkt der Fehler", "in der Stichprobe": de(a.err_ps["in_sample"] * a.err_out["in_sample"], 3),
         "kreuzangepasst": de(a.err_ps["cross_fit"] * a.err_out["cross_fit"], 3)},
    ])


def build_estimates_table(a: Analysis) -> pd.DataFrame:
    out = []
    for r in a.rows:
        out.append({
            "Schätzer": r.name,
            "Schätzung": signed(r.estimate, 3),
            "Fehler": signed(r.error, 3),
            "Standardfehler": de(r.se, 3) if r.se is not None else "–",
            "95-%-Intervall": f"{de(r.lo, 2)} bis {de(r.hi, 2)}" if r.se is not None else "–",
            "Deckt": ("ja" if r.covers else "nein") if r.se is not None else "–",
        })
    return pd.DataFrame(out)


def style_estimates_table(df: pd.DataFrame):
    def row_style(row):
        return ["background-color: rgba(192,57,43,0.15)" if row["Deckt"] == "nein" else ""] * len(row)

    return df.style.apply(row_style, axis=1)


# ----------------------------------------------------------------------------- Experiment 1

def build_double_robust_table(res: dict) -> pd.DataFrame:
    out = []
    for name, _, _ in DR_COMBOS:
        c = res["combos"][name]
        out.append({"Modelle": name, "Plug-in": signed(c["plugin"]["bias"], 2), "Gewichten": signed(c["ipw"]["bias"], 2), "AIPW": signed(c["aipw"]["bias"], 2),
                    "AIPW Streuung": de(c["aipw"]["sd"], 3), "AIPW Deckung": f"{de(100 * c['aipw']['cover'], 0)} %"})
    return pd.DataFrame(out)


def build_double_robust_chart(res: dict) -> go.Figure:
    names = [n for n, _, _ in DR_COMBOS]
    fig = go.Figure()
    for key, label, color in (("plugin", "Plug-in", C.COLOR_NEUTRAL), ("ipw", "Gewichten", C.COLOR_TREATED), ("aipw", "AIPW (doppelt robust)", C.COLOR_GOOD)):
        fig.add_trace(go.Bar(x=names, y=[res["combos"][n][key]["bias"] for n in names], name=label, marker_color=color))
    fig.add_hline(y=0.0, line=dict(color="#888888", width=1))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Verzerrung (Mittel minus 0,5)")
    fig.update_xaxes(tickangle=-12)
    return _fixed(fig, 400)


# ----------------------------------------------------------------------------- Experiment 2

def build_orthogonality_chart(res: dict) -> go.Figure:
    rows = res["rows"]
    base = rows[0]["aipw"]
    only_out = [r for r in rows if r["b2"] == 0.0 and r["b1"] > 0]
    both = [r for r in rows if r["b1"] > 0 and r["b1"] == r["b2"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["b1"] for r in only_out], y=[r["plugin"] for r in only_out], mode="lines+markers", name="Plug-in, nur Ergebnis gestört", line=dict(color=C.COLOR_NEUTRAL, width=2.5)))
    fig.add_trace(go.Scatter(x=[r["b1"] for r in only_out], y=[r["aipw"] - base for r in only_out], mode="lines+markers", name="AIPW, nur Ergebnis gestört", line=dict(color=C.COLOR_GOOD, width=2.5)))
    fig.add_trace(go.Scatter(x=[r["b1"] for r in both], y=[r["aipw"] - base for r in both], mode="lines+markers", name="AIPW, Ergebnis und Zuteilung gestört (b₁ = b₂)",
                             line=dict(color=C.COLOR_BAD, width=2.5)))
    fig.update_xaxes(title="Störung b der Störfunktionen")
    fig.update_yaxes(title="Änderung der Schätzung gegenüber b = 0")
    return _fixed(fig, 380)


def build_orthogonality_table(res: dict) -> pd.DataFrame:
    base = res["rows"][0]
    return pd.DataFrame([{"Störung Ergebnis b₁": de(r["b1"], 1), "Störung Zuteilung b₂": de(r["b2"], 1), "Plug-in": signed(r["plugin"], 4), "Gewichten": signed(r["ipw"], 4),
                          "AIPW": signed(r["aipw"], 4), "AIPW gegenüber b = 0": signed(r["aipw"] - base["aipw"], 4)} for r in res["rows"]])


# ----------------------------------------------------------------------------- Experiment 3

def build_crossfit_table(res: dict) -> pd.DataFrame:
    out = []
    for (leaf, n), run in res["runs"].items():
        label = CF_LABELS[leaf]
        for key, name in (("aipw_plain", "AIPW ohne Kreuzanpassung"), ("aipw_cf_5", "AIPW mit Kreuzanpassung (K = 5)")):
            v = run[key]
            out.append({"Lernverfahren": label, "n": n, "Schätzer": name, "Verzerrung": signed(v["bias"], 3), "Streuung": de(v["sd"], 3), "RMSE": de(v["rmse"], 3),
                        "Deckung": f"{de(100 * v['cover'], 0)} %"})
    return pd.DataFrame(out)


def build_crossfit_chart(res: dict) -> go.Figure:
    keys = list(res["runs"])
    labels = [f"{CF_LABELS[leaf]}, n = {n}" for leaf, n in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * res["runs"][k]["aipw_plain"]["cover"] for k in keys], name="ohne Kreuzanpassung", marker_color=C.COLOR_BAD))
    fig.add_trace(go.Bar(x=labels, y=[100 * res["runs"][k]["aipw_cf_5"]["cover"] for k in keys], name="mit Kreuzanpassung (K = 5)", marker_color=C.COLOR_GOOD))
    fig.add_hline(y=95, line=dict(color="#888888", dash="dot"), annotation_text="versprochen: 95 %", annotation_position="top left")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Deckung der Wahrheit (%)", range=[0, 112])
    return _fixed(fig, 380)


# ----------------------------------------------------------------------------- Experiment 4

def build_rate_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    names = ("Plug-in (Wald)", "AIPW mit Kreuzanpassung (Wald)", "AIPW (linear)", "AIPW (richtige Basis)")
    amps = sorted({r["amp"] for r in res["rows"]})
    for color, name in zip(PALETTE[1:], names):
        for amp, dash in zip(amps, ("solid", "dot")):
            rows = [r for r in res["rows"] if r["amp"] == amp]
            fig.add_trace(go.Scatter(x=[r["n"] for r in rows], y=[r["results"][name]["bias"] for r in rows], mode="lines+markers", name=f"{name}, Nichtlinearität {de(amp, 1)}",
                                     line=dict(color=color, width=2, dash=dash)))
    fig.add_hline(y=0.0, line=dict(color="#888888", width=1))
    fig.update_xaxes(title="Stichprobengröße n", type="log", tickvals=sorted({r["n"] for r in res["rows"]}))
    fig.update_yaxes(title="Verzerrung")
    return _fixed(fig, 440)


def build_rate_table(res: dict) -> pd.DataFrame:
    out = []
    for r in res["rows"]:
        for name, v in r["results"].items():
            out.append({"Nichtlinearität": de(r["amp"], 1), "n": r["n"], "Schätzer": name, "Verzerrung": signed(v["bias"], 3), "RMSE": de(v["rmse"], 3), "RMSE·√n": de(v["rmse_sqrt_n"], 2),
                        "Deckung": f"{de(100 * v['cover'], 0)} %" if "cover" in v else "–"})
    return pd.DataFrame(out)
