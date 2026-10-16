"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from dml_evaluation import Analysis, de, signed, verdict_instance

_REPLACE = {"→": "->", "←": "<-", "–": "-", "—": "-", "−": "-", "€": "EUR", "≥": ">=", "≤": "<=", "„": '"', "“": '"', "×": "x", "γ": "gamma", "τ": "tau", "δ": "delta",
            "**": "", "²": "^2", "≈": "~", "₀": "0", "₁": "1", "√": "sqrt"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Doppelt robust und Double ML"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Lernverfahren {s.learner}, Kreuzanpassung K = {s.k}, Stärke der Nichtlinearität {de(s.amp, 1)}, Zuteilungsstärke γ = {de(s.gamma, 1)}; n = {s.n}, "
                            f"Seed {s.seed}. Alles simuliert. Wahre Wirkung {de(a.truth, 2)}, effektive Stichprobe {de(a.ess, 0)} von {s.n}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    for r in a.rows:
        extra = f", Standardfehler {de(r.se, 3)}, Intervall [{de(r.lo, 2)}; {de(r.hi, 2)}], {'deckt' if r.covers else 'deckt nicht'}" if r.se is not None else ""
        pdf.multi_cell(0, 4.6, _s(f"{r.name}: {de(r.estimate, 3)} (Fehler {signed(r.error, 3)}){extra}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 4.6, _s(f"Fehler der Störfunktionen (Wurzel des mittleren quadratischen Fehlers): Zuteilung {de(a.err_ps['in_sample'], 3)} in der Stichprobe, "
                              f"{de(a.err_ps['cross_fit'], 3)} kreuzangepasst; Ergebnis {de(a.err_out['in_sample'], 3)} bzw. {de(a.err_out['cross_fit'], 3)}."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_instance(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: doppelt-robust-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
