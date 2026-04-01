"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AMEN BANK — Système Intelligent d'Analyse de Risque Crédit                ║
║   Version 7.0 PROFESSIONAL                                                   ║
║   Classification  : Arbre de Décision                                        ║
║   Prédiction      : XGBoost (régression % risque)                            ║
║   Données         : German Credit Dataset                                    ║
║   pip install streamlit pandas numpy scikit-learn xgboost plotly matplotlib  ║
║   streamlit run amen_bank_credit_pro.py                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import datetime
import hashlib
import warnings
import base64 as _b64

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE AMEN BANK
# ══════════════════════════════════════════════════════════════════════════════
VERT       = "#006B3C"
VERT_C     = "#00A651"
OR         = "#F5A623"
VERT_DARK  = "#004D2C"
VERT_BG    = "#E8F5EE"
ROUGE      = "#DC2626"
BLEU       = "#1A4FA0"
BLANC      = "#FFFFFF"
FOND       = "#F0F4F2"

HISTORIQUE_CSV = "amen_bank_historique.csv"

# ══════════════════════════════════════════════════════════════════════════════
#  LOGO SVG (Amen Bank)
# ══════════════════════════════════════════════════════════════════════════════
_LOGO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 260 260'>"
    "<defs><radialGradient id='bg' cx='38%' cy='32%' r='70%'>"
    "<stop offset='0%' stop-color='#FFFFFF'/>"
    "<stop offset='100%' stop-color='#D8E8F0'/></radialGradient>"
    "<filter id='drop'><feDropShadow dx='0' dy='4' stdDeviation='8' flood-opacity='.22'/></filter>"
    "</defs>"
    "<circle cx='130' cy='130' r='122' fill='white' filter='url(#drop)'/>"
    "<circle cx='130' cy='130' r='118' fill='url(#bg)'/>"
    "<path d='M 130 18 C 180 18, 225 55, 238 105 C 248 142, 238 178, 215 202 "
    "C 195 188, 178 168, 172 144 C 166 120, 172 94, 186 76 C 170 50, 152 36, 130 30 Z' fill='#1A5BA0'/>"
    "<path d='M 130 242 C 80 242, 38 208, 24 160 C 14 124, 24 86, 46 62 "
    "C 66 76, 84 96, 90 120 C 96 146, 88 172, 74 190 C 92 218, 110 234, 130 238 Z' fill='#00A651'/>"
    "<circle cx='130' cy='176' r='18' fill='#1A5BA0'/>"
    "<circle cx='130' cy='176' r='10' fill='white'/>"
    "</svg>"
)
LOGO_B64 = _b64.b64encode(_LOGO_SVG.encode()).decode()
LOGO_SRC = f"data:image/svg+xml;base64,{LOGO_B64}"


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Amen Bank — Risque Crédit v8.0",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  CSS GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Cairo:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Cairo',sans-serif;background:{FOND};}}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{{
  background:linear-gradient(180deg,{VERT_DARK} 0%,{VERT} 55%,{VERT_C} 100%);
  border-right:4px solid {OR};
}}
section[data-testid="stSidebar"] *{{color:{BLANC}!important;}}

/* ── Header bannière ── */
.amen-header{{
  background:linear-gradient(135deg,{VERT_DARK} 0%,{VERT} 60%,{VERT_C} 100%);
  border-radius:16px;margin-bottom:1.8rem;
  box-shadow:0 8px 32px rgba(0,107,60,.30);
  border-bottom:4px solid {OR};overflow:hidden;
}}
.amen-header-inner{{display:flex;align-items:center;justify-content:space-between;padding:1rem 2rem;}}
.amen-header h1{{font-family:'Playfair Display',serif;font-size:1.6rem;color:{BLANC};margin:0;font-weight:700;}}
.amen-header .sub{{color:rgba(255,255,255,.7);font-size:.75rem;margin-top:4px;letter-spacing:.5px;}}
.amen-header-logo-zone{{display:flex;align-items:center;gap:16px;}}
.amen-header-logo-zone img{{height:64px;width:64px;border-radius:50%;border:3px solid {OR};
  box-shadow:0 0 0 5px rgba(245,166,35,.2);background:white;padding:3px;}}
.amen-header-brand-name{{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:700;
  color:{OR};letter-spacing:4px;line-height:1;text-align:right;}}
.amen-header-brand-sub{{font-size:.58rem;color:rgba(255,255,255,.6);letter-spacing:3px;
  text-transform:uppercase;margin-top:3px;text-align:right;}}
.amen-header-ticker{{background:rgba(0,0,0,.18);padding:.4rem 2rem;font-size:.7rem;
  color:rgba(255,255,255,.65);border-top:1px solid rgba(255,255,255,.1);display:flex;gap:2rem;}}
.amen-header-ticker span{{color:{OR};font-weight:700;}}

/* ── KPI cards ── */
.kpi-card{{background:{BLANC};border-radius:14px;padding:1.2rem 1rem;text-align:center;
  box-shadow:0 4px 16px rgba(0,107,60,.10);border-top:5px solid {VERT};
  transition:transform .2s,box-shadow .2s;margin-bottom:.5rem;}}
.kpi-card:hover{{transform:translateY(-4px);box-shadow:0 8px 24px rgba(0,107,60,.18);}}
.kpi-icon{{font-size:1.6rem;}}
.kpi-value{{font-size:1.9rem;font-weight:700;color:{VERT};font-family:'Playfair Display',serif;
  margin:4px 0 2px;}}
.kpi-label{{font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:1px;}}
.kpi-danger{{border-top-color:{ROUGE}!important;}}.kpi-danger .kpi-value{{color:{ROUGE}!important;}}
.kpi-success{{border-top-color:{VERT_C}!important;}}.kpi-success .kpi-value{{color:{VERT_C}!important;}}
.kpi-or{{border-top-color:{OR}!important;}}.kpi-or .kpi-value{{color:{OR}!important;}}
.kpi-blue{{border-top-color:{BLEU}!important;}}.kpi-blue .kpi-value{{color:{BLEU}!important;}}

/* ── Section title ── */
.section-title{{font-family:'Playfair Display',serif;font-size:1.2rem;color:{VERT_DARK};
  border-left:5px solid {OR};padding-left:12px;margin:1.6rem 0 1rem;}}

/* ── Boutons ── */
.stButton>button{{
  background:linear-gradient(135deg,{VERT} 0%,{VERT_C} 100%)!important;
  color:{BLANC}!important;border:none!important;border-radius:10px!important;
  font-weight:700!important;font-family:'Cairo',sans-serif!important;
  font-size:.95rem!important;padding:.55rem 1.8rem!important;
  box-shadow:0 4px 14px rgba(0,107,60,.35)!important;transition:all .2s!important;}}
.stButton>button:hover{{transform:translateY(-2px)!important;
  box-shadow:0 7px 20px rgba(0,107,60,.45)!important;}}

/* ── Info bar ── */
.info-bar{{background:{VERT_BG};border-left:4px solid {VERT};border-radius:0 8px 8px 0;
  padding:.7rem 1.2rem;margin-bottom:1rem;font-size:.87rem;color:{VERT_DARK};}}

/* ── Résultat cards ── */
.res-good{{
  background:linear-gradient(135deg,#052e16 0%,#064e23 100%);
  border:2px solid {VERT_C};border-radius:18px;padding:1.8rem 1.5rem;text-align:center;
  box-shadow:0 8px 28px rgba(0,166,81,.25);}}
.res-bad{{
  background:linear-gradient(135deg,#2d0d0d 0%,#4a1212 100%);
  border:2px solid {ROUGE};border-radius:18px;padding:1.8rem 1.5rem;text-align:center;
  box-shadow:0 8px 28px rgba(220,38,38,.25);}}
.res-xgb{{
  background:linear-gradient(135deg,#2d1600 0%,#4a2500 100%);
  border:2px solid {OR};border-radius:18px;padding:1.8rem 1.5rem;text-align:center;
  box-shadow:0 8px 28px rgba(245,166,35,.25);}}
.res-model-label{{font-family:'Cairo',sans-serif;font-size:1rem;font-weight:700;margin-bottom:.6rem;}}
.res-main-val{{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;margin:.4rem 0;}}
.res-sub{{font-size:.8rem;color:rgba(255,255,255,.55);margin-top:.3rem;}}

/* ── Nav sidebar ── */
.nav-section-label{{font-size:.58rem;letter-spacing:2.5px;text-transform:uppercase;
  color:rgba(255,255,255,.4);font-weight:700;padding:.7rem .4rem .3rem;margin-top:.2rem;}}
.nav-item{{display:flex;align-items:center;gap:12px;padding:.65rem 1rem;border-radius:10px;
  margin-bottom:4px;border:1px solid transparent;position:relative;overflow:hidden;}}
.nav-item.active{{
  background:linear-gradient(90deg,rgba(245,166,35,.25),rgba(245,166,35,.06));
  border-color:{OR}77;box-shadow:0 3px 12px rgba(245,166,35,.18);transform:translateX(3px);}}
.nav-item.active::before{{content:'';position:absolute;left:0;top:0;height:100%;
  width:3px;background:{OR};border-radius:0 3px 3px 0;}}
.nav-icon{{font-size:1.1rem;width:26px;text-align:center;flex-shrink:0;}}
.nav-label{{font-size:.84rem;font-weight:600;color:rgba(255,255,255,.88);
  flex:1;letter-spacing:.2px;}}
.nav-item.active .nav-label{{color:{OR};font-weight:700;}}
.nav-badge{{font-size:.54rem;font-weight:800;padding:2px 7px;border-radius:99px;
  background:rgba(245,166,35,.22);color:{OR};border:1px solid {OR}66;letter-spacing:.3px;white-space:nowrap;}}
.nav-divider{{border:none;border-top:1px solid rgba(255,255,255,.08);margin:.7rem 0;}}
.data-stat-chip{{display:inline-flex;align-items:center;gap:5px;
  background:rgba(255,255,255,.08);border-radius:8px;padding:3px 9px;font-size:.7rem;
  color:rgba(255,255,255,.75);border:1px solid rgba(255,255,255,.12);margin:2px;}}

/* ── Cache streamlit nav buttons ── */
[data-testid="stSidebar"] [data-testid="stButton"] button{{
  opacity:0!important;height:44px!important;margin-top:-48px!important;
  margin-bottom:4px!important;padding:0!important;border:none!important;
  box-shadow:none!important;background:transparent!important;
  cursor:pointer!important;display:block!important;width:100%!important;}}

/* ── Footer ── */
.footer{{text-align:center;padding:1.5rem;color:#9CA3AF;font-size:.72rem;
  border-top:1px solid #E5E7EB;margin-top:3rem;}}
.footer span{{color:{VERT};font-weight:600;}}

/* ── Badge ── */
.badge-good{{background:#052e16;color:{VERT_C};border:1px solid #16a34a;
  padding:3px 10px;border-radius:99px;font-size:.8rem;font-weight:600;}}
.badge-bad{{background:#2d0d0d;color:{ROUGE};border:1px solid #b91c1c;
  padding:3px 10px;border-radius:99px;font-size:.8rem;font-weight:600;}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DONNÉES (héritées de app.py)
# ══════════════════════════════════════════════════════════════════════════════
CATEGORICAL = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
FEATURES    = ["Age", "Sex", "Job", "Housing", "Saving accounts",
               "Checking account", "Credit amount", "Duration", "Purpose"]

# ── Dictionnaires de traduction (Anglais → Français) ──────────────────────────
TR_SEX = {"male": "Masculin", "female": "Féminin"}
TR_HOUSING = {"free": "Gratuit", "own": "Propriétaire", "rent": "Locataire"}
TR_SAVING = {
    "little": "Peu", "moderate": "Modéré",
    "quite rich": "Assez riche", "rich": "Riche", "NA": "Sans compte",
}
TR_CHECKING = {
    "little": "Peu", "moderate": "Modéré",
    "rich": "Riche", "NA": "Sans compte",
}
TR_PURPOSE = {
    "business":            "Affaires",
    "car":                 "Voiture",
    "domestic appliances": "Électroménager",
    "education":           "Éducation",
    "furniture/equipment": "Mobilier/Équipement",
    "radio/TV":            "Radio/TV",
    "repairs":             "Réparations",
    "vacation/others":     "Vacances/Autres",
}
TR_RISK = {"good": "Bon Client", "bad": "Client à Risque"}


# ══════════════════════════════════════════════════════════════════════════════
#  AUTHENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

USERS = {
    "koussay": {"hash": _hash("koussay2004"), "role": "Administrateur",  "name": "Koussay Hassana"},
    "bechir":  {"hash": _hash("bechir2001"),  "role": "Analyste Crédit", "name": "Bechir Ghoudi"},
    "tarek":   {"hash": _hash("tarek2026"),   "role": "Directeur",       "name": "Tarek Bouchaddekh"},
}

def check_password(u, p):
    return u in USERS and USERS[u]["hash"] == _hash(p)

for k, v in [
    ("logged_in", False), ("username", ""), ("login_error", ""),
    ("nav_page", "Tableau de Bord"), ("df_uploaded", None),
    ("ia_messages", []), ("ia_api_key", ""), ("ia_pred_messages", []),
    ("ia_auto_done", False), ("groq_api_key", ""),
    ("ia_auto_analysis", ""),      # texte brut de l'analyse initiale → injecté dans PDF
    ("ia_graphs_shown", []),       # liste des types de graphs déjà rendus (évite doublons)
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Regex pour détecter les marqueurs [GRAPH:xxx] dans les réponses IA ──────
import re as _re_graph
_GRAPH_MARKER_RE = _re_graph.compile(r'\[GRAPH:([\w]+)\]', _re_graph.IGNORECASE)

def _extract_graph_markers(text: str) -> list[str]:
    """Extrait tous les types de graphiques demandés par l'IA."""
    return [m.lower() for m in _GRAPH_MARKER_RE.findall(text)]

def _clean_ia_text(text: str) -> str:
    """Supprime les marqueurs [GRAPH:xxx] du texte affiché."""
    return _GRAPH_MARKER_RE.sub("", text).strip()

def _render_ia_graph(gtype: str, df_raw, risk_score, clf, age, credit, duration, job,
                     FEATURES, VERT_C, VERT_DARK, OR, ROUGE, BLEU):
    """Rend un graphique Plotly en fonction du type demandé par l'IA."""
    import plotly.express as _px
    import plotly.graph_objects as _go

    def _layout(fig, h=380, margin=None):
        upd = dict(paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                   font_color="#374151", font_family="Cairo")
        if h: upd["height"] = h
        if margin: upd["margin"] = margin
        fig.update_layout(**upd)
        fig.update_xaxes(gridcolor="#E5E7EB", zeroline=False)
        fig.update_yaxes(gridcolor="#E5E7EB", zeroline=False)
        return fig

    if gtype == "importance":
        _df_fi = pd.DataFrame({"Variable": FEATURES,
                                "Importance": clf.feature_importances_}
                               ).sort_values("Importance", ascending=True)
        fig = _go.Figure(_go.Bar(
            x=_df_fi["Importance"], y=_df_fi["Variable"], orientation="h",
            marker=dict(color=_df_fi["Importance"].values,
                        colorscale=[[0,"#2d1600"],[0.45,OR],[1,"#fcd34d"]],
                        showscale=False),
            text=[f"{v:.3f}" for v in _df_fi["Importance"]], textposition="outside",
        ))
        _layout(fig, h=360, margin=dict(t=10,b=10,r=90))
        fig.update_layout(title="🔑 Importance des Variables — XGBoost",
                          title_font_color=VERT_DARK)
        st.plotly_chart(fig, use_container_width=True)

    elif gtype == "distribution":
        _scores = df_raw["% Risque"].dropna()
        fig = _px.histogram(_scores, nbins=40, color_discrete_sequence=[VERT_C],
                            labels={"value":"Score Risque (%)","count":"Clients"},
                            title="📊 Distribution des Scores de Risque — Portefeuille")
        fig.add_vline(x=risk_score, line_dash="dash", line_color=OR, line_width=2.5,
                      annotation_text=f"  Ce client : {risk_score:.1f}%",
                      annotation_font_color=OR, annotation_font_size=11)
        fig.add_vline(x=50, line_dash="dot", line_color=ROUGE, line_width=1.8,
                      annotation_text="  Seuil 50%",
                      annotation_font_color=ROUGE, annotation_font_size=10)
        _layout(fig, h=400)
        st.plotly_chart(fig, use_container_width=True)

    elif gtype == "radar":
        _labels = ["Âge", "Emploi", "Montant", "Durée", "Score Risque"]
        _vals   = [min(age/80,1), job/3, min(credit/20000,1),
                   min(duration/72,1), risk_score/100]
        fig = _go.Figure(_go.Scatterpolar(
            r=_vals+[_vals[0]], theta=_labels+[_labels[0]],
            fill="toself", name="Profil",
            line=dict(color=OR, width=2.5),
            fillcolor="rgba(245,166,35,0.20)",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1],
                                       gridcolor="#E5E7EB", tickfont_size=8)),
            height=420, paper_bgcolor="white", plot_bgcolor="white",
            title="🕸️ Radar — Profil Normalisé du Client",
            title_font_color=VERT_DARK,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif gtype == "pie":
        _vc = df_raw["Risk"].value_counts()
        fig = _go.Figure(_go.Pie(
            labels=["✅ Bon Client","⚠️ À Risque"],
            values=[int(_vc.get("good",0)), int(_vc.get("bad",0))],
            marker=dict(colors=[VERT_C, ROUGE],
                        line=dict(color="white", width=2)),
            hole=0.42, textinfo="label+percent",
            textfont_size=11,
        ))
        fig.update_layout(height=380, paper_bgcolor="white",
                          title="🥧 Répartition Portefeuille — Bon / À Risque",
                          title_font_color=VERT_DARK)
        st.plotly_chart(fig, use_container_width=True)

    elif gtype == "convergence":
        _good  = df_raw[df_raw["Risk"]=="good"]
        _bad   = df_raw[df_raw["Risk"]=="bad"]
        _cats  = ["Âge moyen","Crédit (k TND)","Durée (mois)","Score risque (%)"]
        _vg    = [_good["Age"].mean(), _good["Credit amount"].mean()/1000,
                  _good["Duration"].mean(), _good["% Risque"].mean()]
        _vb    = [_bad["Age"].mean(),  _bad["Credit amount"].mean()/1000,
                  _bad["Duration"].mean(), _bad["% Risque"].mean()]
        import plotly.graph_objects as _go2
        fig = _go2.Figure()
        fig.add_trace(_go2.Bar(name="✅ Bon Client", x=_cats, y=_vg,
                               marker_color=VERT_C, opacity=0.85))
        fig.add_trace(_go2.Bar(name="⚠️ À Risque",  x=_cats, y=_vb,
                               marker_color=ROUGE,  opacity=0.85))
        fig.update_layout(barmode="group", height=380, paper_bgcolor="white",
                          plot_bgcolor="#FAFAFA", font_color="#374151",
                          title="⚖️ Comparaison Profils — Bon Client vs À Risque",
                          title_font_color=VERT_DARK,
                          legend=dict(orientation="h", y=-0.2))
        fig.update_xaxes(gridcolor="#E5E7EB")
        fig.update_yaxes(gridcolor="#E5E7EB")
        st.plotly_chart(fig, use_container_width=True)

    elif gtype in ("stats", "default"):
        _c1, _c2 = st.columns(2)
        with _c1:
            _render_ia_graph("importance", df_raw, risk_score, clf, age, credit,
                             duration, job, FEATURES, VERT_C, VERT_DARK, OR, ROUGE, BLEU)
        with _c2:
            _render_ia_graph("distribution", df_raw, risk_score, clf, age, credit,
                             duration, job, FEATURES, VERT_C, VERT_DARK, OR, ROUGE, BLEU)


# ══════════════════════════════════════════════════════════════════════════════
#  HISTORIQUE
# ══════════════════════════════════════════════════════════════════════════════
_EMPLOI_LABELS = {
    0: "Sans emploi", 1: "Non qualifié", 2: "Qualifié", 3: "Très qualifié",
    "0": "Sans emploi", "1": "Non qualifié", "2": "Qualifié", "3": "Très qualifié",
}

def save_prediction(client, tree_label, risk_pct, conf_pct, analyste):
    emploi_raw = client["Job"]
    emploi_lbl = _EMPLOI_LABELS.get(emploi_raw, str(emploi_raw))
    row = {
        "Date":             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Analyste":         analyste,
        "Age":              f"{client['Age']} ans",
        "Sexe":             TR_SEX.get(str(client["Sex"]), str(client["Sex"])),
        "Emploi":           emploi_lbl,
        "Logement":         TR_HOUSING.get(str(client["Housing"]), str(client["Housing"])),
        "Epargne":          TR_SAVING.get(str(client["Saving accounts"]), str(client["Saving accounts"])),
        "Compte_Courant":   TR_CHECKING.get(str(client["Checking account"]), str(client["Checking account"])),
        "Montant_Credit":   client["Credit amount"],
        "Duree_Mois":       client["Duration"],
        "Objet":            TR_PURPOSE.get(str(client["Purpose"]), str(client["Purpose"])),
        "Classification":   tree_label,
        "Score_Risque_pct": round(risk_pct, 2),
        "Confiance_pct":    round(conf_pct, 2),
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(HISTORIQUE_CSV):
        df_row.to_csv(HISTORIQUE_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(HISTORIQUE_CSV, mode="w", header=True, index=False)


def load_historique():
    if os.path.exists(HISTORIQUE_CSV):
        df = pd.read_csv(HISTORIQUE_CSV)
        df.columns = df.columns.str.strip()

        # ── Normaliser Classification ─────────────────────────
        if "Classification" in df.columns:
            df["Classification"] = (
                df["Classification"].astype(str).str.strip()
                .replace({
                    "RISQUE ÉLEVÉ":   "CLIENT À RISQUE",
                    "MAUVAIS CLIENT": "CLIENT À RISQUE",
                    "CLIENT À RISQUE":"CLIENT À RISQUE",
                    "BON CLIENT":     "BON CLIENT",
                    "INCONNU":        "CLIENT À RISQUE",   # fallback
                    "nan":            "CLIENT À RISQUE",
                    "":               "CLIENT À RISQUE",
                })
            )
            # Tout ce qui ne correspond pas → déduire du score
            mask_unknown = ~df["Classification"].isin(["BON CLIENT", "CLIENT À RISQUE"])
            if mask_unknown.any() and "Score_Risque_pct" in df.columns:
                scores = pd.to_numeric(df.loc[mask_unknown, "Score_Risque_pct"], errors="coerce").fillna(100)
                df.loc[mask_unknown, "Classification"] = scores.apply(
                    lambda s: "BON CLIENT" if s <= 50 else "CLIENT À RISQUE"
                )

        # ── Nettoyer colonne Âge (éviter "female ans" etc.) ──
        if "Age" in df.columns:
            def clean_age(v):
                v = str(v).strip()
                # Extraire uniquement les chiffres au début
                import re as _re
                m = _re.match(r"(\d+)", v)
                if m:
                    return f"{m.group(1)} ans"
                return v
            df["Age"] = df["Age"].apply(clean_age)

        # ── Nettoyer colonne Emploi (valeurs numériques) ──────
        if "Emploi" in df.columns:
            df["Emploi"] = df["Emploi"].apply(
                lambda v: _EMPLOI_LABELS.get(str(v).strip(), str(v).strip())
                if str(v).strip() in _EMPLOI_LABELS else str(v).strip()
            )

        # ── Normaliser colonne Sexe (traduire male/female) ───
        if "Sexe" in df.columns:
            df["Sexe"] = df["Sexe"].apply(
                lambda v: TR_SEX.get(str(v).strip().lower(), str(v).strip())
            )

        # ── Nettoyer Score_Risque_pct ─────────────────────────
        if "Score_Risque_pct" in df.columns:
            df["Score_Risque_pct"] = pd.to_numeric(
                df["Score_Risque_pct"].astype(str)
                .str.replace("%", "", regex=False)
                .str.strip(),
                errors="coerce"
            ).fillna(0.0)

        return df

    return pd.DataFrame(columns=[
        "Date", "Analyste", "Age", "Sexe", "Emploi", "Logement",
        "Epargne", "Compte_Courant", "Montant_Credit", "Duree_Mois",
        "Objet", "Classification", "Score_Risque_pct", "Confiance_pct",
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT & PRÉTRAITEMENT DES DONNÉES (logique de app.py)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(path=None):
    """Charge le CSV (fichier uploadé ou fichier par défaut) et nettoie."""
    if path is not None:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=";")
    else:
        candidates = [
            ("german_credit_prediction_clean.csv", ","),
            ("german_credit_prediction_1.csv",     ";"),
            ("/mnt/user-data/uploads/german_credit_prediction_clean.csv", ","),
            ("/mnt/user-data/uploads/german_credit_prediction_1.csv",     ";"),
        ]
        df = None
        for fname, sep in candidates:
            try:
                df = pd.read_csv(fname, sep=sep)
                if "Risk" in df.columns:
                    break
            except FileNotFoundError:
                continue
        if df is None:
            raise FileNotFoundError("Aucun fichier de données trouvé.")

    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]
    df.columns = df.columns.str.strip()

    # Nettoyage colonne % Risque
    if "% Risque" in df.columns:
        df["% Risque"] = (
            df["% Risque"].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["% Risque"] = pd.to_numeric(df["% Risque"], errors="coerce")

    # Valeurs manquantes catégorielles
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].fillna("unknown")

    df["Risk"] = df["Risk"].str.strip().str.lower()
    return df[df["Risk"].isin(["good", "bad"])].copy()


def encode_df(df):
    """Encode les catégorielles via LabelEncoder — retourne df encodé + encoders."""
    df2 = df.copy()
    encoders = {}
    for c in CATEGORICAL:
        if c in df2.columns:
            le = LabelEncoder()
            df2[c] = le.fit_transform(df2[c].astype(str))
            encoders[c] = le
    return df2, encoders


@st.cache_resource(show_spinner=False)
def train_tree(df_enc_hash):
    """Entraîne l'Arbre de Décision (classification binaire Bon/Risque)."""
    df_enc = st.session_state["_df_enc_cache"]
    X = df_enc[FEATURES]
    y = df_enc["Risk"].map({"good": 0, "bad": 1})
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = DecisionTreeClassifier(
        max_depth=5, min_samples_split=20,
        min_samples_leaf=10, random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)
    y_pred  = clf.predict(X_te)
    metrics = {
        "accuracy":  round(accuracy_score(y_te, y_pred) * 100, 1),
        "f1":        round(f1_score(y_te, y_pred) * 100, 1),
        "precision": round(precision_score(y_te, y_pred, zero_division=0) * 100, 1),
        "recall":    round(recall_score(y_te, y_pred, zero_division=0) * 100, 1),
        "cv":        round(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean() * 100, 1),
        "report":    classification_report(y_te, y_pred, target_names=["Bon Client", "Client à Risque"]),
        "cm":        confusion_matrix(y_te, y_pred),
    }
    return clf, metrics, (X_tr, X_te, y_tr, y_te)


@st.cache_resource(show_spinner=False)
def train_xgb(df_enc_hash):
    """Entraîne XGBoost en régression pour prédire le % Risque (0–100)."""
    df_enc = st.session_state["_df_enc_cache"]
    X = df_enc[FEATURES]
    y = df_enc["% Risque"]
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
    model = XGBRegressor(
        n_estimators=300, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        "r2":   round(r2_score(y_te, y_pred) * 100, 2),
        "mae":  round(mean_absolute_error(y_te, y_pred), 2),
        "mse":  round(mean_squared_error(y_te, y_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_te, y_pred)), 2),
    }
    return model, scaler, metrics


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def render_header(title, sub="Direction des Risques — Amen Bank Tunisie"):
    st.markdown(f"""
    <div class="amen-header">
      <div class="amen-header-inner">
        <div>
          <h1>🏦 {title}</h1>
          <div class="sub">{sub}</div>
        </div>
        <div class="amen-header-logo-zone">
          <div>
            <div class="amen-header-brand-name">AMEN</div>
            <div class="amen-header-brand-sub">Banque Tunisienne · Since 1967</div>
          </div>
          <img src="{LOGO_SRC}" alt="Amen Bank"/>
        </div>
      </div>
      <div class="amen-header-ticker">
        <span>🏦 Amen Bank</span> · Système Risque Crédit v7.0 Pro &nbsp;|&nbsp;
        <span>EUR</span> 3.315 TND &nbsp;·&nbsp; <span>USD</span> 3.045 TND &nbsp;·&nbsp;
        <span>CAD</span> 2.203 TND &nbsp;|&nbsp;
        <span>🌳 Classification</span> Arbre de Décision &nbsp;·&nbsp;
        <span>⚡ Prédiction</span> XGBoost
      </div>
    </div>""", unsafe_allow_html=True)


def kpi(col, val, lbl, icon, cls=""):
    col.markdown(f"""
    <div class="kpi-card {cls}">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def plotly_layout(fig, height=None, margin=None):
    """Applique le style Amen Bank aux figures Plotly."""
    upd = dict(
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font_color="#374151",
        font_family="Cairo",
    )
    if height:
        upd["height"] = height
    if margin:
        upd["margin"] = margin
    fig.update_layout(**upd)
    fig.update_xaxes(gridcolor="#E5E7EB", zeroline=False, color="#374151")
    fig.update_yaxes(gridcolor="#E5E7EB", zeroline=False, color="#374151")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR + NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("Tableau de Bord",   "🏠", "Tableau de Bord",        "",      ),
    ("Exploration",       "📊", "Exploration des Données", "",      ),
    ("Prédiction Client", "🔮", "Prédiction Client",       "IA+PDF"),
    ("Données",           "📋", "Données & Historique",    "",      ),
]


def render_sidebar():
    u = USERS[st.session_state.username]
    role_icon = {"Administrateur": "👑", "Analyste Crédit": "🔍", "Directeur": "🏛️"}.get(u["role"], "👤")

    with st.sidebar:
        # ── En-tête utilisateur ─────────────────────────────
        st.markdown(f"""
        <div style="background:rgba(0,0,0,.18);margin:-1rem -1rem 0;padding:1.2rem 1rem .8rem;
                    border-bottom:2px solid {OR}55;text-align:center">
          <div style="display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:.5rem">
            <img src="{LOGO_SRC}" style="height:52px;width:52px;border-radius:50%;
              border:2px solid {OR};background:white;padding:2px;
              box-shadow:0 0 0 4px rgba(245,166,35,.2)"/>
            <div style="text-align:left">
              <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;
                color:{OR};letter-spacing:3px;line-height:1">AMEN</div>
              <div style="font-size:.55rem;color:rgba(255,255,255,.5);letter-spacing:2px;
                text-transform:uppercase">Le Partenaire de votre Succès</div>
            </div>
          </div>
          <hr style="border:none;border-top:1px solid rgba(255,255,255,.1);margin:.5rem 0"/>
          <div style="display:flex;align-items:center;justify-content:center;gap:10px">
            <div style="position:relative;display:inline-block">
              <div style="width:42px;height:42px;border-radius:50%;background:{VERT_DARK};
                border:2px solid {OR};display:flex;align-items:center;justify-content:center;
                font-size:1.1rem;box-shadow:0 2px 8px rgba(0,0,0,.3)">{role_icon}</div>
              <div style="position:absolute;bottom:-1px;right:-1px;width:12px;height:12px;
                background:#22C55E;border-radius:50%;border:2px solid {VERT_DARK}"></div>
            </div>
            <div style="text-align:left">
              <div style="font-weight:700;font-size:.85rem;color:{OR}">{u['name']}</div>
              <div style="font-size:.65rem;color:rgba(255,255,255,.55)">{u['role']}</div>
              <div style="font-size:.6rem;color:rgba(255,255,255,.35);margin-top:1px">
                👤 {st.session_state.username} · 🟢 En ligne</div>
            </div>
          </div>
        </div>
        <div style="height:.6rem"></div>
        """, unsafe_allow_html=True)

        # ── Source données (fichier par défaut uniquement) ──
        st.session_state["df_uploaded"] = None

        st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)

        # ── Navigation ──────────────────────────────────────
        st.markdown(f"""
        <div style="font-size:.56rem;letter-spacing:2.5px;text-transform:uppercase;
          color:rgba(255,255,255,.35);font-weight:700;padding:.5rem .4rem .2rem">
          📌 NAVIGATION PRINCIPALE
        </div>""", unsafe_allow_html=True)

        _NAV_GROUPS = {
            "Tableau de Bord":   ("🏠", "Tableau de Bord",        "",        "Vue d'ensemble"),
            "Exploration":       ("📊", "Exploration",             "",        "EDA & Graphiques"),
            "Prédiction Client": ("🔮", "Prédiction Client",       "IA+PDF",  "Analyse + IA + PDF"),
            "Données":           ("📋", "Données & Historique",    "",        "Jeu de données"),
        }

        for _key, (_icon, _label, _badge, _hint) in _NAV_GROUPS.items():
            _is_active  = (st.session_state.nav_page == _key)
            _act_cls    = "active" if _is_active else ""
            _badge_html = f'<span class="nav-badge">{_badge}</span>' if _badge else ""
            st.markdown(f"""
            <div class="nav-item {_act_cls}">
              <span class="nav-icon">{_icon}</span>
              <div style="flex:1;min-width:0">
                <div class="nav-label">{_label}</div>
                <div style="font-size:.6rem;color:rgba(255,255,255,.3);
                  margin-top:1px;letter-spacing:.3px">{_hint}</div>
              </div>
              {_badge_html}
            </div>""", unsafe_allow_html=True)
            if st.button(f"{_icon} {_label}", key=f"nav_{_key}", use_container_width=True):
                st.session_state.nav_page = _key
                if _key == "Prédiction Client":
                    st.session_state.ia_pred_messages = []
                st.rerun()

        st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)

        # ── Moteur & IA Info ─────────────────────────────────
        st.markdown(f"""
        <div style="background:rgba(245,166,35,.07);border-radius:12px;padding:.8rem 1rem;
          border:1px solid {OR}28;margin-bottom:.6rem">
          <div style="font-size:.62rem;color:{OR};font-weight:800;text-transform:uppercase;
            letter-spacing:1px;margin-bottom:.55rem">⚡ Moteurs Actifs</div>
          <div style="font-size:.73rem;color:rgba(255,255,255,.78);line-height:1.8">
            <div style="display:flex;align-items:center;gap:7px;margin-bottom:3px">
              <span style="width:6px;height:6px;background:{VERT_C};border-radius:50%;flex-shrink:0"></span>
              <b>XGBoost</b> — Score % Risque
            </div>
            <div style="display:flex;align-items:center;gap:7px;margin-bottom:3px">
              <span style="width:6px;height:6px;background:{BLEU};border-radius:50%;flex-shrink:0"></span>
              <b>Arbre de Décision</b> — Classification
            </div>
            <div style="display:flex;align-items:center;gap:7px">
              <span style="width:6px;height:6px;background:{OR};border-radius:50%;flex-shrink:0"></span>
              <b>Groq Llama 3.3 70B</b> — IA Analyste
            </div>
          </div>
          <div style="margin-top:.55rem;padding-top:.45rem;border-top:1px solid rgba(255,255,255,.08);
            font-size:.65rem;color:rgba(255,255,255,.35)">
            Seuil décision : ≤ 50% BON &nbsp;|&nbsp; &gt; 50% RISQUE
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Déconnexion ─────────────────────────────────────
        st.markdown(f"""
        <div class="nav-item" style="border-color:rgba(220,38,38,.3)">
          <span class="nav-icon">🚪</span>
          <span class="nav-label" style="color:rgba(255,100,100,.85)!important">
            Se déconnecter</span>
        </div>""", unsafe_allow_html=True)
        if st.button("Se déconnecter", key="nav_btn_logout", use_container_width=True):
            for k in ("logged_in", "username", "nav_page"):
                st.session_state[k] = {"logged_in": False, "username": "", "nav_page": "Tableau de Bord"}[k]
            st.rerun()

        st.markdown(f"""
        <div style="text-align:center;padding:.7rem 0 0;font-size:.6rem;
          color:rgba(255,255,255,.25);border-top:1px solid {OR}18;margin-top:.3rem">
          © 2025 <span style="color:{OR}88">Amen Bank</span>
          &nbsp;·&nbsp; Direction des Risques<br>
          <span style="color:rgba(255,255,255,.18)">Système Risque Crédit · v8.0 Pro · IA Intégrée</span>
        </div>""", unsafe_allow_html=True)

    return st.session_state.nav_page


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE LOGIN
# ══════════════════════════════════════════════════════════════════════════════
def page_login():
    st.markdown(f"""<style>
    section[data-testid="stSidebar"]{{display:none!important}}
    header{{display:none!important}}
    .stApp{{background:linear-gradient(135deg,{VERT_DARK} 0%,{VERT} 40%,#004D30 100%);}}
    </style>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{VERT_DARK};padding:.5rem 2rem;display:flex;align-items:center;
                justify-content:space-between;border-bottom:3px solid {OR}">
      <div style="display:flex;align-items:center;gap:14px">
        <img src="{LOGO_SRC}" style="height:44px;width:44px;border-radius:50%;
          border:2px solid {OR};background:white;padding:2px"/>
        <div>
          <div style="font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:700;
            color:{OR};letter-spacing:3px">AMEN BANK</div>
          <div style="font-size:.58rem;color:rgba(255,255,255,.5);letter-spacing:2px">
            LE PARTENAIRE DE VOTRE SUCCÈS</div>
        </div>
      </div>
      <div style="display:flex;gap:1.5rem;font-size:.72rem;color:rgba(255,255,255,.6)">
        <span>🌐 www.amenbank.com.tn</span>
        <span>📞 71 148 000</span>
        <span>📍 Tunis, Tunisie</span>
      </div>
    </div>
    <div style="background:rgba(0,0,0,.25);padding:.35rem 2rem;font-size:.68rem;
                color:rgba(255,255,255,.55);border-bottom:1px solid rgba(255,255,255,.08)">
      💱 EUR = 3.315 TND &nbsp;·&nbsp; USD = 3.045 TND &nbsp;·&nbsp; CAD = 2.203 TND
      &nbsp;&nbsp;|&nbsp;&nbsp;
      🌳 Classification : Arbre de Décision &nbsp;·&nbsp; ⚡ Prédiction : XGBoost
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown(f"""
        <div style="background:white;border-radius:20px;overflow:hidden;
                    box-shadow:0 24px 64px rgba(0,0,0,.35);">
          <div style="background:linear-gradient(135deg,{VERT_DARK},{VERT},{VERT_C});
                      padding:2rem;text-align:center;border-bottom:4px solid {OR}">
            <img src="{LOGO_SRC}" style="height:80px;width:80px;border-radius:50%;
              border:3px solid {OR};background:white;padding:4px;
              box-shadow:0 0 0 6px rgba(245,166,35,.25),0 8px 24px rgba(0,0,0,.3)"/>
            <div style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;
              color:{OR};letter-spacing:5px;margin-top:.7rem">AMEN BANK</div>
            <div style="font-size:.65rem;color:rgba(255,255,255,.55);letter-spacing:3px;
              text-transform:uppercase;margin-top:4px">Banque Tunisienne · Depuis 1967</div>
          </div>
          <div style="padding:1.5rem 2rem">
            <div style="text-align:center;margin-bottom:1.2rem">
              <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
                color:{VERT_DARK};font-weight:700">Système de Gestion du Risque Crédit</div>
              <div style="font-size:.75rem;color:#9CA3AF;margin-top:4px">
                Direction des Risques · Accès réservé au personnel autorisé</div>
            </div>
            <div style="background:{VERT_BG};border-radius:10px;padding:.7rem 1rem;
              margin-bottom:1rem;border-left:4px solid {VERT_C};font-size:.78rem;color:{VERT_DARK}">
              🌳 <b>Arbre de Décision</b> — Classification &nbsp;|&nbsp; ⚡ <b>XGBoost</b> — Prédiction % Risque
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("👤  Identifiant", placeholder="Entrez votre identifiant")
        password = st.text_input("🔒  Mot de passe", type="password", placeholder="••••••••")
        if st.session_state.login_error:
            st.error(st.session_state.login_error)
        if st.button("🔐  Se connecter — Espace Sécurisé", use_container_width=True):
            if check_password(username, password):
                st.session_state.logged_in  = True
                st.session_state.username   = username
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = "❌ Identifiant ou mot de passe incorrect."
                st.rerun()

        st.markdown(f"""
        <div style="text-align:center;margin-top:1.2rem;font-size:.65rem;color:#9CA3AF">
          © 2025 Amen Bank Tunisie — Tous droits réservés<br>
          <span style="color:{VERT};font-weight:600">Direction des Risques · v7.0 Professional</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE TABLEAU DE BORD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard(df):
    render_header("Tableau de Bord Exécutif")
    bad  = df[df["Risk"] == "bad"]
    good = df[df["Risk"] == "good"]
    pct  = len(bad) / len(df) * 100
    risk_mean   = df["% Risque"].mean() if "% Risque" in df.columns else 0.0
    credit_med  = df["Credit amount"].median()
    dur_mean    = df["Duration"].mean()
    age_mean    = df["Age"].mean()
    credit_max  = df["Credit amount"].max()

    # ── Ligne 1 : KPIs principaux ────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, len(df),                                  "Clients Analysés",   "🏦", "")
    kpi(c2, len(good),                                "Bons Clients",       "✅",  "kpi-success")
    kpi(c3, len(bad),                                 "Clients à Risque",   "⚠️",  "kpi-danger")
    kpi(c4, f"{pct:.1f}%",                            "Taux de Défaut",     "📊",  "kpi-danger")
    kpi(c5, f"{df['Credit amount'].mean():,.0f} TND", "Montant Moyen",     "💰",  "kpi-or")
    kpi(c6, f"{risk_mean:.1f}%",                      "Score Risque Moy",  "⚡",  "kpi-blue")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ligne 2 : KPIs statistiques supplémentaires ──────────
    section("📊 Statistiques Clés du Portefeuille Crédit")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi(k1, f"{credit_med:,.0f} TND",              "Médiane Crédit",       "📉", "kpi-or")
    kpi(k2, f"{credit_max:,.0f} TND",              "Crédit Maximum",       "💎", "kpi-blue")
    kpi(k3, f"{dur_mean:.1f} mois",                "Durée Moyenne",        "📅", "")
    kpi(k4, f"{age_mean:.1f} ans",                 "Âge Moyen",            "👤", "")
    kpi(k5, f"{good['Credit amount'].mean():,.0f}","Montant Moy (Bons)",   "✅", "kpi-success")
    kpi(k6, f"{bad['Credit amount'].mean():,.0f}", "Montant Moy (Risque)", "⚠️", "kpi-danger")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ligne 3 : Graphiques principaux ──────────────────────
    col1, col2, col3 = st.columns([1.2, 1.2, 1])

    with col1:
        section("Répartition Bons Clients / Clients à Risque")
        rc = df["Risk"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["✅ Bon Client", "❌ Client à Risque"],
            values=[rc.get("good", 0), rc.get("bad", 0)],
            hole=0.58, marker_colors=[VERT_C, ROUGE],
            textfont_size=13, pull=[0, 0.06], textinfo="label+percent",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=30),
            showlegend=True, legend=dict(orientation="h", y=-0.12),
            annotations=[dict(
                text=f"<b>{pct:.0f}%</b><br>Défaut",
                x=0.5, y=0.5, font_size=18, showarrow=False, font_color=ROUGE,
            )],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Montant Moyen par Objet du Crédit")
        pa = df.groupby("Purpose")["Credit amount"].mean().sort_values()
        pa.index = [TR_PURPOSE.get(v, v) for v in pa.index]
        fig2 = go.Figure(go.Bar(
            x=pa.values, y=pa.index, orientation="h",
            marker_color=[VERT if i % 2 == 0 else VERT_C for i in range(len(pa))],
            text=[f"{v:,.0f}" for v in pa.values], textposition="outside",
        ))
        plotly_layout(fig2, height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        section("Taux Défaut par Logement")
        grp = df.groupby("Housing")["Risk"].apply(
            lambda x: (x == "bad").sum() / len(x) * 100
        ).reset_index()
        grp.columns = ["Housing", "pct_bad"]
        grp["Housing"] = grp["Housing"].map(lambda v: TR_HOUSING.get(v, v))
        grp = grp.sort_values("pct_bad", ascending=False)
        fig_h = go.Figure(go.Bar(
            x=grp["pct_bad"], y=grp["Housing"], orientation="h",
            marker_color=[ROUGE if v > 30 else OR if v > 20 else VERT_C for v in grp["pct_bad"]],
            text=[f"{v:.1f}%" for v in grp["pct_bad"]], textposition="outside",
        ))
        plotly_layout(fig_h, height=240, margin=dict(t=10, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

        section("Taux Défaut par Tranche d'Âge")
        df2 = df.copy()
        df2["Age_bin"] = pd.cut(df2["Age"], bins=[18, 25, 35, 50, 75],
                                labels=["18-25", "26-35", "36-50", "51+"])
        ar = df2.groupby("Age_bin", observed=True)["Risk"].apply(
            lambda x: (x == "bad").sum() / len(x) * 100
        ).reset_index()
        fig_a = go.Figure(go.Bar(
            x=ar["Age_bin"].astype(str), y=ar["Risk"],
            marker_color=[ROUGE if v > 35 else OR if v > 25 else VERT_C for v in ar["Risk"]],
            text=[f"{v:.1f}%" for v in ar["Risk"]], textposition="outside",
        ))
        plotly_layout(fig_a, height=200, margin=dict(t=10, b=10))
        st.plotly_chart(fig_a, use_container_width=True)

    # ── Ligne 4 : Analyse croisée ─────────────────────────────
    col3b, col4b = st.columns(2)
    with col3b:
        section("Distribution des Âges — Bon Client vs Client à Risque")
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=good["Age"], name="✅ Bon Client",      nbinsx=25, marker_color=VERT_C, opacity=0.75))
        fig3.add_trace(go.Histogram(x=bad["Age"],  name="❌ Client à Risque", nbinsx=25, marker_color=ROUGE,  opacity=0.75))
        plotly_layout(fig3, height=300, margin=dict(t=10))
        fig3.update_layout(barmode="overlay", legend=dict(orientation="h"))
        st.plotly_chart(fig3, use_container_width=True)

    with col4b:
        section("Durée vs Montant Crédit")
        df_sc = df.copy(); df_sc["Risk"] = df_sc["Risk"].map(TR_RISK)
        fig4 = px.scatter(df_sc, x="Duration", y="Credit amount", color="Risk",
                          color_discrete_map={"Bon Client": VERT_C, "Client à Risque": ROUGE},
                          opacity=0.55, trendline="lowess")
        plotly_layout(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Ligne 5 : Distribution du score de risque ─────────────
    if "% Risque" in df.columns:
        col5a, col5b = st.columns(2)
        with col5a:
            section("🔥 Distribution du Score % Risque — Bon Client vs Client à Risque")
            fig5 = go.Figure()
            for risk, clr, lbl in [("good", VERT_C, "Bon Client"), ("bad", ROUGE, "Client à Risque")]:
                fig5.add_trace(go.Histogram(
                    x=df[df["Risk"] == risk]["% Risque"],
                    name=lbl, nbinsx=40, marker_color=clr, opacity=0.7,
                ))
            fig5.add_vline(x=50, line_dash="dash", line_color=OR, line_width=2,
                           annotation_text="Seuil 50%", annotation_font_color=OR)
            plotly_layout(fig5, height=320)
            fig5.update_layout(barmode="overlay", legend=dict(orientation="h"))
            st.plotly_chart(fig5, use_container_width=True)

        with col5b:
            section("📋 Tableau Statistique Comparatif Bon Client vs Client à Risque")
            stats_rows = []
            for col_n, label in [
                ("Age", "Âge"), ("Credit amount", "Montant Crédit"),
                ("Duration", "Durée (mois)"), ("Job", "Catégorie Emploi"),
            ]:
                g_vals = good[col_n]
                b_vals = bad[col_n]
                diff_pct = (b_vals.mean() - g_vals.mean()) / g_vals.mean() * 100 if g_vals.mean() != 0 else 0
                stats_rows.append({
                    "Variable":       label,
                    "Moy. BON":       f"{g_vals.mean():.1f}",
                    "Méd. BON":       f"{g_vals.median():.1f}",
                    "Éc.T BON":       f"{g_vals.std():.1f}",
                    "Moy. RISQUE":    f"{b_vals.mean():.1f}",
                    "Méd. RISQUE":    f"{b_vals.median():.1f}",
                    "Éc.T RISQUE":    f"{b_vals.std():.1f}",
                    "Δ%":             f"{diff_pct:+.1f}%",
                })
            st.dataframe(
                pd.DataFrame(stats_rows),
                use_container_width=True, hide_index=True,
            )

            # Tableau résumé par sexe
            section("Répartition par Sexe")
            sex_grp = df.groupby("Sex").agg(
                Total=("Risk", "count"),
                Bons=("Risk", lambda x: (x=="good").sum()),
                Mauvais=("Risk", lambda x: (x=="bad").sum()),
            ).reset_index()
            sex_grp["Taux Défaut"] = (sex_grp["Mauvais"] / sex_grp["Total"] * 100).round(1).astype(str) + "%"
            sex_grp["Montant Moy"] = df.groupby("Sex")["Credit amount"].mean().round(0).astype(int).values
            sex_grp["Montant Moy"] = sex_grp["Montant Moy"].apply(lambda x: f"{x:,} TND")
            sex_grp["Sex"] = sex_grp["Sex"].map(lambda v: TR_SEX.get(v, v))
            sex_grp = sex_grp.rename(columns={"Sex": "Sexe", "Total": "Total", "Bons": "Bons Clients", "Mauvais": "Clients à Risque"})
            st.dataframe(sex_grp, use_container_width=True, hide_index=True)

    # ── Ligne 6 : Taux défaut par objet ──────────────────────
    section("Taux de Défaut par Objet du Crédit")
    purpose_grp = df.groupby("Purpose").agg(
        Total=("Risk", "count"),
        Mauvais=("Risk", lambda x: (x=="bad").sum()),
    ).reset_index()
    purpose_grp["Taux_Defaut"] = purpose_grp["Mauvais"] / purpose_grp["Total"] * 100
    purpose_grp["Purpose"] = purpose_grp["Purpose"].map(lambda v: TR_PURPOSE.get(v, v))
    purpose_grp = purpose_grp.sort_values("Taux_Defaut", ascending=True)
    fig_pur = go.Figure(go.Bar(
        y=purpose_grp["Purpose"],
        x=purpose_grp["Taux_Defaut"],
        orientation="h",
        marker_color=[ROUGE if v > 35 else OR if v > 25 else VERT_C
                      for v in purpose_grp["Taux_Defaut"]],
        text=[f"  {v:.1f}%  ({int(t)} clients)"
              for v, t in zip(purpose_grp["Taux_Defaut"], purpose_grp["Total"])],
        textposition="inside",
        insidetextanchor="start",
        textfont=dict(color="white", size=12, family="Cairo"),
    ))
    fig_pur.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=360,
        margin=dict(t=20, b=20, l=150, r=40),
        font=dict(color="#374151", family="Cairo"),
        xaxis=dict(
            title="Taux de Défaut (%)",
            gridcolor="#E5E7EB",
            zeroline=False,
            ticksuffix="%",
            color="#374151",
        ),
        yaxis=dict(
            gridcolor="#E5E7EB",
            zeroline=False,
            color="#374151",
            tickfont=dict(size=12),
        ),
    )
    st.plotly_chart(fig_pur, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE EXPLORATION (EDA)
# ══════════════════════════════════════════════════════════════════════════════
def page_eda(df):
    render_header("Analyse Exploratoire des Données")
    NUM = ["Age", "Credit amount", "Duration", "Job"]
    bad  = df[df["Risk"] == "bad"]
    good = df[df["Risk"] == "good"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distributions", "🔗 Corrélations", "🏷️ Catégorielles", "📐 Statistiques",
    ])

    with tab1:
        section("Histogrammes Comparatifs — Bon Client vs Client à Risque")
        fig = make_subplots(rows=2, cols=2, subplot_titles=NUM)
        for i, col in enumerate(NUM):
            r, c = (i // 2) + 1, (i % 2) + 1
            for risk, clr, lbl in [("good", VERT_C, "Bon Client"), ("bad", ROUGE, "Client à Risque")]:
                fig.add_trace(go.Histogram(
                    x=df[df["Risk"] == risk][col], name=lbl,
                    nbinsx=25, marker_color=clr, opacity=0.7, showlegend=(i == 0),
                ), row=r, col=c)
        fig.update_layout(
            barmode="overlay", height=480, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=10),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

        section("Montant Crédit vs Risque (Box Plot)")
        col1, col2 = st.columns(2)
        with col1:
            df_plot = df.copy(); df_plot["Risk"] = df_plot["Risk"].map(TR_RISK)
            fig_b = px.box(df_plot, x="Risk", y="Credit amount", color="Risk",
                           color_discrete_map={"Bon Client": VERT_C, "Client à Risque": ROUGE}, points="outliers")
            plotly_layout(fig_b, height=340)
            st.plotly_chart(fig_b, use_container_width=True)
        with col2:
            fig_v = px.violin(df_plot, x="Risk", y="Duration", color="Risk",
                              color_discrete_map={"Bon Client": BLEU, "Client à Risque": ROUGE}, box=True)
            plotly_layout(fig_v, height=340)
            st.plotly_chart(fig_v, use_container_width=True)

    with tab2:
        section("Matrice de Corrélation")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr = df[num_cols].corr()
        fig3 = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu_r", text=np.round(corr.values, 3),
            texttemplate="<b>%{text}</b>", showscale=True, zmin=-1, zmax=1,
        ))
        fig3.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=10))
        _, mc, _ = st.columns([1, 2, 1])
        with mc:
            st.plotly_chart(fig3, use_container_width=True)

        section("Matrice de Dispersion")
        df_pm = df.copy(); df_pm["Risk"] = df_pm["Risk"].map(TR_RISK)
        fig_pm = px.scatter_matrix(df_pm, dimensions=NUM, color="Risk",
                                   color_discrete_map={"Bon Client": VERT_C, "Client à Risque": ROUGE}, opacity=0.4)
        fig_pm.update_traces(diagonal_visible=False)
        fig_pm.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pm, use_container_width=True)

    with tab3:
        CAT_LABELS = {
            "Housing": "Logement", "Saving accounts": "Compte Épargne",
            "Purpose": "Objet du Crédit", "Sex": "Sexe",
        }
        CAT_TR = {
            "Housing": TR_HOUSING, "Saving accounts": TR_SAVING,
            "Purpose": TR_PURPOSE, "Sex": TR_SEX,
        }
        for cat in ["Housing", "Saving accounts", "Purpose", "Sex"]:
            df_cat = df.copy()
            df_cat[cat] = df_cat[cat].map(lambda v: CAT_TR[cat].get(v, v))
            df_cat["Risk"] = df_cat["Risk"].map(TR_RISK)
            cnt = df_cat.groupby([cat, "Risk"]).size().reset_index(name="count")
            fig4 = px.bar(cnt, x=cat, y="count", color="Risk", barmode="group",
                          color_discrete_map={"Bon Client": VERT_C, "Client à Risque": ROUGE},
                          title=f"Répartition du Risque par {CAT_LABELS.get(cat, cat)}")
            plotly_layout(fig4, height=320)
            st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        section("Statistiques Descriptives — Bon Client vs Client à Risque")
        bad2  = df[df["Risk"] == "bad"]
        good2 = df[df["Risk"] == "good"]
        rows = []
        for col in NUM:
            rows.append({
                "Variable":    col,
                "Moy BON":     f"{good2[col].mean():.2f}",
                "Méd BON":     f"{good2[col].median():.2f}",
                "Éc.T BON":    f"{good2[col].std():.2f}",
                "Moy RISQUE":  f"{bad2[col].mean():.2f}",
                "Méd RISQUE":  f"{bad2[col].median():.2f}",
                "Éc.T RISQUE": f"{bad2[col].std():.2f}",
                "Diff %":      f"{(bad2[col].mean()-good2[col].mean())/good2[col].mean()*100:+.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("📋 Aperçu des données brutes"):
            st.dataframe(df.head(100), use_container_width=True)
        with st.expander("📈 Statistiques descriptives complètes"):
            st.dataframe(df.describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE ARBRE DE DÉCISION
# ══════════════════════════════════════════════════════════════════════════════
def page_tree(df_enc, clf, tree_metrics, splits):
    render_header(
        "🌳 Arbre de Décision — Classification",
        "Classification binaire Bon Client / Client à Risque · max_depth=5 · class_weight=balanced",
    )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"{tree_metrics['accuracy']}%", "Exactitude",      "🎯", "kpi-success")
    kpi(c2, f"{tree_metrics['f1']}%",       "Score F1",        "📊", "kpi-or")
    kpi(c3, f"{tree_metrics['precision']}%","Précision",       "🎯", "kpi-blue")
    kpi(c4, f"{tree_metrics['cv']}%",       "Exactitude Croisée", "🔄", "")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section("Matrice de Confusion")
        cm = tree_metrics["cm"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Prédit Bon", "Prédit Risque"], y=["Réel Bon", "Réel Risque"],
            colorscale=[[0, VERT_BG], [0.5, OR], [1, VERT]],
            text=cm, texttemplate="<b>%{text}</b>", textfont_size=22, showscale=False,
        ))
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=10, b=10))
        _, mc, _ = st.columns([1, 2, 1])
        with mc:
            st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        section("Importance des Variables")
        fi = pd.DataFrame({
            "Feature":    FEATURES,
            "Importance": clf.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"], orientation="h",
            marker=dict(
                color=fi["Importance"].values,
                colorscale=[[0, VERT_DARK], [0.5, VERT], [1, VERT_C]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in fi["Importance"]], textposition="outside",
        ))
        plotly_layout(fig_fi, height=380, margin=dict(t=10, b=10, r=80))
        st.plotly_chart(fig_fi, use_container_width=True)

    # Rapport détaillé
    with st.expander("📋 Rapport de Classification Complet"):
        st.markdown(f"""
        <div style="background:{VERT_BG};border-radius:10px;padding:1rem;">
          <pre style="font-size:.82rem;color:{VERT_DARK};margin:0">{tree_metrics['report']}</pre>
        </div>
        """, unsafe_allow_html=True)

    # Visualisation de l'arbre
    with st.expander("🌳 Visualisation Graphique de l'Arbre"):
        fig_tree, ax = plt.subplots(figsize=(22, 10), facecolor=FOND)
        ax.set_facecolor(FOND)
        plot_tree(
            clf, feature_names=FEATURES, class_names=["Bon Client", "Client à Risque"],
            filled=True, rounded=True, fontsize=7, ax=ax, impurity=False,
        )
        st.pyplot(fig_tree, use_container_width=True)
        plt.close()

    # Paramètres modèle
    with st.expander("⚙️ Paramètres du Modèle"):
        params = clf.get_params()
        df_params = pd.DataFrame(
            [{"Paramètre": k, "Valeur": str(v)} for k, v in params.items()]
        )
        st.dataframe(df_params, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE XGBOOST
# ══════════════════════════════════════════════════════════════════════════════
def page_xgb(df_enc, xgb_model, scaler, xgb_met):
    render_header(
        "⚡ XGBoost — Prédiction du % Risque",
        "Régression du score de risque (0–100 %) · n_estimators=300 · lr=0.04 · max_depth=6",
    )

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"{xgb_met['r2']}%",   "Score R²",            "📈", "kpi-success")
    kpi(c2, f"{xgb_met['mae']}",   "Erreur Abs. Moyenne", "📉", "kpi-or")
    kpi(c3, f"{xgb_met['rmse']}",  "Racine Erreur Quad.", "📊", "kpi-blue")
    kpi(c4, f"{xgb_met['mse']}",   "Erreur Quad. Moyenne","🔢", "")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("Importance des Variables (XGBoost)")
        fi_xgb = pd.DataFrame({
            "Feature":    FEATURES,
            "Importance": xgb_model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig_xfi = go.Figure(go.Bar(
            x=fi_xgb["Importance"], y=fi_xgb["Feature"], orientation="h",
            marker=dict(
                color=fi_xgb["Importance"].values,
                colorscale=[[0, "#2d1600"], [0.5, OR], [1, "#fcd34d"]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in fi_xgb["Importance"]], textposition="outside",
        ))
        plotly_layout(fig_xfi, height=380, margin=dict(t=10, b=10, r=80))
        st.plotly_chart(fig_xfi, use_container_width=True)

    with col2:
        section("Valeurs Réelles vs Prédites")
        X_all    = df_enc[FEATURES]
        X_all_sc = scaler.transform(X_all)
        preds    = xgb_model.predict(X_all_sc)
        actual   = df_enc["% Risque"].values

        df_scatter = pd.DataFrame({"Réel": actual, "Prédit": preds})
        fig_s = px.scatter(df_scatter, x="Réel", y="Prédit", opacity=0.4,
                           color_discrete_sequence=[VERT_C])
        lim = max(actual.max(), preds.max()) * 1.05
        fig_s.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                        line=dict(color=OR, dash="dash", width=2))
        plotly_layout(fig_s, height=380)
        st.plotly_chart(fig_s, use_container_width=True)

    # Distribution des résidus
    section("Distribution des Résidus (Réel − Prédit)")
    residuals = actual - preds
    fig_res = px.histogram(x=residuals, nbins=50, color_discrete_sequence=[BLEU],
                           labels={"x": "Résidu", "y": "Fréquence"}, opacity=0.75)
    fig_res.add_vline(x=0, line_dash="dash", line_color=OR, line_width=2)
    plotly_layout(fig_res, height=320)
    st.plotly_chart(fig_res, use_container_width=True)

    # Paramètres modèle
    with st.expander("⚙️ Paramètres XGBoost"):
        params = xgb_model.get_params()
        df_params = pd.DataFrame(
            [{"Paramètre": k, "Valeur": str(v)} for k, v in params.items()]
        )
        st.dataframe(df_params, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATION PDF RAPPORT DE PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
def generate_prediction_pdf(client_raw, tree_class, tree_proba, risk_score,
                             tree_label, analyste, rec_titre, rec_text,
                             ia_analysis: str = ""):
    """Génère un rapport PDF professionnel de prédiction de risque crédit."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, KeepTogether,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        import io, re

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )

        # Couleurs
        C_VERT  = colors.HexColor("#006B3C")
        C_VERT2 = colors.HexColor("#00A651")
        C_OR    = colors.HexColor("#F5A623")
        C_ROUGE = colors.HexColor("#DC2626")
        C_BLEU  = colors.HexColor("#1A4FA0")
        C_FOND  = colors.HexColor("#E8F5EE")
        C_GRIS  = colors.HexColor("#6B7280")
        C_DARK  = colors.HexColor("#004D2C")
        C_BG    = colors.HexColor("#F0F4F2")

        styles = getSampleStyleSheet()

        def ps(name, **kw):
            return ParagraphStyle(name, **kw)

        style_title  = ps("title",  fontName="Helvetica-Bold", fontSize=22,
                          textColor=C_VERT, alignment=TA_CENTER, spaceAfter=4)
        style_sub    = ps("sub",    fontName="Helvetica",      fontSize=10,
                          textColor=C_GRIS, alignment=TA_CENTER, spaceAfter=10)
        style_h2     = ps("h2",     fontName="Helvetica-Bold", fontSize=13,
                          textColor=C_DARK, spaceBefore=14, spaceAfter=6)
        style_body   = ps("body",   fontName="Helvetica",      fontSize=10,
                          textColor=colors.HexColor("#374151"), leading=16)
        style_center = ps("center", fontName="Helvetica",      fontSize=10,
                          alignment=TA_CENTER, textColor=colors.HexColor("#374151"))
        style_bold   = ps("bold",   fontName="Helvetica-Bold", fontSize=11,
                          textColor=C_VERT)
        style_alert  = ps("alert",  fontName="Helvetica-Bold", fontSize=13,
                          alignment=TA_CENTER)
        style_small  = ps("small",  fontName="Helvetica",      fontSize=8,
                          textColor=C_GRIS, alignment=TA_CENTER)

        story = []
        now   = datetime.datetime.now()

        # ── En-tête ──────────────────────────────────────────────
        header_data = [[
            Paragraph("<b><font color='#F5A623' size='20'>AMEN BANK</font></b><br/>"
                      "<font color='white' size='8'>LE PARTENAIRE DE VOTRE SUCCÈS</font>",
                      ps("hdr", fontName="Helvetica-Bold", alignment=TA_LEFT, leading=22)),
            Paragraph("<font color='white' size='10'><b>RAPPORT D'ANALYSE DU RISQUE CRÉDIT</b></font><br/>"
                      f"<font color='#D1FAE5' size='8'>Référence : RPT-{now.strftime('%Y%m%d%H%M%S')}</font>",
                      ps("hdr2", fontName="Helvetica", alignment=TA_CENTER, leading=20)),
            Paragraph(f"<font color='white' size='9'>{now.strftime('%d/%m/%Y')}</font><br/>"
                      f"<font color='#D1FAE5' size='8'>{now.strftime('%H:%M:%S')}</font>",
                      ps("hdr3", fontName="Helvetica", alignment=TA_RIGHT, leading=18)),
        ]]
        t_hdr = Table(header_data, colWidths=[5*cm, 9*cm, 3.5*cm])
        t_hdr.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), C_DARK),
            ("LEFTPADDING",  (0,0), (-1,-1), 14),
            ("RIGHTPADDING", (0,0), (-1,-1), 14),
            ("TOPPADDING",   (0,0), (-1,-1), 12),
            ("BOTTOMPADDING",(0,0), (-1,-1), 12),
            ("LINEBELOW",    (0,0), (-1,-1), 3, C_OR),
        ]))
        story.append(t_hdr)
        story.append(Spacer(1, 0.5*cm))

        # ── Infos analyste ────────────────────────────────────────
        info_data = [
            ["Analyste", analyste, "Date d'analyse", now.strftime("%d/%m/%Y %H:%M")],
            ["Modèles", "Arbre de Décision + XGBoost", "Version système", "v7.0 Professional"],
        ]
        t_info = Table(info_data, colWidths=[3.5*cm, 6.5*cm, 3.5*cm, 4*cm])
        t_info.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (0,-1), C_FOND),
            ("BACKGROUND",  (2,0), (2,-1), C_FOND),
            ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("TEXTCOLOR",   (0,0), (0,-1), C_DARK),
            ("TEXTCOLOR",   (2,0), (2,-1), C_DARK),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white, colors.HexColor("#F9FAFB")]),
            ("LEFTPADDING",  (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING",   (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ]))
        story.append(t_info)
        story.append(Spacer(1, 0.4*cm))

        # ── Résultat principal ────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=C_OR, spaceAfter=8))
        story.append(Paragraph("RÉSULTAT DE L'ANALYSE", style_h2))

        # Couleur selon résultat
        if tree_label == "BON CLIENT":
            res_color = C_VERT2
            res_icon  = "✓ BON CLIENT"
            res_bg    = colors.HexColor("#D1FAE5")
        else:
            res_color = C_ROUGE
            res_icon  = "✗ CLIENT À RISQUE"
            res_bg    = colors.HexColor("#FEE2E2")

        if risk_score <= 50:
            score_label = "FAIBLE"
            score_color = C_VERT2
        elif risk_score <= 75:
            score_label = "MODÉRÉ"
            score_color = C_OR
        else:
            score_label = "ÉLEVÉ"
            score_color = C_ROUGE

        result_data = [
            [
                Paragraph(f"<b><font size='14'>Classification</font></b><br/>"
                          f"<font size='22'><b>{res_icon}</b></font><br/>"
                          f"<font size='9'>Seuil 50%: Score {'<=' if risk_score<=50 else '>'} 50%</font>",
                          ps("r1", fontName="Helvetica-Bold", alignment=TA_CENTER,
                             textColor=res_color, leading=26)),
                Paragraph(f"<b><font size='14'>Score XGBoost</font></b><br/>"
                          f"<font size='28'><b>{risk_score:.1f}%</b></font><br/>"
                          f"<font size='9'>Risque {score_label}</font>",
                          ps("r2", fontName="Helvetica-Bold", alignment=TA_CENTER,
                             textColor=score_color, leading=26)),
                Paragraph(f"<b><font size='14'>Confiance Arbre</font></b><br/>"
                          f"<font size='22'><b>{max(tree_proba)*100:.1f}%</b></font><br/>"
                          f"<font size='9'>P(Bon)={tree_proba[0]*100:.1f}% | P(Risque)={tree_proba[1]*100:.1f}%</font>",
                          ps("r3", fontName="Helvetica-Bold", alignment=TA_CENTER,
                             textColor=C_BLEU, leading=26)),
            ]
        ]
        t_res = Table(result_data, colWidths=[5.8*cm, 5.8*cm, 5.8*cm])
        t_res.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (0,0), res_bg),
            ("BACKGROUND",  (1,0), (1,0), colors.HexColor("#FEF3C7") if risk_score > 50 else C_FOND),
            ("BACKGROUND",  (2,0), (2,0), colors.HexColor("#EFF6FF")),
            ("GRID",        (0,0), (-1,-1), 1, C_OR),
            ("TOPPADDING",   (0,0), (-1,-1), 14),
            ("BOTTOMPADDING",(0,0), (-1,-1), 14),
            ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ]))
        story.append(t_res)
        story.append(Spacer(1, 0.4*cm))

        # ── Recommandation ────────────────────────────────────────
        rec_clean = re.sub(r'<[^>]+>', '', rec_text)
        rec_data = [[
            Paragraph(f"<b>RECOMMANDATION : {rec_titre}</b><br/><br/>{rec_clean}",
                      ps("rec", fontName="Helvetica", fontSize=10, leading=16,
                         textColor=colors.HexColor("#374151")))
        ]]
        if tree_label == "BON CLIENT":
            rec_bg_pdf = colors.HexColor("#D1FAE5")
            rec_border = C_VERT2
        else:
            rec_bg_pdf = colors.HexColor("#FEE2E2")
            rec_border = C_ROUGE

        t_rec = Table(rec_data, colWidths=[17.5*cm])
        t_rec.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), rec_bg_pdf),
            ("LINEBEFORE",  (0,0), (0,-1),  4, rec_border),
            ("LEFTPADDING",  (0,0), (-1,-1), 14),
            ("RIGHTPADDING", (0,0), (-1,-1), 14),
            ("TOPPADDING",   (0,0), (-1,-1), 12),
            ("BOTTOMPADDING",(0,0), (-1,-1), 12),
        ]))
        story.append(t_rec)
        story.append(Spacer(1, 0.4*cm))

        # ── Données client ────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=C_OR, spaceAfter=6))
        story.append(Paragraph("DONNÉES DU CLIENT", style_h2))

        labels_map = {
            "Age": "Âge", "Sex": "Sexe", "Job": "Catégorie Emploi",
            "Housing": "Logement", "Saving accounts": "Compte Épargne",
            "Checking account": "Compte Courant", "Credit amount": "Montant Crédit (TND)",
            "Duration": "Durée (mois)", "Purpose": "Objet du Crédit",
        }
        client_rows = []
        items = list(client_raw.items())
        for i in range(0, len(items), 3):
            row = []
            for j in range(3):
                if i+j < len(items):
                    k, v = items[i+j]
                    lbl  = labels_map.get(k, k)
                    if k == "Credit amount":
                        v = f"{v:,.0f} TND"
                    elif k == "Duration":
                        v = f"{v} mois"
                    elif k == "Age":
                        v = f"{v} ans"
                    elif k == "Sex":
                        v = TR_SEX.get(str(v), str(v))
                    elif k == "Housing":
                        v = TR_HOUSING.get(str(v), str(v))
                    elif k == "Saving accounts":
                        v = TR_SAVING.get(str(v), str(v))
                    elif k == "Checking account":
                        v = TR_CHECKING.get(str(v), str(v))
                    elif k == "Purpose":
                        v = TR_PURPOSE.get(str(v), str(v))
                    row.append(Paragraph(f"<b>{lbl}</b><br/>{v}",
                                         ps(f"cd{i}{j}", fontName="Helvetica", fontSize=9,
                                            leading=14, textColor=colors.HexColor("#374151"))))
                else:
                    row.append("")
            client_rows.append(row)

        mensualite = client_raw["Credit amount"] / client_raw["Duration"]
        client_rows.append([
            Paragraph(f"<b>Mensualité Estimée</b><br/>{mensualite:,.0f} TND/mois",
                      ps("mens", fontName="Helvetica-Bold", fontSize=9, leading=14,
                         textColor=C_VERT)),
            Paragraph(f"<b>Ratio Durée/Âge</b><br/>{client_raw['Duration']/client_raw['Age']:.2f}",
                      ps("ratio", fontName="Helvetica-Bold", fontSize=9, leading=14,
                         textColor=C_BLEU)),
            "",
        ])

        t_client = Table(client_rows, colWidths=[5.8*cm, 5.8*cm, 5.8*cm])
        t_client.setStyle(TableStyle([
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white, C_FOND]),
            ("LEFTPADDING",  (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING",   (0,0), (-1,-1), 7),
            ("BOTTOMPADDING",(0,0), (-1,-1), 7),
        ]))
        story.append(t_client)
        story.append(Spacer(1, 0.4*cm))

        # ── Barre de risque visuelle ──────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=C_OR, spaceAfter=6))
        story.append(Paragraph("VISUALISATION DU SCORE DE RISQUE", style_h2))
        bar_filled = risk_score / 100
        bar_data = [[
            Paragraph("0%", style_small),
            Paragraph("25%", style_small),
            Paragraph("50%  ← SEUIL", ps("seuil", fontName="Helvetica-Bold", fontSize=8,
                                          textColor=C_OR, alignment=TA_CENTER)),
            Paragraph("75%", style_small),
            Paragraph("100%", style_small),
        ]]
        t_bar_labels = Table(bar_data, colWidths=[3.5*cm]*5)
        t_bar_labels.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),
                                           ("RIGHTPADDING",(0,0),(-1,-1),0)]))
        story.append(t_bar_labels)

        # Barre de progression
        bar_rows = [[""]]
        t_bar_bg = Table(bar_rows, colWidths=[17.5*cm], rowHeights=[0.55*cm])
        t_bar_bg.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#E5E7EB")),
            ("LEFTPADDING",(0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),0),
        ]))
        story.append(t_bar_bg)

        filled_w = 17.5 * bar_filled
        bar_filled_rows = [[""]]
        if filled_w > 0:
            t_bar_filled = Table(bar_filled_rows,
                                  colWidths=[max(0.1, filled_w)*cm],
                                  rowHeights=[0.55*cm])
            fill_color = C_VERT2 if risk_score <= 35 else (C_OR if risk_score <= 60 else C_ROUGE)
            t_bar_filled.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), fill_color),
                ("LEFTPADDING",(0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),0),
            ]))
            story.append(t_bar_filled)

        score_disp = Paragraph(
            f"Score de risque : <b>{risk_score:.1f}%</b> — Seuil de décision : <b>50%</b> "
            f"({'BON CLIENT ≤ 50%' if risk_score <= 50 else 'CLIENT À RISQUE > 50%'})",
            ps("sc", fontName="Helvetica", fontSize=10, textColor=C_DARK, alignment=TA_CENTER))
        story.append(Spacer(1, 0.2*cm))
        story.append(score_disp)
        story.append(Spacer(1, 0.4*cm))

        # ── Section Analyse IA ────────────────────────────────────
        if ia_analysis and ia_analysis.strip():
            story.append(Spacer(1, 0.3*cm))
            story.append(HRFlowable(width="100%", thickness=1.5, color=C_OR, spaceAfter=6))
            story.append(Paragraph("ANALYSE IA — Groq Llama 3.3 70B Versatile", style_h2))
            story.append(Spacer(1, 0.1*cm))

            # Bannière IA
            ia_banner_data = [[
                Paragraph(
                    "<font color='#F5A623' size='10'><b>🤖 IA Analyste Amen Bank</b></font><br/>"
                    "<font color='#9CA3AF' size='8'>Groq · Llama 3.3 70B · Analyse automatique du dossier</font>",
                    ps("iahdr", fontName="Helvetica", alignment=TA_LEFT, leading=16)),
                Paragraph(
                    f"<font color='white' size='8'><b>Score : {risk_score:.1f}%</b></font><br/>"
                    f"<font color='#9CA3AF' size='7'>{tree_label}</font>",
                    ps("iahdr2", fontName="Helvetica", alignment=TA_RIGHT, leading=14)),
            ]]
            t_ia_banner = Table(ia_banner_data, colWidths=[12.5*cm, 5*cm])
            t_ia_banner.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#060D1A")),
                ("LEFTPADDING",  (0,0), (-1,-1), 14),
                ("RIGHTPADDING", (0,0), (-1,-1), 14),
                ("TOPPADDING",   (0,0), (-1,-1), 10),
                ("BOTTOMPADDING",(0,0), (-1,-1), 10),
                ("LINEBELOW",    (0,0), (-1,-1), 2, C_OR),
            ]))
            story.append(t_ia_banner)
            story.append(Spacer(1, 0.2*cm))

            # Texte IA — nettoyage des balises HTML et marqueurs [GRAPH:xxx]
            import re as _re_pdf
            ia_clean = _re_pdf.sub(r'\[GRAPH:[\w]+\]', '', ia_analysis)
            ia_clean = _re_pdf.sub(r'<[^>]+>', '', ia_clean)
            # Couper le texte en paragraphes sur les sauts de ligne
            ia_paragraphs = [p.strip() for p in ia_clean.split('\n') if p.strip()]
            style_ia_body = ps("iabd", fontName="Helvetica", fontSize=9.5,
                               textColor=colors.HexColor("#374151"), leading=15,
                               spaceBefore=4)
            style_ia_bold = ps("iabld", fontName="Helvetica-Bold", fontSize=9.5,
                               textColor=C_DARK, leading=15, spaceBefore=6)
            for para in ia_paragraphs:
                # Détecter les titres de section (commencent par 1️⃣, 2️⃣ etc ou par ━)
                is_title = (para.startswith(("1️", "2️", "3️", "4️", "━", "•"))
                            or len(para) < 60 and para.isupper())
                use_style = style_ia_bold if is_title else style_ia_body
                story.append(Paragraph(para, use_style))

            story.append(Spacer(1, 0.3*cm))

        # ── Pied de page ──────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_GRIS))
        story.append(Spacer(1, 0.2*cm))
        footer_data = [[
            Paragraph("© 2025 Amen Bank Tunisie — Direction des Risques",
                      ps("fl", fontName="Helvetica", fontSize=8, textColor=C_GRIS)),
            Paragraph(f"Système Risque Crédit v7.0 Professional",
                      ps("fc", fontName="Helvetica", fontSize=8,
                         textColor=C_GRIS, alignment=TA_CENTER)),
            Paragraph(f"Document généré le {now.strftime('%d/%m/%Y à %H:%M')}",
                      ps("fr", fontName="Helvetica", fontSize=8,
                         textColor=C_GRIS, alignment=TA_RIGHT)),
        ]]
        t_footer = Table(footer_data, colWidths=[6*cm, 5.5*cm, 6*cm])
        t_footer.setStyle(TableStyle([
            ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ]))
        story.append(t_footer)

        doc.build(story)
        return buffer.getvalue()

    except Exception as e:
        import io as _io, traceback
        tb = traceback.format_exc()
        # Fallback: simple text PDF
        try:
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.pagesizes import A4
            buf2 = _io.BytesIO()
            c = rl_canvas.Canvas(buf2, pagesize=A4)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 800, "Amen Bank - Rapport de Prediction")
            c.setFont("Helvetica", 10)
            c.drawString(50, 780, f"Classification: {tree_label}")
            c.drawString(50, 765, f"Score Risque: {risk_score:.1f}%")
            c.drawString(50, 750, f"Analyste: {analyste}")
            c.drawString(50, 735, f"Date: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
            y = 710
            for k, v in client_raw.items():
                c.drawString(50, y, f"{k}: {v}")
                y -= 15
            c.save()
            return buf2.getvalue()
        except Exception:
            return b""


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE PRÉDICTION CLIENT
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE PRÉDICTION CLIENT
# ══════════════════════════════════════════════════════════════════════════════
def page_prediction(df, df_raw, df_enc, encoders, clf, xgb_model, scaler):
    render_header(
        "🔮 Prédiction Intelligente du Risque",
        "Analyse combinée : Arbre de Décision (classification) + XGBoost (score % risque)",
    )
    u = USERS[st.session_state.username]

    st.markdown(f"""
    <div class="info-bar">
      🌳 <b>Arbre de Décision</b> — Classification Bon Client / Client à Risque &nbsp;|&nbsp;
      ⚡ <b>XGBoost Régression</b> — Score de Risque 0–100% &nbsp;|&nbsp;
      🎯 <b>Règle de décision :</b> Score ≤ 50% → BON CLIENT &nbsp;·&nbsp; Score > 50% → CLIENT À RISQUE
    </div>""", unsafe_allow_html=True)

    section("📋 Informations du Client")
    c1, c2, c3 = st.columns(3)

    def get_vals(col):
        return sorted(df_raw[col].dropna().unique().tolist())

    with c1:
        st.markdown("**👤 Données Personnelles**")
        age     = st.slider("Âge", 18, 80, 35)
        sex     = st.selectbox("Sexe", get_vals("Sex"),
                               format_func=lambda x: TR_SEX.get(x, x))
        job     = st.selectbox("Catégorie Emploi", [0, 1, 2, 3],
                               format_func=lambda x: {
                                   0: "0 – Sans emploi", 1: "1 – Non qualifié",
                                   2: "2 – Qualifié", 3: "3 – Très qualifié",
                               }[x])
        housing = st.selectbox("Logement", get_vals("Housing"),
                               format_func=lambda x: TR_HOUSING.get(x, x))

    with c2:
        st.markdown("**💳 Situation Financière**")
        saving   = st.selectbox("Compte Épargne",  get_vals("Saving accounts"),
                                format_func=lambda x: TR_SAVING.get(x, x))
        checking = st.selectbox("Compte Courant",  get_vals("Checking account"),
                                format_func=lambda x: TR_CHECKING.get(x, x))
        credit   = st.number_input("Montant Crédit (TND)", 250, 20_000, 3_000, step=100)

    with c3:
        st.markdown("**📋 Détails du Crédit**")
        duration = st.slider("Durée (mois)", 4, 72, 24)
        purpose  = st.selectbox("Objet du Crédit", get_vals("Purpose"),
                                format_func=lambda x: TR_PURPOSE.get(x, x))
        mensualite = credit / duration
        st.markdown(f"""
        <div style="background:{VERT_BG};border-radius:10px;padding:.9rem;
          border:1px solid {VERT_C}55;margin-top:.5rem">
          <div style="font-size:.7rem;color:{VERT_DARK};font-weight:700">📊 Résumé Crédit</div>
          <div style="font-size:.82rem;color:{VERT_DARK};margin-top:4px">
            Mensualité : <b>{mensualite:,.0f} TND/mois</b><br>
            Âge : <b>{age} ans</b> · Durée : <b>{duration} mois</b><br>
            Ratio durée/âge : <b>{duration/age:.2f}</b>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  BOUTON : calcule + sauvegarde TOUT dans session_state
    # ════════════════════════════════════════════════════════
    if st.button("🚀  Lancer l'Analyse du Risque", use_container_width=True):
        client_raw = {
            "Age": age, "Sex": sex, "Job": job, "Housing": housing,
            "Saving accounts": saving, "Checking account": checking,
            "Credit amount": credit, "Duration": duration, "Purpose": purpose,
        }
        client_df = pd.DataFrame([client_raw])

        for c in CATEGORICAL:
            le2 = encoders[c]
            val = client_df[c].astype(str).values[0]
            if val in le2.classes_:
                client_df[c] = le2.transform([val])
            else:
                client_df[c] = le2.transform(["unknown"]) if "unknown" in le2.classes_ else 0

        X_client    = client_df[FEATURES].values
        tree_class  = clf.predict(X_client)[0]
        tree_proba  = clf.predict_proba(X_client)[0]
        X_client_sc = scaler.transform(X_client)
        risk_score  = float(xgb_model.predict(X_client_sc)[0])
        risk_score  = max(0.0, min(100.0, risk_score))

        # ── Tout stocker dans session_state ──────────────────
        st.session_state["pred_result"] = {
            "client_raw":    client_raw,
            "tree_class":    tree_class,
            "tree_proba":    tree_proba,
            "risk_score":    risk_score,
            "analyste":      u["name"],
        }
        st.session_state.ia_pred_messages = []
        st.session_state.ia_pending       = True

        # Sauvegarder historique immédiatement
        decision_tmp = "BON CLIENT" if risk_score <= 50 else "CLIENT À RISQUE"
        save_prediction(client_raw, decision_tmp, risk_score, 100 - risk_score, u["name"])

    # ════════════════════════════════════════════════════════
    #  AFFICHAGE — lu depuis session_state, TOUJOURS visible
    #  même après rerun (entrée clé API, boutons IA, etc.)
    # ════════════════════════════════════════════════════════
    if "pred_result" not in st.session_state:
        return  # Rien à afficher, l'utilisateur n'a pas encore lancé d'analyse

    R = st.session_state["pred_result"]
    client_raw  = R["client_raw"]
    tree_class  = R["tree_class"]
    tree_proba  = R["tree_proba"]
    risk_score  = R["risk_score"]

    # Variables locales reconstituées
    age      = client_raw["Age"]
    sex      = client_raw["Sex"]
    job      = client_raw["Job"]
    housing  = client_raw["Housing"]
    saving   = client_raw["Saving accounts"]
    checking = client_raw["Checking account"]
    credit   = client_raw["Credit amount"]
    duration = client_raw["Duration"]
    purpose  = client_raw["Purpose"]

    decision_finale = "BON CLIENT" if risk_score <= 50 else "CLIENT À RISQUE"
    arbre_avis      = "BON CLIENT" if tree_class == 0 else "CLIENT À RISQUE"
    convergence     = (decision_finale == arbre_avis)

    if decision_finale == "BON CLIENT":
        dec_color = VERT_C; dec_bg_grad = "linear-gradient(135deg,#052e16,#064e23)"
        dec_border = VERT_C; dec_icon = "✅"
    else:
        dec_color = ROUGE; dec_bg_grad = "linear-gradient(135deg,#2d0d0d,#4a1212)"
        dec_border = ROUGE; dec_icon = "⚠️"

    st.markdown("---")
    section("📊 Résultats de l'Analyse")

    # ── BANNIÈRE DÉCISION FINALE ──────────────────────────
    st.markdown(f"""
    <div style="background:{dec_bg_grad};border:3px solid {dec_border};
                border-radius:20px;padding:1.6rem 2rem;text-align:center;
                box-shadow:0 10px 36px rgba(0,0,0,.35);margin-bottom:1.2rem">
      <div style="font-size:.75rem;color:rgba(255,255,255,.5);letter-spacing:3px;
                  text-transform:uppercase;margin-bottom:.4rem">
        🎯 DÉCISION FINALE — Basée sur Score XGBoost (Seuil 50%)
      </div>
      <div style="font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:700;
                  color:{dec_color};letter-spacing:2px;margin:.3rem 0">
        {dec_icon} {decision_finale}
      </div>
      <div style="font-size:1rem;color:rgba(255,255,255,.65);margin-top:.3rem">
        Score de risque XGBoost : <b style="color:{dec_color};font-size:1.3rem">{risk_score:.1f}%</b>
        &nbsp;{'≤' if risk_score<=50 else '>'}&nbsp; seuil 50%
      </div>
      <div style="margin-top:.8rem;font-size:.78rem;color:rgba(255,255,255,.4)">
        {'✅ Les deux modèles convergent vers la même conclusion' if convergence
         else f'⚠️ Note : L\'Arbre de Décision indique {arbre_avis} — la décision finale suit le score XGBoost'}
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_xgb, col_gauge = st.columns([1.2, 1])

    with col_xgb:
        if risk_score <= 35:
            score_color, score_label, score_zone = VERT_C, "Risque Faible",   "Zone Verte ✅"
        elif risk_score <= 50:
            score_color, score_label, score_zone = VERT_C, "BON CLIENT",      "Zone Verte ✅"
        elif risk_score <= 75:
            score_color, score_label, score_zone = OR,     "CLIENT À RISQUE", "Zone Orange ⚠️"
        else:
            score_color, score_label, score_zone = ROUGE,  "CLIENT À RISQUE", "Zone Rouge 🚫"

        st.markdown(f"""
        <div class="res-xgb" style="position:relative;padding:2rem 1.8rem">
          <div style="position:absolute;top:12px;right:14px;font-size:.62rem;
            background:rgba(245,166,35,.22);padding:3px 10px;border-radius:99px;
            color:{OR};font-weight:700;letter-spacing:1px">⚡ DÉCISION FINALE</div>
          <div class="res-model-label" style="color:{OR};font-size:1rem;margin-bottom:.4rem">
            ⚡ XGBoost — Score de Risque
          </div>
          <div class="res-main-val" style="color:{score_color};font-size:3rem;margin:.5rem 0">
            {risk_score:.1f}%
          </div>
          <div style="font-size:1rem;font-weight:700;color:{score_color};margin-bottom:.6rem">
            {score_label} &nbsp;·&nbsp;
            <span style="color:rgba(255,255,255,.45);font-size:.82rem">{score_zone}</span>
          </div>
          <div style="background:rgba(0,0,0,.25);border-radius:99px;
                      height:12px;overflow:hidden;position:relative;margin-top:.6rem">
            <div style="width:{min(risk_score,100):.0f}%;height:100%;
              background:linear-gradient(90deg,{VERT_C},{OR},{ROUGE});
              border-radius:99px;transition:width .6s ease"></div>
            <div style="position:absolute;left:50%;top:-3px;width:3px;height:18px;
              background:{OR};opacity:.9;border-radius:2px"></div>
          </div>
          <div style="display:flex;justify-content:space-between;
            font-size:.7rem;color:rgba(255,255,255,.4);margin-top:5px">
            <span>0%</span>
            <span style="color:{OR};font-weight:700">▲ Seuil 50%</span>
            <span>100%</span>
          </div>
          <div style="margin-top:1rem;font-size:.78rem;color:rgba(255,255,255,.5);
                      background:rgba(255,255,255,.05);border-radius:8px;padding:.5rem .8rem">
            Règle : Score ≤ 50% → <b style="color:{VERT_C}">BON CLIENT</b>
            &nbsp;|&nbsp; Score &gt; 50% → <b style="color:{ROUGE}">CLIENT À RISQUE</b>
          </div>
        </div>""", unsafe_allow_html=True)

    with col_gauge:
        needle_color = VERT_C if risk_score <= 50 else ROUGE
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "%", "font": {"size": 34, "color": "#1A1A1A",
                                             "family": "Playfair Display"}},
            gauge={
                "axis":   {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6B7280",
                           "tickvals": [0, 25, 50, 75, 100]},
                "bar":    {"color": needle_color, "thickness": 0.28},
                "bgcolor": "white", "bordercolor": "#E5E7EB", "borderwidth": 1,
                "threshold": {"line": {"color": OR, "width": 5}, "thickness": 0.85, "value": 50},
                "steps": [
                    {"range": [0,  50],  "color": "#D1FAE5"},
                    {"range": [50, 75],  "color": "#FEF3C7"},
                    {"range": [75, 100], "color": "#FEE2E2"},
                ],
            },
        ))
        fig_g.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=5),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(f"""
        <div style="text-align:center;margin-top:-10px;font-size:.85rem;
                    color:{needle_color};font-weight:700;letter-spacing:1px">
          {dec_icon} {decision_finale}
        </div>""", unsafe_allow_html=True)

    # ── RECOMMANDATION & SOLUTIONS ────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("💡 Recommandation & Plan d'Action")

    if risk_score <= 35:
        rec_bg = VERT_BG; rec_bdr = VERT; rec_icon = "✅"
        rec_titre = "Crédit Accordé — Profil Très Fiable"
        rec_text  = (
            f"Le score XGBoost de <b>{risk_score:.1f}%</b> est très bas. "
            "Ce client présente un excellent profil de solvabilité. "
            "Les deux modèles convergent : risque minimal détecté."
        )
        solutions = [
            ("✅", VERT_C, "Accorder le crédit",       "Le dossier peut être approuvé immédiatement sans conditions particulières."),
            ("📋", BLEU,   "Suivi standard",            "Appliquer le suivi mensuel habituel. Aucune garantie supplémentaire requise."),
            ("💰", VERT,   "Montant accordable",        f"Le montant de <b>{credit:,.0f} TND</b> sur <b>{duration} mois</b> est cohérent."),
            ("📈", VERT_C, "Opportunité fidélisation",  "Profil idéal pour proposer épargne, assurance crédit, produits complémentaires."),
        ]
    elif risk_score <= 50:
        rec_bg = VERT_BG; rec_bdr = VERT; rec_icon = "✅"
        rec_titre = "Crédit Accordé — BON CLIENT (Score ≤ 50%)"
        rec_text  = (
            f"Le score XGBoost de <b>{risk_score:.1f}%</b> reste sous le seuil de 50%. "
            "Ce client est classifié <b>BON CLIENT</b>. "
            f"{'⚠️ Nota : l\'Arbre de Décision est plus pessimiste — vigilance recommandée.' if not convergence else 'Les deux modèles confirment ce profil acceptable.'}"
        )
        solutions = [
            ("✅", VERT_C, "Crédit approuvé",          "Le dossier peut être accepté conformément à la règle de décision (seuil 50%)."),
            ("🔍", OR,     "Vérification recommandée", f"Score proche du seuil ({risk_score:.1f}% vs 50%). Vérifier les justificatifs de revenus."),
            ("📅", BLEU,   "Suivi renforcé",           "Suivi mensuel les 6 premiers mois pour s'assurer du remboursement."),
            ("💳", OR,     "Garantie optionnelle",     "Suggérer une assurance crédit ou un co-emprunteur pour sécuriser le dossier."),
        ]
    elif risk_score <= 75:
        rec_bg = "#FEF3C7"; rec_bdr = "#D97706"; rec_icon = "⚠️"
        rec_titre = "Crédit Refusé — CLIENT À RISQUE (Score > 50%)"
        rec_text  = (
            f"Le score XGBoost de <b>{risk_score:.1f}%</b> dépasse le seuil de 50%. "
            "Ce client est classifié <b>CLIENT À RISQUE</b>. "
            f"{'Les deux modèles confirment ce risque.' if not convergence else '⚠️ Nota : L\'Arbre est plus optimiste, mais le score XGBoost prime.'}"
        )
        solutions = [
            ("🚫", OR,    "Refus provisoire",       "Le dossier doit être refusé en l'état. Des conditions supplémentaires sont requises."),
            ("🏦", BLEU,  "Garantie obligatoire",   "Exiger une garantie réelle (bien immobilier, véhicule) ou un garant solvable."),
            ("💰", OR,    "Réduction du montant",   f"Proposer {credit*0.6:,.0f} TND au lieu de {credit:,.0f} TND pour diminuer l'exposition."),
            ("📅", ROUGE, "Durée raccourcie",       f"Réduire la durée à {max(6, duration//2)} mois pour limiter le risque de défaut."),
            ("🔁", BLEU,  "Réévaluation 6 mois",   "Inviter le client à améliorer son épargne et revenir dans 6 mois."),
        ]
    else:
        rec_bg = "#FEE2E2"; rec_bdr = "#B91C1C"; rec_icon = "🚫"
        rec_titre = "Crédit Refusé — Risque Très Élevé (Score > 75%)"
        rec_text  = (
            f"Le score XGBoost atteint <b>{risk_score:.1f}%</b> — risque très significatif. "
            "Ce profil présente des signaux d'alerte multiples. "
            "Un refus ferme est fortement recommandé."
        )
        solutions = [
            ("🚫", ROUGE, "Refus ferme",            "Ce dossier doit être refusé. Le niveau de risque est trop élevé pour tout financement."),
            ("📋", ROUGE, "Analyse anti-fraude",    "Vérifier l'authenticité des documents et croiser avec la centrale des risques."),
            ("🏦", BLEU,  "Microcrédit uniquement", f"Si financement envisagé : microcrédit garanti max {min(credit, 1500):,.0f} TND."),
            ("🔁", OR,    "Plan d'assainissement",  "Régularisation des comptes, épargne forcée 12 mois."),
            ("📞", BLEU,  "Orientation conseiller", "Diriger le client vers un conseiller financier."),
        ]

    st.markdown(f"""
    <div style="background:{rec_bg};border-left:6px solid {rec_bdr};
                border-radius:0 16px 16px 0;padding:1.3rem 1.8rem;margin-bottom:1.2rem">
      <div style="font-weight:700;font-size:1.05rem;color:{rec_bdr};margin-bottom:.5rem">
        {rec_icon} {rec_titre}
      </div>
      <div style="font-size:.9rem;color:#374151;line-height:1.8">{rec_text}</div>
    </div>
    """, unsafe_allow_html=True)

    n_sol    = len(solutions)
    cols_sol = st.columns(min(n_sol, 3)) if n_sol <= 3 else st.columns(3)
    for idx, (icon, color, titre, detail) in enumerate(solutions):
        with cols_sol[idx % 3]:
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:1rem;
                        border-top:4px solid {color};
                        box-shadow:0 3px 12px rgba(0,0,0,.08);
                        margin-bottom:.7rem;min-height:110px">
              <div style="font-size:1.2rem;margin-bottom:.3rem">{icon}</div>
              <div style="font-weight:700;font-size:.85rem;color:{color};
                          margin-bottom:.3rem">{titre}</div>
              <div style="font-size:.78rem;color:#4B5563;line-height:1.5">{detail}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("📋 Détail des données saisies"):
        df_disp = pd.DataFrame([client_raw]).T.rename(columns={0: "Valeur"})
        st.dataframe(df_disp, use_container_width=True)

    st.success(f"✅ Analyse enregistrée dans l'historique — {datetime.datetime.now().strftime('%H:%M:%S')}")

    # ── PDF ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    pdf_bytes = generate_prediction_pdf(
        client_raw=client_raw, tree_class=tree_class, tree_proba=tree_proba,
        risk_score=risk_score, tree_label=decision_finale,
        analyste=u["name"], rec_titre=rec_titre, rec_text=rec_text,
    )
    _pc1, _pc2 = st.columns([3, 1])
    with _pc1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{VERT_DARK},{VERT});
          border-radius:14px;padding:1rem 1.5rem;border:1.5px solid {OR}55;
          display:flex;align-items:center;gap:12px;
          box-shadow:0 6px 20px rgba(0,107,60,.25)">
          <div style="font-size:2rem">📄</div>
          <div>
            <div style="font-family:'Playfair Display',serif;font-size:.95rem;
              color:white;font-weight:700">Rapport d'Analyse Complet — Amen Bank</div>
            <div style="font-size:.7rem;color:rgba(255,255,255,.55);margin-top:2px">
              Décision · Score XGBoost · Recommandations · Données client · Barre de risque
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    with _pc2:
        if pdf_bytes:
            st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ PDF",
                data=pdf_bytes,
                file_name=f"amen_bank_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="pdf_top",
            )

    # ══════════════════════════════════════════════════════
    #  IA ANALYSTE — Groq llama-3.3-70b-versatile  v2.0
    #  Auto-analyse → PDF  |  Graphiques auto  |  Chat libre
    # ══════════════════════════════════════════════════════
    import requests as _req_ia

    IA_MODEL  = "llama-3.3-70b-versatile"
    _emp_lbl  = {0:"Sans emploi",1:"Non qualifié",2:"Qualifié",3:"Très qualifié"}
    _mens     = credit / duration
    _ratio    = duration / age

    # ════════════════════════════════════════════════════════
    #  PROMPT SYSTÈME ULTRA-RICHE
    # ════════════════════════════════════════════════════════
    _sys_prompt = f"""Tu es Dr. Analyste, expert senior en risque crédit à Amen Bank Tunisie (30 ans d'expérience).
Réponds TOUJOURS en français, avec précision chirurgicale et profondeur analytique.
Sois chiffré, actionnable, professionnel. Structure avec des emojis de section. Max 500 mots.

══════════ DOSSIER CLIENT COMPLET ══════════
👤 PROFIL PERSONNEL
  • Âge : {age} ans  |  Sexe : {TR_SEX.get(str(sex), str(sex))}
  • Emploi : {_emp_lbl.get(job, str(job))} (niveau {job}/3)
  • Logement : {TR_HOUSING.get(str(housing), str(housing))}

💰 SITUATION FINANCIÈRE
  • Épargne : {TR_SAVING.get(str(saving), str(saving))}
  • Compte courant : {TR_CHECKING.get(str(checking), str(checking))}
  • Crédit demandé : {credit:,.0f} TND sur {duration} mois
  • Mensualité estimée : {_mens:,.0f} TND/mois
  • Objet : {TR_PURPOSE.get(str(purpose), str(purpose))}
  • Ratio durée/âge : {_ratio:.3f} {"⚠️ ÉLEVÉ" if _ratio > 0.6 else "✅ OK"}

📊 SCORES & MODÈLES
  • XGBoost : {risk_score:.1f}% → {'✅ BON CLIENT (≤50%)' if risk_score<=50 else '❌ CLIENT À RISQUE (>50%)'}
  • Arbre de Décision : {arbre_avis}
  • Convergence modèles : {'✅ OUI — Les deux modèles s\'accordent' if convergence else '⚠️ NON — Divergence à analyser'}
  • DÉCISION FINALE : {decision_finale}

⚖️ RÈGLE BCT : Score XGBoost ≤ 50% = BON CLIENT  /  > 50% = CLIENT À RISQUE
══════════════════════════════════════════

INSTRUCTIONS GRAPHIQUES IMPORTANTES :
Quand une visualisation renforcerait ton analyse, insère UN de ces marqueurs exacts sur une ligne seule :
[GRAPH:importance]   → importance des 9 variables XGBoost
[GRAPH:distribution] → distribution des scores + position de ce client
[GRAPH:radar]        → radar normalisé du profil client
[GRAPH:pie]          → camembert bon client vs à risque du portefeuille
[GRAPH:convergence]  → comparaison profils bon vs mauvais client (barres groupées)
⚠️ Un seul marqueur maximum par réponse, uniquement si vraiment pertinent.

POUR L'ANALYSE INITIALE AUTOMATIQUE :
Structure exactement en 4 blocs avec ces titres :
1️⃣ DÉCISION ET JUSTIFICATION
2️⃣ POINTS FORTS DU DOSSIER
3️⃣ FACTEURS DE RISQUE IDENTIFIÉS
4️⃣ ACTIONS RECOMMANDÉES — AMEN BANK
Sois très précis : cite les chiffres exacts du dossier."""

    # ── CSS bloc IA v2 ───────────────────────────────────
    st.markdown(f"""
    <style>
    @keyframes blink{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.35;transform:scale(.8)}}}}
    @keyframes slidein{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:none}}}}
    @keyframes pulse{{0%,100%{{box-shadow:0 0 0 0 rgba(245,166,35,.4)}}50%{{box-shadow:0 0 0 8px rgba(245,166,35,0)}}}}

    .ia-card{{
      background:linear-gradient(160deg,#060D1A 0%,#0D1829 60%,#060D1A 100%);
      border-radius:24px;border:1.5px solid rgba(245,166,35,.4);
      box-shadow:0 24px 72px rgba(0,0,0,.7),inset 0 1px 0 rgba(255,255,255,.04);
      overflow:hidden;margin:2rem 0;animation:slidein .4s ease;
    }}
    .ia-card-hdr{{
      background:linear-gradient(90deg,{VERT_DARK} 0%,{VERT} 50%,{VERT_C} 100%);
      padding:1.2rem 2rem;border-bottom:3px solid {OR};
      display:flex;align-items:center;justify-content:space-between;
    }}
    .ia-card-av{{
      width:54px;height:54px;border-radius:50%;
      background:rgba(255,255,255,.08);border:2.5px solid {OR};
      display:flex;align-items:center;justify-content:center;font-size:1.6rem;
      box-shadow:0 0 0 6px rgba(245,166,35,.12);flex-shrink:0;animation:pulse 3s infinite;
    }}
    .ia-card-t{{font-family:'Playfair Display',serif;font-size:1.15rem;
      color:white;font-weight:700;line-height:1.2}}
    .ia-card-s{{font-size:.6rem;color:rgba(255,255,255,.42);
      letter-spacing:.9px;margin-top:4px;text-transform:uppercase;}}
    .ia-pill-live{{
      display:flex;align-items:center;gap:8px;
      background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);
      border-radius:99px;padding:6px 14px;
    }}
    .ia-dot-live{{
      width:9px;height:9px;background:#22C55E;border-radius:50%;
      box-shadow:0 0 0 3px rgba(34,197,94,.25);animation:blink 2s infinite;
    }}
    .ia-msgs-box{{
      min-height:320px;max-height:580px;overflow-y:auto;
      padding:1.2rem 1rem .9rem;
      background:rgba(255,255,255,.014);
      border-radius:16px;border:1px solid rgba(255,255,255,.05);
      margin-bottom:1rem;
      scrollbar-width:thin;scrollbar-color:#1F3350 transparent;
    }}
    .ia-row-u{{display:flex;justify-content:flex-end;margin-bottom:1.1rem;
      gap:10px;align-items:flex-end;animation:slidein .25s ease}}
    .ia-row-a{{display:flex;justify-content:flex-start;margin-bottom:1.1rem;
      gap:10px;align-items:flex-end;animation:slidein .25s ease}}
    .ia-bub-u{{
      background:linear-gradient(135deg,{VERT} 0%,{VERT_C} 100%);
      color:white;border-radius:20px 20px 5px 20px;
      padding:.8rem 1.2rem;max-width:78%;font-size:.86rem;line-height:1.74;
      box-shadow:0 6px 20px rgba(0,107,60,.4);
    }}
    .ia-bub-a{{
      background:linear-gradient(160deg,#141E30,#1a2744);
      border:1px solid rgba(100,130,200,.25);color:#C8D8EE;
      border-radius:20px 20px 20px 5px;
      padding:.85rem 1.2rem;max-width:88%;font-size:.86rem;line-height:1.8;
      box-shadow:0 6px 22px rgba(0,0,0,.35);
    }}
    .ia-bub-a b, .ia-bub-a strong{{color:{OR};}}
    .ia-ico{{width:36px;height:36px;border-radius:50%;flex-shrink:0;
      display:flex;align-items:center;justify-content:center;font-size:.95rem}}
    .ia-ico-u{{background:{VERT};border:2px solid {VERT_C};
      box-shadow:0 3px 10px rgba(0,107,60,.4)}}
    .ia-ico-a{{background:linear-gradient(135deg,#111827,#1e293b);
      border:2px solid {OR}55;box-shadow:0 3px 10px rgba(0,0,0,.4)}}
    .ia-empty-s{{display:flex;flex-direction:column;align-items:center;
      padding:3.5rem 1rem;color:#2D3F56;text-align:center;}}
    .ia-stat-bar{{
      display:flex;justify-content:space-between;align-items:center;
      padding:.6rem 1.2rem;background:rgba(0,0,0,.35);
      border-top:1px solid rgba(255,255,255,.05);
      font-size:.67rem;color:#3D5068;letter-spacing:.3px;
    }}
    .ia-sugg-lbl{{font-size:.6rem;color:#3D5068;letter-spacing:2px;
      text-transform:uppercase;font-weight:700;margin:.5rem 0 .6rem}}
    .ia-pdf-badge{{
      display:inline-flex;align-items:center;gap:6px;
      background:rgba(0,166,81,.15);border:1px solid rgba(0,166,81,.35);
      border-radius:8px;padding:5px 12px;font-size:.72rem;
      color:{VERT_C};font-weight:700;margin-top:.4rem;
    }}
    .ia-graph-wrap{{
      background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
      border-radius:14px;padding:.8rem;margin:.6rem 0 .3rem;
    }}
    </style>
    <div class="ia-card">
      <div class="ia-card-hdr">
        <div style="display:flex;align-items:center;gap:15px">
          <div class="ia-card-av">🤖</div>
          <div>
            <div class="ia-card-t">Dr. Analyste IA — Amen Bank</div>
            <div class="ia-card-s">Groq · Llama 3.3 · 70B · Analyse de risque crédit · 100% Gratuit</div>
          </div>
        </div>
        <div style="text-align:right">
          <div class="ia-pill-live">
            <div class="ia-dot-live"></div>
            <span style="font-size:.67rem;color:#22C55E;font-weight:700;letter-spacing:.6px">EN LIGNE</span>
          </div>
          <div style="font-size:.6rem;color:rgba(255,255,255,.3);margin-top:5px;letter-spacing:.5px">
            ANALYSE INTÉGRÉE AU PDF ✦
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Champ clé API ────────────────────────────────────
    _kc1, _kc2 = st.columns([5, 1])
    with _kc1:
        _new_key = st.text_input(
            "🔑 Clé API Groq — 100% gratuite sur console.groq.com → API Keys → Create Key",
            value=st.session_state.get("groq_api_key", ""),
            type="password",
            placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            key="ia_key_field",
            help="Gratuit, sans carte bancaire. 30 secondes pour créer un compte.",
        )
        if _new_key and _new_key != st.session_state.get("groq_api_key", ""):
            st.session_state.groq_api_key     = _new_key
            st.session_state.ia_pred_messages = []
            st.session_state.ia_auto_analysis = ""
            st.session_state.ia_graphs_shown  = []
            st.session_state.ia_pending       = True

    with _kc2:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Reset IA", key="ia_rst", use_container_width=True):
            st.session_state.ia_pred_messages = []
            st.session_state.ia_auto_analysis = ""
            st.session_state.ia_graphs_shown  = []
            st.session_state.ia_pending       = True
            st.rerun()

    _api_key = st.session_state.get("groq_api_key", "")

    # ── Pas de clé : invite ───────────────────────────────
    if not _api_key:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1A1200,#211800);
          border:1.5px solid {OR}45;border-radius:16px;
          padding:1.5rem 2rem;margin:.5rem 0 1.2rem;
          display:flex;align-items:flex-start;gap:18px">
          <div style="font-size:2.5rem;flex-shrink:0;margin-top:2px">🔑</div>
          <div>
            <div style="font-weight:700;color:{OR};font-size:1rem;margin-bottom:8px">
              Activez Dr. Analyste IA avec votre clé Groq gratuite
            </div>
            <div style="font-size:.82rem;color:rgba(255,200,50,.65);line-height:1.8">
              1. Allez sur <b style="color:{OR}AA">console.groq.com</b><br>
              2. Créez un compte — <b>gratuit, sans carte bancaire, 30 secondes</b><br>
              3. Menu <b>API Keys</b> → <b>Create API Key</b> → Copiez → Collez ci-dessus<br>
              <span style="color:rgba(255,255,255,.3);font-size:.73rem;margin-top:4px;display:block">
                ✦ L'IA analysera automatiquement le dossier et son analyse sera incluse dans le PDF.
              </span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        # ════════════════════════════════════════════════════
        #  AUTO-ANALYSE (une seule fois par dossier)
        # ════════════════════════════════════════════════════
        if st.session_state.get("ia_pending", False):
            st.session_state.ia_pending       = False
            st.session_state.ia_graphs_shown  = []
            _auto_q = (
                "Lance l'analyse complète de ce dossier en 4 sections structurées. "
                "Cite les chiffres exacts. Sois précis et actionnable. "
                "Si tu veux montrer l'importance des variables après la section 3, "
                "place le marqueur approprié."
            )
            st.session_state.ia_pred_messages = [{"role": "user", "content": _auto_q}]
            with st.spinner("🤖 Dr. Analyste IA prépare l'analyse complète du dossier…"):
                try:
                    _r = _req_ia.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {_api_key}",
                                 "Content-Type": "application/json"},
                        json={"model": IA_MODEL,
                              "messages": [{"role": "system", "content": _sys_prompt},
                                           {"role": "user",   "content": _auto_q}],
                              "max_tokens": 1800, "temperature": 0.55},
                        timeout=60,
                    )
                    _reply = (_r.json()["choices"][0]["message"]["content"]
                              if _r.status_code == 200
                              else f"⚠️ Erreur API ({_r.status_code}) — vérifiez votre clé Groq.")
                except Exception as _ex:
                    _reply = f"⚠️ Connexion impossible : {str(_ex)}"

                st.session_state.ia_pred_messages.append(
                    {"role": "assistant", "content": _reply})
                # ✦ Sauvegarder l'analyse nettoyée pour le PDF
                st.session_state.ia_auto_analysis = _clean_ia_text(_reply)
            st.rerun()

        # ════════════════════════════════════════════════════
        #  RENDU DU CHAT
        # ════════════════════════════════════════════════════
        _msgs      = st.session_state.get("ia_pred_messages", [])
        _graphs_to_render = []   # (index_après_msg, gtype)
        _html      = ""

        for _i, _m in enumerate(_msgs):
            if _m["role"] == "user" and _i == 0:
                continue  # cacher la question auto initiale
            if _m["role"] == "user":
                _html += f"""
                <div class="ia-row-u">
                  <div class="ia-bub-u">{_m['content']}</div>
                  <div class="ia-ico ia-ico-u">👤</div>
                </div>"""
            else:
                # Nettoyer le texte et détecter les marqueurs graphiques
                _raw_content = _m["content"]
                _gtypes      = _extract_graph_markers(_raw_content)
                _clean_ct    = _clean_ia_text(_raw_content).replace("\n", "<br>")

                # Mettre en forme les titres de section (1️⃣ … )
                import re as _re_fmt
                _clean_ct = _re_fmt.sub(
                    r'(1️⃣|2️⃣|3️⃣|4️⃣|━+)',
                    r'<b style="color:#F5A623">\1</b>',
                    _clean_ct,
                )
                _html += f"""
                <div class="ia-row-a">
                  <div class="ia-ico ia-ico-a">🤖</div>
                  <div class="ia-bub-a">{_clean_ct}</div>
                </div>"""

                if _gtypes:
                    for _gt in _gtypes:
                        if _gt not in st.session_state.ia_graphs_shown:
                            _graphs_to_render.append((_i, _gt))

        if not _html:
            _html = """<div class="ia-empty-s">
              <div style="font-size:3rem;margin-bottom:.8rem">⏳</div>
              <div style="color:#3D5068;font-size:.9rem">Génération en cours…</div>
            </div>"""

        st.markdown(f'<div class="ia-msgs-box">{_html}</div>', unsafe_allow_html=True)

        # ── Rendre les graphiques demandés par l'IA ───────────────
        if _graphs_to_render:
            for (_msg_idx, _gtype) in _graphs_to_render:
                if _gtype not in st.session_state.ia_graphs_shown:
                    st.session_state.ia_graphs_shown.append(_gtype)
                    with st.container():
                        st.markdown('<div class="ia-graph-wrap">', unsafe_allow_html=True)
                        _render_ia_graph(
                            _gtype, df_raw, risk_score, clf,
                            age, credit, duration, job,
                            FEATURES, VERT_C, VERT_DARK, OR, ROUGE, BLEU,
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

        # ── Traitement question user en attente ───────────
        if _msgs and _msgs[-1]["role"] == "user":
            _nu = sum(1 for _m in _msgs if _m["role"] == "user")
            _na = sum(1 for _m in _msgs if _m["role"] == "assistant")
            if _nu > _na:
                with st.spinner("🤖 Dr. Analyste IA réfléchit…"):
                    try:
                        _r2 = _req_ia.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {_api_key}",
                                     "Content-Type": "application/json"},
                            json={"model": IA_MODEL,
                                  "messages": [{"role": "system", "content": _sys_prompt}] + [
                                      {"role": _m["role"], "content": _m["content"]}
                                      for _m in _msgs],
                                  "max_tokens": 1600, "temperature": 0.58},
                            timeout=60,
                        )
                        _rep2 = (_r2.json()["choices"][0]["message"]["content"]
                                 if _r2.status_code == 200
                                 else f"⚠️ Erreur ({_r2.status_code})")
                    except Exception as _ex2:
                        _rep2 = f"⚠️ Erreur : {str(_ex2)}"
                    st.session_state.ia_pred_messages.append(
                        {"role": "assistant", "content": _rep2})
                    st.rerun()

        # ── Suggestions contextuelles intelligentes ───────
        st.markdown('<div class="ia-sugg-lbl">⚡ Questions de suivi — Cliquez pour envoyer</div>',
                    unsafe_allow_html=True)
        _purpose_fr = TR_PURPOSE.get(str(purpose), str(purpose))
        _sugg = [
            ("📊", "importance",
             "Montre l'importance des variables XGBoost pour ce dossier [GRAPH:importance]"),
            ("📈", "distribution",
             f"Où se situe ce score de {risk_score:.0f}% par rapport au portefeuille ? [GRAPH:distribution]"),
            ("🔍", "risque_specifique",
             f"Quels sont les 3 principaux facteurs de risque pour un crédit {_purpose_fr} de {credit:,.0f} TND ?"),
            ("💡", "garanties",
             f"Quelles garanties concrètes exiger pour un client de {age} ans avec ce profil ?"),
            ("⚖️", "comparaison",
             f"Compare ce profil aux clients similaires dans le portefeuille [GRAPH:convergence]"),
            ("💰", "montant_max",
             f"Quel est le montant maximum de crédit sécurisé pour ce profil précis ?"),
            ("🏛️", "bct",
             "Analyse la conformité de ce dossier avec la réglementation BCT Tunisie 2024."),
            ("🕸️", "radar",
             "Montre le radar normalisé du profil client [GRAPH:radar]"),
        ]
        _row1, _row2 = st.columns(4), st.columns(4)
        for _i, (_ic, _key, _txt) in enumerate(_sugg):
            _col = _row1[_i] if _i < 4 else _row2[_i - 4]
            with _col:
                _display = _clean_ia_text(_txt)
                _lbl     = f"{_ic} {_display[:30]}…" if len(_display) > 30 else f"{_ic} {_display}"
                if st.button(_lbl, key=f"ia_sg_{_key}", use_container_width=True):
                    st.session_state.ia_pred_messages.append(
                        {"role": "user", "content": _display})
                    st.rerun()

        # ── Saisie libre ──────────────────────────────────
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        _ci, _cb = st.columns([5, 1])
        with _ci:
            _fq = st.text_input(
                "_ia_fq",
                placeholder="💬 Posez n'importe quelle question sur ce dossier… L'IA peut aussi générer des graphiques.",
                label_visibility="collapsed",
                key="ia_input_libre",
            )
        with _cb:
            _fsnd = st.button("Envoyer ➤", use_container_width=True, key="ia_send_libre")
        if _fsnd and _fq.strip():
            st.session_state.ia_pred_messages.append(
                {"role": "user", "content": _fq.strip()})
            st.rerun()

        # ── Indicateur PDF ────────────────────────────────
        _has_ia = bool(st.session_state.get("ia_auto_analysis", "").strip())
        st.markdown(f"""
        <div style="margin-top:.6rem;display:flex;align-items:center;
          justify-content:space-between;flex-wrap:wrap;gap:.5rem">
          <div class="ia-stat-bar" style="border-radius:8px;flex:1">
            🤖 <b style="color:#4A6080">Groq</b> ·
            <code style="color:{OR};font-size:.63rem">{IA_MODEL}</code> ·
            {sum(1 for _m in _msgs if _m["role"]=="user" and _msgs.index(_m)>0)} échange(s)
          </div>
          {'<div class="ia-pdf-badge">✅ Analyse IA incluse dans le PDF</div>'
           if _has_ia
           else '<div style="font-size:.7rem;color:#3D5068;padding:.3rem .6rem">⏳ Analyse en attente d\'intégration PDF</div>'}
        </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    #  REGÉNÉRER LE PDF avec l'analyse IA (si disponible)
    # ════════════════════════════════════════════════════
    _ia_txt = st.session_state.get("ia_auto_analysis", "")
    if _ia_txt:
        pdf_bytes = generate_prediction_pdf(
            client_raw=client_raw, tree_class=tree_class, tree_proba=tree_proba,
            risk_score=risk_score, tree_label=decision_finale,
            analyste=u["name"], rec_titre=rec_titre, rec_text=rec_text,
            ia_analysis=_ia_txt,
        )

    # ── PDF en bas (toujours visible) ────────────────────
    if pdf_bytes:
        st.markdown("<br>", unsafe_allow_html=True)
        _contains_ia = bool(_ia_txt)
        _pdf_desc    = (
            "Décision · Score XGBoost · Recommandations · Données client · ✦ Analyse IA Groq"
            if _contains_ia else
            "Décision · Score XGBoost · Recommandations · Données client"
        )
        _pdf_lbl = "📄  Télécharger le Rapport PDF" + (" + Analyse IA ✦" if _contains_ia else "")
        _pdc1, _pdc2 = st.columns([3, 1])
        with _pdc1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{VERT_DARK},{VERT});
              border-radius:14px;padding:1rem 1.5rem;border:1.5px solid {OR}55;
              display:flex;align-items:center;gap:14px;
              box-shadow:0 6px 22px rgba(0,107,60,.28)">
              <div style="font-size:2.2rem">📄</div>
              <div>
                <div style="font-family:'Playfair Display',serif;font-size:.98rem;
                  color:white;font-weight:700">
                  Rapport Complet — Amen Bank
                  {"<span style='font-size:.65rem;background:rgba(245,166,35,.2);" +
                   "color:" + OR + ";border-radius:4px;padding:2px 7px;margin-left:8px;'" +
                   ">✦ IA Groq</span>" if _contains_ia else ""}
                </div>
                <div style="font-size:.7rem;color:rgba(255,255,255,.5);margin-top:3px">
                  {_pdf_desc}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        with _pdc2:
            st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ PDF",
                data=pdf_bytes,
                file_name=f"amen_bank_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="pdf_bottom",
            )

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE DONNÉES & HISTORIQUE
# ══════════════════════════════════════════════════════════════════════════════
def _badge_classification(val):
    """Retourne le HTML badge coloré selon la classification."""
    val = str(val).strip()
    if val == "BON CLIENT":
        return (
            "<span style='background:#D1FAE5;color:#065F46;border:1.5px solid #34D399;"
            "padding:3px 12px;border-radius:99px;font-size:.75rem;font-weight:700;"
            "white-space:nowrap'>✅ BON CLIENT</span>"
        )
    elif val == "CLIENT À RISQUE":
        return (
            "<span style='background:#FEE2E2;color:#991B1B;border:1.5px solid #F87171;"
            "padding:3px 12px;border-radius:99px;font-size:.75rem;font-weight:700;"
            "white-space:nowrap'>⚠️ CLIENT À RISQUE</span>"
        )
    return f"<span style='color:#6B7280'>{val}</span>"


def _score_bar(score):
    """Retourne une mini barre HTML avec score %."""
    try:
        s = float(score)
    except Exception:
        return str(score)
    color = "#00A651" if s <= 50 else ("#F5A623" if s <= 75 else "#DC2626")
    return (
        f"<div style='display:flex;align-items:center;gap:6px'>"
        f"<div style='flex:1;background:#E5E7EB;border-radius:99px;height:7px;min-width:60px;overflow:hidden'>"
        f"<div style='width:{min(s,100):.0f}%;height:100%;background:{color};border-radius:99px'></div>"
        f"</div>"
        f"<span style='font-size:.78rem;font-weight:700;color:{color};min-width:38px'>{s:.1f}%</span>"
        f"</div>"
    )


def page_data(df):
    render_header("Données & Historique des Analyses")
    tab1, tab2 = st.tabs(["📊 Jeu de données", "📜 Historique des analyses"])

    # ══════════════════════════════════════════════════════════
    #  TAB 1 — JEU DE DONNÉES
    # ══════════════════════════════════════════════════════════
    with tab1:
        bad  = df[df["Risk"] == "bad"]
        good = df[df["Risk"] == "good"]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        kpi(c1, len(df),                                    "Total clients",   "🏦", "")
        kpi(c2, len(good),                                  "Bons clients",    "✅", "kpi-success")
        kpi(c3, len(bad),                                   "Clients à Risque","⚠️", "kpi-danger")
        kpi(c4, f"{len(bad)/len(df)*100:.1f}%",             "Taux de défaut",  "📊", "kpi-or")
        kpi(c5, f"{df['Credit amount'].median():,.0f} TND", "Médiane Crédit",  "💰", "kpi-blue")
        kpi(c6, f"{df['Duration'].mean():.1f} mois",        "Durée Moy.",      "📅", "")

        st.markdown("<br>", unsafe_allow_html=True)
        section("Aperçu du jeu de données")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_risk_fr = st.selectbox("Filtrer par risque", ["Tous", "Bon Client", "Client à Risque"])
            filter_risk = {"Bon Client": "good", "Client à Risque": "bad"}.get(filter_risk_fr, "Tous")
        with col_f2:
            sex_options_fr = ["Tous"] + [TR_SEX.get(v, v) for v in sorted(df["Sex"].dropna().unique().tolist())]
            sex_raw_map    = {TR_SEX.get(v, v): v for v in sorted(df["Sex"].dropna().unique().tolist())}
            filter_sex_fr  = st.selectbox("Filtrer par sexe", sex_options_fr)
            filter_sex     = sex_raw_map.get(filter_sex_fr, None)
        with col_f3:
            purp_options_fr = ["Tous"] + [TR_PURPOSE.get(v, v) for v in sorted(df["Purpose"].dropna().unique().tolist())]
            purp_raw_map    = {TR_PURPOSE.get(v, v): v for v in sorted(df["Purpose"].dropna().unique().tolist())}
            filter_purp_fr  = st.selectbox("Filtrer par objet", purp_options_fr)
            filter_purp     = purp_raw_map.get(filter_purp_fr, None)

        df_view = df.copy()
        if filter_risk != "Tous": df_view = df_view[df_view["Risk"]    == filter_risk]
        if filter_sex  is not None: df_view = df_view[df_view["Sex"]   == filter_sex]
        if filter_purp is not None: df_view = df_view[df_view["Purpose"] == filter_purp]
        st.dataframe(df_view.head(500), use_container_width=True)
        st.caption(f"{len(df_view)} lignes affichées sur {len(df)}")

        with st.expander("📈 Statistiques descriptives complètes"):
            st.dataframe(df.describe().round(2), use_container_width=True)
        with st.expander("🔢 Valeurs manquantes par colonne"):
            missing = df.isnull().sum().reset_index()
            missing.columns = ["Colonne", "Manquantes"]
            missing["Taux %"] = (missing["Manquantes"] / len(df) * 100).round(2)
            st.dataframe(missing, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    #  TAB 2 — HISTORIQUE DES ANALYSES
    # ══════════════════════════════════════════════════════════
    with tab2:
        hist = load_historique()   # déjà normalisé par load_historique()

        # Sécurité colonnes manquantes
        if "Classification" not in hist.columns:
            hist["Classification"] = "CLIENT À RISQUE"
        if "Score_Risque_pct" not in hist.columns:
            hist["Score_Risque_pct"] = 0.0

        if len(hist) == 0:
            st.markdown(f"""
            <div style="background:white;border-radius:18px;padding:3rem;text-align:center;
                        border:2px dashed #D1D5DB;margin:2rem 0">
              <div style="font-size:3rem;margin-bottom:1rem">📭</div>
              <div style="font-size:1.1rem;font-weight:700;color:#374151;margin-bottom:.5rem">
                Aucune analyse enregistrée</div>
              <div style="font-size:.87rem;color:#9CA3AF">
                Rendez-vous dans <b>Prédiction Client</b> pour lancer votre première analyse.</div>
            </div>""", unsafe_allow_html=True)
            return

        n_bon    = (hist["Classification"] == "BON CLIENT").sum()
        n_risque = (hist["Classification"] == "CLIENT À RISQUE").sum()
        score_moy = hist["Score_Risque_pct"].mean() if hist["Score_Risque_pct"].notna().any() else 0.0
        taux_app  = n_bon / len(hist) * 100 if len(hist) > 0 else 0
        score_min = hist["Score_Risque_pct"].min()
        score_max = hist["Score_Risque_pct"].max()

        # ── KPIs ────────────────────────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpi(k1, len(hist),            "Analyses totales",   "🔮", "")
        kpi(k2, n_bon,                "BON CLIENT",         "✅", "kpi-success")
        kpi(k3, n_risque,             "CLIENT À RISQUE",    "⚠️", "kpi-danger")
        kpi(k4, f"{taux_app:.1f}%",   "Taux approbation",   "📈", "kpi-success")
        kpi(k5, f"{score_moy:.1f}%",  "Score Moyen",        "⚡", "kpi-blue")
        kpi(k6, f"{score_min:.1f}–{score_max:.1f}%", "Plage Scores", "📊", "kpi-or")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bouton SUPPRIMER l'historique ───────────────────────
        with st.expander("🗑️  Supprimer l'historique", expanded=False):
            st.markdown(f"""
            <div style="background:#FEF2F2;border:1.5px solid #FECACA;border-radius:12px;
                        padding:1rem 1.2rem;margin-bottom:.8rem">
              <div style="font-weight:700;color:#991B1B;font-size:.9rem;margin-bottom:.3rem">
                ⚠️ Action irréversible
              </div>
              <div style="font-size:.82rem;color:#7F1D1D">
                Cette action supprimera définitivement les <b>{len(hist)}</b> analyses enregistrées.
                Cette opération ne peut pas être annulée.
              </div>
            </div>""", unsafe_allow_html=True)
            col_del1, col_del2 = st.columns([1, 2])
            with col_del1:
                if st.button("🗑️  Confirmer la suppression", type="primary",
                             use_container_width=True, key="btn_delete_hist"):
                    if os.path.exists(HISTORIQUE_CSV):
                        os.remove(HISTORIQUE_CSV)
                    st.success("✅ Historique supprimé avec succès.")
                    st.rerun()
            with col_del2:
                st.markdown(
                    "<div style='padding:.5rem;font-size:.78rem;color:#6B7280'>"
                    "💡 Vous pouvez aussi exporter l'historique avant de le supprimer.</div>",
                    unsafe_allow_html=True
                )

        # ── Filtres historique ──────────────────────────────────
        section("🔍 Filtres & Recherche")
        hf1, hf2, hf3, hf4 = st.columns(4)
        with hf1:
            f_cls = st.selectbox("Classification", ["Toutes", "BON CLIENT", "CLIENT À RISQUE"],
                                 key="hf_cls")
        with hf2:
            f_ana = st.selectbox("Analyste", ["Tous"] + sorted(hist["Analyste"].dropna().unique().tolist()),
                                 key="hf_ana")
        with hf3:
            score_range = st.slider("Plage score (%)", 0.0, 100.0,
                                    (float(max(0, score_min-1)), float(min(100, score_max+1))),
                                    0.5, key="hf_score")
        with hf4:
            f_objet = st.selectbox("Objet du crédit",
                                   ["Tous"] + (sorted(hist["Objet"].dropna().unique().tolist())
                                               if "Objet" in hist.columns else []),
                                   key="hf_obj")

        hist_f = hist.copy()
        if f_cls  != "Toutes": hist_f = hist_f[hist_f["Classification"] == f_cls]
        if f_ana  != "Tous":   hist_f = hist_f[hist_f["Analyste"] == f_ana]
        hist_f = hist_f[
            (hist_f["Score_Risque_pct"] >= score_range[0]) &
            (hist_f["Score_Risque_pct"] <= score_range[1])
        ]
        if f_objet != "Tous" and "Objet" in hist_f.columns:
            hist_f = hist_f[hist_f["Objet"] == f_objet]

        hist_f = hist_f.sort_values("Date", ascending=False).reset_index(drop=True)

        st.markdown(f"""
        <div style="background:#F0F9FF;border-left:4px solid #0EA5E9;border-radius:0 8px 8px 0;
                    padding:.6rem 1rem;margin-bottom:.8rem;font-size:.83rem;color:#0C4A6E">
          🔎 <b>{len(hist_f)}</b> analyse(s) affichée(s) sur <b>{len(hist)}</b> au total
        </div>""", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        #  TABLEAU HISTORIQUE COMPLET — HTML personnalisé
        # ══════════════════════════════════════════════════════
        section("📋 Tableau Historique Complet")

        col_map = {
            "Date":             "📅 Date & Heure",
            "Analyste":         "👤 Analyste",
            "Age":              "🎂 Âge",
            "Sexe":             "⚥ Sexe",
            "Emploi":           "💼 Emploi",
            "Logement":         "🏠 Logement",
            "Epargne":          "💰 Épargne",
            "Montant_Credit":   "💳 Montant (TND)",
            "Duree_Mois":       "📅 Durée",
            "Objet":            "🎯 Objet",
            "Score_Risque_pct": "⚡ Score",
            "Classification":   "🏷️ Décision",
            "Confiance_pct":    "🎯 Confiance",
        }
        cols_exist = [c for c in col_map if c in hist_f.columns]

        header_html = "".join([
            f"<th style='background:linear-gradient(135deg,{VERT_DARK},{VERT});"
            f"color:white;padding:10px 14px;text-align:left;font-size:.72rem;"
            f"font-weight:700;letter-spacing:.5px;white-space:nowrap;"
            f"border-right:1px solid rgba(255,255,255,.1)'>{col_map[c]}</th>"
            for c in cols_exist
        ])

        rows_html = ""
        for idx, row in hist_f.iterrows():
            bg      = "white" if idx % 2 == 0 else "#F9FAFB"
            cls_val = str(row.get("Classification", ""))

            cells = ""
            for c in cols_exist:
                val = row.get(c, "")
                style = (f"padding:9px 14px;font-size:.78rem;color:#374151;"
                         f"border-right:1px solid #F3F4F6;vertical-align:middle;"
                         f"white-space:nowrap")

                if c == "Classification":
                    cell_content = _badge_classification(val)

                elif c == "Score_Risque_pct":
                    cell_content = _score_bar(val)

                elif c == "Confiance_pct":
                    try:
                        cf = float(val)
                        color_cf = VERT_C if cf >= 70 else (OR if cf >= 50 else ROUGE)
                        cell_content = (
                            f"<div style='display:flex;align-items:center;gap:5px'>"
                            f"<div style='width:36px;height:36px;border-radius:50%;"
                            f"background:conic-gradient({color_cf} {cf*3.6:.0f}deg,#E5E7EB 0deg);"
                            f"display:flex;align-items:center;justify-content:center'>"
                            f"<div style='width:26px;height:26px;border-radius:50%;background:white;"
                            f"display:flex;align-items:center;justify-content:center;"
                            f"font-size:.6rem;font-weight:700;color:{color_cf}'>{cf:.0f}%</div>"
                            f"</div></div>"
                        )
                    except Exception:
                        cell_content = str(val)

                elif c == "Montant_Credit":
                    try:
                        cell_content = f"<b style='color:{BLEU}'>{float(val):,.0f}</b>"
                    except Exception:
                        cell_content = str(val)

                elif c == "Duree_Mois":
                    cell_content = f"<span style='background:#EFF6FF;color:{BLEU};padding:2px 8px;border-radius:99px;font-size:.72rem;font-weight:700'>{val} mois</span>"

                elif c == "Date":
                    parts = str(val).split(" ")
                    if len(parts) == 2:
                        cell_content = (f"<div style='line-height:1.3'>"
                                        f"<div style='font-weight:600;color:#374151;font-size:.77rem'>{parts[0]}</div>"
                                        f"<div style='color:#9CA3AF;font-size:.68rem'>{parts[1]}</div></div>")
                    else:
                        cell_content = f"<span style='color:#6B7280;font-size:.72rem'>{val}</span>"

                elif c == "Analyste":
                    initials = "".join([w[0].upper() for w in str(val).split()[:2]])
                    cell_content = (
                        f"<div style='display:flex;align-items:center;gap:7px'>"
                        f"<div style='width:28px;height:28px;border-radius:50%;"
                        f"background:linear-gradient(135deg,{VERT},{VERT_C});"
                        f"display:flex;align-items:center;justify-content:center;"
                        f"font-size:.65rem;font-weight:700;color:white;flex-shrink:0'>{initials}</div>"
                        f"<span style='font-weight:600;color:{VERT_DARK}'>{val}</span></div>"
                    )

                elif c == "Sexe":
                    icon = "👨" if str(val).lower() in ["male", "masculin", "homme"] else "👩"
                    fr_val = TR_SEX.get(str(val).lower(), val)
                    cell_content = f"{icon} {fr_val}"

                elif c == "Emploi":
                    cell_content = f"<span style='color:#374151'>{val}</span>"

                else:
                    cell_content = str(val) if pd.notna(val) else "—"

                cells += f"<td style='{style}'>{cell_content}</td>"

            row_border = ""
            if cls_val == "BON CLIENT":
                row_border = "border-left:4px solid #00A651"
            elif cls_val == "CLIENT À RISQUE":
                row_border = "border-left:4px solid #DC2626"

            rows_html += (
                f"<tr style='background:{bg};{row_border};transition:background .15s' "
                f"onmouseover=\"this.style.background='#EFF6FF'\" "
                f"onmouseout=\"this.style.background='{bg}'\">"
                f"{cells}</tr>"
            )

        empty_row = ("<tr><td colspan='13' style='text-align:center;padding:2.5rem;"
                     "color:#9CA3AF;font-size:.9rem'>📭 Aucun résultat pour ces filtres</td></tr>")

        st.markdown(f"""
        <div style="overflow-x:auto;border-radius:14px;
                    box-shadow:0 4px 20px rgba(0,0,0,.10);
                    border:1px solid #E5E7EB;margin-bottom:1rem">
          <table style="width:100%;border-collapse:collapse;font-family:'Cairo',sans-serif">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html if rows_html else empty_row}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

        # ── Export ────────────────────────────────────────────
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                "⬇️  Exporter la sélection (CSV)",
                data=hist_f.to_csv(index=False).encode("utf-8"),
                file_name=f"historique_selection_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True,
            )
        with col_exp2:
            st.download_button(
                "⬇️  Exporter tout l'historique (CSV)",
                data=hist.to_csv(index=False).encode("utf-8"),
                file_name=f"historique_complet_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True,
            )

        # ══════════════════════════════════════════════════════
        #  GRAPHIQUES ANALYTIQUES
        # ══════════════════════════════════════════════════════
        if len(hist) >= 2:
            st.markdown("<br>", unsafe_allow_html=True)
            section("📊 Analyse Visuelle de l'Historique")

            ga, gb = st.columns(2)
            with ga:
                fig_pie = go.Figure(go.Pie(
                    labels=["✅ BON CLIENT", "⚠️ CLIENT À RISQUE"],
                    values=[n_bon, n_risque],
                    hole=0.60, marker_colors=[VERT_C, ROUGE],
                    textinfo="label+percent+value", textfont_size=11, pull=[0.04, 0],
                ))
                fig_pie.update_layout(
                    title=dict(text="Répartition des décisions", font_size=13,
                               font_color=VERT_DARK),
                    paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(t=40, b=10, l=10, r=10),
                    legend=dict(orientation="h", y=-0.08),
                    annotations=[dict(
                        text=f"<b>{taux_app:.0f}%</b><br>approuvés",
                        x=0.5, y=0.5, font_size=16, showarrow=False, font_color=VERT_C,
                    )],
                    height=300,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with gb:
                fig_dist = go.Figure()
                for cls, clr in [("BON CLIENT", VERT_C), ("CLIENT À RISQUE", ROUGE)]:
                    sub = hist[hist["Classification"] == cls]["Score_Risque_pct"].dropna()
                    if len(sub) > 0:
                        fig_dist.add_trace(go.Histogram(
                            x=sub, name=cls, marker_color=clr, opacity=0.75, nbinsx=15,
                        ))
                fig_dist.add_vline(x=50, line_dash="dash", line_color=OR, line_width=2.5,
                                   annotation_text="Seuil 50%", annotation_font_color=OR,
                                   annotation_font_size=11)
                fig_dist.update_layout(
                    title=dict(text="Distribution des scores XGBoost", font_size=13,
                               font_color=VERT_DARK),
                    barmode="overlay", paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                    font_color="#374151",
                    xaxis=dict(title="Score (%)", gridcolor="#E5E7EB", zeroline=False,
                               ticksuffix="%"),
                    yaxis=dict(title="Nombre d'analyses", gridcolor="#E5E7EB", zeroline=False),
                    legend=dict(orientation="h", y=1.12),
                    height=300, margin=dict(t=50, b=20, l=40, r=20),
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            if len(hist) >= 4:
                hist_sorted = hist.sort_values("Date").reset_index(drop=True)
                hist_sorted["N°"] = range(1, len(hist_sorted) + 1)
                fig_evo = go.Figure()
                fig_evo.add_hrect(y0=0, y1=50, fillcolor=VERT_C, opacity=0.05, line_width=0,
                                  annotation_text="Zone BON CLIENT",
                                  annotation_position="top left",
                                  annotation_font_color=VERT_C, annotation_font_size=10)
                fig_evo.add_hrect(y0=50, y1=105, fillcolor=ROUGE, opacity=0.04, line_width=0,
                                  annotation_text="Zone CLIENT À RISQUE",
                                  annotation_position="bottom left",
                                  annotation_font_color=ROUGE, annotation_font_size=10)
                fig_evo.add_trace(go.Scatter(
                    x=hist_sorted["N°"], y=hist_sorted["Score_Risque_pct"],
                    mode="lines+markers",
                    line=dict(color=BLEU, width=2.5, shape="spline"),
                    marker=dict(
                        color=[VERT_C if c == "BON CLIENT" else ROUGE
                               for c in hist_sorted["Classification"]],
                        size=10, line=dict(color="white", width=2),
                    ),
                    name="Score XGBoost",
                    hovertemplate="<b>Analyse #%{x}</b><br>Score : %{y:.1f}%<extra></extra>",
                ))
                fig_evo.add_hline(y=50, line_dash="dash", line_color=OR, line_width=2,
                                  annotation_text="⬅ Seuil 50%",
                                  annotation_font_color=OR, annotation_font_size=11)
                fig_evo.update_layout(
                    title=dict(text="📈 Évolution chronologique des scores",
                               font_size=13, font_color=VERT_DARK),
                    paper_bgcolor="white", plot_bgcolor="#FAFAFA", font_color="#374151",
                    xaxis=dict(title="N° Analyse", gridcolor="#E5E7EB", zeroline=False,
                               tickmode="linear"),
                    yaxis=dict(title="Score (%)", gridcolor="#E5E7EB", zeroline=False,
                               range=[0, 105], ticksuffix="%"),
                    height=340, margin=dict(t=50, b=30, l=50, r=30), showlegend=False,
                )
                st.plotly_chart(fig_evo, use_container_width=True)

            if hist["Analyste"].nunique() > 1:
                section("👤 Performance par Analyste")
                ana_grp = hist.groupby("Analyste").agg(
                    Total=("Classification", "count"),
                    Bons=("Classification", lambda x: (x == "BON CLIENT").sum()),
                    Risques=("Classification", lambda x: (x == "CLIENT À RISQUE").sum()),
                    Score_Moy=("Score_Risque_pct", "mean"),
                ).reset_index()
                ana_grp["Taux Approbation"] = (ana_grp["Bons"] / ana_grp["Total"] * 100).round(1).astype(str) + "%"
                ana_grp["Score Moyen"]      = ana_grp["Score_Moy"].round(1).astype(str) + "%"
                ana_grp = ana_grp.drop(columns=["Score_Moy"])
                ana_grp.columns = ["Analyste", "Total", "✅ BON CLIENT",
                                   "⚠️ CLIENT À RISQUE", "Taux Approbation", "Score Moyen"]
                st.dataframe(ana_grp, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE ANALYSTE IA — GROQ (100% GRATUIT)
# ══════════════════════════════════════════════════════════════════════════════
def page_ia_chat(df_raw):
    """Page Analyste IA — Chat avec Groq API (100% gratuit)"""
    import requests as _req

    render_header("🤖 Analyste IA — Groq (100% Gratuit)", "Assistant Intelligent d'Analyse Crédit · Sans crédit · Sans carte bancaire")

    # ── CSS (identique à avant, je le garde) ─────────────────
    st.markdown(f"""
    <style>
    .ia-chat-wrap {{
      background: white; border-radius: 18px; border: 1.5px solid #E2E8F0;
      box-shadow: 0 6px 28px rgba(0,107,60,.10); overflow: hidden; margin-bottom: 1rem;
    }}
    .ia-chat-header {{
      background: linear-gradient(135deg,{VERT_DARK} 0%,{VERT} 60%,{VERT_C} 100%);
      padding: .85rem 1.4rem; display: flex; align-items: center; gap: 12px;
      border-bottom: 3px solid {OR};
    }}
    .ia-chat-header .ia-avatar {{
      width: 40px; height: 40px; border-radius: 50%; background: rgba(255,255,255,.15);
      border: 2px solid {OR}; display: flex; align-items: center; justify-content: center;
      font-size: 1.25rem;
    }}
    .ia-chat-header .ia-title {{
      font-family: 'Playfair Display', serif; font-size: 1.05rem;
      color: white; font-weight: 700; line-height: 1;
    }}
    .ia-chat-header .ia-sub {{
      font-size: .65rem; color: rgba(255,255,255,.55); letter-spacing: .5px; margin-top: 2px;
    }}
    .ia-online-dot {{
      width: 9px; height: 9px; background: #22C55E; border-radius: 50%;
      box-shadow: 0 0 0 3px rgba(34,197,94,.25); margin-left: auto;
      animation: pulse-dot 2s infinite;
    }}
    @keyframes pulse-dot {{ 0%,100% {{opacity:1}} 50% {{opacity:.4}} }}
    .ia-messages {{
      min-height: 340px; max-height: 420px; overflow-y: auto;
      padding: 1.2rem 1.2rem .5rem; background: #F8FAFC;
    }}
    .ia-msg-user {{ display: flex; justify-content: flex-end; margin-bottom: .85rem; gap: 8px; align-items: flex-end; }}
    .ia-msg-ai   {{ display: flex; justify-content: flex-start;  margin-bottom: .85rem; gap: 8px; align-items: flex-end; }}
    .ia-bubble-user {{
      background: linear-gradient(135deg,{VERT} 0%,{VERT_C} 100%);
      color: white; border-radius: 18px 18px 4px 18px;
      padding: .7rem 1rem; max-width: 76%; font-size: .87rem; line-height: 1.65;
      box-shadow: 0 3px 12px rgba(0,107,60,.22);
    }}
    .ia-bubble-ai {{
      background: white; border: 1px solid #E2E8F0; color: #1A202C;
      border-radius: 18px 18px 18px 4px;
      padding: .7rem 1rem; max-width: 76%; font-size: .87rem; line-height: 1.65;
      box-shadow: 0 3px 12px rgba(0,0,0,.06);
    }}
    .ia-av {{
      width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center; font-size: .9rem;
    }}
    .ia-av-user {{ background: {VERT}; }}
    .ia-av-ai   {{ background: linear-gradient(135deg,#1A1A2E,#16213E); }}
    .ia-empty {{
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      padding: 2.5rem 1rem; color: #9CA3AF; text-align: center;
    }}
    .ia-empty .ia-empty-icon {{ font-size: 2.8rem; margin-bottom: .6rem; }}
    .ia-empty .ia-empty-title {{ font-size: 1rem; font-weight: 700; color: #4B5563; margin-bottom: .3rem; }}
    </style>
    """, unsafe_allow_html=True)

    # ── Stats du portefeuille ───────────────────────────────
    n_total    = len(df_raw)
    n_good     = int((df_raw["Risk"] == "good").sum())
    n_bad      = int((df_raw["Risk"] == "bad").sum())
    avg_age    = df_raw["Age"].mean()
    avg_credit = df_raw["Credit amount"].mean()
    avg_dur    = df_raw["Duration"].mean()
    taux_bon   = n_good / n_total * 100

    # ── KPI rapides ─────────────────────────────────────────
    section("📊 Contexte du Portefeuille Analysé")
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, f"{n_total:,}",          "Clients Totaux",        "👥")
    kpi(k2, f"{taux_bon:.1f}%",      "Taux Bons Clients",     "✅", "kpi-success")
    kpi(k3, f"{n_bad/n_total*100:.1f}%", "Taux de Risque",   "⚠️", "kpi-danger")
    kpi(k4, f"{avg_credit:,.0f}",    "Crédit Moyen (TND)",    "💰", "kpi-or")
    kpi(k5, f"{avg_dur:.0f} mois",   "Durée Moyenne",         "📅", "kpi-blue")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Système prompt ──────────────────────────────────────
    ctx_stats = (
        f"Portefeuille : {n_total} clients | "
        f"{n_good} bons clients ({taux_bon:.1f}%) | "
        f"{n_bad} clients à risque ({n_bad/n_total*100:.1f}%) | "
        f"Âge moyen : {avg_age:.1f} ans | "
        f"Crédit moyen : {avg_credit:,.0f} TND | "
        f"Durée moyenne : {avg_dur:.1f} mois."
    )
    purpose_dist = df_raw["Purpose"].value_counts().head(4).to_dict()
    purpose_str  = " | ".join(f"{k}: {v}" for k, v in purpose_dist.items())

    SYSTEM_PROMPT = f"""Tu es un analyste expert en risque crédit pour Amen Bank Tunisie.
Réponds TOUJOURS en français, de façon professionnelle et structurée.
Utilise des emojis pour organiser ta réponse.

═══ DONNÉES ACTUELLES DU PORTEFEUILLE ═══
{ctx_stats}
Top objets de crédit : {purpose_str}

═══ MODÈLES DÉPLOYÉS ═══
• Arbre de Décision : classification Bon Client / Client à Risque
• XGBoost : prédit un score de risque (0-100%)
  Règle : Score ≤ 50% → BON CLIENT | Score > 50% → CLIENT À RISQUE

═══ CONSIGNE ═══
Sois concis, chiffré et actionnable. Recommandations conformes à la réglementation BCT."""

    # ── Config API GROQ ─────────────────────────────────────
    col_key, col_clear = st.columns([4, 1])
    with col_key:
        api_key = st.text_input(
            "🔑 Clé API Groq (Gratuite)",
            value=st.session_state.get("groq_api_key", ""),
            type="password",
            placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            help="Obtenez votre clé sur console.groq.com (gratuit, sans carte bancaire)",
        )
        if api_key != st.session_state.get("groq_api_key", ""):
            st.session_state.groq_api_key = api_key
    with col_clear:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Effacer chat", use_container_width=True):
            st.session_state.ia_messages = []
            st.rerun()

    # Modèle garanti actif
    MODEL_NAME = "llama-3.3-70b-versatile"

    st.markdown(f"""
    <div style="background:{VERT_BG};border:1px solid {VERT};border-radius:8px;
                padding:.45rem .8rem;font-size:.78rem;color:{VERT_DARK};margin-bottom:1rem">
      🤖 <b>Modèle actif :</b> <code>{MODEL_NAME}</code> — 100% gratuit · Supporté par Groq
    </div>""", unsafe_allow_html=True)

    if not api_key:
        st.markdown(f"""
        <div style="background:#FEF3C7;border-left:4px solid {OR};border-radius:0 10px 10px 0;
                    padding:.75rem 1.2rem;font-size:.87rem;color:#92400E;margin-bottom:1rem">
          🎉 <b>GROQ API 100% GRATUITE</b> — Créez un compte sur 
          <a href="https://console.groq.com" target="_blank" style="color:{BLEU}">console.groq.com</a>
          (sans carte bancaire), copiez votre clé et collez-la ci-dessus.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Suggestions rapides ─────────────────────────────────
    section("⚡ Questions Rapides")
    SUGGESTIONS = [
        ("📊", "Analyse le profil type du client à risque dans ce portefeuille"),
        ("💡", "Quels sont les 3 facteurs qui augmentent le plus le risque de défaut ?"),
        ("🎯", "Donne-moi 5 recommandations concrètes pour réduire le taux de risque"),
        ("⚖️", "Compare les profils des bons clients vs clients à risque"),
        ("💰", "Quels objets de crédit présentent le plus grand risque ?"),
    ]
    cols_s = st.columns(3)
    for i, (icon, txt) in enumerate(SUGGESTIONS):
        with cols_s[i % 3]:
            if st.button(f"{icon} {txt[:45]}…" if len(txt) > 45 else f"{icon} {txt}",
                         key=f"ia_sug_{i}", use_container_width=True):
                st.session_state.ia_messages.append({"role": "user", "content": txt})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    section("💬 Fenêtre de Chat IA")

    # ── Rendu du chat ───────────────────────────────────────
    msgs = st.session_state.ia_messages
    bubbles_html = ""
    if not msgs:
        bubbles_html = f"""
        <div class="ia-empty">
          <div class="ia-empty-icon">🤖</div>
          <div class="ia-empty-title">Groq — Analyste Risque Crédit</div>
          <div style="font-size:.83rem">Posez votre question ou cliquez sur une suggestion ci-dessus.<br>🔥 <b>100% gratuit</b> — Pas de crédits, pas de carte bancaire !</div>
        </div>"""
    else:
        for m in msgs:
            if m["role"] == "user":
                bubbles_html += f"""
                <div class="ia-msg-user">
                  <div class="ia-bubble-user">{m['content']}</div>
                  <div class="ia-av ia-av-user">👤</div>
                </div>"""
            else:
                content = m["content"].replace("\n", "<br>")
                bubbles_html += f"""
                <div class="ia-msg-ai">
                  <div class="ia-av ia-av-ai">🤖</div>
                  <div class="ia-bubble-ai">{content}</div>
                </div>"""

    st.markdown(f"""
    <div class="ia-chat-wrap">
      <div class="ia-chat-header">
        <div class="ia-avatar">🤖</div>
        <div>
          <div class="ia-title">Groq Analyste — Amen Bank</div>
          <div class="ia-sub">Risque Crédit · 100% GRATUIT · Modèle Llama 3.3 70B</div>
        </div>
        <div class="ia-online-dot" title="En ligne"></div>
      </div>
      <div class="ia-messages">{bubbles_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Traitement de la réponse ────────────────────────────
    if msgs and msgs[-1]["role"] == "user":
        if not api_key:
            st.warning("⚠️ Entrez votre clé API Groq pour recevoir une réponse.")
        else:
            with st.spinner("🤖 Groq analyse votre demande (100% gratuit)…"):
                try:
                    payload = {
                        "model": MODEL_NAME,
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + [
                            {"role": m["role"] if m["role"] != "assistant" else "assistant",
                             "content": m["content"]} for m in msgs
                        ],
                        "max_tokens": 1200,
                        "temperature": 0.65,
                    }
                    resp = _req.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        reply = resp.json()["choices"][0]["message"]["content"]
                        st.session_state.ia_messages.append({"role": "assistant", "content": reply})
                        st.rerun()
                    else:
                        st.error(f"❌ Erreur API ({resp.status_code}) : {resp.text[:500]}")
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")

    # ── Zone de saisie ──────────────────────────────────────
    col_inp, col_send = st.columns([5, 1])
    with col_inp:
        user_input = st.text_input(
            "msg",
            placeholder="💬 Ex : Analyse les clients de plus de 50 ans dans ce portefeuille…",
            label_visibility="collapsed",
            key="ia_text_input",
        )
    with col_send:
        send_btn = st.button("Envoyer ➤", use_container_width=True, key="ia_send")

    if send_btn and user_input.strip():
        st.session_state.ia_messages.append({"role": "user", "content": user_input.strip()})
        st.rerun()

    n_msgs = len([m for m in msgs if m["role"] == "user"])
    st.markdown(f"""
    <div style="margin-top:1.2rem;padding:.8rem 1.2rem;background:#F8FAFC;
                border-radius:10px;border:1px solid #E2E8F0;font-size:.73rem;color:#6B7280">
      🤖 <b>Groq API — 100% GRATUIT</b> · Modèle : <code>{MODEL_NAME}</code>
      · Messages : <b>{n_msgs}</b> · Contexte : {n_total} clients · Amen Bank Tunisie
      <span style="float:right;color:#00A651;font-weight:700">✅ Aucun crédit nécessaire</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Auth ────────────────────────────────────────────────
    if not st.session_state.logged_in:
        page_login()
        return

    page = render_sidebar()

    # ── Chargement des données ───────────────────────────────
    try:
        df_raw = load_data(st.session_state["df_uploaded"])
    except FileNotFoundError:
        st.error(
            "❌ Fichier de données introuvable. "
            "Placez `german_credit_prediction_clean.csv` dans le même dossier "
            "ou importez votre CSV via la barre latérale."
        )
        return

    # ── Encodage + entraînement ──────────────────────────────
    df_enc, encoders = encode_df(df_raw)

    # Stocker df_enc pour les fonctions cachées
    st.session_state["_df_enc_cache"] = df_enc
    df_hash = str(len(df_enc)) + "_" + str(df_enc.columns.tolist())

    with st.spinner("⚙️ Entraînement des modèles en cours…"):
        clf, tree_metrics, splits    = train_tree(df_hash)
        xgb_model, scaler, xgb_met  = train_xgb(df_hash)

    # ── Routage des pages ────────────────────────────────────
    if page == "Tableau de Bord":
        page_dashboard(df_raw)

    elif page == "Exploration":
        page_eda(df_raw)

    elif page == "Prédiction Client":
        page_prediction(df_raw, df_raw, df_enc, encoders, clf, xgb_model, scaler)

    elif page == "Données":
        page_data(df_raw)

    # ── Footer ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="footer">
      © 2025 <span>Amen Bank Tunisie</span> — Direction des Risques<br>
      Système Risque Crédit v8.0 Professional &nbsp;·&nbsp;
      ⚡ <span>XGBoost</span> — Prédiction % Risque &nbsp;·&nbsp;
      🤖 <span>Groq Llama 3.3 70B</span> — IA Analyste Intégrée &nbsp;·&nbsp;
      Seuil de décision : Score ≤ 50% → BON CLIENT &nbsp;|&nbsp; Score &gt; 50% → CLIENT À RISQUE
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()