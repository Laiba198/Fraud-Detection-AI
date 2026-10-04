import os, requests
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from catboost import CatBoostClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score)

st.set_page_config(page_title="Fraud Detector", page_icon="🛡️", layout="wide")

BG, SIDE, CYAN, VIOLET, RED, GREEN = "#07111F", "#0D1B2A", "#35D5E8", "#8B7CFF", "#FF647C", "#32D6A0"
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] {{font-family:'Inter',sans-serif;}}
.stApp {{background:{BG}; color:#E6EDF5;}}
[data-testid="stSidebar"] {{background:{SIDE}; border-right:1px solid #1B2D42;}}
h1,h2,h3 {{color:#fff;}}
.card {{background:linear-gradient(145deg,#0F2236,#0B1A2B); border:1px solid #1E3450; border-radius:14px;
        padding:16px 18px; animation:fade .6s ease; transition:.2s;}}
.card:hover {{border-color:{CYAN}; transform:translateY(-2px);}}
.card .l {{font-size:.72rem; color:#8FA3BA; text-transform:uppercase; letter-spacing:.07em;}}
.card .v {{font-size:1.8rem; font-weight:700;}}
.stButton>button {{background:{CYAN}; color:#07111F; font-weight:600; border:0; border-radius:10px;}}
@keyframes fade {{from {{opacity:0; transform:translateY(8px);}} to {{opacity:1; transform:none;}}}}
</style>""", unsafe_allow_html=True)

REQ = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
FEATS = REQ + ["errorOrig", "errorDest"]
MAX_MB = 50


def cards(items):
    for col, (label, value, color) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="card"><div class="l">{label}</div>'
                     f'<div class="v" style="color:{color}">{value}</div></div>', unsafe_allow_html=True)


def style(fig, h=320):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=45, b=10), font=dict(family="Inter"))
    return fig


def prep(d):
    d = d.copy()
    d["errorOrig"] = d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"]
    d["errorDest"] = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]
    return d[FEATS]


@st.cache_resource(show_spinner="Training CatBoost model...")
def train(_df, key):
    d = _df[_df["type"].isin(["TRANSFER", "CASH_OUT"])]
    tr, te = d[d.step <= 600], d[d.step > 600]
    legit = tr[tr.isFraud == 0]
    tr = pd.concat([tr[tr.isFraud == 1], legit.sample(min(200000, len(legit)), random_state=1)])
    m = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                           auto_class_weights="Balanced", cat_features=["type"], verbose=0)
    m.fit(prep(tr), tr.isFraud)
    yt = te.isFraud.values
    pr = m.predict_proba(prep(te))[:, 1]
    p, r, t = precision_recall_curve(yt, pr)
    ok = np.where(p[:-1] >= 0.90)[0]
    thr = float(t[ok[np.argmax(r[:-1][ok])]]) if len(ok) else 0.5
    pred = pr >= thr
    return dict(model=m, thr=thr, probs=pr, y=yt, curve=(p, r),
                metrics=dict(pr_auc=average_precision_score(yt, pr), precision=precision_score(yt, pred, zero_division=0),
                             recall=recall_score(yt, pred), f1=f1_score(yt, pred)),
                stats=dict(rows=len(_df), fraud_rate=_df.isFraud.mean(), n_test=len(yt), test_fraud=int(yt.sum())),
                imp=dict(zip(FEATS, m.get_feature_importance())))


def groq_explain(tx, prob, thr, imp):
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return "Add GROQ_API_KEY to .streamlit/secrets.toml (or Streamlit Cloud secrets) to enable insights."
    top = ", ".join(f"{k} ({v:.0f}%)" for k, v in sorted(imp.items(), key=lambda x: -x[1])[:4])
    prompt = (f"A fraud model scored a mobile-money transaction.\nFeatures: {tx}\n"
              f"Fraud probability: {prob:.1%}; alert threshold: {thr:.1%}.\nMost important model features: {top}.\n"
              "In under 120 words, explain in plain language which signals look suspicious or normal, "
              "and suggest one next step for a human analyst. Do not claim certainty.")
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": "llama-3.3-70b-versatile", "temperature": 0.2, "max_tokens": 350,
                                "messages": [{"role": "system", "content": "You are a careful fraud-analysis assistant."},
                                             {"role": "user", "content": prompt}]}, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Groq request failed: {e}"


# ---------------- Sidebar ----------------
st.sidebar.markdown("## 🛡️ Fraud Detector")
page = st.sidebar.radio("Navigate", ["Overview", "Single Prediction", "Batch Analysis",
                                     "AI Insights", "Model Performance", "About & Methodology"])
st.sidebar.markdown("---")
up = st.sidebar.file_uploader("PaySim training CSV", type="csv")
path = st.sidebar.text_input("...or file path (Colab)", "PS_20174392719_1491204439457_log.csv")
if st.sidebar.button("Load & train"):
    src = up if up is not None else (path if os.path.exists(path) else None)
    if src is None:
        st.sidebar.error("No CSV found.")
    else:
        try:
            cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
            df = pd.read_csv(src, usecols=cols)
            st.session_state.art = train(df, str(getattr(src, "name", src)))
            st.sidebar.success("Model ready")
        except Exception as e:
            st.sidebar.error(f"Could not train: {e}")
st.sidebar.caption("Research and decision-support tool. Not an automatic payment-blocking system.")

art = st.session_state.get("art")
st.title(page)
if page != "About & Methodology" and not art:
    st.info("Upload the PaySim CSV in the sidebar (or enter its path) and click **Load & train** to begin.")
    st.stop()

# ---------------- Pages ----------------
if page == "Overview":
    s, mt = art["stats"], art["metrics"]
    cards([("Transactions in file", f"{s['rows']:,}", CYAN), ("Fraud rate", f"{s['fraud_rate']:.3%}", RED),
           ("PR-AUC", f"{mt['pr_auc']:.3f}", VIOLET), ("Alert threshold", f"{art['thr']:.2f}", GREEN)])
    st.write("")
    pr, y, thr = art["probs"], art["y"], art["thr"]
    c1, c2 = st.columns(2)
    low = (pr < min(0.3, thr)).sum(); high = (pr >= thr).sum(); med = len(pr) - low - high
    fig = go.Figure(go.Pie(labels=["Low", "Medium", "High"], values=[low, med, high], hole=.6,
                           marker=dict(colors=[GREEN, VIOLET, RED])))
    c1.plotly_chart(style(fig.update_layout(title="Risk distribution (test period)")), use_container_width=True)
    fig = go.Figure()
    fig.add_histogram(x=pr[y == 0], name="Legitimate", marker_color=GREEN, nbinsx=40)
    fig.add_histogram(x=pr[y == 1], name="Fraud", marker_color=RED, nbinsx=40)
    fig.update_layout(barmode="overlay", title="Score distribution", yaxis_type="log").update_traces(opacity=.75)
    c2.plotly_chart(style(fig), use_container_width=True)

elif page == "Single Prediction":
    t = st.selectbox("Type", ["TRANSFER", "CASH_OUT"])
    amt = st.number_input("Amount", min_value=0.0, value=5000.0)
    a, b = st.columns(2)
    oo = a.number_input("Sender balance before", min_value=0.0, value=5000.0)
    no = a.number_input("Sender balance after", min_value=0.0, value=0.0)
    od = b.number_input("Receiver balance before", min_value=0.0, value=0.0)
    nd = b.number_input("Receiver balance after", min_value=0.0, value=0.0)
    if st.button("Check transaction"):
        row = pd.DataFrame([dict(type=t, amount=amt, oldbalanceOrg=oo, newbalanceOrig=no,
                                 oldbalanceDest=od, newbalanceDest=nd)])
        p = float(art["model"].predict_proba(prep(row))[0, 1])
        st.session_state.last = dict(tx=row.iloc[0].to_dict(), prob=p)
        flagged = p >= art["thr"]
        g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number=dict(suffix="%"),
                                   gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=RED if flagged else GREEN),
                                              threshold=dict(line=dict(color=CYAN, width=3), value=art["thr"] * 100))))
        st.plotly_chart(style(g, 280), use_container_width=True)
        (st.error if flagged else st.success)("🚨 Flagged: suspicious transaction" if flagged else "✅ Looks legitimate")

elif page == "Batch Analysis":
    f = st.file_uploader("Upload transactions CSV", type="csv")
    st.caption(f"Required columns: {', '.join(REQ)}. Max {MAX_MB} MB. Processed in memory only.")
    if f:
        if f.size > MAX_MB * 1024 * 1024:
            st.error(f"File exceeds {MAX_MB} MB."); st.stop()
        d = pd.read_csv(f)
        miss = [c for c in REQ if c not in d.columns]
        if miss:
            st.error(f"Missing columns: {miss}"); st.stop()
        num = REQ[1:]
        d[num] = d[num].apply(pd.to_numeric, errors="coerce")
        if d[num].isna().any().any():
            st.error("Non-numeric or empty values found in numeric columns."); st.stop()
        if (d[num] < 0).any().any():
            st.error("Negative values found."); st.stop()
        if not d["type"].isin(["TRANSFER", "CASH_OUT"]).all():
            st.warning("Rows with types other than TRANSFER / CASH_OUT were removed.")
            d = d[d["type"].isin(["TRANSFER", "CASH_OUT"])]
        d["fraud_prob"] = art["model"].predict_proba(prep(d))[:, 1]
        d["flagged"] = d["fraud_prob"] >= art["thr"]
        d = d.sort_values("fraud_prob", ascending=False).reset_index(drop=True)
        st.session_state.batch = d
        cards([("Rows", f"{len(d):,}", CYAN), ("Flagged", f"{int(d.flagged.sum()):,}", RED),
               ("Flag rate", f"{d.flagged.mean():.2%}", VIOLET)])
        st.write("")
        st.dataframe(d.head(200), use_container_width=True)
        st.download_button("Download results (CSV)", d.to_csv(index=False), "fraud_results.csv")

elif page == "AI Insights":
    st.caption("Groq is called only when you press the button. Only numeric features are sent: no IDs or personal data.")
    options = {}
    if "last" in st.session_state:
        options["Last single prediction"] = st.session_state.last
    if "batch" in st.session_state:
        for i, r in st.session_state.batch.head(10).iterrows():
            options[f"Batch row #{i + 1} ({r.fraud_prob:.1%})"] = dict(tx={k: r[k] for k in REQ}, prob=float(r.fraud_prob))
    if not options:
        st.info("Run a single prediction or a batch analysis first."); st.stop()
    choice = st.selectbox("Transaction to explain", list(options))
    if st.button("Generate explanation"):
        sel = options[choice]
        with st.spinner("Asking Groq..."):
            txt = groq_explain(sel["tx"], sel["prob"], art["thr"], art["imp"])
        st.markdown(f'<div class="card">{txt}</div>', unsafe_allow_html=True)

elif page == "Model Performance":
    m = art["metrics"]
    cards([("PR-AUC", f"{m['pr_auc']:.3f}", CYAN), ("Precision", f"{m['precision']:.3f}", GREEN),
           ("Recall", f"{m['recall']:.3f}", VIOLET), ("F1", f"{m['f1']:.3f}", RED)])
    st.caption(f"Time-based test: steps > 600, {art['stats']['n_test']:,} transactions, {art['stats']['test_fraud']:,} fraud cases.")
    c1, c2 = st.columns(2)
    p, r = art["curve"]
    c1.plotly_chart(style(go.Figure(go.Scatter(x=r, y=p, line=dict(color=CYAN))).update_layout(
        title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision")), use_container_width=True)
    imp = sorted(art["imp"].items(), key=lambda x: x[1])
    c2.plotly_chart(style(go.Figure(go.Bar(x=[v for _, v in imp], y=[k for k, _ in imp], orientation="h",
                                           marker_color=VIOLET)).update_layout(title="Feature importance")), use_container_width=True)

else:
    st.markdown("""
**Dataset:** PaySim, a synthetic mobile-money dataset. Fraud occurs only in TRANSFER and CASH_OUT, so only those are modelled.

**Model:** CatBoost binary classifier with class balancing and engineered balance-error features
(`errorOrig`, `errorDest`). Training uses all fraud rows plus a sample of legitimate rows from steps up to 600;
evaluation uses every transaction after step 600 (time-based split, no sampling).

**Metrics:** PR-AUC, precision, recall and F1. Accuracy is not used because fraud is about 0.1% of rows.
The alert threshold maximises recall while keeping precision at or above 0.90.

**AI Insights:** Groq (Llama) writes plain-language explanations on request, from numeric features only.

**Limitations:** PaySim is synthetic and its balance-error signal is unusually strong, so real-world performance will be lower.
This is a research and decision-support tool, not an automatic payment-blocking system.
""")
