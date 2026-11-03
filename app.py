from pathlib import Path
from datetime import datetime
import html
import re
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "spam_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"

st.set_page_config(page_title="Spam Email Detection | MailGuard", page_icon="📨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--navy:#050b18;--panel:#0b1628;--panel2:#14243b;--line:#344963;--blue:#4ca6ff;--red:#ff6378;--muted:#9aadc6}
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:radial-gradient(ellipse at 48% 0%,#142e50 0%,#081326 44%,#030711 100%);color:#f4f7ff}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1900px;padding:1.05rem 1.25rem 1.5rem}
footer{visibility:hidden}
h1,h2,h3,h4,p,label{color:#f1f6ff}
.title{text-align:center;font-size:clamp(2rem,3.7vw,3.15rem);font-weight:800;letter-spacing:-.045em;text-shadow:0 0 30px #3e9eff55;margin:0}
.subtitle{text-align:center;color:#a9c2e1;font-size:.82rem;letter-spacing:.19em;text-transform:uppercase;margin:.35rem 0 1.25rem}
[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(150deg,rgba(12,24,43,.97),rgba(5,12,25,.98));border:1px solid rgba(125,166,213,.42);border-radius:17px;padding:1rem 1rem .95rem;box-shadow:0 15px 45px #0005}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]{gap:.65rem}
.panel-title{font-weight:750;font-size:1rem;margin:0 0 .25rem;color:#f3f7ff}
.kicker{font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:#61baff;font-weight:800}
.muted{color:#9aadc6;font-size:.8rem}
.result-row{display:flex;align-items:center;gap:10px;padding:10px 11px;margin:7px 0;background:linear-gradient(90deg,#1b2d48,#111f34);border:1px solid #7b9bc022;border-radius:9px}
.mail-icon{width:34px;height:30px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:19px;background:#e6f2ff;color:#173d68;flex-shrink:0}
.mail-icon.spam{background:#ff687d;color:#531323}
.row-label{font-weight:650;font-size:.86rem;flex:1}
.pill{font-size:.68rem;padding:4px 8px;border-radius:30px;background:#ff637822;color:#ff9aa8;border:1px solid #ff637855;white-space:nowrap}
.pill.ham{background:#28d5a51b;color:#6ce3c4;border-color:#28d5a555}
.statbox{background:linear-gradient(145deg,#14243a,#0c1729);border:1px solid #6e8bad33;border-radius:10px;padding:11px 5px;text-align:center}
.statbox b{display:block;font-size:1.35rem;color:#f3f7ff}
.statbox span{font-size:.66rem;color:#a8bad0;text-transform:uppercase;letter-spacing:.06em}
.email-paper{background:#f5f7fb;color:#1a2638;border-radius:8px;padding:14px;min-height:235px;font-size:.77rem;line-height:1.55;overflow-wrap:anywhere;box-shadow:0 5px 18px #0003}
.email-paper b{color:#14243b}
.highlight{background:#ff6478;color:#fff;padding:1px 4px;border-radius:3px;font-weight:750}
.conf-track{height:9px;background:#26374f;border-radius:20px;overflow:hidden;margin-top:7px}
.conf-fill{height:100%;background:linear-gradient(90deg,#3497ff,#5bd4e8);border-radius:20px}
div[data-testid="stSelectbox"]>div>div,div[data-testid="stTextArea"] textarea{background:#111f34!important;color:#eff6ff!important;border:1px solid #344d6d!important;border-radius:9px!important}
.stButton>button{background:linear-gradient(100deg,#287fce,#168cae);border:1px solid #65baff;color:#fff;border-radius:9px;font-weight:750;min-height:2.55rem}
.stButton>button:hover{border-color:#c2e7ff;color:white;box-shadow:0 0 18px #267ed066}
[data-testid="stMetric"]{background:#122139;border:1px solid #344963;border-radius:10px;padding:10px}
hr{border-color:#344963}
div[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#3497ff,#5bd4e8)}
@media(max-width:900px){.subtitle{font-size:.68rem;letter-spacing:.1em}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("Model files are missing. Run `python train_model.py` first.")
    st.stop()

model = load_model()
metrics = load_metrics()

if "latest" not in st.session_state:
    st.session_state.latest = None
if "history" not in st.session_state:
    st.session_state.history = []

def classify(text):
    pred = str(model.predict([text])[0]).lower()
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    return pred, float(probs[classes.index(pred)])

def highlighted_email(text):
    safe = html.escape(text).replace("\n", "<br>")
    phrases = ["urgent", "click here", "winner", "claim", "fee", "limited time", "prize", "free", "verify", "won"]
    for phrase in phrases:
        safe = re.sub(rf"(?i)({re.escape(phrase)})", r"<span class='highlight'>\1</span>", safe)
    return safe

# Reference-style title
st.markdown("<h1 class='title'>Spam Email Detection</h1><div class='subtitle'>REAL-TIME INTELLIGENT MESSAGE CLASSIFICATION</div>", unsafe_allow_html=True)

# Default examples are classified by the actual saved model, not hard-coded labels.
demo_emails = [
    "Congratulations! You have won a prize. Click here to claim your free reward now!",
    "URGENT: Your account needs verification. Pay a processing fee immediately to receive your prize.",
    "Hi team, our project meeting is scheduled for tomorrow at 10 AM. Please bring your notes."
]
demo_results = [classify(x) for x in demo_emails]
latest_text = st.session_state.latest["message"] if st.session_state.latest else demo_emails[0]
latest_pred, latest_conf = classify(latest_text)
if st.session_state.latest:
    latest_pred = st.session_state.latest["prediction"].lower()
    latest_conf = st.session_state.latest["confidence"]

left, center, right = st.columns([0.92, 2.18, 0.98], gap="small")

with left:
    with st.container(border=True):
        st.markdown("<div class='panel-title'>✉ Email Network</div><div class='muted'>AI-powered message flow</div>", unsafe_allow_html=True)
        svg = """<svg viewBox="0 0 320 560" width="100%" xmlns="http://www.w3.org/2000/svg">
        <defs><radialGradient id="g"><stop stop-color="#4ba9ff" stop-opacity=".65"/><stop offset="1" stop-color="#4ba9ff" stop-opacity="0"/></radialGradient><linearGradient id="e" x2="1" y2="1"><stop stop-color="#9bd5ff"/><stop offset="1" stop-color="#2879d2"/></linearGradient></defs>
        <ellipse cx="160" cy="270" rx="145" ry="170" fill="url(#g)"/>
        <g stroke="#4da9ff" stroke-opacity=".5" stroke-width="1"><path d="M150 25L65 100 245 145 72 220 252 280 85 360 195 455 150 25M65 100L252 280M245 145L85 360M72 220L195 455M150 25L252 280"/></g>
        <g fill="#69baff">""" + "".join(f'<circle cx="{x}" cy="{y}" r="{r}"/>' for x,y,r in [(150,25,3),(65,100,3),(245,145,3),(72,220,3),(252,280,3),(85,360,3),(195,455,3),(160,270,4),(115,175,2),(220,330,2),(120,410,2),(275,90,2),(40,300,2)]) + """</g>
        <g stroke="#dceeff" stroke-width="2" stroke-linejoin="round">
        <g transform="translate(115 40) rotate(-13)"><rect width="74" height="48" rx="6" fill="url(#e)"/><path d="M2 4L37 29 72 4M2 44L27 23M72 44L47 23" fill="none"/></g>
        <g transform="translate(25 135) rotate(12)"><rect width="65" height="43" rx="6" fill="#2e83d8"/><path d="M2 3L32 26 63 3M2 40L25 21M63 40L40 21" fill="none"/></g>
        <g transform="translate(195 190) rotate(-8)"><rect width="70" height="46" rx="6" fill="#e65c70"/><path d="M2 3L35 28 68 3M2 43L27 22M68 43L43 22" fill="none"/></g>
        <g transform="translate(112 245) rotate(8)"><rect width="78" height="51" rx="6" fill="#398bdc"/><path d="M2 3L39 31 76 3M2 48L29 25M76 48L49 25" fill="none"/></g>
        <g transform="translate(28 330) rotate(-12)"><rect width="68" height="45" rx="6" fill="#e86a7c"/><path d="M2 3L34 27 66 3M2 42L25 22M66 42L43 22" fill="none"/></g>
        <g transform="translate(190 385) rotate(10)"><rect width="76" height="49" rx="6" fill="url(#e)"/><path d="M2 3L38 30 74 3M2 46L28 24M74 46L48 24" fill="none"/></g>
        <g transform="translate(95 465) rotate(-6)"><rect width="60" height="40" rx="5" fill="#eaf4ff"/><path d="M2 3L30 24 58 3M2 37L23 20M58 37L37 20" fill="none" stroke="#8ba8c8"/></g></g>
        <g fill="none" stroke="#80c7ff" stroke-width="2" opacity=".7"><ellipse cx="13" cy="270" rx="11" ry="75"/><ellipse cx="307" cy="270" rx="11" ry="75"/><path d="M20 270H300" stroke-dasharray="3 6"/></g></svg>"""
        st.markdown(svg, unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;color:#83b5e6;font-size:.68rem;letter-spacing:.14em'>INCOMING EMAIL SIGNALS</div>", unsafe_allow_html=True)

with center:
    with st.container(border=True):
        st.markdown("<div class='panel-title'>Real Time Classification Results</div>", unsafe_allow_html=True)
        st.markdown("<div class='kicker'>EMAIL SCANNER</div>", unsafe_allow_html=True)
        examples = {
            "Write my own": "",
            "Suspicious prize notification": "Congratulations! You have won a free gift card. Click here now to claim your prize. Limited time offer!",
            "Bank security alert": "We noticed an unusual login attempt. Please review your account activity using the official banking app.",
            "Normal project meeting": "Hi team, our machine learning project meeting is scheduled for tomorrow at 10 AM. Please bring your notes.",
            "Job offer with registration fee": "You are selected for a remote job earning ₹85,000 per month. Pay a refundable registration fee immediately to confirm your position."
        }
        choice = st.selectbox("Try an example", list(examples), label_visibility="collapsed")
        email_input = st.text_area("Email message", value=examples[choice], height=250, placeholder="Paste an email message here...", label_visibility="collapsed")
        if st.button("⚡ ANALYSE EMAIL", use_container_width=True):
            if not email_input.strip():
                st.warning("Paste an email message first.")
            else:
                pred, conf = classify(email_input)
                record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          "prediction": "SPAM" if pred == "spam" else "HAM",
                          "confidence": conf, "message": email_input.strip()}
                st.session_state.latest = record
                st.session_state.history.insert(0, record)
                st.rerun()

        st.markdown("<div class='kicker'>LATEST CLASSIFICATION</div>", unsafe_allow_html=True)
        display_items = []
        if st.session_state.history:
            display_items = st.session_state.history[:3]
        else:
            display_items = [{"prediction": "SPAM" if p == "spam" else "HAM", "confidence": c, "message": t}
                             for t,(p,c) in zip(demo_emails,demo_results)]
        for item in display_items:
            spam = item["prediction"] == "SPAM"
            st.markdown(f"""<div class="result-row"><div class="mail-icon {'spam' if spam else ''}">✉</div>
            <div class="row-label">{'⚠ ' if spam else '✓ '}{item['prediction']}</div>
            <span class="pill {' ' if spam else 'ham'}">{item['confidence']:.0%}</span></div>""", unsafe_allow_html=True)

        st.markdown("<div class='kicker' style='margin-top:12px'>SPAM VS HAM</div><div class='muted'>Latest model prediction confidence</div>", unsafe_allow_html=True)
        gauge_col, chart_col = st.columns([1,1.1], gap="small")
        with gauge_col:
            conf = max(0,min(1,latest_conf))
            st.markdown(f"""<div style="width:145px;height:145px;border-radius:50%;margin:8px auto 5px;
            background:conic-gradient(#4ca6ff 0deg,#ff5e72 {conf*360:.1f}deg,#25364f {conf*360:.1f}deg 360deg);
            display:flex;align-items:center;justify-content:center"><div style="width:119px;height:119px;border-radius:50%;background:#0b1628;
            display:flex;align-items:center;justify-content:center;font-size:1.85rem;font-weight:800;color:white">{conf:.0%}</div></div>
            <div class='muted' style='text-align:center'>Confidence</div>""", unsafe_allow_html=True)
        with chart_col:
            counts = metrics.get("class_counts", {})
            st.markdown("<div class='muted' style='margin:.5rem 0'>Dataset distribution</div>", unsafe_allow_html=True)
            st.bar_chart(pd.DataFrame({"Messages": counts}), height=140)
        st.markdown("<div class='kicker'>MODEL PERFORMANCE</div>", unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.markdown(f"<div class='statbox'><b>{metrics['accuracy']:.1%}</b><span>Accuracy</span></div>",unsafe_allow_html=True)
        m2.markdown(f"<div class='statbox'><b>{metrics['precision_spam']:.1%}</b><span>Precision</span></div>",unsafe_allow_html=True)
        m3.markdown(f"<div class='statbox'><b>{metrics['recall_spam']:.1%}</b><span>Recall</span></div>",unsafe_allow_html=True)

with right:
    with st.container(border=True):
        st.markdown("<div class='panel-title'>Sample Analysis</div><div class='muted'>Email content preview</div>", unsafe_allow_html=True)
        preview = st.session_state.latest["message"] if st.session_state.latest else demo_emails[0]
        preview_pred, preview_conf = classify(preview)
        st.markdown(f"<div class='email-paper'><b>{'Latest email' if st.session_state.latest else 'Sample spam analysis'}</b><hr style='border-color:#dce2eb'>{highlighted_email(preview)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex;justify-content:space-between;margin-top:10px'><span class='muted'>{'Spam' if preview_pred == 'spam' else 'Ham'} confidence</span><b>{preview_conf:.0%}</b></div><div class='conf-track'><div class='conf-fill' style='width:{preview_conf*100:.1f}%'></div></div>", unsafe_allow_html=True)
        st.markdown("<div class='kicker' style='margin-top:18px'>DETECTION SIGNALS</div>", unsafe_allow_html=True)
        terms = [w for w in ["urgent","click here","winner","claim","fee","limited time","prize","free","verify","won"] if w in preview.lower()]
        if terms:
            st.markdown(" ".join(f"<span class='highlight'>{html.escape(t)}</span>" for t in terms), unsafe_allow_html=True)
        else:
            st.markdown("<span class='muted'>No common demo keywords found.</span>", unsafe_allow_html=True)
        st.markdown("<hr><div class='kicker'>MODEL ENGINE</div><div class='muted' style='line-height:1.9'>TF-IDF Vectorizer<br>Multinomial Naive Bayes<br>50 illustrative sample messages</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;color:#7187a5;font-size:.7rem;letter-spacing:.13em;padding:14px 0'>MAILGUARD · TF-IDF + MULTINOMIAL NAIVE BAYES · ACADEMIC PROJECT</div>", unsafe_allow_html=True)
with st.expander("📊 Full model analytics and confusion matrix"):
    st.caption("Metrics from the held-out test split of the 50-message sample dataset; not a real-world performance guarantee.")
    cm = metrics["confusion_matrix"]
    st.dataframe(pd.DataFrame(cm,index=["Actual Ham","Actual Spam"],columns=["Predicted Ham","Predicted Spam"]),use_container_width=True)
with st.expander("🕘 Test history"):
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist,use_container_width=True,hide_index=True)
        st.download_button("Download history CSV",hist.to_csv(index=False),"mailguard_history.csv","text/csv")
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.latest = None
            st.rerun()
    else:
        st.info("No emails tested yet.")
