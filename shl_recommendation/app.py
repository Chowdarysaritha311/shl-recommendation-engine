"""
SHL Assessment Recommendation — Streamlit Web App
Run locally: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import os

# ─── Config ──────────────────────────────────────────────────────────────────

# Detect API URL: Use environment variable first, fallback to localhost
API_URL = os.getenv("API_URL") or "https://shl-recommendation-api-lf5m.onrender.com"

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
)

# ─── Styling ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #003087 0%, #0066CC 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #cce0ff; margin: 5px 0 0; }
    .assessment-card {
        border: 1px solid #e0e0e0;
        border-left: 5px solid #0066CC;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        background: #fafcff;
    }
    .assessment-card h4 { margin: 0 0 6px; color: #003087; font-size: 1.05rem; }
    .assessment-card a  { color: #0066CC; font-size: 0.85rem; word-break: break-all; }
    .badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a56db;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px 4px 2px 0;
        font-weight: 500;
    }
    .badge-green  { background: #dcfce7; color: #166534; }
    .badge-orange { background: #fef9c3; color: #854d0e; }
    .stTextArea textarea { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🎯 SHL Assessment Recommender</h1>
  <p>Enter a job description or natural language query to get the most relevant SHL assessments.</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://www.shl.com/assets/header-graphics/SHL-logo-colour-update.svg",
             width=120, output_format="auto")
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
1. **Type** your job description or query  
2. Click **Get Recommendations**  
3. See the most relevant SHL assessments  

The engine uses **Google Gemini** + **RAG** to semantically understand your query 
and match it against SHL's full catalogue of 377+ assessments.
""")
    st.markdown("---")
    st.markdown("### 🔧 API")
    st.code(f"POST {API_URL}/recommend\n" + '{"query": "your query"}', language="bash")
    st.markdown(f"[API Docs ↗]({API_URL}/docs)")

# ─── Sample Queries ───────────────────────────────────────────────────────────

SAMPLES = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
    "Hiring an analyst — want to screen using cognitive and personality tests within 45 minutes.",
    "Need assessments for a sales manager who leads a team and deals with clients.",
]

st.markdown("#### 💡 Try a sample query")
cols = st.columns(len(SAMPLES))
selected_sample = ""
for i, (col, sample) in enumerate(zip(cols, SAMPLES)):
    if col.button(f"Sample {i+1}", use_container_width=True, help=sample):
        selected_sample = sample

# ─── Input ───────────────────────────────────────────────────────────────────

default_query = selected_sample if selected_sample else ""
query = st.text_area(
    "📝 Your query or job description",
    value=default_query,
    height=160,
    placeholder="e.g. I am hiring for Java developers who are team players...",
)

col1, col2, _ = st.columns([1.5, 1, 5])
run_btn = col1.button("🔍 Get Recommendations", type="primary", use_container_width=True)
clear_btn = col2.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()

# ─── Recommendation ───────────────────────────────────────────────────────────

if run_btn and query.strip():
    with st.spinner("Analysing your query and searching the catalogue..."):
        try:
            resp = requests.post(
                f"{API_URL}/recommend",
                json={"query": query},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                assessments = data.get("recommended_assessments", [])

                st.success(f"✅ Found {len(assessments)} relevant assessments")
                st.markdown("---")

                # Summary table
                table_data = []
                for i, a in enumerate(assessments):
                    table_data.append({
                        "#": i + 1,
                        "Assessment Name": a["name"],
                        "Test Type": ", ".join(a.get("test_type", [])),
                        "Duration (min)": a.get("duration") or "—",
                        "Remote": a.get("remote_support", "—"),
                        "Adaptive": a.get("adaptive_support", "—"),
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

                st.markdown("### 📋 Detailed Results")

                for i, a in enumerate(assessments):
                    types_html = "".join(
                        f'<span class="badge">{t}</span>'
                        for t in a.get("test_type", [])
                    )
                    remote = a.get("remote_support", "No")
                    adaptive = a.get("adaptive_support", "No")
                    remote_badge = f'<span class="badge badge-green">Remote ✓</span>' if remote == "Yes" else ""
                    adaptive_badge = f'<span class="badge badge-orange">Adaptive ✓</span>' if adaptive == "Yes" else ""
                    dur = a.get("duration")
                    dur_text = f"⏱ {dur} min" if dur else ""
                    desc = a.get("description", "")[:250]

                    st.markdown(f"""
<div class="assessment-card">
  <h4>{i+1}. {a['name']}</h4>
  <a href="{a['url']}" target="_blank">{a['url']}</a><br><br>
  {types_html} {remote_badge} {adaptive_badge}
  {'<br>' if dur_text else ''}<span style="color:#555; font-size:0.85rem;">{dur_text}</span>
  {'<br><p style="margin:8px 0 0; font-size:0.88rem; color:#444;">' + desc + '</p>' if desc else ''}
</div>
""", unsafe_allow_html=True)

            else:
                st.error(f"API Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Cannot connect to API at `{API_URL}`. "
                "Make sure the API is running: `uvicorn api:app --reload`"
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")

elif run_btn and not query.strip():
    st.warning("Please enter a query first.")

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8rem;'>"
    "Built with FastAPI · Streamlit · Google Gemini · RAG"
    "</div>",
    unsafe_allow_html=True,
)
