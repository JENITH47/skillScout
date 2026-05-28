"""
SkillScout v2.0 — AI Resume Ranking System
=============================================
Premium Streamlit UI with downloadable results.
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import io
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Page Config ──
st.set_page_config(
    page_title="SkillScout — AI Resume Ranking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for premium dark UI ──
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Root variables */
    :root {
        --bg-primary: #0f1117;
        --bg-secondary: #1a1d27;
        --bg-card: #1e2130;
        --bg-card-hover: #252839;
        --accent-blue: #4f8ff7;
        --accent-purple: #8b5cf6;
        --accent-green: #22c55e;
        --accent-yellow: #eab308;
        --accent-red: #ef4444;
        --accent-orange: #f97316;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-color: #2d3348;
        --gradient-1: linear-gradient(135deg, #4f8ff7 0%, #8b5cf6 100%);
        --gradient-2: linear-gradient(135deg, #22c55e 0%, #14b8a6 100%);
    }

    /* Global */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero header */
    .hero-container {
        background: linear-gradient(135deg, #1a1d3a 0%, #0f1117 50%, #1a1025 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 20%, rgba(79,143,247,0.08) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(139,92,246,0.06) 0%, transparent 50%);
        z-index: 0;
    }
    .hero-container > * { position: relative; z-index: 1; }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f8ff7 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.6;
    }

    /* Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* Score badges */
    .score-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }
    .score-excellent { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .score-good { background: rgba(79,143,247,0.15); color: #4f8ff7; border: 1px solid rgba(79,143,247,0.3); }
    .score-moderate { background: rgba(234,179,8,0.15); color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
    .score-weak { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

    /* Result cards */
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .result-card:hover {
        border-color: var(--accent-blue);
        box-shadow: 0 4px 20px rgba(79,143,247,0.08);
    }

    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .result-filename {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .result-rank {
        font-size: 0.85rem;
        color: var(--text-muted);
        font-weight: 500;
    }

    .result-score-big {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .result-score-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Skill tags */
    .skill-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0.15rem;
    }
    .skill-matched { background: rgba(34,197,94,0.12); color: #22c55e; border: 1px solid rgba(34,197,94,0.25); }
    .skill-missing { background: rgba(239,68,68,0.12); color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }
    .skill-extra { background: rgba(79,143,247,0.12); color: #4f8ff7; border: 1px solid rgba(79,143,247,0.25); }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4f8ff7, #8b5cf6);
    }

    /* Section divider */
    .section-divider {
        height: 1px;
        background: var(--border-color);
        margin: 1.5rem 0;
    }

    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4f8ff7 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
    }

    /* Streamlit expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Text area / file uploader */
    .stTextArea textarea {
        border-radius: 10px !important;
        border-color: var(--border-color) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(79,143,247,0.2) !important;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f8ff7 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Info/success/error boxes */
    .stAlert {
        border-radius: 10px !important;
    }

    /* Feedback list */
    .feedback-item {
        padding: 0.4rem 0;
        font-size: 0.9rem;
        line-height: 1.5;
        color: var(--text-secondary);
    }

    /* Role suggestion badge */
    .role-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0.15rem;
        background: rgba(139,92,246,0.12);
        color: #a78bfa;
        border: 1px solid rgba(139,92,246,0.25);
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Resources ──
@st.cache_resource
def get_inference_pipeline():
    from skillscout.inference_pipeline import InferencePipeline
    return InferencePipeline()

@st.cache_resource
def get_bert_model():
    from skillscout.ranking_feedback_nlp import load_bert
    return load_bert()


# ── Helper Functions ──

def get_score_color(score: float) -> str:
    if score >= 8.0: return "#22c55e"
    if score >= 6.5: return "#4f8ff7"
    if score >= 5.0: return "#eab308"
    return "#ef4444"

def get_score_class(score: float) -> str:
    if score >= 8.0: return "score-excellent"
    if score >= 6.5: return "score-good"
    if score >= 5.0: return "score-moderate"
    return "score-weak"

def get_score_label(score: float) -> str:
    if score >= 8.0: return "Excellent"
    if score >= 6.5: return "Good"
    if score >= 5.0: return "Moderate"
    return "Weak"

def build_csv_report(results: list, jd_text: str) -> str:
    """Build a downloadable CSV report from results."""
    rows = []
    for r in results:
        fb = r.get("feedback", {})
        feats = r.get("features", {})
        rows.append({
            "Rank": r["rank"],
            "File": os.path.basename(r["file_path"]),
            "Score (0-10)": round(r["predicted_score"], 2),
            "Raw Model Score": round(r.get("raw_model_score", 0), 2),
            "Verdict": fb.get("verdict", ""),
            "Eligibility": fb.get("eligibility_label", ""),
            "Confidence %": fb.get("ranking_confidence", 0),
            "Requirement Fit %": fb.get("requirement_fit_pct", 0),
            "Experience (yrs)": feats.get("experience_years", 0),
            "Required Experience": feats.get("required_experience_years", 0),
            "Education Level": feats.get("education_level", 0),
            "Total Skills": feats.get("num_skills", 0),
            "CS Skills": feats.get("cs_skill_count", 0),
            "Data Skills": feats.get("data_skill_count", 0),
            "Semantic Match": round(feats.get("bert_sim_score", 0), 2),
            "Extraction Quality": round(feats.get("text_quality_score", 0) * 100, 1),
            "Embedding Model": feats.get("embedding_model", ""),
            "Skill Match %": fb.get("skill_match_pct", 0),
            "Matched Skills": ", ".join(fb.get("matched_skills", [])),
            "Missing Skills": ", ".join(fb.get("missing_skills", [])),
            "Missing Mandatory Skills": ", ".join(fb.get("missing_mandatory_skills", [])),
            "Strengths": " | ".join(fb.get("strengths", [])),
            "Gaps": " | ".join(fb.get("gaps", [])),
            "Recommendations": " | ".join(fb.get("recommendations", [])),
            "Suggested Roles": ", ".join([f"{role} ({pct}%)" for role, pct in fb.get("suggested_roles", [])]),
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def render_skill_tags(skills: list, tag_class: str, max_show: int = 15) -> str:
    """Render skill tags as HTML."""
    html = ""
    for skill in skills[:max_show]:
        html += f'<span class="skill-tag {tag_class}">{skill}</span>'
    if len(skills) > max_show:
        html += f'<span class="skill-tag {tag_class}">+{len(skills)-max_show} more</span>'
    return html


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎯 SkillScout</div>
    <div class="hero-subtitle">
        AI-Powered Resume Ranking System — Upload candidate resumes and a job description
        to get intelligent rankings, skill gap analysis, and actionable feedback.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input Section ──
col_jd, col_upload = st.columns([3, 2], gap="large")

with col_jd:
    st.markdown("##### 📝 Job Description")
    jd_text = st.text_area(
        "Paste the full job description here",
        height=280,
        placeholder="Paste the job description including required skills, experience, qualifications...",
        label_visibility="collapsed",
    )

with col_upload:
    st.markdown("##### 📄 Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF resumes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.caption(f"✅ {len(uploaded_files)} resume{'s' if len(uploaded_files) > 1 else ''} uploaded")
    else:
        st.caption("Drag and drop PDF files or click to browse")

# ── Action Button ──
st.markdown("")
col_btn, _, _ = st.columns([1, 2, 1])
with col_btn:
    run_button = st.button("🚀 Rank Resumes", type="primary", use_container_width=True)

# ── Processing & Results ──
if run_button:
    if not jd_text or not jd_text.strip():
        st.error("❌ Please paste a Job Description before ranking.")
        st.stop()

    if not uploaded_files:
        st.error("❌ Please upload at least one resume PDF.")
        st.stop()

    # Save uploaded files to temp dir
    pdf_paths = []
    for file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.read())
        pdf_paths.append(temp_path)

    # Progress bar
    progress_bar = st.progress(0, text="🔄 Initializing AI models...")

    # Pre-load models during the progress display
    from skillscout.ranking_feedback_nlp import rank_resumes

    def update_progress(current, total, msg):
        pct = int((current / total) * 100) if total > 0 else 0
        progress_bar.progress(pct, text=f"🔄 {msg}")

    # Run ranking
    results = rank_resumes(
        jd_text,
        pdf_paths,
        micro_dict=None,
        college_tier="Unknown",
        progress_callback=update_progress,
    )
    progress_bar.progress(100, text="✅ Ranking complete!")

    # ── Summary Metrics ──
    st.markdown("---")
    st.markdown("### 📊 Summary")

    scores = [r["predicted_score"] for r in results if r["predicted_score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    excellent_count = sum(1 for s in scores if s >= 8.0)
    good_count = sum(1 for s in scores if 6.5 <= s < 8.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(results)}</div>
            <div class="metric-label">Total Resumes</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {get_score_color(max_score)}">{max_score:.1f}</div>
            <div class="metric-label">Top Score</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_score:.1f}</div>
            <div class="metric-label">Avg Score</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #22c55e">{excellent_count}</div>
            <div class="metric-label">Excellent</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #4f8ff7">{good_count}</div>
            <div class="metric-label">Good</div>
        </div>""", unsafe_allow_html=True)

    # ── Download Button ──
    st.markdown("")
    csv_data = build_csv_report(results, jd_text)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dl_col1, dl_col2, _ = st.columns([1, 1, 2])
    with dl_col1:
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv_data,
            file_name=f"skillscout_ranking_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_col2:
        # Quick summary table as downloadable too
        quick_df = pd.DataFrame([{
            "Rank": r["rank"],
            "File": os.path.basename(r["file_path"]),
            "Score": round(r["predicted_score"], 2),
            "Raw Score": round(r.get("raw_model_score", 0), 2),
            "Confidence %": r.get("feedback", {}).get("ranking_confidence", 0),
            "Requirement Fit %": r.get("feedback", {}).get("requirement_fit_pct", 0),
            "Experience": r["features"].get("experience_years", 0),
            "Skills": r["features"].get("num_skills", 0),
            "Match %": r.get("feedback", {}).get("skill_match_pct", 0),
        } for r in results])

        st.download_button(
            label="📋 Download Quick Summary (CSV)",
            data=quick_df.to_csv(index=False),
            file_name=f"skillscout_summary_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Rankings Table ──
    st.markdown("---")
    st.markdown("### 🏆 Rankings")

    for r in results:
        score = r["predicted_score"]
        filename = os.path.basename(r["file_path"])
        rank = r["rank"]
        fb = r.get("feedback", {})
        feats = r.get("features", {})

        score_color = get_score_color(score)
        score_class = get_score_class(score)
        score_label = get_score_label(score)

        # Medal emoji for top 3
        medal = ""
        if rank == 1: medal = "🥇 "
        elif rank == 2: medal = "🥈 "
        elif rank == 3: medal = "🥉 "

        with st.container():
            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <div>
                        <span class="result-rank">{medal}#{rank}</span>
                        <span class="result-filename" style="margin-left: 0.5rem;">{filename}</span>
                    </div>
                    <div style="text-align: right;">
                        <div class="result-score-big" style="color: {score_color}">{score:.1f}</div>
                        <span class="score-badge {score_class}">{score_label}</span>
                    </div>
                </div>
                <div style="display: flex; gap: 2rem; flex-wrap: wrap; margin-top: 0.5rem;">
                    <div>
                        <span style="color: var(--text-muted); font-size: 0.78rem;">EXPERIENCE</span><br>
                        <span style="color: var(--text-primary); font-weight: 600;">{feats.get('experience_years', 0):.1f} yrs</span>
                    </div>
                    <div>
                        <span style="color: var(--text-muted); font-size: 0.78rem;">SKILLS</span><br>
                        <span style="color: var(--text-primary); font-weight: 600;">{feats.get('num_skills', 0)}</span>
                    </div>
                    <div>
                        <span style="color: var(--text-muted); font-size: 0.78rem;">SKILL MATCH</span><br>
                        <span style="color: var(--text-primary); font-weight: 600;">{fb.get('skill_match_pct', 0):.0f}%</span>
                    </div>
                    <div>
                        <span style="color: var(--text-muted); font-size: 0.78rem;">CONFIDENCE</span><br>
                        <span style="color: var(--text-primary); font-weight: 600;">{fb.get('ranking_confidence', 0):.0f}%</span>
                    </div>
                    <div>
                        <span style="color: var(--text-muted); font-size: 0.78rem;">REQ FIT</span><br>
                        <span style="color: var(--text-primary); font-weight: 600;">{fb.get('requirement_fit_pct', 0):.0f}%</span>
                    </div>
                    <div>
                        <span style="color: var(--text-muted); font-size: 0.78rem;">SEMANTIC</span><br>
                        <span style="color: var(--text-primary); font-weight: 600;">{feats.get('bert_sim_score', 0):.1f}/10</span>
                    </div>
                    <div>
                        <span style="color: var(--text-muted); font-size: 0.78rem;">EDUCATION</span><br>
                        <span style="color: var(--text-primary); font-weight: 600;">{['—','HS','Dip','BSc','MSc','PhD','PostDoc'][min(feats.get('education_level', 0), 6)]}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Expandable detailed feedback
            with st.expander(f"📋 Detailed Analysis — {filename}", expanded=(rank == 1)):

                # Verdict
                st.markdown(f"**{fb.get('verdict', '')}** — {fb.get('verdict_msg', '')}")
                st.caption(
                    f"Eligibility: {fb.get('eligibility_label', 'Unknown')} | "
                    f"Confidence: {fb.get('ranking_confidence', 0):.0f}% | "
                    f"Requirement fit: {fb.get('requirement_fit_pct', 0):.0f}% | "
                    f"Extraction quality: {feats.get('text_quality_score', 0) * 100:.0f}%"
                )
                if r.get("raw_model_score") is not None:
                    st.caption(
                        f"Final score: {score:.1f}/10 | Raw model score: {r.get('raw_model_score', 0):.1f}/10 | "
                        f"Embedding: {feats.get('embedding_model', 'n/a')}"
                    )

                # Two columns: Strengths + Gaps
                col_s, col_g = st.columns(2)

                with col_s:
                    st.markdown("**💪 Strengths**")
                    for s in fb.get("strengths", []):
                        st.markdown(f"<div class='feedback-item'>{s}</div>", unsafe_allow_html=True)
                    if not fb.get("strengths"):
                        st.caption("No notable strengths detected")

                with col_g:
                    st.markdown("**⚠️ Gaps**")
                    for g in fb.get("gaps", []):
                        st.markdown(f"<div class='feedback-item'>{g}</div>", unsafe_allow_html=True)
                    if not fb.get("gaps"):
                        st.caption("No significant gaps found")

                # Recommendations
                recs = fb.get("recommendations", [])
                if recs:
                    st.markdown("**💡 Recommendations**")
                    for rec in recs:
                        st.markdown(f"<div class='feedback-item'>{rec}</div>", unsafe_allow_html=True)

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                # Skills breakdown
                st.markdown("**🔧 Skills Analysis**")

                matched = fb.get("matched_skills", [])
                missing = fb.get("missing_skills", [])
                missing_mandatory = fb.get("missing_mandatory_skills", [])
                extra = fb.get("extra_skills", [])

                if missing_mandatory:
                    st.markdown(f"**Missing Mandatory Skills** ({len(missing_mandatory)})")
                    st.markdown(render_skill_tags(missing_mandatory, "skill-missing"), unsafe_allow_html=True)

                if matched:
                    st.markdown(f"**Matched with JD** ({len(matched)})")
                    st.markdown(render_skill_tags(matched, "skill-matched"), unsafe_allow_html=True)

                if missing:
                    st.markdown(f"**Missing from Resume** ({len(missing)})")
                    st.markdown(render_skill_tags(missing, "skill-missing"), unsafe_allow_html=True)

                if extra:
                    st.markdown(f"**Additional Skills** ({len(extra)})")
                    st.markdown(render_skill_tags(extra, "skill-extra"), unsafe_allow_html=True)

                # Suggested Roles
                roles = fb.get("suggested_roles", [])
                if roles:
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown("**🎯 Suggested Job Roles**")
                    roles_html = ""
                    for role, pct in roles:
                        roles_html += f'<span class="role-badge">{role} ({pct}%)</span>'
                    st.markdown(roles_html, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.caption(f"🎯 SkillScout v2.0 — Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
