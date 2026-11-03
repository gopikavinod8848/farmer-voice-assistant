# Main Streamlit Interface - AI Voice Assistant for Farmers
# Group Members: Akash B L (TRV23CE010), Gopika L V (TRV23CE039), Karthika B Nair (TRV23CE042)

import io
import os

import streamlit as st
from rag_engine import KeralaAgriRAGEngine

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

st.set_page_config(
    page_title="AI Voice Assistant for Farmers - Kerala",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching index.html theme and typography
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Malayalam:wght@400;500;600;700&display=swap');
  
  html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans Malayalam', sans-serif;
  }
  
  .stApp {
    background-color: #f4f7f4;
    color: #1c2e24;
  }
  
  .header-banner {
    background: linear-gradient(135deg, #022c22 0%, #064e3b 50%, #15803d 100%);
    color: white;
    padding: 18px 24px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(2, 44, 34, 0.15);
  }
  
  .header-title {
    font-size: 22px;
    font-weight: 700;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  
  .header-badge {
    background-color: #f59e0b;
    color: #022c22;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 9999px;
  }
  
  .header-sub {
    font-size: 13px;
    color: #a7f3d0;
    margin-top: 4px;
  }
  
  .header-team {
    font-size: 11px;
    color: #fef3c7;
    background: rgba(2, 44, 34, 0.4);
    display: inline-block;
    padding: 4px 12px;
    border-radius: 8px;
    border: 1px solid rgba(16, 185, 129, 0.3);
    margin-top: 8px;
  }
  
  .pipeline-card {
    background: white;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
  }
  .pipeline-num {
    font-size: 12px;
    font-weight: 700;
    color: #064e3b;
  }
  .pipeline-desc {
    font-size: 10px;
    color: #64748b;
    margin-top: 2px;
  }
  
  div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    font-size: 12px;
    transition: all 0.2s ease;
  }
  div.stButton > button:hover {
    border-color: #15803d;
    color: #15803d;
  }
  
  div[data-testid="stFormSubmitButton"] > button {
    background-color: #15803d !important;
    color: white !important;
    border: none !important;
    padding: 10px !important;
    font-size: 14px !important;
    border-radius: 12px !important;
  }
  div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #166534 !important;
  }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_engine():
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    return KeralaAgriRAGEngine(kb_path)

engine = get_rag_engine()

if "user_query" not in st.session_state:
    st.session_state.user_query = "നെല്ലിലെ മുഞ്ഞ (BPH) എങ്ങനെ നിയന്ത്രിക്കാം?"

def set_query(text):
    st.session_state.user_query = text

st.markdown("""
<div class="header-banner">
  <div class="header-title">
    <span>🌾 AI Voice Assistant for Farmers</span>
    <span class="header-badge">കേരള കർഷക സഹായി</span>
  </div>
  <div class="header-sub">
    Bilingual (Malayalam/English) RAG Advisory &amp; Government Scheme Matcher for Kerala
  </div>
  <div class="header-team">
    <strong>Group:</strong> Akash B L (TRV23CE010) • Gopika L V (TRV23CE039) • Karthika B Nair (TRV23CE042)
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("##### ⚡ 6-Step RAG Execution Pipeline (Slide 05)")
p1, p2, p3, p4, p5, p6 = st.columns(6)
p1.markdown('<div class="pipeline-card"><div class="pipeline-num">1. Speak / Type</div><div class="pipeline-desc">Malayalam / English</div></div>', unsafe_allow_html=True)
p2.markdown('<div class="pipeline-card"><div class="pipeline-num">2. Speech-to-Text</div><div class="pipeline-desc">Audio Transcription</div></div>', unsafe_allow_html=True)
p3.markdown('<div class="pipeline-card"><div class="pipeline-num">3. RAG Retrieval</div><div class="pipeline-desc">Cosine Similarity</div></div>', unsafe_allow_html=True)
p4.markdown('<div class="pipeline-card"><div class="pipeline-num">4. Grounded LLM</div><div class="pipeline-desc">Answers from KAU</div></div>', unsafe_allow_html=True)
p5.markdown('<div class="pipeline-card"><div class="pipeline-num">5. Scheme Check</div><div class="pipeline-desc">Profile Matcher</div></div>', unsafe_allow_html=True)
p6.markdown('<div class="pipeline-card"><div class="pipeline-num">6. Text-to-Speech</div><div class="pipeline-desc">Audio Read-Aloud</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 👨‍🌾 Farmer Profile")
    st.caption("Parameters used to auto-match eligible government schemes")
    district = st.selectbox(
        "District (ജില്ല)",
        ["Kollam", "Thiruvananthapuram", "Alappuzha (Kuttanad)", "Thrissur", "Palakkad", "Wayanad", "Idukki", "Kottayam", "Ernakulam"]
    )
    selected_crops = st.multiselect(
        "Cultivated Crops (കൃഷി ചെയ്യുന്ന വിളകൾ)",
        ["Paddy", "Coconut", "Black Pepper", "Banana", "Vegetables", "Tuber Crops", "Soil & Fertilizer"],
        default=["Paddy", "Coconut", "Black Pepper", "Banana"]
    )
    land_ha = st.slider("Land Area (Hectares / ഹെക്ടർ)", 0.1, 10.0, 0.8, 0.1)
    st.caption(f"Equivalent: **{(land_ha * 2.471):.1f} Acres**")
    is_owner = st.checkbox("Owner Cultivator (സ്വന്തം ഭൂമി / പട്ടയം)", value=True)

    st.divider()
    st.markdown("### ⚙️ Language & Voice Settings")
    language_mode = st.radio(
        "Output Language",
        ["Bilingual (Malayalam + English)", "Malayalam Only", "English Only"]
    )
    tts_lang = st.selectbox("TTS Voice Language", ["ml (Malayalam)", "en (English)"])
    api_key = st.text_input("External LLM API Key (Optional)", type="password", help="Leave blank for local offline KAU RAG synthesizer.")

    st.divider()
    st.caption("Developed by: Akash B L, Gopika L V, Karthika B Nair")

col1, col2 = st.columns([7, 5])

with col1:
    st.markdown("### 🎙️ Ask Any Farming Question (ചോദ്യം ചോദിക്കുക)")
    st.caption("Click a sample question or type your custom query below:")

    c1, c2, c3 = st.columns(3)
    if c1.button("🌾 നെല്ലിലെ മുഞ്ഞ (BPH)", use_container_width=True):
        set_query("നെല്ലിലെ മുഞ്ഞ എങ്ങനെ നിയന്ത്രിക്കാം? ഏത് വിത്താണ് നല്ലത്?")
    if c2.button("🥥 തെങ്ങിലെ ചെല്ലി", use_container_width=True):
        set_query("തെങ്ങിലെ ചെമ്പൻ ചെല്ലി, കൊമ്പൻ ചെല്ലി എന്നിവയെ എങ്ങനെ നിയന്ത്രിക്കാം?")
    if c3.button("🌿 കുരുമുളക് വാട്ടം", use_container_width=True):
        set_query("കുരുമുളകിലെ ദ്രുതവാട്ടം തടയാൻ ട്രൈക്കോഡെർമയും ബോർഡോ മിശ്രിതവും എങ്ങനെ ഉപയോഗിക്കണം?")

    c4, c5, c6 = st.columns(3)
    if c4.button("🍌 വാഴയിലെ പിണ്ടിപ്പുഴു", use_container_width=True):
        set_query("വാഴയിലെ പിണ്ടിപ്പുഴു നിയന്ത്രണവും വളപ്രയോഗവും എങ്ങനെ?")
    if c5.button("🥬 പച്ചക്കറി കായീച്ച", use_container_width=True):
        set_query("പാവലിലെയും പയറിലെയും കായീച്ചയും കീടങ്ങളും എങ്ങനെ തടയാം?")
    if c6.button("🏛️ കുമ്മായം & സബ്സിഡി", use_container_width=True):
        set_query("മണ്ണിന്റെ അമ്ലത കുറയ്ക്കാൻ കുമ്മായം എത്ര ചേർക്കണം? സുഭിക്ഷ കേരളം സബ്സിഡി എങ്ങനെ ലഭിക്കും?")

    with st.form(key="query_form"):
        query_input = st.text_area(
            "Type or paste your question in Malayalam or English:",
            value=st.session_state.user_query,
            height=85,
            placeholder="e.g. വാഴയിലെ പുഴു / how to control coconut bud rot / കുമ്മായം പ്രയോഗം..."
        )
        submit_btn = st.form_submit_button("🔍 Run RAG Advisory Search", use_container_width=True)

    with st.expander("🎙️ Or Upload Recorded Voice File (.wav)"):
        audio_upload = st.file_uploader("Upload Audio (.wav):", type=["wav"], key="audio_uploader")
        if audio_upload is not None and HAS_SR:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_upload) as source:
                audio_data = recognizer.record(source)
                try:
                    lang_code = "ml-IN" if "Malayalam" in language_mode else "en-IN"
                    transcribed = recognizer.recognize_google(audio_data, language=lang_code)
                    st.success(f"Transcribed Voice: **{transcribed}**")
                    query_input = transcribed
                    submit_btn = True
                except Exception as err:
                    st.warning(f"Voice transcription notice: {err}")

    if submit_btn or query_input:
        active_query = query_input.strip()
        if not active_query:
            st.warning("Please type a question above.")
        else:
            retrieved = engine.retrieve(active_query, top_k=2)
            schemes = engine.match_schemes(selected_crops or ["Paddy"], land_ha, is_owner)
            response_text = engine.generate_answer(
                active_query,
                retrieved,
                schemes,
                language=language_mode,
                api_key=api_key
            )

            st.markdown(response_text)

            if HAS_GTTS and retrieved and retrieved[0].get("similarity_score", 0.0) >= 0.18:
                try:
                    lang_code = "ml" if tts_lang.startswith("ml") else "en"
                    speech_text = retrieved[0]["content_ml"] if lang_code == "ml" else retrieved[0]["content_en"]
                    tts = gTTS(text=speech_text, lang=lang_code)
                    buf = io.BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    st.markdown("#### 🔊 Step 6: Listen to Advisory (Text-to-Speech)")
                    st.audio(buf, format="audio/mp3")
                except Exception:
                    pass

            with st.expander("🔎 Retrieved Vector Knowledge Base Chunks (RAG Cosine Similarity)", expanded=True):
                for idx, doc in enumerate(retrieved, 1):
                    score = doc.get("similarity_score", 0.0)
                    st.markdown(f"**{idx}. [{doc['id']}] {doc['title_en']}** (Cosine Sim: `{score}`)")
                    st.caption(f"Source: {doc['source_name']} - {doc['source_url']}")
                    st.write(doc["content_en"])

with col2:
    st.markdown("### 🏛️ Eligible Government Schemes (Step 5)")
    st.caption(f"Profile: **{district}** | **{land_ha} Ha** | Crops: **{', '.join(selected_crops)}**")
    matched = engine.match_schemes(selected_crops or ["Paddy"], land_ha, is_owner)
    if not matched:
        st.info("No matching schemes found for the current crop/land filter.")
    else:
        for s in matched:
            with st.container(border=True):
                st.markdown(f"#### {s['name_en']}")
                st.markdown(f"**{s['name_ml']}**")
                st.write(f"**Benefit:** {s['benefit_en']}")
                st.caption(f"**How to Apply:** {s['how_to_apply_en']} | [Official Portal]({s['source_url']})")