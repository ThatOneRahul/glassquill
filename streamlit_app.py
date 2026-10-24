"""Plan C front end for Streamlit Community Cloud. Same Detector, CPU only."""
import streamlit as st
from src.detector import Detector

st.set_page_config(page_title="GlassQuill")


@st.cache_resource                      # load the models once, not on every click
def load_detector():
    return Detector("models", device="cpu")


st.title("GlassQuill")
st.caption("Explainable detection of AI-generated text. Study project only.")
text = st.text_area("Paste text (50 to 400 words works best)", height=250)

if st.button("Analyse", type="primary"):
    if len(text.split()) < 50:
        st.error("Please paste at least 50 words.")
    else:
        r = load_detector().analyze(text)
        st.metric("Chance it is AI-generated (XGBoost)", f"{r['p_ai']:.0%}")
        st.caption(f"Word model (TF-IDF + LR): {r['p_ai_lr']:.0%} AI")
        st.markdown("**Top reasons**\n\n" + r["reasons"])
        st.pyplot(r["shap_fig"])
