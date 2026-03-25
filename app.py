import streamlit as st
import streamlit.components.v1 as components
import os
import json
from moodmate_pipeline import moodmate

# 1. Component Declaration
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(parent_dir, "my_component")
_moodmate_component = components.declare_component("moodmate", path=build_path)

# FIX: Added 'height' to the arguments here
def moodmate_ui(mood_data=None, height=800, key=None):
    return _moodmate_component(mood_data=mood_data, height=height, key=key, default=None)

# 2. Page Setup
st.set_page_config(page_title="MoodMate", layout="wide")

# Hide standard Streamlit header/footer for a cleaner look
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding: 0px !important;}
    iframe {display: block;}
    </style>
    """, unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None

# 3. Execution
# Now 'height' is a valid argument
user_input = moodmate_ui(mood_data=st.session_state.results, height=1000, key="moodmate_app")

if user_input:
    with st.spinner("Analyzing your vibe..."):
        try:
            final_output = moodmate(user_input)
            st.session_state.results = final_output
            # Rerun to push the results into the HTML component
            st.rerun()
        except Exception as e:
            st.error(f"Error processing mood: {e}")