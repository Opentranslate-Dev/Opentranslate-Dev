"""
OpenTranslate web interface
"""

import streamlit as st
import requests
from typing import Optional, Dict
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from opentranslate.api.schemas import (
    TranslationRequest, TranslationResponse,
    ValidationRequest, ValidationResponse,
    TranslatorProfile, TranslationStats
)

# Configure page
st.set_page_config(
    page_title="OpenTranslate",
    page_icon="🌐",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000"
SUPPORTED_LANGUAGES = [
    "en", "zh", "es", "fr", "de", "it", "pt", "ru",
    "ja", "ko", "ar", "hi"
]
SUPPORTED_DOMAINS = [
    "general", "technical", "medical", "legal",
    "business", "academic", "literary"
]

def get_auth_token() -> Optional[str]:
    """Get authentication token from session state"""
    return st.session_state.get("auth_token")

def set_auth_token(token: str):
    """Set authentication token in session state"""
    st.session_state.auth_token = token

def login(address: str, password: str) -> bool:
    """Login user and get authentication token"""
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={"username": address, "password": password}
        )
        if response.status_code == 200:
            token = response.json()["access_token"]
            set_auth_token(token)
            return True
        return False
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False

def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    domain: Optional[str] = None
) -> Optional[Dict]:
    """Submit translation request"""
    try:
        headers = {"Authorization": f"Bearer {get_auth_token()}"}
        data = {
            "source_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "domain": domain
        }
        response = requests.post(
            f"{API_URL}/translations",
            headers=headers,
            json=data
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Translation failed: {str(e)}")
        return None

def get_translation_stats() -> Optional[Dict]:
    """Get translation statistics"""
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Failed to get statistics: {str(e)}")
        return None

def render_login_page():
    """Render login page"""
    st.title("OpenTranslate Login")
    
    with st.form("login_form"):
        address = st.text_input("Wallet Address")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if login(address, password):
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")

def render_translation_page():
    """Render translation page"""
    st.title("OpenTranslate")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source Text")
        source_lang = st.selectbox(
            "Source Language",
            SUPPORTED_LANGUAGES,
            format_func=lambda x: x.upper()
        )
        source_text = st.text_area("Enter text to translate")
        
        domain = st.selectbox(
            "Domain (Optional)",
            [None] + SUPPORTED_DOMAINS,
            format_func=lambda x: x.capitalize() if x else "Auto-detect"
        )
        
        target_lang = st.selectbox(
            "Target Language",
            SUPPORTED_LANGUAGES,
            format_func=lambda x: x.upper()
        )
        
        if st.button("Translate"):
            if source_text:
                with st.spinner("Translating..."):
                    result = translate_text(
                        source_text,
                        source_lang,
                        target_lang,
                        domain
                    )
                    if result:
                        st.session_state.translation_result = result
    
    with col2:
        st.subheader("Translation")
        if "translation_result" in st.session_state:
            result = st.session_state.translation_result
            st.text_area(
                "Translated Text",
                result["target_text"],
                height=200
            )
            
            if result.get("score"):
                st.metric(
                    "Quality Score",
                    f"{result['score']:.2f}"
                )
            
            if result.get("metrics"):
                metrics = result["metrics"]
                st.subheader("Translation Metrics")
                for key, value in metrics.items():
                    st.metric(key, f"{value:.2f}")

def render_stats_page():
    """Render statistics page"""
    st.title("Translation Statistics")
    
    stats = get_translation_stats()
    if stats:
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Translations",
                stats["total_translations"]
            )
        with col2:
            st.metric(
                "Active Translators",
                stats["active_translators"]
            )
        with col3:
            st.metric(
                "Average Score",
                f"{stats['average_score']:.2f}"
            )
        
        # Language distribution
        st.subheader("Language Distribution")
        lang_df = pd.DataFrame(
            list(stats["languages"].items()),
            columns=["Language", "Count"]
        )
        fig = px.bar(
            lang_df,
            x="Language",
            y="Count",
            title="Translations by Language"
        )
        st.plotly_chart(fig)
        
        # Domain distribution
        st.subheader("Domain Distribution")
        domain_df = pd.DataFrame(
            list(stats["domains"].items()),
            columns=["Domain", "Count"]
        )
        fig = px.pie(
            domain_df,
            values="Count",
            names="Domain",
            title="Translations by Domain"
        )
        st.plotly_chart(fig)
        
        # Stake and rewards
        st.subheader("Token Economics")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Total Stake",
            x=["Stake"],
            y=[stats["total_stake"]]
        ))
        fig.add_trace(go.Bar(
            name="Total Rewards",
            x=["Rewards"],
            y=[stats["total_rewards"]]
        ))
        fig.update_layout(title="Token Distribution")
        st.plotly_chart(fig)

def main():
    """Main application"""
    # Check authentication
    if not get_auth_token():
        render_login_page()
        return
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Translation", "Statistics"]
    )
    
    if page == "Translation":
        render_translation_page()
    else:
        render_stats_page()

if __name__ == "__main__":
    main() 