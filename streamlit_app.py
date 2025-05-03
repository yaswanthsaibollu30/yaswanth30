import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
from gtts import gTTS
import base64
from googletrans import Translator
from deep_translator import GoogleTranslator

# Set Page Config
st.set_page_config(
    page_title="AI Healthcare Companion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and dataset
@st.cache_resource
def load_model_and_data():
    with open('random_forest.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    mlb = model_data['mlb']
    suggestions_data = pd.read_csv('symptoms.csv')
    return model, mlb, suggestions_data

# Load data
model, mlb, suggestions_data = load_model_and_data()

# Load Lottie animations
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

lottie_health = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_5njp3vgg.json")
lottie_doctor = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_5njp3vgg.json")
lottie_prediction = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_5njp3vgg.json")

# Custom CSS for a modern and soothing design
st.markdown("""
    <style>
        /* General Styling */
        body { background-color: #f0f2f6; }
        .title { color: #2E86C1; text-align: center; font-size: 48px; font-weight: bold; font-family: 'Arial', sans-serif; }
        .subheader { color: #2E86C1; font-size: 28px; font-weight: bold; font-family: 'Arial', sans-serif; }
        
        /* Card Styling */
        .info-card { 
            background-color: #FFFFFF; 
            padding: 25px; 
            border-radius: 15px; 
            margin: 20px 0; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
        }
        .info-card p { 
            color: #333333; /* Dark gray color for <p> tags */
            font-size: 18px; 
            line-height: 1.6; 
        }
        .prediction-box { 
            background-color: #E8F8F5; 
            padding: 30px; 
            border-radius: 15px; 
            text-align: center; 
            font-size: 28px; 
            font-weight: bold; 
            color: #1A5276; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
        }
        .details-box { 
            background-color: #FCF3CF; 
            padding: 25px; 
            border-radius: 15px; 
            margin: 20px 0; 
            color: #7D6608; 
            font-size: 20px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
        }
        .stButton>button { 
            background-color: #2E86C1; 
            color: white; 
            font-size: 18px; 
            border-radius: 10px; 
            padding: 10px 20px; 
        }
        .stButton>button:hover { 
            background-color: #1B4F72; 
        }
        .sidebar .sidebar-content { 
            background-color: #2E86C1; 
            color: white; 
        }
        .stMarkdown h3 { 
            color: #2E86C1; 
        }
    </style>
""", unsafe_allow_html=True)

# Initialize translator
translator = Translator()

# Function to translate text
def translate_text(text, dest_language):
    try:
        translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
        return translated
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return text  # Return original text if translation fails

# Function to convert text to speech and play audio
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    audio_file = open("output.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

# Multi-Page Navigation
with st.sidebar:
    st.markdown('<h2 style="color: white;">Navigation</h2>', unsafe_allow_html=True)
    selected_page = option_menu(
        menu_title=None,
        options=["Home", "Predict Disease", "FAQs", "Contact Us", "About"],
        icons=["house", "heart-pulse", "question-circle", "envelope", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#2E86C1"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
            "nav-link-selected": {"background-color": "#1B4F72"},
        }
    )

# Home Page
if selected_page == "Home":
    # Cover Template
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<h1 class="title">🤖 AI Healthcare Companion</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-card">
                <h3>Welcome to the Future of Healthcare</h3>
                <p>Our AI-powered platform helps you understand potential health conditions based on your symptoms. 
                Get instant predictions and personalized recommendations to take control of your health.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        if lottie_health:
            st_lottie(lottie_health, height=300, key="health")

    # Features Section
    st.markdown("---")
    st.markdown('<h2 class="subheader">✨ Key Features</h2>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>📊 Symptom Analysis</h3>
                <p>Input your symptoms and get a detailed analysis of potential health conditions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="info-card">
                <h3>💊 Personalized Recommendations</h3>
                <p>Receive tailored prescriptions, diet plans, and workout suggestions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col5:
        st.markdown(
            """
            <div class="info-card">
                <h3>🩺 AI-Powered Predictions</h3>
                <p>Our advanced machine learning model ensures accurate and reliable predictions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Predict Disease Page
elif selected_page == "Predict Disease":
    st.markdown('<h1 class="title">🔍 Predict Disease</h1>', unsafe_allow_html=True)
    if lottie_prediction:
        st_lottie(lottie_prediction, height=200, key="prediction")

    # Language Selection
    st.markdown('<h2 class="subheader">🌐 Select Language</h2>', unsafe_allow_html=True)
    languages = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese (Simplified)": "zh-cn",
        "Hindi": "hi",
        "Arabic": "ar",
        "Russian": "ru",
        "Japanese": "ja",
        "Portuguese": "pt",
        "Telugu": "te"
    }
    selected_language = st.selectbox("Choose your preferred language:", list(languages.keys()))

    # Select Symptoms
    st.markdown('<h2 class="subheader">Select Symptoms</h2>', unsafe_allow_html=True)
    selected_symptoms = st.multiselect("Choose from the list:", options=mlb.classes_ if mlb else [])

    # Prediction Button
    if st.button("🔮 Predict Disease", disabled=not selected_symptoms):
        if selected_symptoms:
            # Convert symptoms to input vector
            input_vector = np.zeros(len(mlb.classes_))
            for symptom in selected_symptoms:
                if symptom in mlb.classes_:
                    input_vector[mlb.classes_.tolist().index(symptom)] = 1
            
            # Predict disease and trim whitespace
            predicted_disease = model.predict([input_vector])[0].strip()
            
            # Fetch recommendations
            suggestions = suggestions_data[suggestions_data['Disease'].str.strip() == predicted_disease]
            
            if not suggestions.empty:
                suggestions = suggestions.iloc[0]
                st.markdown(f'<div class="prediction-box">Predicted Disease: <b>{predicted_disease}</b></div>', unsafe_allow_html=True)
                
                # Translate disease details
                dest_language = languages[selected_language]
                description = translate_text(suggestions["Description"], dest_language)
                prescription = translate_text(suggestions["Prescription"], dest_language)
                precautions = translate_text(suggestions["Precautions"], dest_language)
                diet_plan = translate_text(suggestions["Diet Plan"], dest_language)
                workouts = translate_text(suggestions["Workouts"], dest_language)

                # Display Disease Details
                st.markdown("<h3>📌 Disease Information</h3>", unsafe_allow_html=True)
                st.markdown(f'<div class="details-box"><b>Description:</b> {description}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="details-box"><b>Prescription:</b> {prescription}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="details-box"><b>Precautions:</b> {precautions}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="details-box"><b>Diet Plan:</b> {diet_plan}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="details-box"><b>Workouts:</b> {workouts}</div>', unsafe_allow_html=True)

                # Generate text for audio
                audio_text = f"""
                    Predicted Disease: {predicted_disease}.
                    Description: {description}.
                    Prescription: {prescription}.
                    Precautions: {precautions}.
                    Diet Plan: {diet_plan}.
                    Workouts: {workouts}.
                """
                # Play audio
                st.markdown("<h3>🎧 Listen to Recommendations</h3>", unsafe_allow_html=True)
                text_to_speech(audio_text, lang=dest_language)
            else:
                st.markdown(f'<div class="prediction-box">Predicted Disease: <b>{predicted_disease}</b></div>', unsafe_allow_html=True)
                st.warning("No recommendations found for this disease.")
        else:
            st.warning("⚠ Please select at least one symptom.")

# FAQs Page
elif selected_page == "FAQs":
    st.markdown('<h1 class="title">❓ FAQs</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card">
            <h3>1. How does the AI Healthcare Companion work?</h3>
            <p>The AI uses a machine learning model trained on a dataset of symptoms and diseases to predict the most likely condition based on your input.</p>
            <h3>2. Is the prediction accurate?</h3>
            <p>While the model is trained on reliable data, it is not a substitute for professional medical advice. Always consult a doctor for a diagnosis.</p>
            <h3>3. Can I use this app for emergencies?</h3>
            <p>No, this app is not designed for emergencies. In case of an emergency, contact your local healthcare provider immediately.</p>
        </div>
    """, unsafe_allow_html=True)

# Contact Us Page
elif selected_page == "Contact Us":
    st.markdown('<h1 class="title">📧 Contact Us</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card">
            <h3>Have questions or feedback?</h3>
            <p>We'd love to hear from you! Reach out to us at:</p>
            <ul>
                <li>Email: <a href="b.yaswanth.s24@gmail.com">b.yaswanth.s24@gmail.com</a></li>
                <li>Phone: 000000000</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# About Page
elif selected_page == "About":
    st.markdown('<h1 class="title">📚 About</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card">
            <h3>About the Project</h3>
            <p>The AI Healthcare Companion is a project aimed at providing quick and accessible healthcare insights using machine learning. It is designed for educational purposes and to assist users in understanding potential health conditions based on symptoms.</p>
            <h3>Our Mission</h3>
            <p>To make healthcare information more accessible and empower users with knowledge about their health.</p>
            <h3>Team</h3>
            <p>This project was developed by a team of AI enthusiasts and healthcare professionals.</p>
        </div>
    """, unsafe_allow_html=True)