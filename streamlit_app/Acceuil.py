"""
Application Streamlit - Page d'accueil
Projet MNIST CNN Classification

Auteur : ALLOUKOUTOU Tundé Lionel Alex
Description : Interface web pour tester un modèle CNN de classification de chiffres manuscrits MNIST
"""

import streamlit as st
import numpy as np
import keras
import os
import base64
import io
from PIL import Image as PILImage
from utils.style import apply_style, create_card, create_metric, create_link_card

# Configuration de la page Streamlit
st.set_page_config(
    page_title="MNIST CNN - Classification de chiffres",
    page_icon="🔢",
    layout="wide"
)

# Appliquer le style global (Fond animé, Glassmorphism, etc.)
apply_style()

# Fonction pour charger le dataset MNIST
@st.cache_data
def load_mnist_samples():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    samples = {}
    for digit in range(10):
        indices = np.where(y_train == digit)[0]
        selected_indices = np.random.choice(indices, size=2, replace=False)
        samples[digit] = x_train[selected_indices]
    return samples

def img_to_html(img_array):
    img = PILImage.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" />'

# --- EN-TÊTE PRINCIPAL ---
col_logo, col_title = st.columns([1, 4])

avatar_path = os.path.join(os.path.dirname(__file__), 'assets', 'profile.jpg')
avatar_html = ""
if os.path.exists(avatar_path):
    with open(avatar_path, "rb") as f:
        avatar_data = base64.b64encode(f.read()).decode()
        avatar_html = f'<img src="data:image/jpeg;base64,{avatar_data}" class="author-avatar" alt="ALLOUKOUTOU Tundé Lionel Alex" />'

with col_title:
    st.markdown('<h1 class="main-title animate-enter" style="text-align: center; color: #1e3a8a; font-size: 2.5rem; margin-top: 0;">Classification MNIST avec CNN</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-size: 1.1rem; color: #6b7280; margin-bottom: 0;" class="animate-enter delay-100">Par <strong>ALLOUKOUTOU Tundé Lionel Alex</strong></p>', unsafe_allow_html=True)

with col_logo:
    st.markdown(f'<div class="animate-enter" style="display: flex; justify-content: center; align-items: center; height: 100%;">{avatar_html}</div>', unsafe_allow_html=True)

st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)

# --- VUE D'ENSEMBLE ---
st.markdown(create_card(
    "🎯 Vue d'ensemble du projet",
    """
    Ce projet implémente un <strong>réseau de neurones convolutif (CNN)</strong> optimisé pour la
    classification des chiffres manuscrits du dataset MNIST. Le modèle atteint une précision entre
    <strong>99.6% et 99.7%</strong> avec une architecture légère de ~300K paramètres.
    <p style="margin-top: 1rem;">
        <strong>🌐 Testez l'application en ligne :</strong>
        <a href="https://cnn-mnist-ise.streamlit.app/" target="_blank" style="color: #3b82f6; font-weight: 600;">
            cnn-mnist-ise.streamlit.app
        </a>
    </p>
    """,
    icon="🚀"
), unsafe_allow_html=True)

# --- DATASET MNIST ---
st.markdown('<div class="section-header">Dataset MNIST</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(create_metric("60 000", "Images d'entraînement"), unsafe_allow_html=True)
    st.markdown(create_metric("10 000", "Images de test"), unsafe_allow_html=True)
with col2:
    st.markdown(create_metric("28×28", "Dimensions (pixels)"), unsafe_allow_html=True)
    st.markdown(create_metric("784", "Pixels par image"), unsafe_allow_html=True)
with col3:
    st.markdown(create_metric("10", "Classes (chiffres 0-9)"), unsafe_allow_html=True)
    st.markdown(create_metric("Grayscale", "Format d'image"), unsafe_allow_html=True)

# --- EXEMPLES DU DATASET ---
st.markdown('<div class="section-header">Exemples du dataset MNIST</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <p><strong>📊 Variabilité des styles d'écriture :</strong></p>
    <p>Le dataset MNIST contient des milliers de styles d'écriture différents. Le modèle doit apprendre à généraliser sur tous ces styles pour éviter les confusions (1↔7, 3↔5, 4↔9, etc.).</p>
</div>
""", unsafe_allow_html=True)

# Charger les données
mnist_samples = load_mnist_samples()

st.markdown("**Voici un exemple d'écriture pour chaque chiffre :**")
st.markdown("<br>", unsafe_allow_html=True)

# --- Affichage Grille Responsive ---
html_content = '<div class="mnist-grid">'
for digit in range(10):
    img1 = img_to_html(mnist_samples[digit][0])
    html_content += f'<div class="mnist-item"><div class="mnist-label">{digit}</div>{img1}</div>'
html_content += '</div>'
st.markdown(html_content, unsafe_allow_html=True)

st.markdown("""
<div class="glass-card" style="margin-top: 2rem; border-left: 4px solid #3b82f6;">
    <h3 style="margin-top: 0; color: #1e3a8a;">🎯 Constatation importante</h3>
    <p>Même pour un humain, certains chiffres sont difficiles à distinguer ! Le modèle CNN apprend à reconnaître les patterns communs, ce qui lui permet d'atteindre <strong>99.6-99.7% de précision</strong>.</p>
</div>
""", unsafe_allow_html=True)

# --- NAVIGATION ---
st.markdown('<div class="section-header">Explorer le projet</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(create_link_card("Prédiction", "Testez le modèle avec vos propres images via upload ou caméra", "🎯", "Prediction"), unsafe_allow_html=True)
with col2:
    st.markdown(create_link_card("Architecture", "Découvrez la structure du CNN et les choix techniques", "🏗️", "Architecture"), unsafe_allow_html=True)
with col3:
    st.markdown(create_link_card("Performances", "Consultez les résultats et l'évolution des métriques", "📊", "Performances"), unsafe_allow_html=True)

# --- POINTS CLÉS ---
st.markdown('<div class="section-header">Points clés du projet</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(create_card("Architecture optimisée", "4 blocs convolutifs, Batch Normalization, Dropout et GlobalAveragePooling (~300K params).", icon="🎨"), unsafe_allow_html=True)
    st.markdown(create_card("Techniques avancées", "Data Augmentation, Label Smoothing (0.1) et Callbacks adaptatifs.", icon="🔧"), unsafe_allow_html=True)
with col2:
    st.markdown(create_card("Performances", "Précision de 99.6% - 99.7%, architecture légère et excellente généralisation.", icon="⚡"), unsafe_allow_html=True)
    st.markdown(create_card("Application interactive", "4 modes de prédiction, TTA et score de qualité en temps réel.", icon="🚀"), unsafe_allow_html=True)

# --- FOOTER ---
st.markdown('<div class="app-footer" style="text-align: center; color: #6b7280; padding: 2rem 0; border-top: 1px solid #e5e7eb; margin-top: 3rem;">Projet de Deep Learning - Classification MNIST avec CNN<br>ALLOUKOUTOU Tundé Lionel Alex</div>', unsafe_allow_html=True)