"""
Application Streamlit - Page Architecture
Projet MNIST CNN Classification

Auteur : ALLOUKOUTOU Tundé Lionel Alex
Description : Détails de l'architecture du réseau CNN, choix techniques et justifications
"""

import streamlit as st
import os
import base64
import sys

# Configuration des chemins pour imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.style import apply_style

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Architecture - MNIST CNN",
    page_icon="🏗️",
    layout="wide"
)

# Appliquer le style global (Poppins)
apply_style()

# CSS personnalisé - Design professionnel moderne
st.markdown("""
<style>
    /* Palette de couleurs professionnelle */
    :root {
        --primary-color: #1e3a8a;
        --accent-color: #3b82f6;
        --text-dark: #1f2937;
        --text-light: #6b7280;
        --white: #ffffff;
    }

    /* Background avec dégradé subtil */
    .main .block-container {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }

    /* Animations modernes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Avatar arrondi */
    .author-avatar-small {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        border: 2px solid var(--accent-color);
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        display: inline-block;
        vertical-align: middle;
        margin-right: 12px;
        transition: transform 0.3s ease;
    }

    .author-avatar-small:hover {
        transform: scale(1.1);
    }

    /* Titre principal avec animation */
    .main-title {
        color: var(--primary-color);
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.6s ease;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: var(--text-light);
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease;
    }

    .author-name {
        color: var(--primary-color);
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        color: var(--primary-color);
        font-size: 1.75rem;
        font-weight: 600;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--accent-color);
    }

    /* Code block container */
    .code-container {
        background: var(--white);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid var(--accent-color);
        margin: 1.5rem 0;
    }

    /* Metric cards */
    .metric-card {
        background: var(--white);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 3px solid var(--accent-color);
    }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border: 1px solid var(--accent-color);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
    }

    .info-box p {
        color: var(--text-dark);
        margin: 0;
        line-height: 1.7;
    }

    /* Warning box */
    .warning-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .warning-box p {
        color: #92400e;
        margin: 0;
        font-weight: 500;
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: var(--text-light);
        font-size: 0.9rem;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: white;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
# En-tête avec avatar
avatar_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'profile.jpg')
avatar_html = ""
if os.path.exists(avatar_path):
    with open(avatar_path, "rb") as f:
        avatar_data = base64.b64encode(f.read()).decode()
        avatar_html = f'<img src="data:image/jpeg;base64,{avatar_data}" class="author-avatar-small" />'

st.markdown('<h1 class="main-title">🏗️ Architecture du modèle</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Par {avatar_html}<span class="author-name">ALLOUKOUTOU Tundé Lionel Alex</span></p>', unsafe_allow_html=True)

# Structure générale
st.markdown('<div class="section-header">Structure générale du CNN</div>', unsafe_allow_html=True)

st.code("""
Input (28×28×1)
    │
    ▼
[Data Augmentation] (uniquement à l'entraînement)
    │
    ▼
Normalisation (μ=33.32, σ=78.57)
    │
    ▼
Conv2D 32 filtres (3×3) → BatchNorm → ReLU → MaxPool (28→14)
    │
    ▼
Conv2D 64 filtres (3×3) → BatchNorm → ReLU → MaxPool (14→7)
    │
    ▼
Conv2D 128 filtres (3×3) → BatchNorm → ReLU
    │
    ▼
Conv2D 256 filtres (3×3) → BatchNorm → ReLU
    │
    ▼
GlobalAveragePooling2D (7×7×256 → 256)
    │
    ▼
Dropout (0.3)
    │
    ▼
Dense 10 (softmax)
""", language="text")

# Paramètres
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2rem; font-weight: 700; color: var(--primary-color); margin-bottom: 0.5rem;">~300K</div>
        <div style="color: var(--text-light); font-size: 0.95rem;">Paramètres totaux</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-color); margin-bottom: 0.5rem;">32→64→128→256</div>
        <div style="color: var(--text-light); font-size: 0.95rem;">Filtres par bloc</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2rem; font-weight: 700; color: var(--primary-color); margin-bottom: 0.5rem;">0.3</div>
        <div style="color: var(--text-light); font-size: 0.95rem;">Taux de dropout</div>
    </div>
    """, unsafe_allow_html=True)

# Choix architecturaux
st.markdown('<div class="section-header">Choix architecturaux et justifications</div>', unsafe_allow_html=True)

# Tabs pour organiser les informations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 CNN simple",
    "📊 Normalisation",
    "🔄 Batch Norm",
    "🎲 Augmentation",
    "🛡️ Régularisation"
])

with tab1:
    st.markdown("#### Pourquoi un CNN simple plutôt qu'un ResNet ?")
    st.markdown("""
    MNIST est un problème relativement **simple**. Un ResNet32 (~470K paramètres) serait surdimensionné :

    - ❌ Temps d'entraînement plus long sans gain significatif
    - ❌ Risque d'overfitting accru
    - ✅ Un CNN de ~300K paramètres atteint des performances comparables

    **Notre CNN est optimisé** pour le problème MNIST spécifiquement.
    """)

with tab2:
    st.markdown("#### Normalisation des données")
    st.markdown("Les statistiques de normalisation sont calculées sur le dataset d'entraînement :")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Moyenne (μ)", "33.3184", help="Échelle 0-255")
    with col2:
        st.metric("Écart-type (σ)", "78.5675", help="Échelle 0-255")

    st.markdown("""
    <div class="info-box">
        <p>💡 La normalisation est intégrée directement dans le modèle, garantissant une cohérence entre l'entraînement et l'inférence.</p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("#### Batch Normalization")
    st.markdown("""
    Chaque bloc convolutif inclut une couche de BatchNormalization après la convolution :

    ✅ **Stabilise l'entraînement**
    ✅ **Permet des learning rates plus élevés**
    ✅ **Agit comme régularisateur léger**

    La BatchNorm normalise les activations entre les batches, facilitant la convergence.
    """)

with tab4:
    st.markdown("#### Data Augmentation")
    st.markdown("Trois transformations légères appliquées **uniquement pendant l'entraînement** :")

    import pandas as pd
    data = pd.DataFrame({
        "Transformation": ["RandomRotation", "RandomTranslation", "RandomZoom"],
        "Paramètre": ["±18° (0.05)", "±10%", "±10%"],
        "Justification": [
            "Simule l'inclinaison naturelle",
            "Simule le décalage du chiffre",
            "Simule les variations de taille"
        ]
    })
    st.table(data)

    st.markdown("""
    <div class="warning-box">
        <p>⚠️ <strong>Précaution MNIST</strong> : Les rotations excessives sont évitées car elles peuvent transformer un 6 en 9 (et vice versa).</p>
    </div>
    """, unsafe_allow_html=True)

with tab5:
    st.markdown("#### Régularisation")

    st.markdown("**1. Label Smoothing (0.1)**")
    st.markdown("""
    - Au lieu de `[0, 0, 1, 0, ...]`, les labels deviennent `[0.01, 0.01, 0.91, 0.01, ...]`
    - Réduit l'overconfidence du modèle
    - Améliore la généralisation
    """)

    st.markdown("**2. Dropout (0.3)**")
    st.markdown("""
    - Appliqué avant la couche Dense finale
    - Valeur réduite car la data augmentation régularise déjà
    - Actif uniquement pendant l'entraînement
    """)

    st.markdown("**3. GlobalAveragePooling**")
    st.markdown("""
    ✅ Réduit les paramètres (7×7×256 = 12 544 → 256)
    ✅ Moins sujet à l'overfitting
    ✅ Standard des architectures modernes
    """)

# Configuration d'entraînement
st.markdown('<div class="section-header">Configuration d\'entraînement</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Hyperparamètres")
    hyperparams = {
        "Optimizer": "Adam",
        "Learning rate initial": "1e-3",
        "Batch size": "128",
        "Epochs max": "50",
        "Loss": "CategoricalCrossentropy",
        "Label smoothing": "0.1"
    }

    params_text = "\n".join([f"**{key}** : {value}" for key, value in hyperparams.items()])
    st.markdown(params_text)

with col2:
    st.markdown("#### Callbacks")

    st.markdown("**ReduceLROnPlateau**")
    st.markdown("""
    - Surveille : validation loss
    - Patience : 3 epochs
    - Facteur : 0.5
    - LR minimum : 1e-6
    """)

    st.markdown("**EarlyStopping**")
    st.markdown("""
    - Surveille : validation loss
    - Patience : 10 epochs
    - Restaure les meilleurs poids
    """)

# Footer
st.markdown("""
<div class="footer-note">
    <p>Architecture optimisée pour MNIST - ~300K paramètres</p>
    <p style="margin-top: 0.5rem;">ALLOUKOUTOU Tundé Lionel Alex</p>
</div>
""", unsafe_allow_html=True)
