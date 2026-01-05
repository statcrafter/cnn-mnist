"""
Application Streamlit - Page Performances
Projet MNIST CNN Classification

Auteur : ALLOUKOUTOU Tundé Lionel Alex
Description : Affichage des performances, métriques et résultats d'entraînement du modèle CNN
"""

import streamlit as st
import pandas as pd
import os
import base64
import sys
from pathlib import Path

# Configuration des chemins pour imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.style import apply_style

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Performances - MNIST CNN",
    page_icon="📊",
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
        --success-color: #059669;
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

    /* Metric cards */
    .metric-card {
        background: var(--white);
        padding: 1.75rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 3px solid var(--accent-color);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.95rem;
        color: var(--text-light);
        font-weight: 500;
    }

    .metric-delta {
        font-size: 0.85rem;
        color: var(--success-color);
        margin-top: 0.25rem;
    }

    /* Success box */
    .success-box {
        background: #d1fae5;
        border: 1px solid var(--success-color);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1.5rem 0;
    }

    .success-box p {
        color: #065f46;
        font-weight: 600;
        margin: 0;
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

    /* Footer */
    .footer-note {
        text-align: center;
        color: var(--text-light);
        font-size: 0.9rem;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
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

st.markdown('<h1 class="main-title">📊 Performances du modèle</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Par {avatar_html}<span class="author-name">ALLOUKOUTOU Tundé Lionel Alex</span></p>', unsafe_allow_html=True)

# Résultats principaux du modèle
st.markdown('<div class="section-header">Résultats obtenus</div>', unsafe_allow_html=True)

# Note explicative sur la variation de l'accuracy
st.markdown("""
<div class="info-box">
    <p>💡 <strong>Note importante :</strong> La précision du modèle oscille entre 99.6% et 99.7% d'un entraînement à l'autre.
    Cette variation est normale et attendue, due à l'initialisation aléatoire des poids et à l'augmentation de données pendant l'entraînement.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">99.6-99.7%</div>
        <div class="metric-label">Précision Test</div>
        <div class="metric-delta">Oscille selon l'entraînement</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">~0.52</div>
        <div class="metric-label">Loss Test</div>
        <div class="metric-delta">CategoricalCrossentropy</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">~300K</div>
        <div class="metric-label">Paramètres</div>
        <div class="metric-delta">Architecture légère</div>
    </div>
    """, unsafe_allow_html=True)

# Progression des performances
st.markdown('<div class="section-header">Progression des performances</div>', unsafe_allow_html=True)

st.markdown("Évolution de l'accuracy au fur et à mesure des améliorations architecturales :")

# Tableau de progression
progress_data = pd.DataFrame({
    "Configuration": [
        "CNN simple (32→64→128)",
        "+ Data augmentation",
        "+ Bloc supplémentaire (256)",
        "+ Label smoothing"
    ],
    "Précision": [
        "99.1%",
        "99.3%",
        "99.44%",
        "99.6-99.7%"
    ],
    "Gain": [
        "Baseline",
        "+0.2%",
        "+0.14%",
        "+0.2-0.3%"
    ]
})

st.table(progress_data)

st.markdown("""
<div class="success-box">
    <p>🎯 Objectif largement dépassé : 99.6-99.7% > 99.4% (cible)</p>
</div>
""", unsafe_allow_html=True)

# Analyse des améliorations
st.markdown('<div class="section-header">Analyse des améliorations</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "CNN de base",
    "Data Augmentation",
    "Bloc Conv 256",
    "Label Smoothing"
])

with tab1:
    st.markdown("#### CNN simple (32→64→128)")
    st.markdown("""
    **Architecture initiale** :
    - 3 blocs convolutifs (32, 64, 128 filtres)
    - BatchNormalization après chaque convolution
    - GlobalAveragePooling pour réduire les paramètres
    - Dropout 0.3

    **Résultat** : 99.1% accuracy

    Cette baseline déjà solide prouve qu'une architecture simple bien conçue peut donner d'excellents résultats sur MNIST.
    """)

with tab2:
    st.markdown("#### + Data Augmentation")
    st.markdown("""
    **Transformations ajoutées** :
    - Rotation aléatoire (±18°)
    - Translation aléatoire (±10%)
    - Zoom aléatoire (±10%)

    **Impact** : +0.2% → 99.3% accuracy

    La data augmentation améliore la généralisation en exposant le modèle à plus de variations.
    Les rotations sont limitées pour éviter les confusions (6↔9).
    """)

with tab3:
    st.markdown("#### + Bloc convolutif supplémentaire (256)")
    st.markdown("""
    **Modification** :
    - Ajout d'un 4ème bloc avec 256 filtres
    - Augmente la capacité du modèle à capturer des features complexes

    **Impact** : +0.14% → 99.44% accuracy

    L'augmentation progressive du nombre de filtres (32→64→128→256) permet au modèle
    d'apprendre des représentations de plus en plus abstraites.
    """)

with tab4:
    st.markdown("#### + Label Smoothing (0.1)")
    st.markdown("""
    **Technique** :
    - Labels "adoucis" : [0, 0, 1, 0, ...] → [0.01, 0.01, 0.91, 0.01, ...]
    - Réduit l'overconfidence du modèle

    **Impact** : +0.2-0.3% → **99.6-99.7% précision**

    Le label smoothing améliore la généralisation en encourageant le modèle à être
    moins certain de ses prédictions, ce qui le rend plus robuste.
    """)

# Pour aller plus loin
st.markdown('<div class="section-header">Pour aller plus loin (99.7% - 99.9%)</div>', unsafe_allow_html=True)

st.markdown("### Ensemble de modèles")
st.markdown("L'utilisation d'un **ensemble de 5 modèles** entraînés avec des seeds différentes permettrait d'atteindre 99.7% - 99.9% :")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Principe** :
    - Entraîner 5 modèles identiques avec des seeds aléatoires différentes
    - Chaque modèle fait des erreurs différentes
    - Moyenner les prédictions des 5 modèles
    """)

with col2:
    st.markdown("""
    **Avantages** :
    - Réduit les erreurs individuelles
    - Plus robuste aux variations
    - Technique éprouvée en compétition

    **Inconvénient** :
    - 5× plus lent à l'inférence
    """)

st.markdown("""
<div class="info-box">
    <p>💡 Cette technique est couramment utilisée en compétition Kaggle pour gagner les derniers points de précision.</p>
</div>
""", unsafe_allow_html=True)

# Comparaison avec d'autres approches
st.markdown('<div class="section-header">Comparaison avec d\'autres approches</div>', unsafe_allow_html=True)

comparison_data = pd.DataFrame({
    "Approche": [
        "MLP simple",
        "CNN de base",
        "Notre CNN optimisé",
        "ResNet32",
        "Ensemble (5 modèles)"
    ],
    "Paramètres": [
        "~100K",
        "~100K",
        "~300K",
        "~470K",
        "~1M"
    ],
    "Accuracy estimée": [
        "~97%",
        "~98.5%",
        "99.63%",
        "~99.6%",
        "~99.8%"
    ],
    "Commentaire": [
        "Trop simple",
        "Bon mais peut mieux faire",
        "Optimal pour MNIST",
        "Surdimensionné",
        "Maximum sans erreurs humaines"
    ]
})

st.table(comparison_data)

st.markdown("""
<div class="success-box">
    <p>✅ Notre modèle offre le meilleur ratio performance/complexité pour MNIST</p>
</div>
""", unsafe_allow_html=True)

# Résultats d'entraînement - Courbes et Matrice de confusion
st.markdown('<div class="section-header">Résultats d\'entraînement</div>', unsafe_allow_html=True)

st.markdown("""
Les graphiques ci-dessous montrent l'évolution de l'entraînement et les performances détaillées du modèle :
""")

# Chemins vers les images d'entraînement
# Les images sont dans training/notebooks/
base_path = Path(__file__).parent.parent.parent  # Remonte de 3 niveaux (pages -> streamlit_app -> cnn_mnist)
training_curves_path = base_path / "training" / "notebooks" / "training_curves.png"
confusion_matrix_path = base_path / "training" / "notebooks" / "confusion_matrix.png"

# Affichage des images côte à côte si elles existent
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Courbes d'entraînement")
    if training_curves_path.exists():
        st.image(str(training_curves_path), use_container_width=True)
        st.caption("Évolution de la précision et de la loss pendant l'entraînement")
    else:
        st.warning("Image des courbes d'entraînement non trouvée")

with col2:
    st.markdown("### 🎯 Matrice de confusion")
    if confusion_matrix_path.exists():
        st.image(str(confusion_matrix_path), use_container_width=True)
        st.caption("Matrice de confusion sur l'ensemble de test")
    else:
        st.warning("Image de la matrice de confusion non trouvée")

# Explication des résultats
st.markdown("""
<div class="info-box">
    <p><strong>Interprétation :</strong></p>
    <ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
        <li><strong>Courbes d'entraînement :</strong> Montrent la convergence du modèle et l'absence de surapprentissage</li>
        <li><strong>Matrice de confusion :</strong> Révèle les confusions les plus fréquentes (ex: 4↔9, 3↔5, 7↔9)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-note">
    <p>99.6-99.7% de précision - Performance exceptionnelle pour un modèle unique</p>
    <p style="margin-top: 0.5rem;">ALLOUKOUTOU Tundé Lionel Alex</p>
</div>
""", unsafe_allow_html=True)
