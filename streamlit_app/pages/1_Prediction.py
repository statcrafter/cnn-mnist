"""
Application Streamlit - Page Prédiction
Projet MNIST CNN Classification

Auteur : ALLOUKOUTOU Tundé Lionel Alex
Description : Interface de prédiction interactive avec 4 modes :
              - Upload d'image
              - Capture webcam
              - Dessin sur canvas
              - Test avec dataset MNIST

Fonctionnalités :
- Visualisation des étapes de prétraitement (11 étapes optimisées)
- Affichage du top 3 des prédictions avec confiance
- TTA (Test-Time Augmentation) optionnel pour +0.2-0.4% précision
- Score de qualité du preprocessing (contraste, taille, aspect ratio)
- 3 modèles rembg disponibles (u2netp, u2net, isnet-general-use)
"""

import streamlit as st
from PIL import Image
import keras
import sys
import os
import numpy as np
import base64
from streamlit_drawable_canvas import st_canvas

# Ajouter les répertoires au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.inference import predict_mnist
# Importer la classe du modèle pour le chargement
from training.utils.model_definition import SimpleCNN_MNIST
from utils.style import apply_style

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédiction - MNIST CNN",
    page_icon="🎯",
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

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.9;
            transform: scale(1.02);
        }
    }

    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
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

    /* Carte d'instructions */
    .info-box {
        background: #eff6ff;
        border: 1px solid var(--accent-color);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
    }

    .info-box h4 {
        color: var(--primary-color);
        margin-top: 0;
        margin-bottom: 1rem;
    }

    .info-box ul {
        color: var(--text-dark);
        line-height: 1.7;
        margin: 0.5rem 0 0 0;
    }

    /* Carte de résultat principal avec glassmorphism */
    .result-card {
        background: linear-gradient(135deg, rgba(239, 246, 255, 0.95) 0%, rgba(219, 234, 254, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border: 2px solid var(--accent-color);
        margin: 1rem 0 1.5rem 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        animation: fadeInUp 0.5s ease;
        transition: all 0.3s ease;
    }

    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25);
    }

    .result-label {
        color: var(--text-light);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }

    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.25rem 0;
        line-height: 1;
        animation: pulse 2s ease-in-out infinite;
    }

    .confidence-label {
        color: var(--success-color);
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* Séparateur */
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #e5e7eb, transparent);
        margin: 2rem 0 1.5rem 0;
    }

    /* Top 3 header */
    .top3-header {
        color: var(--primary-color);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Quality score card avec glassmorphism */
    .quality-card {
        background: rgba(249, 250, 251, 0.9);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(229, 231, 235, 0.6);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        animation: fadeInUp 0.7s ease;
        transition: all 0.3s ease;
    }

    .quality-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: var(--accent-color);
    }

    .quality-header {
        color: var(--primary-color);
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .quality-score {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .quality-score-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color);
    }

    .quality-level {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .quality-level.excellente {
        background: #d1fae5;
        color: #065f46;
    }

    .quality-level.bonne {
        background: #dbeafe;
        color: #1e40af;
    }

    .quality-level.moyenne {
        background: #fef3c7;
        color: #92400e;
    }

    .quality-level.faible {
        background: #fee2e2;
        color: #991b1b;
    }

    .quality-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.5rem;
        font-size: 0.75rem;
    }

    .quality-metric {
        background: white;
        padding: 0.5rem;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
    }

    .quality-metric-label {
        color: var(--text-light);
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .quality-metric-value {
        color: var(--text-dark);
        font-weight: 700;
    }

    /* Section header */
    .section-header {
        color: var(--primary-color);
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-color);
    }

    /* Image container */
    .image-container {
        background: var(--white);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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

    /* Ajustements pour les boutons radio */
    .stRadio > label {
        font-weight: 600;
        color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# Charger le modèle CNN
@st.cache_resource
def load_model():
    # Remonter de streamlit_app/pages/ vers la racine puis aller dans models/
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(root_dir, 'models', 'mnist_cnn.keras')
    return keras.models.load_model(model_path)

# Charger le dataset MNIST
@st.cache_data
def load_mnist_dataset():
    """Charge le dataset MNIST pour le mode test"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return (x_test, y_test)  # On utilise le test set

# Charger des exemples de confusion 1/7
@st.cache_data
def load_confusing_examples():
    """Charge des exemples de 1 et 7 qui peuvent être confondus"""
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    # Trouver des exemples de 1 et 7
    ones = x_train[y_train == 1]
    sevens = x_train[y_train == 7]

    # Prendre quelques exemples qui peuvent être ambigus
    # (on prend juste les premiers pour la démo, idéalement on filtrerait)
    return ones[15], sevens[8]  # Exemples qui se ressemblent visuellement

model = load_model()

# Fonction helper pour afficher le score de qualité
def display_quality_score(quality_score):
    """Affiche le score de qualité du preprocessing de manière visuelle"""
    if quality_score is None:
        return

    # Déterminer la classe CSS pour le niveau de qualité
    level_class = quality_score['quality_level'].lower()

    st.markdown(f"""
    <div class="quality-card">
        <div class="quality-header">📊 Qualité du Preprocessing</div>
        <div class="quality-score">
            <span class="quality-score-value">{quality_score['global_score']}/1.0</span>
            <span class="quality-level {level_class}">{quality_score['quality_level']}</span>
        </div>
        <div class="quality-metrics">
            <div class="quality-metric">
                <div class="quality-metric-label">Contraste</div>
                <div class="quality-metric-value">{quality_score['contrast']:.1f} ({quality_score['contrast_score']:.0%})</div>
            </div>
            <div class="quality-metric">
                <div class="quality-metric-label">Taille</div>
                <div class="quality-metric-value">{quality_score['size']}px ({quality_score['size_score']:.0%})</div>
            </div>
            <div class="quality-metric">
                <div class="quality-metric-label">Aspect ratio</div>
                <div class="quality-metric-value">{quality_score['aspect_ratio']:.2f} ({quality_score['aspect_score']:.0%})</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# En-tête avec avatar
avatar_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'profile.jpg')
avatar_html = ""
if os.path.exists(avatar_path):
    with open(avatar_path, "rb") as f:
        avatar_data = base64.b64encode(f.read()).decode()
        avatar_html = f'<img src="data:image/jpeg;base64,{avatar_data}" class="author-avatar-small" />'

st.markdown('<h1 class="main-title">🎯 Prédiction de chiffres manuscrits</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Par {avatar_html}<span class="author-name">ALLOUKOUTOU Tundé Lionel Alex</span></p>', unsafe_allow_html=True)

# Instructions
with st.expander("ℹ️ Comment utiliser cette application", expanded=False):
    st.markdown("""
    **Modes disponibles**
    - **📤 Upload** : Téléchargez une image de chiffre manuscrit (PNG, JPG, JPEG)
    - **📷 Caméra** : Prenez une photo d'un chiffre écrit sur papier
    - **✏️ Dessiner** : Dessinez un chiffre directement sur le canvas
    - **🎲 Dataset MNIST** : Testez sur les 10 000 images du dataset MNIST original

    **Conseils pour de meilleurs résultats**
    - Écrivez le chiffre en noir sur fond blanc (ou inversement)
    - Assurez-vous que le chiffre est bien visible et net
    - Pour le mode dessin : tracez un chiffre épais et clair
    - Évitez les ombres et reflets (modes Upload/Caméra)

    **Note** : Le système utilise l'IA (rembg) pour supprimer automatiquement l'arrière-plan,
    donc il fonctionne même avec des fonds complexes !
    """)

with st.expander("⚙️ Paramètres avancés", expanded=False):
    rembg_model = st.selectbox(
        "Modèle de suppression d'arrière-plan (rembg)",
        options=["u2netp", "u2net", "isnet-general-use"],
        index=0,
        help="""
        - **u2netp** (Défaut, recommandé pour MNIST) : Léger, rapide et performant (~4.7 MB)
        - **u2net** : Bon équilibre qualité/vitesse (~176 MB)
        - **isnet-general-use** : Plus récent, meilleure qualité générale, bordures plus nettes
        """
    )

    use_tta = st.checkbox(
        "🎯 Activer TTA (Test-Time Augmentation)",
        value=False,
        help="""
        Le TTA améliore la précision en moyennant 5 prédictions avec rotations légères (-5°, -3°, 0°, +3°, +5°).

        **Avantages** : +0.2-0.4% de précision, plus robuste aux rotations
        **Inconvénient** : 5× plus lent (~5 secondes au lieu de ~1 seconde)

        Recommandé pour les images ambiguës ou critiques.
        """
    )

with st.expander("💡 Conseils importants et confusions fréquentes", expanded=False):
    st.markdown("#### ⚠️ Confusion 1 ↔ 7")

    # Exemples visuels avec explications
    col1, col2, col3 = st.columns([1, 1, 4])

    example_1, example_7 = load_confusing_examples()

    with col1:
        st.markdown("<div style='text-align: center; font-size: 1.5rem; font-weight: 700; color: var(--primary-color);'>1</div>", unsafe_allow_html=True)
        st.image(example_1, width=70)

    with col2:
        st.markdown("<div style='text-align: center; font-size: 1.5rem; font-weight: 700; color: var(--primary-color);'>7</div>", unsafe_allow_html=True)
        st.image(example_7, width=70)

    with col3:
        st.markdown("""
        <div style='padding: 0.75rem; background: #fef3c7; border-radius: 6px; border-left: 3px solid #f59e0b; margin-top: 0.5rem;'>
            <p style='margin: 0; color: #92400e; font-size: 0.95rem; line-height: 1.5;'>
                <strong>Pourquoi la confusion ?</strong><br>
                Un <strong>1</strong> écrit comme une simple barre verticale, mais avec une petite barre
                horizontale en haut (ou un trait), peut facilement être confondu avec un <strong>7</strong>
                selon le style d'écriture !
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 1.25rem 0 0.75rem 0; border-top: 1px solid #e5e7eb;'></div>", unsafe_allow_html=True)

    st.markdown("#### 🔍 Conseils pour de meilleures prédictions")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        **📦 Modèle rembg**
        - Le modèle **u2netp** (le plus léger, ~4.7 MB) donne paradoxalement les **meilleurs résultats** pour MNIST
        - Pour des raisons inconnues, il surpasse les modèles plus lourds
        - **Conseil** : Testez les 3 modèles en cas de doute
        """)

    with col_b:
        st.markdown("""
        **🔬 Debug des erreurs**
        - Ouvrez **"🔬 Voir les étapes de transformation"** en cas de mauvaise prédiction
        - Cela vous permet de comprendre ce qui s'est passé pendant le preprocessing
        - Identifiez si le problème vient de l'image ou du traitement
        """)

st.markdown("<br>", unsafe_allow_html=True)

# Choix du mode
mode = st.radio(
    "**Choisissez un mode**",
    ["📤 Upload", "📷 Caméra", "✏️ Dessiner", "🎲 Dataset MNIST"],
    horizontal=True
)

st.markdown("<br>", unsafe_allow_html=True)

if mode == "📤 Upload":
    uploaded_file = st.file_uploader("Choisir une image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-header">Image originale</div>', unsafe_allow_html=True)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image := Image.open(uploaded_file), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header">Résultats de l\'analyse</div>', unsafe_allow_html=True)

            with st.spinner("🔍 Analyse en cours..."):
                top3, steps, quality_score = predict_mnist(
                    image, model,
                    return_steps=True,
                    rembg_model=rembg_model,
                    use_tta=use_tta,
                    return_quality=True
                )

            # Résultat principal
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prédiction</div>
                <div class="prediction-value">{top3[0][0]}</div>
                <div class="confidence-label">{top3[0][1]*100:.1f}% de confiance</div>
            </div>
            """, unsafe_allow_html=True)

            # Séparateur
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Top 3 détaillé
            st.markdown('<div class="top3-header">Détail des prédictions</div>', unsafe_allow_html=True)
            for idx, (digit, conf) in enumerate(top3, 1):
                st.progress(float(conf), text=f"#{idx} - Chiffre {digit} : {conf*100:.1f}%")

            # Affichage du score de qualité
            display_quality_score(quality_score)

        # Étapes de transformation (en pleine largeur)
        with st.expander("🔬 Voir les étapes de transformation MNIST", expanded=False):
            st.markdown("**Pipeline de prétraitement appliqué à l'image :**")

            # Afficher les étapes ligne par ligne pour préserver l'ordre sur mobile
            step_list = [
                ('0_background_removed', f'0️⃣ Suppression fond ({rembg_model})'),
                ('1_grayscale', '1️⃣ Grayscale'),
                ('2_blurred', '2️⃣ Débruitage'),
                ('3_binary_detection', '3️⃣ Détection (binaire temp)'),
                ('4_cropped_grayscale', '4️⃣ Extraction + normalisation'),
                ('5_resized', '5️⃣ Resize 20×20'),
                ('6_final_28x28', '6️⃣ Final 28×28 (entrée modèle)')
            ]

            # Afficher 3 étapes par ligne pour ordre correct sur mobile
            for i in range(0, len(step_list), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(step_list):
                        key, title = step_list[i + j]
                        with cols[j]:
                            st.markdown(f"**{title}**")
                            st.image(steps[key], use_container_width=True, clamp=True)

elif mode == "📷 Caméra":
    camera_input = st.camera_input("📸 Prendre une photo du chiffre")

    if camera_input:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-header">Photo capturée</div>', unsafe_allow_html=True)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image := Image.open(camera_input), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header">Résultats de l\'analyse</div>', unsafe_allow_html=True)

            with st.spinner("🔍 Analyse en cours..."):
                top3, steps, quality_score = predict_mnist(
                    image, model,
                    return_steps=True,
                    rembg_model=rembg_model,
                    use_tta=use_tta,
                    return_quality=True
                )

            # Résultat principal
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prédiction</div>
                <div class="prediction-value">{top3[0][0]}</div>
                <div class="confidence-label">{top3[0][1]*100:.1f}% de confiance</div>
            </div>
            """, unsafe_allow_html=True)

            # Séparateur
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Top 3 détaillé
            st.markdown('<div class="top3-header">Détail des prédictions</div>', unsafe_allow_html=True)
            for idx, (digit, conf) in enumerate(top3, 1):
                st.progress(float(conf), text=f"#{idx} - Chiffre {digit} : {conf*100:.1f}%")

            # Affichage du score de qualité
            display_quality_score(quality_score)

        # Étapes de transformation (en pleine largeur)
        with st.expander("🔬 Voir les étapes de transformation MNIST", expanded=False):
            st.markdown("**Pipeline de prétraitement appliqué à l'image :**")

            # Afficher les étapes ligne par ligne pour préserver l'ordre sur mobile
            step_list = [
                ('0_background_removed', f'0️⃣ Suppression fond ({rembg_model})'),
                ('1_grayscale', '1️⃣ Grayscale'),
                ('2_blurred', '2️⃣ Débruitage'),
                ('3_binary_detection', '3️⃣ Détection (binaire temp)'),
                ('4_cropped_grayscale', '4️⃣ Extraction + normalisation'),
                ('5_resized', '5️⃣ Resize 20×20'),
                ('6_final_28x28', '6️⃣ Final 28×28 (entrée modèle)')
            ]

            # Afficher 3 étapes par ligne pour ordre correct sur mobile
            for i in range(0, len(step_list), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(step_list):
                        key, title = step_list[i + j]
                        with cols[j]:
                            st.markdown(f"**{title}**")
                            st.image(steps[key], use_container_width=True, clamp=True)

elif mode == "✏️ Dessiner":
    st.markdown("**Dessinez un chiffre dans le canvas ci-dessous**")

    # Canvas de dessin
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Bouton pour lancer la prédiction (évite de prédire à chaque trait)
    predict_button = st.button("🔮 Prédire le chiffre", type="primary", use_container_width=True)

    if canvas_result.image_data is not None and predict_button:
        # Vérifier si quelque chose a été dessiné
        if np.any(canvas_result.image_data[:, :, :3] != 255):  # Si pas tout blanc
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown('<div class="section-header">Votre dessin</div>', unsafe_allow_html=True)
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(canvas_result.image_data, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="section-header">Résultats de l\'analyse</div>', unsafe_allow_html=True)

                with st.spinner("🔍 Analyse en cours..."):
                    # Convertir l'image du canvas en PIL
                    img_array = canvas_result.image_data[:, :, :3]  # RGB seulement
                    image = Image.fromarray(img_array.astype('uint8'), 'RGB')

                    top3, steps, quality_score = predict_mnist(
                        image, model,
                        return_steps=True,
                        rembg_model=rembg_model,
                        use_tta=use_tta,
                        return_quality=True
                    )

                # Résultat principal
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Prédiction</div>
                    <div class="prediction-value">{top3[0][0]}</div>
                    <div class="confidence-label">{top3[0][1]*100:.1f}% de confiance</div>
                </div>
                """, unsafe_allow_html=True)

                # Séparateur
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # Top 3 détaillé
                st.markdown('<div class="top3-header">Détail des prédictions</div>', unsafe_allow_html=True)
                for idx, (digit, conf) in enumerate(top3, 1):
                    st.progress(float(conf), text=f"#{idx} - Chiffre {digit} : {conf*100:.1f}%")

                # Affichage du score de qualité
                display_quality_score(quality_score)

            # Étapes de transformation (en pleine largeur)
            with st.expander("🔬 Voir les étapes de transformation MNIST", expanded=False):
                st.markdown("**Pipeline de prétraitement appliqué à l'image :**")

                # Afficher les étapes en grille 3×3 (7 étapes) - ORDRE GARANTI
                step_list = [
                    ('0_background_removed', f'0️⃣ Suppression fond ({rembg_model})'),
                    ('1_grayscale', '1️⃣ Grayscale'),
                    ('2_blurred', '2️⃣ Débruitage'),
                    ('3_binary_detection', '3️⃣ Détection (binaire temp)'),
                    ('4_cropped_grayscale', '4️⃣ Extraction + normalisation'),
                    ('5_resized', '5️⃣ Resize 20×20'),
                    ('6_final_28x28', '6️⃣ Final 28×28 (entrée modèle)')
                ]

                cols = st.columns(3)
                for idx, (key, title) in enumerate(step_list):
                    with cols[idx % 3]:
                        st.markdown(f"**{title}**")
                        st.image(steps[key], use_container_width=True, clamp=True)
        else:
            st.info("👆 Dessinez un chiffre puis cliquez sur 'Prédire le chiffre'")
    elif canvas_result.image_data is not None:
        st.info("👆 Dessinez un chiffre puis cliquez sur 'Prédire le chiffre'")

else:  # Dataset MNIST
    st.markdown("**Testez le modèle sur le dataset MNIST original**")

    # Charger le dataset
    x_test, y_test = load_mnist_dataset()

    # Initialiser l'index dans session_state si pas présent
    if 'mnist_index' not in st.session_state:
        st.session_state.mnist_index = 0

    col_selector, col_button = st.columns([3, 1])

    with col_button:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Image aléatoire"):
            # Mettre à jour le session_state avec un index aléatoire
            st.session_state.mnist_index = np.random.randint(0, len(x_test))
            st.rerun()

    with col_selector:
        # Sélection de l'image - utilise la valeur du session_state
        image_index = st.slider("Index de l'image", 0, len(x_test) - 1,
                               st.session_state.mnist_index)
        # Synchroniser le session_state avec le slider
        st.session_state.mnist_index = image_index

    # Récupérer l'image et le label
    mnist_img = x_test[image_index]
    true_label = y_test[image_index]

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-header">Image MNIST</div>', unsafe_allow_html=True)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(mnist_img, use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"**Label réel : {true_label}**")

    with col2:
        st.markdown('<div class="section-header">Résultats de l\'analyse</div>', unsafe_allow_html=True)

        with st.spinner("🔍 Analyse en cours..."):
            # Préparer l'image pour le modèle (déjà 28×28 grayscale)
            # Le modèle attend (batch, 28, 28, 1) avec valeurs [0, 255]
            img_array = mnist_img.astype('float32')
            img_array = img_array[np.newaxis, ..., np.newaxis]  # (1, 28, 28, 1)

            # Prédiction directe (pas de preprocessing, déjà au format MNIST)
            predictions = model.predict(img_array, verbose=0)[0]
            top3_indices = np.argsort(predictions)[::-1][:3]
            top3_confidences = predictions[top3_indices]
            top3 = list(zip(top3_indices, top3_confidences))

        # Résultat principal avec indication de succès/échec
        is_correct = top3[0][0] == true_label
        result_color = "#059669" if is_correct else "#dc2626"
        result_icon = "✅" if is_correct else "❌"

        st.markdown(f"""
        <div class="result-card" style="border-color: {result_color};">
            <div class="result-label">Prédiction {result_icon}</div>
            <div class="prediction-value">{top3[0][0]}</div>
            <div class="confidence-label" style="color: {result_color};">{top3[0][1]*100:.1f}% de confiance</div>
        </div>
        """, unsafe_allow_html=True)

        if not is_correct:
            st.warning(f"⚠️ Erreur de prédiction ! Label réel : **{true_label}**")

        # Séparateur
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Top 3 détaillé
        st.markdown('<div class="top3-header">Détail des prédictions</div>', unsafe_allow_html=True)
        for idx, (digit, conf) in enumerate(top3, 1):
            st.progress(float(conf), text=f"#{idx} - Chiffre {digit} : {conf*100:.1f}%")

# Footer
st.markdown("""
<div class="footer-note">
    <p>Pipeline optimisé avec 11 étapes : rembg, composition adaptative, CLAHE, morphologie, centrage par centre de masse</p>
    <p>Fonctionnalités : TTA (Test-Time Augmentation) • Score de qualité du preprocessing</p>
    <p style="margin-top: 0.5rem;">ALLOUKOUTOU Tundé Lionel Alex</p>
</div>
""", unsafe_allow_html=True)
