# Classification MNIST avec CNN

**Par ALLOUKOUTOU Tundé Lionel Alex**

Ce projet utilise un réseau de neurones convolutif (CNN) pour reconnaître automatiquement des chiffres manuscrits. Le modèle est entraîné sur la base de données MNIST et déployé dans une application web interactive développée avec Streamlit.

## 🌐 Démo en ligne

**Testez l'application directement :** https://cnn-mnist-ise.streamlit.app/

## 📊 Performances du modèle

- **Précision** : 99.6% - 99.7% sur les données de test
- **Taille** : ~300 000 paramètres
- **Objectif** : ✅ Objectif de 99.4% dépassé

> **Note** : La précision oscille légèrement entre 99.6% et 99.7% d'un entraînement à l'autre en raison de l'initialisation aléatoire des poids et de l'augmentation de données pendant l'entraînement. Cette variation est normale et attendue.

## 📁 Structure du projet

Le projet est organisé en 3 parties principales :

### 1. `training/` - Entraînement du modèle
- `notebooks/cnn_mnist.ipynb` : Notebook Jupyter contenant tout le code d'entraînement
- `notebooks/training_curves.png` : Graphiques de progression de l'entraînement
- `notebooks/confusion_matrix.png` : Matrice de confusion des prédictions
- `utils/model_definition.py` : Définition de l'architecture du réseau

### 2. `models/` - Modèle entraîné
- `mnist_cnn.keras` : Le modèle CNN final prêt à être utilisé

### 3. `streamlit_app/` - Application web interactive
- `Home.py` : Page d'accueil de l'application
- `pages/1_Prediction.py` : Interface de prédiction (4 modes disponibles)
- `pages/2_Architecture.py` : Visualisation de l'architecture du modèle
- `pages/3_Performances.py` : Résultats et métriques de performance
- `utils/inference.py` : Fonctions de prétraitement et prédiction

> **Note** : Certains fichiers et dossiers ont été supprimés de la version finale pour ne garder que l'essentiel du projet.

## 🚀 Comment utiliser le projet

### Étape 1 : Installation des dépendances

```bash
pip install -r requirements.txt
```

### Étape 2 : Lancer l'application web

```bash
cd streamlit_app
streamlit run Home.py
```

L'application s'ouvrira dans votre navigateur et propose **4 modes de prédiction** :

1. **📤 Upload** : Télécharger une image de chiffre manuscrit
2. **📷 Caméra** : Prendre une photo en temps réel
3. **✏️ Dessin** : Dessiner un chiffre directement sur l'interface
4. **🎲 Dataset MNIST** : Tester avec les vraies images du dataset MNIST

Pour chaque prédiction, l'application affiche le **top 3 des prédictions** avec leur niveau de confiance.

### Entraînement du modèle

Le code complet d'entraînement se trouve dans le notebook `training/notebooks/cnn_mnist.ipynb`. Les résultats de l'entraînement sont visibles dans les images `training_curves.png` et `confusion_matrix.png`.

## 🚀 Déploiement

L'application est actuellement déployée sur **Streamlit Cloud** et accessible à l'adresse :
**https://cnn-mnist-ise.streamlit.app/**

Pour déployer votre propre version :
1. Créer un compte sur [share.streamlit.io](https://share.streamlit.io)
2. Connecter votre dépôt GitHub
3. Sélectionner le fichier `streamlit_app/Home.py` comme point d'entrée

## 🏗️ Architecture du réseau CNN

Le réseau est composé de **4 blocs convolutifs** qui extraient progressivement des caractéristiques de plus en plus complexes :

1. **Bloc 1** (32 filtres) : Détecte des formes simples (traits, courbes)
2. **Bloc 2** (64 filtres) : Combine les formes simples en motifs
3. **Bloc 3** (128 filtres) : Identifie des parties de chiffres
4. **Bloc 4** (256 filtres) : Reconnaît des structures complètes

Chaque bloc utilise :
- Une **couche de convolution** pour extraire les caractéristiques
- Une **normalisation** pour stabiliser l'apprentissage
- Une **fonction d'activation ReLU** pour introduire de la non-linéarité
- Un **Max Pooling** pour réduire la taille (blocs 1 et 2 uniquement)

Enfin, une **couche dense** avec 10 neurones (un par chiffre 0-9) produit la prédiction finale.

### Techniques d'optimisation

- **Data Augmentation** : Rotations et déformations aléatoires pendant l'entraînement pour améliorer la robustesse
- **Batch Normalization** : Stabilise et accélère l'apprentissage
- **Dropout (30%)** : Évite le surapprentissage en désactivant aléatoirement des neurones
- **Label Smoothing** : Rend le modèle moins sûr de lui pour éviter la surconfiance

## 📈 Évolution des performances

Le modèle a été amélioré progressivement par l'ajout de différentes techniques :

| Étape | Précision | Amélioration |
|-------|-----------|--------------|
| CNN de base (3 blocs) | 99.1% | Point de départ |
| + Augmentation de données | 99.3% | +0.2% |
| + 4ème bloc convolutif (256 filtres) | 99.44% | +0.14% |
| + Label smoothing | **99.6% - 99.7%** | +0.2% |

Chaque amélioration successive a permis de gagner en précision tout en gardant le modèle relativement léger (~300K paramètres).

## 🛠️ Technologies utilisées

- **Deep Learning** : TensorFlow 2.20 / Keras 3.13
- **Interface web** : Streamlit 1.50
- **Traitement d'images** : OpenCV, PIL, rembg u2netp (suppression de fond par IA légère et performante)
- **Optimisations** : CLAHE (contraste adaptatif), morphologie OpenCV, session caching
- **Dataset** : MNIST (60 000 images d'entraînement, 10 000 images de test)

## 📝 Points importants

- **Prétraitement automatique optimisé** : L'application transforme automatiquement les photos pour les rendre compatibles avec MNIST
  - 📄 **Détails techniques** : Voir [PREPROCESSING.md](PREPROCESSING.md) pour comprendre les **11 étapes du pipeline optimisé**
  - 💻 **Implémentation** : `streamlit_app/utils/inference.py`
  - ⚡ **Optimisations** :
    - Session rembg cachée (+30-50% de vitesse)
    - Composition adaptative (gère fond noir/blanc automatiquement)
    - CLAHE pour contraste optimal
    - Validation de détection (rejette formes aberrantes)
    - Post-processing morphologique pour meilleur match MNIST
- **🎯 TTA (Test-Time Augmentation)** : Option pour améliorer la précision (+0.2-0.4%) en moyennant 5 prédictions avec rotations légères
  - Particulièrement utile pour les images ambiguës (ex: confusion 1/7)
  - Inconvénient : 5× plus lent, à réserver aux cas critiques
- **📊 Score de qualité du preprocessing** : Évaluation automatique de la qualité avec 3 métriques (contraste, taille, aspect ratio)
  - Affichage visuel avec badge de niveau (Excellente/Bonne/Moyenne/Faible)
  - Permet de détecter les images problématiques avant prédiction
- **4 modes de test** : Permet de tester le modèle dans différentes conditions (upload, caméra, dessin, dataset MNIST)
- **Visualisation des étapes** : Possibilité de voir toutes les étapes de prétraitement appliquées à l'image en temps réel

---

**ALLOUKOUTOU Tundé Lionel Alex** - Projet de Deep Learning
