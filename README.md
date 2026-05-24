# 🎵 Musical Recommender System — Version 2

> **Système de recommandation musicale intelligent basé sur l'IA, avec interface graphique moderne et base de données vectorielle.**

🔗 **Version 1** : *[lien à venir]*

---

## 📋 Table des matières

- [Aperçu](#-aperçu)
- [Fonctionnement](#-fonctionnement)
- [Architecture](#-architecture)
- [Pipeline ML](#-pipeline-ml)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Base de données](#-base-de-données)
- [Algorithme MMR](#-algorithme-mmr)
- [Interface utilisateur](#-interface-utilisateur)
- [Différences avec la v1](#-différences-avec-la-v1)
- [Développement](#-développement)
- [Crédits](#-crédits)
- [Licence](#-licence)

---

## 🎯 Aperçu

Un système de recommandation musicale **local et hors-ligne**, basé sur l'analyse audio par deep learning. Il génère automatiquement une playlist personnalisée à partir de vos fichiers audio en apprenant de vos goûts via un système de likes.

- 🖥️ Interface graphique moderne (CustomTkinter) avec thème clair/sombre
- 🧠 Embeddings audio via CNN14 (ONNX Runtime, quantisé INT8)
- 🗄️ Base de données vectorielle locale (LanceDB) — pas de serveur requis
- 🎲 Algorithme MMR pour équilibrer pertinence et diversité
- 🎧 Génération de playlist M3U8 avec lancement automatique VLC

---

## ✨ Fonctionnement

1. **Importez** vos fichiers audio (MP3, FLAC, WAV…)
2. **Likez** au moins 3 morceaux que vous aimez
3. **Cliquez sur "Commencez"** — le système analyse vos fichiers en arrière-plan
4. **VLC s'ouvre** automatiquement avec la playlist ordonnée

Le système extrait un embedding pour chaque fichier via **CNN14**, calcule le vecteur moyen de vos morceaux likés (votre profil musical), puis classe l'ensemble de vos fichiers par pertinence et diversité grâce à l'algorithme **MMR**.

---

## 🏗️ Architecture

```
musical-recommender-v2/
│
├── main.py                  # Point d'entrée
├── app.py                   # Interface graphique (CustomTkinter)
├── func.py                  # Utilitaires UI (open_file, show_toast)
├── func_ia.py               # Pipeline ML : embeddings, DB, MMR, playlist
├── schema.py                # Schéma LanceDB (TrackEmbeddingModel)
│
├── cnn14_int8.onnx          # Modèle CNN14 quantisé INT8
├── requirements.txt
│
├── assets/                  # Icônes et images UI
│   ├── light_upload.png
│   ├── dark_upload.png
│   ├── coeur_gris.png
│   ├── coeur_rouge.png
│   ├── streamline--delete-1-remix.png
│   └── loader.png
│
└── MusicRecommenderDB/      # Base LanceDB (créée automatiquement)
    └── audio_embeddings.lance
```

---

## 🧠 Pipeline ML

```
Fichier audio
     │
     ▼
ffmpeg → PCM float32 (32 kHz, mono)
     │
     ▼
Découpage en segments de 10s
→ sélection des 3 segments les plus énergétiques (RMS)
     │
     ▼
CNN14 INT8 (ONNX Runtime) → 3 embeddings
     │
     ▼
Mean pooling + normalisation L2 → vecteur 512d
     │
     ▼
LanceDB (cache BLAKE3) → recalcul évité si déjà indexé
     │
     ▼
Profil utilisateur = moyenne des vecteurs likés
     │
     ▼
MMR Ranking (λ=0.7) → playlist ordonnée
```

### Optimisations ONNX Runtime

```python
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = max(1, cpu // 2)  # parallélisme contrôlé
so.inter_op_num_threads = 1                  # séquentialité inter-op
```

---

## ⚙️ Installation

### Prérequis système

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) — accessible dans le PATH (décodage audio)
- [VLC](https://www.videolan.org/) — accessible dans le PATH (lecture playlist)

### Dépendances principales

| Package         | Usage                       |
| --------------- | --------------------------- |
| `customtkinter` | Interface graphique         |
| `onnxruntime`   | Inférence CNN14             |
| `lancedb`       | Base de données vectorielle |
| `blake3`        | Hash rapide des fichiers    |
| `numpy`         | Calculs numériques          |
| `Pillow`        | Gestion des icônes          |

### Setup

```bash
git clone https://github.com/<user>/musical-recommender-v2.git
cd musical-recommender-v2

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Placer `cnn14_int8.onnx` à la racine du projet, puis vérifier FFmpeg :

```bash
ffmpeg -version
```

---

## 🚀 Utilisation

```bash
python main.py
```

### Workflow

| Étape             | Action                                                  |
| ----------------- | ------------------------------------------------------- |
| 1. Accueil        | Cliquer *"Importer mes musiques"*                       |
| 2. Sélection      | Choisir vos fichiers audio dans l'explorateur           |
| 3. Likes          | Cliquer ❤️ sur au moins 3 morceaux                      |
| 4. Recommandation | Cliquer *"Commencez..!"* (s'active à partir de 3 likes) |
| 5. Résultat       | `playlist.m3u8` généré, VLC s'ouvre automatiquement     |

### Contrôles

| Élément               | Action                                    |
| --------------------- | ----------------------------------------- |
| 🔍 Barre de recherche | Filtrer la bibliothèque par nom           |
| 🗑️ Icône poubelle    | Retirer un fichier de la sélection        |
| ❤️ Cœur               | Like / Unlike un morceau                  |
| 🔄 Réinitialiser      | Vider la sélection et revenir à l'accueil |

---

## 🗄️ Base de données

Les embeddings sont stockés localement dans `MusicRecommenderDB/` (LanceDB). Chaque fichier est identifié par son hash **BLAKE3** sur le contenu — si vous déplacez ou renommez un fichier, son embedding est retrouvé en base et le chemin est mis à jour automatiquement. Le modèle CNN14 ne tourne qu'**une seule fois par fichier**.

```python
class TrackEmbeddingModel(LanceModel):
    file_name: str
    file_path: str
    file_hash: str       # BLAKE3 — identifiant unique
    file_size_bytes: int
    vector: Vector(512)  # Embedding CNN14
```

---

## 🎲 Algorithme MMR

Le **Maximal Marginal Relevance** équilibre pertinence et diversité à chaque sélection :

```
MMR_score = λ × sim(profil, candidat) − (1−λ) × max_sim(candidat, déjà_sélectionnés)
```

| λ              | Comportement                           |
| -------------- | -------------------------------------- |
| `0.7` (défaut) | 70% pertinence, 30% diversité          |
| `1.0`          | Pure pertinence (risque de redondance) |
| `0.0`          | Pure diversité (exploration maximale)  |

Les morceaux likés sont inclus dans la playlist finale — leur haute similarité avec le profil utilisateur les fait naturellement remonter en tête.

---

## 🖥️ Interface utilisateur

- **Layout** : `grid` exclusif sur `main_frame` (aucun mix `pack`/`place`)
- **Deux boutons contextuels** : `center_button` (accueil) ↔ `import_button` (barre du bas)
- **Threading** : l'inférence tourne dans un thread dédié — l'UI reste réactive
- **Loader animé** : 24 frames à 50ms, fond cohérent avec `main_frame`
- **Palette** : Orange `#FF8E25` · Hover `#F36C19`

| État           | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| Accueil        | Bouton import centré, label accroche                         |
| Import         | Loader animé + "Chargement..."                               |
| Bibliothèque   | Barre recherche · compteur · liste scrollable · barre du bas |
| Recommandation | Loader plein écran → toast succès → VLC                      |

---

## 🔄 Différences avec la v1

|             | v1                   | v2                                                        |
| ----------- | -------------------- | --------------------------------------------------------- |
| Interface   | *(voir v1)*          | CustomTkinter redesigné, loader animé, recherche, reset   |
| Layout      | `place()` uniquement | `grid` sur `main_frame`, deux boutons distincts           |
| Threading   | —                    | Thread dédié pour l'inférence, UI non bloquée             |
| Playlist    | —                    | Tous les fichiers classés, likés inclus                   |
| Cache       | —                    | LanceDB + hash BLAKE3, chemins mis à jour automatiquement |
| Requêtes DB | `find_one`           | `search().where().to_pandas()` + `iloc[0]`                |

---

## 🔧 Développement

```bash
# Vérifier le modèle
python -c "from func_ia import load_cnn14; s = load_cnn14(); print('CNN14 OK')"

# Vérifier la base
python -c "from func_ia import initialize_database; t = initialize_database('.'); print(t.count_rows(), 'entrées')"

# Tester un embedding
python -c "
from func_ia import compute_embedding, load_cnn14
s = load_cnn14()
v = compute_embedding('test.mp3', s)
print(v.shape if v is not None else 'None')
"
```

---

## 🙏 Crédits

- **Modèle CNN14** : [Qiuqiang Kong — PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- **Interface** : [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) (Tom Schimansky)
- **Base de données** : [LanceDB](https://lancedb.com/)
- **Hashing** : [BLAKE3](https://github.com/BLAKE3-team/BLAKE3)

---

## 📄 Licence

MIT

---

> *"Écoutez. Likez. Découvrez. Votre musique, parfaitement orchestrée."*
