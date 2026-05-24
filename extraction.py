from typing import Any, Optional
import lancedb
import numpy as np
import subprocess
import onnxruntime as ort
import os
from schema import TrackEmbeddingModel
from blake3 import blake3
from pathlib import Path



def get_file_hash(filepath: str):
    hasher = blake3()
    hasher.update_mmap(filepath)
    return hasher.hexdigest()


def load_cnn14(onnx_path: str="cnn14_int8.onnx") -> ort.InferenceSession:

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    cpu = os.cpu_count() or 2

    so.intra_op_num_threads = max(1, cpu // 2)
    so.inter_op_num_threads = 1

    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True

    session = ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=[
            "CPUExecutionProvider"
            ]
    )


    return session


def load_audio(path: str, sr: int = 32000) -> np.ndarray:

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-i", path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-"
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        audio = np.frombuffer(
            result.stdout,
            dtype=np.float32
        )

        return audio

    except (subprocess.CalledProcessError, FileNotFoundError):
        return np.array([], dtype=np.float32)


def extract_top_3_rms_batch(audio: np.ndarray, sr: int = 32000, segment_duration: int = 10) -> np.ndarray:
    """
    Découpe l'audio en blocs de 10s, calcule le RMS de chaque bloc,
    et retourne un batch (3, taille_segment) des 3 blocs les plus puissants.
    """
    if len(audio) == 0:
        return None

    samples_per_segment = sr * segment_duration
    total_samples = len(audio)
    num_segments = total_samples // samples_per_segment

    if num_segments < 3:
        padding_needed = (samples_per_segment * 3) - total_samples
        if padding_needed > 0:
            audio = np.concatenate([audio, np.zeros(padding_needed, dtype=np.float32)])
            num_segments = 3

    audio_trimmed = audio[:num_segments * samples_per_segment]
    segments = audio_trimmed.reshape(num_segments, samples_per_segment)
    rms_values = np.sqrt(np.mean(segments ** 2, axis=1))
    top_3_indices = np.argsort(rms_values)[-3:]
    batch = segments[top_3_indices]
    return batch


def l2_normalize(v):
    return v / (np.linalg.norm(v) + 1e-10)


def compute_embedding(path: str, session: ort.InferenceSession) -> Optional[np.ndarray]:
    """Calcule l'embedding d'un fichier audio (ne touche pas à la base)."""
    file = Path(path)
    if not file.is_file():
        return None

    audio = load_audio(path, sr=32000)
    if audio is None or audio.size == 0:
        return None

    batch = extract_top_3_rms_batch(audio, sr=32000, segment_duration=10)
    if batch is None:
        return None

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch.astype(np.float32)})
    embeddings = outputs[0]
    vector = np.mean(embeddings, axis=0)
    return l2_normalize(vector.astype(np.float32))


def get_or_compute_embedding(path: str, session: ort.InferenceSession, table: lancedb.table.Table) -> Optional[np.ndarray]:
    """
    Récupère ou calcule l'embedding d'un fichier, et l'ajoute en base si nouveau.
    Retourne le vecteur normalisé ou None.
    """
    file = Path(path)
    if not file.is_file():
        return None

    file_hash = get_file_hash(path)

    df = (
        table.search()
        .where(f"file_hash = '{file_hash}'")
        .limit(1)
        .to_pandas()
    )



    # Déjà en base : retourne le vecteur stocké, met à jour le chemin si besoin
    if not df.empty:
        row = df.iloc[0]

        if row['file_path'] != path:
            table.update(where=f"file_hash = '{file_hash}'", values={"file_path": path})

    # Nouveau : calcule et stocke
    vector = compute_embedding(path, session)
    if vector is None:
        return None

    table.add([{
        "file_name": file.name,
        "file_path": path,
        "file_hash": file_hash,
        "file_size_bytes": file.stat().st_size,
        "vector": vector,
    }])
    return vector


def compute_mean_liked_vector(path_dict: dict, session: ort.InferenceSession, table: lancedb.table.Table) -> Optional[np.ndarray]:
    """
    Calcule le vecteur moyen des chansons likées.
    Seuls les vecteurs likés sont récupérés/calculés.
    """
    liked_vectors = []

    for path, is_liked in path_dict.items():
        if not is_liked:
            continue

        try:
            vector = get_or_compute_embedding(path, session, table)
            if vector is not None:
                liked_vectors.append(vector)
        except Exception:
            continue

    if not liked_vectors:
        return None

    final_matrix = np.stack(liked_vectors)
    return np.mean(final_matrix, axis=0)


def index_non_liked_files(path_dict: dict, session: ort.InferenceSession, table: lancedb.table.Table) -> None:
    """
    Indexe en base les fichiers non-likés qui ne sont pas déjà présents.
    Ne retourne aucun vecteur (pas besoin pour le calcul moyen).
    """
    for path, is_liked in path_dict.items():
        if is_liked:
            continue  # Déjà traités par compute_mean_liked_vector

        try:
            file = Path(path)
            if not file.is_file():
                continue

            file_hash = get_file_hash(path)
            result = table.find_one(file_hash=file_hash)

            # Déjà en base : juste mettre à jour le chemin si besoin
            if result is not None and result["file_hash"] == file_hash:
                if result["file_path"] != path:
                    table.update(where=f"file_hash == '{file_hash}'", values={"file_path": path})
                continue

            # Nouveau : calcule et stocke (sans récupérer le vecteur)
            vector = compute_embedding(path, session)
            if vector is not None:
                table.add([{
                    "file_name": file.name,
                    "file_path": path,
                    "file_hash": file_hash,
                    "file_size_bytes": file.stat().st_size,
                    "vector": vector,
                }])

        except Exception:
            continue


def mmr_ranking(
    query_vector: np.ndarray,
    candidates: list[dict],
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Classe TOUS les candidats par ordre MMR (du meilleur au pire).
    
    Args:
        query_vector: Vecteur moyen des likés
        candidates: Liste de dicts avec 'file_path' et 'vector'
        lambda_param: 0.7 = 70% pertinence, 30% diversité
    
    Returns:
        Liste des candidats classés par MMR (tous les candidats, reordonnés)
    """
    if not candidates:
        return []
    
    n = len(candidates)
    
    # Convertir les vecteurs en matrice numpy
    candidate_vectors = np.stack([c["vector"] for c in candidates])
    
    # Similarité cosinus avec le query (produit scalaire car L2-normalisé)
    query_sims = candidate_vectors @ query_vector  # shape: (n_candidates,)
    
    # Matrice de similarité entre candidats
    sim_matrix = candidate_vectors @ candidate_vectors.T  # shape: (n, n)
    np.fill_diagonal(sim_matrix, -np.inf)  # s'auto-exclure
    
    selected = []
    selected_indices = set()
    remaining = set(range(n))
    
    # Itérativement : à chaque tour, choisir le meilleur MMR parmi les restants
    while remaining:
        best_mmr_score = -np.inf
        best_idx = None
        
        for idx in remaining:
            # Pertinence : similitude avec le query
            relevance = query_sims[idx]
            
            # Diversité : max similitude avec les déjà sélectionnés
            # Si c'est le premier, diversity = 0 (pas encore de sélectionnés)
            if selected_indices:
                diversity = np.max(sim_matrix[idx, list(selected_indices)])
            else:
                diversity = 0.0
            
            # Score MMR
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(candidates[best_idx])
            selected_indices.add(best_idx)
            remaining.remove(best_idx)
    
    return selected


def recommend_playlist(
    path_dict: dict,
    session: ort.InferenceSession,
    table: lancedb.table.Table,
    lambda_mmr: float = 0.7,
) -> list[str]:
    """
    Génère une playlist complète (tous les non-likés du dictionnaire)
    classée par MMR à partir du vecteur moyen des likés.
    
    Args:
        path_dict: {chemin_fichier: is_liked (bool)}
        session: Session ONNX Runtime
        table: Table LanceDB
        lambda_mmr: Ratio MMR (0.7 = 70% pertinence, 30% diversité)
    
    Returns:
        Liste des chemins des chansons recommandées (tous les non-likés, classés)
    """
    # 1. Calcule le vecteur moyen des likés
    mean_vector = compute_mean_liked_vector(path_dict, session, table)
    if mean_vector is None:
        return []
    
    # 2. Indexe les non-likés (et met à jour les likés si besoin)
    index_non_liked_files(path_dict, session, table)
    
    
    # 4. Construit la liste de tous les fichiers avec leurs vecteurs
    candidates = []
    
    for path, is_liked in path_dict.items():

        try:
            file = Path(path)
            if not file.is_file():
                continue
            
            vector = get_or_compute_embedding(path, session, table)
            if vector is None:
                continue
            
            candidates.append({
                "file_path": path,
                "file_name": file.name,
                "file_hash": get_file_hash(path),
                "vector": vector
            })
            
        except Exception:
            continue
    
    # 5. Applique MMR pour classer TOUS les non-likés
    ranked = mmr_ranking(mean_vector, candidates, lambda_param=lambda_mmr)
    
    # 6. Retourne les chemins dans l'ordre MMR
    playlist_paths = [c["file_path"] for c in ranked]
    
    return playlist_paths


def extract_cnn14_embedding(path_dict: dict, session: ort.InferenceSession, table: lancedb.table.Table) -> Optional[np.ndarray]:
    """
    Fonction principale :
    1. Calcule le vecteur moyen des likés (récupère/calcule leurs vecteurs)
    2. Indexe les non-likés en arrière-plan (sans charger les vecteurs en mémoire)
    """
    # Étape 1 : calcul du vecteur moyen (seuls les likés passent par l'inférence si absents)
    mean_vector = compute_mean_liked_vector(path_dict, session, table)

    # Étape 2 : indexation des non-likés (optionnel, peut être async ou différé)
    index_non_liked_files(path_dict, session, table)

    return mean_vector


def initialize_database(db_path: str) -> lancedb.table.Table:
    """Initialise la connexion à LanceDB et retourne la table d'embeddings."""
    # Crée la base si elle n'existe pas, sinon l'ouvre
    db = lancedb.connect(db_path)

    # Crée la table "audio_embeddings" si elle n'existe pas, sinon l'ouvre
    if "audio_embeddings" not in db.table_names():
        table = db.create_table("audio_embeddings", schema=TrackEmbeddingModel)
        table.create_index("file_hash")  # Index sur le hash pour accélérer les recherches
    else:
        table = db.open_table("audio_embeddings")
    
    # Retourne la table pour les opérations de lecture/écriture
    return table


def make_m3u(playlist_paths: list[str], output_path: str) -> None:
    """Génère un fichier .m3u à partir d'une liste de chemins de fichiers audio."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for path in playlist_paths:
            p = Path(path)
            f.write(f"#EXTINF:-1,{p.stem}\n")
            f.write(p.as_uri() + "\n")