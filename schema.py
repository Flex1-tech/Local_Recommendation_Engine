import os
import lancedb
from uuid import uuid4
from pydantic import Field
from lancedb.pydantic import LanceModel, Vector

class TrackEmbeddingModel(LanceModel):
    """
    Modèle de données pour stocker les embeddings des pistes audio dans LanceDB.
    Chaque instance représente une piste audio avec son embedding et ses métadonnées associées. 
    """
    # Identifiant unique de la piste
    file_hash: str = Field(primary_key=True, description="Blake3 hash du fichier audio, utilisé comme identifiant unique")

    # Métadonnées générales
    file_name: str
    file_path: str
    file_size_bytes: int

    # Embedding de la piste
    vector: Vector(2048) = Field(description="Embedding audio de dimenson 2048")  # type: ignore