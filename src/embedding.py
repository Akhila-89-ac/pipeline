import os
import pickle
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from src.config import CACHE_FOLDER, BATCH_SIZE

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(model_name: str, device: str) -> SentenceTransformer:
    print(f"[INFO] Loading model '{model_name}' on device: {device}")
    model = SentenceTransformer(model_name, cache_folder=str(CACHE_FOLDER))
    return model.to(device)

def generate_embeddings(model: SentenceTransformer, texts: List[str]):
    # sentence-transformers handles batching internally via encode()
    return model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE
    )

def save_embeddings(embeddings: torch.Tensor, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(embeddings.detach().cpu(), f)

def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
