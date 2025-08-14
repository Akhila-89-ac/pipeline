from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# paths
DATA_PATH = BASE_DIR / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
MODEL_PATH = BASE_DIR / "models"
CACHE_FOLDER = BASE_DIR / ".model_cache"

# files
ARTICLE_CSV = RAW_DATA_PATH / "abstract_all.csv"
PUBLICATIONS_FILES = [
    RAW_DATA_PATH / "PublicationsData(in).csv",
    RAW_DATA_PATH / "PublicinationsData_update_04242025(in).csv".replace("Publicinations","Publications")
]
CATEGORY_JSON = DATA_PATH / "category_descriptions.json"

# outputs
USER_EMB_PATH = MODEL_PATH / "user_embeddings_v3.pkl"
ITEM_EMB_PATH = MODEL_PATH / "item_embeddings_product_v3.pkl"
FINE_TUNED_MODEL_DIR = MODEL_PATH / "fine_tuned_two_tower_model_v3"

# model + runtime
SENTENCE_MODEL = "allenai/specter2_base"
DEVICE = "cpu"     # "cuda" if NVIDIA, "cpu" as fallback
BATCH_SIZE = 16    # used when encoding in batches
NUM_EPOCHS = 1
TOP_K = 10
SAMPLE_N = None    # e.g., 500 for quick dev runs
