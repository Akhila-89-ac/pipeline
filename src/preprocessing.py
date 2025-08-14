import re
import pandas as pd

def build_id_map(article_df: pd.DataFrame) -> dict:
    return dict(zip(article_df["pmid"].astype(str), article_df["abstract"].astype(str)))

def _clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()

def filter_valid_publications(publications_df: pd.DataFrame, id_map: dict) -> pd.DataFrame:
    df = publications_df.copy()
    df = df.dropna(subset=["articleTitle", "Instrument", "Category", "sciLeadsSuperResearcherId", "Classification 1"])
    df = df[df["pmid"].astype(str).str.isnumeric()]
    df["raw_abstract"] = df["pmid"].astype(str).map(id_map)
    df["abstract"] = df["raw_abstract"].map(_clean_text)
    return df

def add_clean_instrument_column(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "LSM880": "LSM 990",
        "LSM 880": "LSM 990",
        "LSM880 NLO": "LSM 990",
        "LSM 880 NLO": "LSM 990",
    }
    out = df.copy()
    out["clean_instrument"] = out["Instrument"].astype(str).str.strip().map(lambda x: mapping.get(x, x))
    return out
