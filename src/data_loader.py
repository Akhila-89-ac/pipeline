import json
import pandas as pd

def load_article_data(path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df.rename(columns={"PMID": "pmid", "Abstract": "abstract"})

def load_publication_data(paths) -> pd.DataFrame:
    dfs = [pd.read_csv(p, low_memory=False) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def load_category_descriptions(path: str) -> dict[str, str]:
    with open(path, "r") as f:
        data = json.load(f)
    # allow list or dict in JSON
    if isinstance(data, list):
        return {k: k for k in data}
    return data
