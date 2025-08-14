from flask import Blueprint, jsonify
from src.config import ARTICLE_CSV, PUBLICATIONS_FILES, CATEGORY_JSON, SAMPLE_N
from src.data_loader import load_article_data, load_publication_data, load_category_descriptions
from src.preprocessing import build_id_map, filter_valid_publications, add_clean_instrument_column
from routes import pipeline_data

data_bp = Blueprint("data", __name__)

@data_bp.route("/load", methods=["GET"])
def load_data():
    article_df = load_article_data(ARTICLE_CSV)
    publications_df = load_publication_data(PUBLICATIONS_FILES)

    if SAMPLE_N:
        article_df = article_df.sample(n=min(SAMPLE_N, len(article_df)), random_state=42)
        publications_df = publications_df.sample(n=min(SAMPLE_N, len(publications_df)), random_state=42)

    pipeline_data["article_df"] = article_df
    pipeline_data["publications_df"] = publications_df

    return jsonify({
        "articles_rows": len(article_df),
        "publications_rows": len(publications_df)
    })

@data_bp.route("/preprocess", methods=["GET"])
def preprocess():
    if "article_df" not in pipeline_data or "publications_df" not in pipeline_data:
        return jsonify({"error": "Call /data/load first"}), 400

    id_map = build_id_map(pipeline_data["article_df"])
    filtered_df = filter_valid_publications(pipeline_data["publications_df"], id_map)
    pre_df = add_clean_instrument_column(filtered_df)

    category_descriptions = load_category_descriptions(CATEGORY_JSON)

    pipeline_data["id_map"] = id_map
    pipeline_data["pre_df"] = pre_df
    pipeline_data["category_descriptions"] = category_descriptions

    return jsonify({
        "processed_rows": len(pre_df),
        "categories": list(category_descriptions.keys())
    })
