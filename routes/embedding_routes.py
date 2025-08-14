from flask import Blueprint, jsonify
from src.config import SENTENCE_MODEL, USER_EMB_PATH, ITEM_EMB_PATH
from routes import pipeline_data
from src.embedding import load_model, generate_embeddings, save_embeddings, get_device

embedding_bp = Blueprint("embedding", __name__)

@embedding_bp.route("/generate", methods=["GET"])
def generate():
    if "pre_df" not in pipeline_data or "category_descriptions" not in pipeline_data:
        return jsonify({"error": "Call /data/preprocess first"}), 400

    pre_df = pipeline_data["pre_df"]
    cat_desc = pipeline_data["category_descriptions"]
    category_texts = [f"{k}: {v}" for k, v in cat_desc.items()]

    device = get_device()
    model = load_model(SENTENCE_MODEL, device)

    user_emb = generate_embeddings(model, pre_df["abstract"].tolist())
    item_emb = generate_embeddings(model, category_texts)

    save_embeddings(user_emb, USER_EMB_PATH)
    save_embeddings(item_emb, ITEM_EMB_PATH)

    pipeline_data["user_emb"] = user_emb
    pipeline_data["item_emb"] = item_emb

    return jsonify({
        "status": "ok",
        "user_emb_saved": str(USER_EMB_PATH),
        "item_emb_saved": str(ITEM_EMB_PATH)
    })
