from flask import Blueprint, jsonify
from sentence_transformers import util
from routes import pipeline_data
from src.evaluation import calculate_topk_accuracy, mean_average_precision, mean_reciprocal_rank

evaluation_bp = Blueprint("evaluation", __name__)

@evaluation_bp.route("/run", methods=["GET"])
def run_eval():
    for key in ("pre_df", "user_emb", "item_emb", "category_descriptions"):
        if key not in pipeline_data:
            return jsonify({"error": f"Missing '{key}'. Run previous steps."}), 400

    pre_df = pipeline_data["pre_df"]
    user_emb = pipeline_data["user_emb"]
    item_emb = pipeline_data["item_emb"]
    cat_desc = pipeline_data["category_descriptions"]

    true_labels = pre_df["clean_instrument"].tolist()
    category_list = list(cat_desc.keys())

    sim = util.cos_sim(user_emb, item_emb)
    top1_acc, topk_acc, top_k_categories = calculate_topk_accuracy(sim, true_labels, category_list)

    return jsonify({
        "top1_accuracy": round(top1_acc, 4),
        "topk_accuracy": round(topk_acc, 4),
        "map": round(mean_average_precision(true_labels, top_k_categories), 4),
        "mrr": round(mean_reciprocal_rank(true_labels, top_k_categories), 4)
    })
