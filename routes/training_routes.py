from flask import Blueprint, jsonify
from routes import pipeline_data
from src.config import SENTENCE_MODEL, FINE_TUNED_MODEL_DIR
from src.training import create_training_examples, train_model

training_bp = Blueprint("training", __name__)

@training_bp.route("/fine-tune", methods=["GET"])
def fine_tune():
    if "pre_df" not in pipeline_data or "category_descriptions" not in pipeline_data:
        return jsonify({"error": "Call /data/preprocess first"}), 400

    train_examples = create_training_examples(
        pipeline_data["pre_df"],
        pipeline_data["category_descriptions"]
    )

    model = train_model(SENTENCE_MODEL, train_examples)
    FINE_TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(FINE_TUNED_MODEL_DIR))

    return jsonify({"status": "ok", "saved_to": str(FINE_TUNED_MODEL_DIR)})
