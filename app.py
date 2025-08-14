from flask import Flask
from routes.data_routes import data_bp
from routes.embedding_routes import embedding_bp
from routes.evaluation_routes import evaluation_bp
from routes.training_routes import training_bp
#from api.pipeline_routes import pipeline_bp

app = Flask(__name__)

app.register_blueprint(data_bp, url_prefix="/data")
app.register_blueprint(embedding_bp, url_prefix="/embedding")
app.register_blueprint(evaluation_bp, url_prefix="/evaluation")
app.register_blueprint(training_bp, url_prefix="/training")
#app.register_blueprint(pipeline_bp, url_prefix="/pipeline")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
