import json

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

# MultiLabelTopN must be importable as __main__.MultiLabelTopN for pickle to restore the model.
from assets_data_prep import MultiLabelTopN, prepare_data  # noqa: F401

app = Flask(__name__)

# Load the trained model once at startup
model = joblib.load("trained_model.pkl")

REQUIRED_FIELDS = ["startYear", "runtimeMinutes", "num_actors", "num_genres", "genres", "Language", "Country"]
NUMERIC_FIELDS  = ["startYear", "runtimeMinutes", "num_actors", "num_genres"]


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        # Validate all required fields are present and non-empty
        missing = [f for f in REQUIRED_FIELDS if f not in data or str(data[f]).strip() == ""]
        if missing:
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

        # Parse and validate numeric fields
        parsed = {}
        for field in REQUIRED_FIELDS:
            if field in NUMERIC_FIELDS:
                try:
                    parsed[field] = float(data[field])
                except (ValueError, TypeError):
                    return jsonify({"error": f"Field '{field}' must be a number, got: {data[field]!r}"}), 400
            else:
                parsed[field] = str(data[field]).strip()

        num_actors = int(parsed["num_actors"])

        # Build a DataFrame with the columns prepare_data expects.
        # - tconst:          dummy (cache misses return sensible defaults)
        # - primaryTitle:    dummy (used only for sequel detection)
        # - lead_actors_ids: N dummy nconst IDs so num_actors is computed correctly
        dummy_actor_ids = json.dumps([f"nm{i:07d}" for i in range(1, num_actors + 1)])

        row = {
            "tconst":          "tt0000000",
            "primaryTitle":    "Unknown",
            "startYear":       parsed["startYear"],
            "runtimeMinutes":  parsed["runtimeMinutes"],
            "genres":          parsed["genres"],
            "Language":        parsed["Language"],
            "Country":         parsed["Country"],
            "lead_actors_ids": dummy_actor_ids,
        }

        df = pd.DataFrame([row])

    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    try:
        processed = prepare_data(df)
        prediction = model.predict(processed)
        rating = round(float(max(0.0, min(10.0, prediction[0]))), 1)
        return jsonify({"predicted_rating": rating})
    except Exception as e:
        app.logger.exception("Internal error during prediction")
        return jsonify({"error": "An internal server error occurred. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True)
