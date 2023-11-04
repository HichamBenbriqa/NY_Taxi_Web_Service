"""_summary_
"""
import os
import neptune as neptune

from dotenv import load_dotenv
from flask import Flask, jsonify, request  # pylint: disable=0E401
from joblib import load

load_dotenv()

PORT = os.getenv("PORT")
PROJECT_NAME = os.getenv("PROJECT_NAME")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")


def get_model_from_neptune():
    """_summary_

    :return: _description_"""
    model = neptune.init_model(
        project=PROJECT_NAME,
        api_token=NEPTUNE_API_TOKEN,
        with_id="MLOPS-RF1",
    )

    model_versions_df = model.fetch_model_versions_table().to_pandas()

    production_models = model_versions_df[
        model_versions_df["sys/stage"] == "production"
    ]
    print(PROJECT_NAME)
    print(NEPTUNE_API_TOKEN)
    for _, model_version in production_models.iterrows():
        print(model_version)
        version_id = model_version["sys/id"]
        model_version = neptune.init_model_version(
            project=PROJECT_NAME, api_token=NEPTUNE_API_TOKEN, with_id=version_id
        )
        print("downloading model")
        model_version["model/binary"].download(f"models/{version_id}_pipeline.joblib")
        print("done downloading model")

    model.stop()
    model_version.stop()
    print("done")
    return load(f"models/{version_id}_pipeline.joblib")


def prepare_features(ride):
    """_summary_

    :param ride: _description_
    :type ride: _type_
    :return: _description_
    :rtype: _type_
    """
    features = {}
    features["PU_DO"] = "%s_%s" % (ride["PULocationID"], ride["DOLocationID"])
    features["trip_distance"] = ride["trip_distance"]
    return features


def predict(features):
    """_summary_

    :param features: _description_
    :type features: _type_
    :return: _description_
    :rtype: _type_
    """
    if os.path.exists("models/MLOPS-RF1-2_pipeline.joblib"):
        model = load("models/MLOPS-RF1-2_pipeline.joblib")
    else:
        model = get_model_from_neptune()
    preds = model.predict(features)
    return float(preds[0])


app = Flask("duration-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {"ride": ride, "duration": pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)
