import joblib
import json
import os

def init():
    global model
    # Load the model file from the registered model directory
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)

def run(data):
    try:
        # Parse input JSON
        inputs = json.loads(data)
        prediction = model.predict([inputs["text"]])
        return {"result": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
