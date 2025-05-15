import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.preprocessing import load_and_preprocess_data  # Import preprocessing function

app = FastAPI()

# ✅ API Home
@app.get("/")
def home():
    return {"message": "Divorce Prediction API is running!"}

# ✅ Request body model for prediction
class DivorcePredictionRequest(BaseModel):
    features: list[float]  # List of 54 feature values

# ✅ Function to evaluate a single model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

# ✅ Function to evaluate all models
def evaluate_all_models(X_test, y_test):
    model_files = ["logistic_regression.pkl", "decision_tree.pkl", "random_forest.pkl", "svm.pkl", "xgboost.pkl"]
    results = {}

    for model_file in model_files:
        try:
            model = joblib.load(f"models/{model_file}")
            model_name = model_file.replace("_", " ").replace(".pkl", "").title()
            results[model_name] = evaluate_model(model, X_test, y_test)
        except FileNotFoundError:
            results[model_file] = "Model file not found"

    return results

# ✅ Endpoint to evaluate all models
@app.get("/evaluate")
def evaluate_models():
    try:
        _, X_test, _, y_test = load_and_preprocess_data("data/divorce_data.csv")
        results = evaluate_all_models(X_test, y_test)

        # Save results
        with open("models/evaluation_results.json", "w") as f:
            json.dump(results, f, indent=4)

        return {"message": "Evaluation complete!", "results": results}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found!")

# ✅ Endpoint to fetch saved evaluation results
@app.get("/results")
def get_results():
    try:
        with open("models/evaluation_results.json", "r") as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run /evaluate first!")

# ✅ New Endpoint: Predict Divorce Probability
@app.post("/predict")
def predict_divorce(data: DivorcePredictionRequest):
    try:
        model_path = "models/random_forest.pkl"  # Change to your best model
        model = joblib.load(model_path)

        # Convert input data to numpy array and reshape for prediction
        input_data = np.array(data.features).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of divorce

        return {
            "prediction": int(prediction),  # 1 = Divorce likely, 0 = No Divorce
            "probability": round(probability, 4)
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found!")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
