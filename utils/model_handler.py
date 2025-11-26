import pickle
import os
from utils.preprocessing import preprocess_text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "svm.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "vectorizer.pkl")

# Load SVM + vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as f:
        svm_model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
else:
    svm_model, vectorizer = None, None


def predict(judul: str, abstrak: str):
    if svm_model is None or vectorizer is None:
        raise ValueError("Model belum tersedia. Harap upload svm.pkl dan vectorizer.pkl")

    text = preprocess_text(judul + " " + abstrak)

    x = vectorizer.transform([text])

    y_pred = svm_model.predict(x)[0]
    y_proba = svm_model.predict_proba(x)[0]
    confidence = float(max(y_proba) * 100)

    return y_pred, confidence
