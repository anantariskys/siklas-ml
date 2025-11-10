import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from app.preprocessing import preprocess_text


# contoh: load model yang sudah dilatih
MODEL_PATH = "app/model/svm.pkl"
VECTORIZER_PATH = "app/model/vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as f:
        svm_model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
else:
    svm_model, vectorizer = None, None

def predict(judul: str, abstrak: str):
    if not svm_model or not vectorizer:
        raise ValueError("Model belum tersedia. Harap latih model terlebih dahulu.")

    # Preprocess input
    text = preprocess_text(judul + " " + abstrak)
    
    # TF-IDF transform
    x = vectorizer.transform([text])
    
    # Predict
    y_pred = svm_model.predict(x)[0]
    y_proba = svm_model.predict_proba(x)[0]
    y_conf = max(y_proba) * 100
    
    return y_pred, float(y_conf)
