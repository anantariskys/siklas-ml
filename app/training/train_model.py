import os
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from app.preprocessing import preprocess_text 

print("Script started...", flush=True)

# ====== Konfigurasi path ======
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "svm_best.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
PREPROCESSED_PATH = os.path.join(MODEL_DIR, "preprocessed_dataset.xlsx")
PREDICTION_REPORT_PATH = os.path.join(MODEL_DIR, "prediction_report.xlsx")

# ====== Load dataset ======
dataset_path = os.path.join(BASE_DIR, "dataset_skripsi.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset tidak ada di path: {dataset_path}")

print("📂 Loading dataset...")
data = pd.read_csv(dataset_path)
print(f"✅ Dataset berhasil diload, jumlah baris: {len(data)}")

# ====== Preprocessing ======
print("📝 Preprocessing...")
tqdm.pandas()
data["teks"] = (data["Judul Skripsi"].fillna("") + " " + data["Abstrak"].fillna("")).progress_apply(preprocess_text)

data.to_excel(PREPROCESSED_PATH, index=False)
print(f"✅ Dataset hasil preprocessing disimpan di: {PREPROCESSED_PATH}")

# ====== Siapkan fitur dan label ======
X = data["teks"]
y = data["Bidang Penelitian"]

# ====== Filter kelas dengan data sangat sedikit ======
print("🔍 Mengecek distribusi kelas...")
class_counts = y.value_counts()
valid_classes = class_counts[class_counts > 1].index
data = data[data["Bidang Penelitian"].isin(valid_classes)]
X = data["teks"]
y = data["Bidang Penelitian"]

print("Jumlah data setelah filter:", len(data))
print("Distribusi kelas:\n", y.value_counts())

# ====== Train-test split ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== TF-IDF ======
print("🔡 Membuat representasi TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    smooth_idf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ====== Oversampling + SVC (linear dan rbf) + GridSearch ======
print("⚙️ Hyperparameter tuning hanya untuk kernel linear dan rbf...")

pipeline = Pipeline([
    ('oversample', RandomOverSampler(random_state=42)),
    ('svc', SVC(probability=True))
])

# Hanya gunakan kernel linear dan rbf
param_grid = {
    'svc__kernel': ['linear', 'rbf'],
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto'],  # gamma diabaikan untuk linear
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=3
)

grid.fit(X_train_vec, y_train)

print("\n✅ Best Params:", grid.best_params_)
print("✅ Best F1 Score (CV):", grid.best_score_)

# ====== Kalibrasi probabilitas ======
print("\n🔧 Mengkalibrasi probabilitas...")
svm_calibrated = CalibratedClassifierCV(grid.best_estimator_, cv=3)
svm_calibrated.fit(X_train_vec, y_train)
svm = svm_calibrated

# ====== Evaluasi Model ======
print("\n🔎 Evaluasi Model SVM...")
y_pred = svm.predict(X_test_vec)
y_proba = svm.predict_proba(X_test_vec)
confidence = y_proba.max(axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n📊 Evaluasi Model SVM")
print("Akurasi :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ====== Confusion Matrix ======
cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=svm.classes_, yticklabels=svm.classes_)
plt.title("Confusion Matrix SVM (Linear & RBF)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ====== Simpan Evaluasi ke Excel ======
print("📑 Menyimpan evaluasi model ke Excel...")

metrics_df = pd.DataFrame({
    "Metrik": ["Akurasi", "Precision", "Recall", "F1-score"],
    "Nilai": [accuracy, precision, recall, f1]
})

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "Kelas"})

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {cls}" for cls in svm.classes_],
    columns=[f"Predicted: {cls}" for cls in svm.classes_]
)

EVALUATION_PATH = os.path.join(MODEL_DIR, "svm_evaluation_linear_rbf.xlsx")
with pd.ExcelWriter(EVALUATION_PATH) as writer:
    metrics_df.to_excel(writer, sheet_name="Summary Metrics", index=False)
    report_df.to_excel(writer, sheet_name="Classification Report", index=False)
    cm_df.to_excel(writer, sheet_name="Confusion Matrix")

print(f"✅ Laporan evaluasi model disimpan di: {EVALUATION_PATH}")

# ====== Simpan model & vectorizer ======
print("💾 Menyimpan model dan vectorizer...")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(svm, f)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"✅ Model disimpan di: {MODEL_PATH}")
print(f"✅ Vectorizer disimpan di: {VECTORIZER_PATH}")

# ====== Simpan hasil prediksi ======
print("🧾 Menyimpan hasil prediksi ke Excel...")
result_df = pd.DataFrame({
    "Judul Skripsi": X_test.index.map(lambda i: data.loc[i, "Judul Skripsi"] if "Judul Skripsi" in data.columns else ""),
    "Abstrak": X_test.index.map(lambda i: data.loc[i, "Abstrak"] if "Abstrak" in data.columns else ""),
    "Teks Preprocessed": X_test,
    "Bidang Asli": y_test.values,
    "Prediksi SVM": y_pred,
    "Confidence": confidence
})

result_df.sort_values(by="Confidence", ascending=False, inplace=True)
result_df.to_excel(PREDICTION_REPORT_PATH, index=False)
print(f"✅ Laporan hasil prediksi disimpan di: {PREDICTION_REPORT_PATH}")
