import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

from utils.preprocessing import preprocess_text

plt.switch_backend("Agg")

print("Script started...", flush=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_CLASS_SAMPLES = 5
CV_FOLDS = 5
CALIBRATION_FOLDS = 3
PRIMARY_SCORING = "f1_macro"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"
REPORTS_DIR = MODEL_DIR / "training_reports"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = REPORTS_DIR / RUN_ID
IMAGES_DIR = RUN_DIR / "images"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "svm.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
WORKBOOK_PATH = RUN_DIR / "training_documentation.xlsx"
PREPROCESSED_PATH = RUN_DIR / "preprocessed_dataset.xlsx"
PREPROCESSED_LATEST_PATH = MODEL_DIR / "preprocessed_dataset.xlsx"
PREDICTION_REPORT_PATH = RUN_DIR / "prediction_report.xlsx"
GRIDSEARCH_EXPORT_PATH = RUN_DIR / "grid_search_results.xlsx"
DATASET_PATH = BASE_DIR / "dataset_skripsi.csv"

SCORING_METRICS = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
}

PARAM_GRID = [
    {
        "vectorizer__max_features": [7000, 10000],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__min_df": [2, 3],
        "vectorizer__sublinear_tf": [True],
        "vectorizer__smooth_idf": [True],
        "svc__kernel": ["linear"],
        "svc__C": [0.1, 1, 10],
        "svc__class_weight": [None, "balanced"],
    },
    {
        "vectorizer__max_features": [7000, 10000],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__min_df": [2, 3],
        "vectorizer__sublinear_tf": [True],
        "vectorizer__smooth_idf": [True],
        "svc__kernel": ["rbf"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", "auto", 0.1],
        "svc__class_weight": [None, "balanced"],
    },
]


def build_experiment_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer()),
            ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
            ("svc", SVC(probability=False)),
        ]
    )


def normalize_cv_results(cv_results: dict) -> pd.DataFrame:
    cv_results_df = pd.DataFrame(cv_results).copy()
    cv_results_df["vectorizer_max_features"] = cv_results_df["param_vectorizer__max_features"]
    cv_results_df["vectorizer_ngram_range"] = cv_results_df["param_vectorizer__ngram_range"].astype(str)
    cv_results_df["vectorizer_min_df"] = cv_results_df["param_vectorizer__min_df"]
    cv_results_df["svc_kernel"] = cv_results_df["param_svc__kernel"]
    cv_results_df["svc_C"] = cv_results_df["param_svc__C"]
    cv_results_df["svc_class_weight"] = cv_results_df["param_svc__class_weight"].astype(str)
    cv_results_df["svc_gamma"] = cv_results_df["param_svc__gamma"].astype(str)
    cv_results_df["parameter_signature"] = cv_results_df.apply(
        lambda row: (
            f"tfidf(max_features={row['vectorizer_max_features']}, "
            f"ngram={row['vectorizer_ngram_range']}, min_df={row['vectorizer_min_df']}) | "
            f"svc(kernel={row['svc_kernel']}, C={row['svc_C']}, "
            f"gamma={row['svc_gamma']}, class_weight={row['svc_class_weight']})"
        ),
        axis=1,
    )
    return cv_results_df.sort_values(
        by=["rank_test_f1_macro", "mean_test_f1_weighted", "mean_test_accuracy"]
    ).reset_index(drop=True)


def save_horizontal_bar(series: pd.Series, title: str, output_path: Path, xlabel: str) -> None:
    plt.figure(figsize=(12, max(6, len(series) * 0.35)))
    sns.barplot(x=series.values, y=series.index, orient="h", palette="Blues_r")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Kelas")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_preprocessing_histogram(
    before_tokens: pd.Series, after_tokens: pd.Series, output_path: Path
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(before_tokens, bins=30, alpha=0.6, label="Sebelum Preprocessing", color="#fb9233")
    plt.hist(after_tokens, bins=30, alpha=0.6, label="Sesudah Preprocessing", color="#262e43")
    plt.title("Distribusi Panjang Dokumen Sebelum dan Sesudah Preprocessing")
    plt.xlabel("Jumlah Token")
    plt.ylabel("Jumlah Dokumen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_grid_search_plot(grid_results: pd.DataFrame, output_path: Path) -> None:
    top_results = grid_results.head(20).iloc[::-1]
    plt.figure(figsize=(14, 10))
    sns.barplot(
        data=top_results,
        x="mean_test_f1_macro",
        y="parameter_signature",
        hue="svc_kernel",
        dodge=False,
        palette="deep",
    )
    plt.title("20 Kombinasi GridSearchCV Terbaik Berdasarkan Macro F1")
    plt.xlabel("Mean CV Macro F1")
    plt.ylabel("Kombinasi Parameter")
    plt.legend(title="Kernel", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_heatmap(cm: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix Model SVM Terpilih")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confidence_histogram(confidence: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(confidence, bins=20, color="#10b981", edgecolor="white")
    plt.title("Distribusi Confidence Prediksi")
    plt.xlabel("Confidence")
    plt.ylabel("Jumlah Prediksi")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_actual_vs_predicted_plot(
    y_true: pd.Series, y_pred: pd.Series, output_path: Path
) -> None:
    comparison_df = pd.DataFrame(
        {
            "Actual": y_true.value_counts().sort_index(),
            "Predicted": pd.Series(y_pred).value_counts().sort_index(),
        }
    ).fillna(0)
    comparison_df = comparison_df.sort_values("Actual", ascending=False)
    ax = comparison_df.head(15).plot(
        kind="bar",
        figsize=(14, 7),
        color=["#262e43", "#fb9233"],
    )
    ax.set_title("15 Kelas Teratas: Distribusi Aktual vs Prediksi")
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah Data")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset tidak ada di path: {DATASET_PATH}")

print("Loading dataset...")
data = pd.read_csv(DATASET_PATH)
required_columns = {"Judul Skripsi", "Abstrak", "Bidang Penelitian"}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
    raise ValueError(f"Kolom dataset tidak lengkap: {sorted(missing_columns)}")

print(f"Dataset berhasil dimuat: {len(data)} baris")

data["Judul Skripsi"] = data["Judul Skripsi"].fillna("")
data["Abstrak"] = data["Abstrak"].fillna("")
data["raw_text"] = (data["Judul Skripsi"] + " " + data["Abstrak"]).str.strip()
data["raw_token_count"] = data["raw_text"].str.split().str.len()

print("Menjalankan preprocessing teks...")
tqdm.pandas()
data["teks"] = data["raw_text"].progress_apply(preprocess_text)
data["clean_token_count"] = data["teks"].str.split().str.len()
data["is_text_empty_after_preprocessing"] = data["teks"].eq("")

data.to_excel(PREPROCESSED_PATH, index=False)
data.to_excel(PREPROCESSED_LATEST_PATH, index=False)
print(f"Dataset hasil preprocessing disimpan di: {PREPROCESSED_PATH}")

class_distribution_before = data["Bidang Penelitian"].value_counts().sort_values(ascending=False)
valid_classes = class_distribution_before[class_distribution_before >= MIN_CLASS_SAMPLES].index
filtered_out_classes = class_distribution_before[
    class_distribution_before < MIN_CLASS_SAMPLES
].reset_index()
filtered_out_classes.columns = ["Bidang Penelitian", "Jumlah Data"]

filtered_data = data[data["Bidang Penelitian"].isin(valid_classes)].copy()
class_distribution_after = filtered_data["Bidang Penelitian"].value_counts().sort_values(ascending=False)

print(f"Jumlah data setelah filter minimal {MIN_CLASS_SAMPLES}: {len(filtered_data)}")
print("Distribusi kelas setelah filter:\n", class_distribution_after)

X = filtered_data["teks"]
y = filtered_data["Bidang Penelitian"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print("Menjalankan GridSearchCV dengan dokumentasi lengkap setiap kombinasi...")
grid = GridSearchCV(
    estimator=build_experiment_pipeline(),
    param_grid=PARAM_GRID,
    scoring=SCORING_METRICS,
    refit=PRIMARY_SCORING,
    cv=cv,
    n_jobs=-1,
    verbose=3,
    return_train_score=True,
)
grid.fit(X_train, y_train)

grid_results_df = normalize_cv_results(grid.cv_results_)
grid_results_df.to_excel(GRIDSEARCH_EXPORT_PATH, index=False)

best_params = grid.best_params_.copy()
print("\nBest Params:", best_params)
print("Best CV Macro F1:", grid.best_score_)

vectorizer_params = {
    key.replace("vectorizer__", ""): value
    for key, value in best_params.items()
    if key.startswith("vectorizer__")
}
svc_params = {
    key.replace("svc__", ""): value
    for key, value in best_params.items()
    if key.startswith("svc__")
}
svc_params["probability"] = False

vectorizer = TfidfVectorizer(**vectorizer_params)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

final_pipeline = Pipeline(
    steps=[
        ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
        ("svc", SVC(**svc_params)),
    ]
)

print("Mengkalibrasi probabilitas model terbaik...")
svm = CalibratedClassifierCV(estimator=final_pipeline, cv=CALIBRATION_FOLDS)
svm.fit(X_train_vec, y_train)

print("Evaluasi model pada test set...")
y_pred = svm.predict(X_test_vec)
y_proba = svm.predict_proba(X_test_vec)
confidence = pd.Series(y_proba.max(axis=1), index=y_test.index, name="confidence")

metrics_summary_df = pd.DataFrame(
    {
        "Metrik": [
            "Accuracy",
            "Balanced Accuracy",
            "Precision Weighted",
            "Recall Weighted",
            "F1 Weighted",
            "Precision Macro",
            "Recall Macro",
            "F1 Macro",
        ],
        "Nilai": [
            accuracy_score(y_test, y_pred),
            balanced_accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average="weighted", zero_division=0),
            recall_score(y_test, y_pred, average="weighted", zero_division=0),
            f1_score(y_test, y_pred, average="weighted", zero_division=0),
            precision_score(y_test, y_pred, average="macro", zero_division=0),
            recall_score(y_test, y_pred, average="macro", zero_division=0),
            f1_score(y_test, y_pred, average="macro", zero_division=0),
        ],
    }
)

print(metrics_summary_df.to_string(index=False))

classification_report_df = (
    pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    )
    .transpose()
    .reset_index()
    .rename(columns={"index": "Kelas"})
)

labels = list(svm.classes_)
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {label}" for label in labels],
    columns=[f"Predicted: {label}" for label in labels],
)

prediction_report_df = filtered_data.loc[y_test.index, ["Judul Skripsi", "Abstrak", "teks"]].copy()
prediction_report_df["Bidang Asli"] = y_test.values
prediction_report_df["Prediksi SVM"] = y_pred
prediction_report_df["Confidence"] = confidence.values
prediction_report_df["Benar"] = prediction_report_df["Bidang Asli"] == prediction_report_df["Prediksi SVM"]
prediction_report_df = prediction_report_df.sort_values(
    by=["Benar", "Confidence"], ascending=[True, False]
).reset_index(drop=True)
prediction_report_df.to_excel(PREDICTION_REPORT_PATH, index=False)

feature_importance_df = pd.DataFrame(
    {
        "feature": vectorizer.get_feature_names_out(),
        "idf": vectorizer.idf_,
    }
).sort_values(by=["idf", "feature"]).reset_index(drop=True)

dataset_overview_df = pd.DataFrame(
    {
        "Parameter": [
            "Run ID",
            "Dataset Path",
            "Jumlah Baris Awal",
            "Jumlah Kelas Awal",
            "Jumlah Data Setelah Filter",
            "Jumlah Kelas Setelah Filter",
            "Jumlah Data Train",
            "Jumlah Data Test",
            "Test Size",
            "Random State",
            "Minimum Sampel per Kelas",
            "CV Folds",
            "Calibration Folds",
            "Primary Scoring",
            "Jumlah Kombinasi GridSearch",
            "Jumlah Fitur TF-IDF Terpilih",
            "Dokumen Kosong Setelah Preprocessing",
            "Judul Duplikat",
            "Abstrak Duplikat",
            "Judul+Abstrak Duplikat",
        ],
        "Nilai": [
            RUN_ID,
            str(DATASET_PATH),
            len(data),
            data["Bidang Penelitian"].nunique(),
            len(filtered_data),
            filtered_data["Bidang Penelitian"].nunique(),
            len(X_train),
            len(X_test),
            TEST_SIZE,
            RANDOM_STATE,
            MIN_CLASS_SAMPLES,
            CV_FOLDS,
            CALIBRATION_FOLDS,
            PRIMARY_SCORING,
            len(grid_results_df),
            len(vectorizer.get_feature_names_out()),
            int(data["is_text_empty_after_preprocessing"].sum()),
            int(data.duplicated(subset=["Judul Skripsi"]).sum()),
            int(data.duplicated(subset=["Abstrak"]).sum()),
            int(data.duplicated(subset=["Judul Skripsi", "Abstrak"]).sum()),
        ],
    }
)

preprocessing_summary_df = pd.DataFrame(
    {
        "Metrik": [
            "Rata-rata token sebelum preprocessing",
            "Median token sebelum preprocessing",
            "Rata-rata token sesudah preprocessing",
            "Median token sesudah preprocessing",
            "Dokumen kosong setelah preprocessing",
            "Kelas yang dibuang karena support rendah",
        ],
        "Nilai": [
            round(data["raw_token_count"].mean(), 2),
            round(data["raw_token_count"].median(), 2),
            round(data["clean_token_count"].mean(), 2),
            round(data["clean_token_count"].median(), 2),
            int(data["is_text_empty_after_preprocessing"].sum()),
            len(filtered_out_classes),
        ],
    }
)

preprocessing_samples_df = data[
    [
        "Judul Skripsi",
        "Abstrak",
        "raw_text",
        "teks",
        "raw_token_count",
        "clean_token_count",
        "Bidang Penelitian",
    ]
].head(50)

class_before_df = class_distribution_before.reset_index()
class_before_df.columns = ["Bidang Penelitian", "Jumlah Data"]

class_after_df = class_distribution_after.reset_index()
class_after_df.columns = ["Bidang Penelitian", "Jumlah Data"]

split_summary_df = pd.DataFrame(
    {
        "Subset": ["Train", "Test"],
        "Jumlah Data": [len(X_train), len(X_test)],
        "Persentase": [
            round(len(X_train) / len(filtered_data) * 100, 2),
            round(len(X_test) / len(filtered_data) * 100, 2),
        ],
    }
)

best_model_df = pd.DataFrame(
    {
        "Kunci": [
            "Best Score (CV Macro F1)",
            "Best Params JSON",
            "Vectorizer Params JSON",
            "SVC Params JSON",
        ],
        "Nilai": [
            grid.best_score_,
            json.dumps(best_params, ensure_ascii=True, default=str),
            json.dumps(vectorizer_params, ensure_ascii=True, default=str),
            json.dumps(svc_params, ensure_ascii=True, default=str),
        ],
    }
)

image_paths = {
    "class_distribution_before": IMAGES_DIR / "class_distribution_before.png",
    "class_distribution_after": IMAGES_DIR / "class_distribution_after.png",
    "preprocessing_histogram": IMAGES_DIR / "preprocessing_histogram.png",
    "grid_search_top20": IMAGES_DIR / "grid_search_top20.png",
    "confusion_matrix": IMAGES_DIR / "confusion_matrix.png",
    "confidence_histogram": IMAGES_DIR / "confidence_histogram.png",
    "actual_vs_predicted_top15": IMAGES_DIR / "actual_vs_predicted_top15.png",
}

save_horizontal_bar(
    class_distribution_before,
    "Distribusi Kelas Sebelum Filtering",
    image_paths["class_distribution_before"],
    "Jumlah Data",
)
save_horizontal_bar(
    class_distribution_after,
    "Distribusi Kelas Setelah Filtering",
    image_paths["class_distribution_after"],
    "Jumlah Data",
)
save_preprocessing_histogram(
    data["raw_token_count"],
    data["clean_token_count"],
    image_paths["preprocessing_histogram"],
)
save_grid_search_plot(grid_results_df, image_paths["grid_search_top20"])
save_confusion_matrix_heatmap(cm_df, image_paths["confusion_matrix"])
save_confidence_histogram(confidence, image_paths["confidence_histogram"])
save_actual_vs_predicted_plot(
    y_test,
    pd.Series(y_pred, index=y_test.index),
    image_paths["actual_vs_predicted_top15"],
)

artifacts_df = pd.DataFrame(
    {
        "Artefak": [
            "Workbook Dokumentasi",
            "Preprocessed Dataset",
            "Prediction Report",
            "Grid Search Export",
            "Model SVM",
            "Vectorizer",
            *image_paths.keys(),
        ],
        "Path": [
            str(WORKBOOK_PATH),
            str(PREPROCESSED_PATH),
            str(PREDICTION_REPORT_PATH),
            str(GRIDSEARCH_EXPORT_PATH),
            str(MODEL_PATH),
            str(VECTORIZER_PATH),
            *[str(path) for path in image_paths.values()],
        ],
    }
)

top_grid_results_df = grid_results_df.head(20).copy()

print("Menyimpan dokumentasi training ke Excel...")
with pd.ExcelWriter(WORKBOOK_PATH) as writer:
    dataset_overview_df.to_excel(writer, sheet_name="Ringkasan", index=False)
    preprocessing_summary_df.to_excel(writer, sheet_name="Preprocessing", index=False)
    preprocessing_samples_df.to_excel(writer, sheet_name="Contoh Preprocessing", index=False)
    class_before_df.to_excel(writer, sheet_name="Kelas Sebelum Filter", index=False)
    class_after_df.to_excel(writer, sheet_name="Kelas Setelah Filter", index=False)
    filtered_out_classes.to_excel(writer, sheet_name="Kelas Dibuang", index=False)
    split_summary_df.to_excel(writer, sheet_name="Split Data", index=False)
    best_model_df.to_excel(writer, sheet_name="Best Model", index=False)
    metrics_summary_df.to_excel(writer, sheet_name="Metrics Summary", index=False)
    classification_report_df.to_excel(writer, sheet_name="Classification Report", index=False)
    cm_df.to_excel(writer, sheet_name="Confusion Matrix")
    top_grid_results_df.to_excel(writer, sheet_name="GridSearch Top20", index=False)
    grid_results_df.to_excel(writer, sheet_name="GridSearch Full", index=False)
    prediction_report_df.to_excel(writer, sheet_name="Prediction Report", index=False)
    feature_importance_df.head(500).to_excel(writer, sheet_name="TFIDF Vocabulary", index=False)
    artifacts_df.to_excel(writer, sheet_name="Artefak", index=False)

print("Menyimpan model dan vectorizer...")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(svm, f)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)
with open(RUN_DIR / "svm.pkl", "wb") as f:
    pickle.dump(svm, f)
with open(RUN_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(f"Workbook dokumentasi: {WORKBOOK_PATH}")
print(f"Grid search lengkap: {GRIDSEARCH_EXPORT_PATH}")
print(f"Prediction report: {PREDICTION_REPORT_PATH}")
print(f"Model: {MODEL_PATH}")
print(f"Vectorizer: {VECTORIZER_PATH}")
