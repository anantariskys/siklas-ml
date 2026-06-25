from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

plt.switch_backend("Agg")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"
OUTPUT_DIR = MODEL_DIR / "hasil_pengujian"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = MODEL_DIR / "preprocessed_dataset.xlsx"
TABLE_PATH = OUTPUT_DIR / "tabel_hasil_pengujian_model.xlsx"
DETAIL_PATH = OUTPUT_DIR / "detail_30_eksperimen_pengujian_model.xlsx"
FIGURE_PATH = OUTPUT_DIR / "grafik_akurasi_model.png"
METRIC_FIGURE_PATH = OUTPUT_DIR / "grafik_metrik_rata_rata_model.png"
DOCUMENTATION_PATH = OUTPUT_DIR / "dokumentasi_pengujian_model.md"

RANDOM_STATE_START = 42
REPETITIONS_PER_SCENARIO = 10
CALIBRATION_FOLDS = 3

TFIDF_PARAMS = {
    "max_features": 7000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "smooth_idf": True,
}

SPLIT_SCENARIOS = {
    "70:30": 0.30,
    "80:20": 0.20,
    "90:10": 0.10,
}


def decimal_id(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def markdown_table(summary_df: pd.DataFrame) -> str:
    headers = [
        "Skenario Pembagian Data",
        "Jumlah Pengujian",
        "Jumlah Data Latih",
        "Jumlah Data Uji",
        "Akurasi Rata-rata (%)",
        "Precision Rata-rata (%)",
        "Recall Rata-rata (%)",
        "F1-Score Rata-rata (%)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Skenario Pembagian Data"]),
                    str(int(row["Jumlah Pengujian"])),
                    str(int(row["Jumlah Data Latih"])),
                    str(int(row["Jumlah Data Uji"])),
                    decimal_id(row["Akurasi Rata-rata (%)"]),
                    decimal_id(row["Precision Rata-rata (%)"]),
                    decimal_id(row["Recall Rata-rata (%)"]),
                    decimal_id(row["F1-Score Rata-rata (%)"]),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def build_model(random_state: int) -> CalibratedClassifierCV:
    pipeline = Pipeline(
        [
            ("oversample", RandomOverSampler(random_state=random_state)),
            ("svm", SVC(kernel="linear", probability=False)),
        ]
    )
    return CalibratedClassifierCV(pipeline, cv=CALIBRATION_FOLDS)


def run_single_experiment(
    scenario: str,
    test_size: float,
    repetition: int,
    random_state: int,
    X: pd.Series,
    y: pd.Series,
) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = build_model(random_state)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    return {
        "Skenario Pembagian Data": scenario,
        "Pengujian Ke": repetition,
        "Random State": random_state,
        "Jumlah Data Latih": len(X_train),
        "Jumlah Data Uji": len(X_test),
        "Akurasi (%)": accuracy_score(y_test, y_pred) * 100,
        "Precision (%)": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        * 100,
        "Recall (%)": recall_score(y_test, y_pred, average="weighted", zero_division=0)
        * 100,
        "F1-Score (%)": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        * 100,
    }


def summarize_results(detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    variation_rows = []

    for scenario in SPLIT_SCENARIOS:
        scenario_df = detail_df[detail_df["Skenario Pembagian Data"] == scenario]

        summary_rows.append(
            {
                "Skenario Pembagian Data": scenario,
                "Jumlah Pengujian": len(scenario_df),
                "Jumlah Data Latih": int(round(scenario_df["Jumlah Data Latih"].mean())),
                "Jumlah Data Uji": int(round(scenario_df["Jumlah Data Uji"].mean())),
                "Akurasi Rata-rata (%)": scenario_df["Akurasi (%)"].mean(),
                "Precision Rata-rata (%)": scenario_df["Precision (%)"].mean(),
                "Recall Rata-rata (%)": scenario_df["Recall (%)"].mean(),
                "F1-Score Rata-rata (%)": scenario_df["F1-Score (%)"].mean(),
            }
        )

        variation_rows.append(
            {
                "Skenario Pembagian Data": scenario,
                "Std Akurasi (%)": scenario_df["Akurasi (%)"].std(),
                "Std Precision (%)": scenario_df["Precision (%)"].std(),
                "Std Recall (%)": scenario_df["Recall (%)"].std(),
                "Std F1-Score (%)": scenario_df["F1-Score (%)"].std(),
                "Min Akurasi (%)": scenario_df["Akurasi (%)"].min(),
                "Max Akurasi (%)": scenario_df["Akurasi (%)"].max(),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(variation_rows)


def autosize_excel_columns(writer: pd.ExcelWriter) -> None:
    for worksheet in writer.book.worksheets:
        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter
            for cell in column_cells:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))
            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 48)


def save_workbooks(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    variation_df: pd.DataFrame,
    config_df: pd.DataFrame,
) -> None:
    rounded_summary = summary_df.round(2)
    rounded_detail = detail_df.round(2)
    rounded_variation = variation_df.round(2)

    with pd.ExcelWriter(TABLE_PATH, engine="openpyxl") as writer:
        rounded_summary.to_excel(writer, sheet_name="Ringkasan Rata-rata", index=False)
        rounded_detail.to_excel(writer, sheet_name="Detail 30 Eksperimen", index=False)
        rounded_variation.to_excel(writer, sheet_name="Standar Deviasi", index=False)
        config_df.to_excel(writer, sheet_name="Konfigurasi", index=False)
        autosize_excel_columns(writer)

    rounded_detail.to_excel(DETAIL_PATH, index=False)


def save_accuracy_plot(summary_df: pd.DataFrame, variation_df: pd.DataFrame) -> None:
    scenarios = summary_df["Skenario Pembagian Data"]
    accuracy = summary_df["Akurasi Rata-rata (%)"]
    accuracy_std = variation_df["Std Akurasi (%)"]
    colors = ["#2563eb", "#16a34a", "#f97316"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        scenarios,
        accuracy,
        yerr=accuracy_std,
        capsize=6,
        color=colors,
        edgecolor="#1f2937",
        linewidth=0.8,
    )

    for bar, value in zip(bars, accuracy):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{decimal_id(value)}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    y_min = max(0, float(accuracy.min()) - 3)
    y_max = min(100, float(accuracy.max()) + 4)
    plt.ylim(y_min, y_max)
    plt.title("Rata-rata Akurasi Model dari 10 Pengujian per Skenario")
    plt.xlabel("Skenario Pembagian Data")
    plt.ylabel("Akurasi Rata-rata (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close()


def save_metric_plot(summary_df: pd.DataFrame) -> None:
    scenarios = summary_df["Skenario Pembagian Data"].tolist()
    metrics = [
        ("Akurasi", "Akurasi Rata-rata (%)"),
        ("Precision", "Precision Rata-rata (%)"),
        ("Recall", "Recall Rata-rata (%)"),
        ("F1-Score", "F1-Score Rata-rata (%)"),
    ]
    x = np.arange(len(scenarios))
    width = 0.19

    plt.figure(figsize=(10, 5.5))
    for idx, (label, column) in enumerate(metrics):
        offset = (idx - 1.5) * width
        plt.bar(x + offset, summary_df[column], width, label=label)

    metric_columns = [column for _, column in metrics]
    plt.xticks(x, scenarios)
    plt.ylim(max(0, summary_df[metric_columns].min().min() - 4), 100)
    plt.title("Perbandingan Rata-rata Metrik Model")
    plt.xlabel("Skenario Pembagian Data")
    plt.ylabel("Nilai Rata-rata (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(METRIC_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close()


def save_documentation(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    variation_df: pd.DataFrame,
    config_df: pd.DataFrame,
) -> None:
    best_row = summary_df.loc[summary_df["Akurasi Rata-rata (%)"].idxmax()]
    best_scenario = best_row["Skenario Pembagian Data"]
    best_accuracy = decimal_id(best_row["Akurasi Rata-rata (%)"])
    best_precision = decimal_id(best_row["Precision Rata-rata (%)"])
    best_recall = decimal_id(best_row["Recall Rata-rata (%)"])
    best_f1 = decimal_id(best_row["F1-Score Rata-rata (%)"])
    total_experiments = len(detail_df)
    random_states = ", ".join(str(state) for state in detail_df["Random State"].unique())

    documentation = f"""# Dokumentasi Pengujian Model

Pengujian model klasifikasi bidang penelitian dilakukan dengan tiga skenario pembagian data, yaitu 70:30, 80:20, dan 90:10. Sesuai revisi penguji, setiap skenario diulang sebanyak {REPETITIONS_PER_SCENARIO} kali dengan variasi `random_state` {random_states}. Dengan demikian, total eksperimen yang dilakukan adalah {total_experiments} eksperimen. Nilai pada tabel berikut merupakan rata-rata dari {REPETITIONS_PER_SCENARIO} kali pengujian pada masing-masing skenario.

## Tabel 5.19 Hasil Pengujian Model Klasifikasi Bidang Penelitian

{markdown_table(summary_df)}

## Gambar 5.5 Perbandingan Akurasi Model

![Gambar 5.5 Perbandingan Akurasi Model]({FIGURE_PATH.name})

Gambar 5.5 menunjukkan perbandingan rata-rata akurasi model pada setiap skenario pembagian data. Error bar pada grafik menunjukkan standar deviasi akurasi dari {REPETITIONS_PER_SCENARIO} kali pengujian.

## Ringkasan Narasi Revisi 5.4.1

Berdasarkan Tabel 5.19, skenario pembagian data {best_scenario} menghasilkan performa terbaik berdasarkan nilai rata-rata akurasi sebesar {best_accuracy}%. Skenario tersebut memperoleh nilai rata-rata precision sebesar {best_precision}%, recall sebesar {best_recall}%, dan F1-score sebesar {best_f1}%. Hasil tersebut diperoleh dari rata-rata {REPETITIONS_PER_SCENARIO} kali pengujian, sehingga evaluasi model tidak hanya bergantung pada satu kali pembagian data.

## Artefak

- Tabel ringkasan dan detail pengujian: `{TABLE_PATH.name}`
- Detail 30 eksperimen: `{DETAIL_PATH.name}`
- Grafik rata-rata akurasi: `{FIGURE_PATH.name}`
- Grafik rata-rata seluruh metrik: `{METRIC_FIGURE_PATH.name}`

## Konfigurasi Pengujian

{config_df.to_string(index=False)}

## Standar Deviasi Metrik

{variation_df.round(2).to_string(index=False)}
"""

    DOCUMENTATION_PATH.write_text(documentation, encoding="utf-8")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset hasil preprocessing tidak ditemukan: {DATASET_PATH}"
        )

    data = pd.read_excel(DATASET_PATH)

    required_columns = {"teks", "Bidang Penelitian"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Kolom dataset tidak lengkap: {sorted(missing_columns)}")

    class_counts = data["Bidang Penelitian"].value_counts()
    valid_classes = class_counts[class_counts > 1].index
    filtered_data = data[data["Bidang Penelitian"].isin(valid_classes)].copy()

    X = filtered_data["teks"].fillna("")
    y = filtered_data["Bidang Penelitian"]

    print("Jumlah data setelah filtering:", len(filtered_data))
    print("Jumlah kelas setelah filtering:", y.nunique())
    print("Distribusi kelas:\n", y.value_counts())

    results = []
    for scenario, test_size in SPLIT_SCENARIOS.items():
        for repetition in range(1, REPETITIONS_PER_SCENARIO + 1):
            random_state = RANDOM_STATE_START + repetition - 1
            print(
                f"Pengujian {scenario} ke-{repetition}/{REPETITIONS_PER_SCENARIO} "
                f"(random_state={random_state})"
            )
            results.append(
                run_single_experiment(
                    scenario=scenario,
                    test_size=test_size,
                    repetition=repetition,
                    random_state=random_state,
                    X=X,
                    y=y,
                )
            )

    detail_df = pd.DataFrame(results)
    summary_df, variation_df = summarize_results(detail_df)

    config_df = pd.DataFrame(
        {
            "Parameter": [
                "Dataset",
                "Jumlah data awal",
                "Jumlah data setelah filtering",
                "Jumlah kelas setelah filtering",
                "Filter kelas",
                "Skenario pembagian data",
                "Pengulangan per skenario",
                "Total eksperimen",
                "Random state",
                "TF-IDF max_features",
                "TF-IDF ngram_range",
                "TF-IDF min_df",
                "SVM kernel",
                "Calibration folds",
            ],
            "Nilai": [
                str(DATASET_PATH),
                len(data),
                len(filtered_data),
                y.nunique(),
                "Kelas dengan jumlah data > 1",
                ", ".join(SPLIT_SCENARIOS.keys()),
                REPETITIONS_PER_SCENARIO,
                len(detail_df),
                f"{RANDOM_STATE_START}-{RANDOM_STATE_START + REPETITIONS_PER_SCENARIO - 1}",
                TFIDF_PARAMS["max_features"],
                str(TFIDF_PARAMS["ngram_range"]),
                TFIDF_PARAMS["min_df"],
                "linear",
                CALIBRATION_FOLDS,
            ],
        }
    )

    save_workbooks(summary_df, detail_df, variation_df, config_df)
    save_accuracy_plot(summary_df, variation_df)
    save_metric_plot(summary_df)
    save_documentation(summary_df.round(2), detail_df.round(2), variation_df, config_df)

    print("\nRingkasan rata-rata pengujian:")
    print(summary_df.round(2).to_string(index=False))
    print(f"\nTabel hasil pengujian disimpan di: {TABLE_PATH}")
    print(f"Detail 30 eksperimen disimpan di: {DETAIL_PATH}")
    print(f"Grafik akurasi disimpan di: {FIGURE_PATH}")
    print(f"Grafik metrik disimpan di: {METRIC_FIGURE_PATH}")
    print(f"Dokumentasi pengujian disimpan di: {DOCUMENTATION_PATH}")


if __name__ == "__main__":
    main()

"""
Legacy single-run implementation retained only as historical reference.
The revised implementation above is the script entry point.
# =====================================================
# PROSES PENGUJIAN
# =====================================================
for scenario, test_size in split_scenarios.items():
    print(f"🔍 Pengujian model dengan skenario {scenario}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=7000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        smooth_idf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Pipeline SVM + Oversampling
    pipeline = Pipeline([
        ("oversample", RandomOverSampler(random_state=42)),
        ("svm", SVC(kernel="linear", probability=True))
    ])

    # Kalibrasi probabilitas
    model = CalibratedClassifierCV(pipeline, cv=3)
    model.fit(X_train_vec, y_train)

    # Prediksi
    y_pred = model.predict(X_test_vec)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    results.append({
        "Skenario Pembagian Data": scenario,
        "Jumlah Data Latih": len(X_train),
        "Jumlah Data Uji": len(X_test),
        "Akurasi (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2)
    })

# =====================================================
# SIMPAN HASIL KE EXCEL
# =====================================================
df_results = pd.DataFrame(results)
df_results.to_excel(TABLE_PATH, index=False)

print(f"✅ Tabel hasil pengujian disimpan di: {TABLE_PATH}")

# =====================================================
# VISUALISASI GRAFIK AKURASI
# =====================================================
plt.figure(figsize=(8, 5))
plt.plot(
    df_results["Skenario Pembagian Data"],
    df_results["Akurasi (%)"],
    marker="o"
)

plt.title("Perbandingan Akurasi Model pada Berbagai Skenario Pembagian Data")
plt.xlabel("Skenario Pembagian Data")
plt.ylabel("Akurasi (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_PATH)
plt.close()

print(f"📊 Grafik akurasi disimpan di: {FIGURE_PATH}")
"""
