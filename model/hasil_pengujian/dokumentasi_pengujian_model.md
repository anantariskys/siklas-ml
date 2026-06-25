# Dokumentasi Pengujian Model

Pengujian model klasifikasi bidang penelitian dilakukan dengan tiga skenario pembagian data, yaitu 70:30, 80:20, dan 90:10. Sesuai revisi penguji, setiap skenario diulang sebanyak 10 kali dengan variasi `random_state` 42, 43, 44, 45, 46, 47, 48, 49, 50, 51. Dengan demikian, total eksperimen yang dilakukan adalah 30 eksperimen. Nilai pada tabel berikut merupakan rata-rata dari 10 kali pengujian pada masing-masing skenario.

## Tabel 5.19 Hasil Pengujian Model Klasifikasi Bidang Penelitian

| Skenario Pembagian Data | Jumlah Pengujian | Jumlah Data Latih | Jumlah Data Uji | Akurasi Rata-rata (%) | Precision Rata-rata (%) | Recall Rata-rata (%) | F1-Score Rata-rata (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 70:30 | 10 | 2156 | 925 | 86,01 | 84,90 | 86,01 | 84,79 |
| 80:20 | 10 | 2464 | 617 | 86,09 | 84,81 | 86,09 | 84,86 |
| 90:10 | 10 | 2772 | 309 | 86,60 | 85,69 | 86,60 | 85,52 |

## Gambar 5.5 Perbandingan Akurasi Model

![Gambar 5.5 Perbandingan Akurasi Model](grafik_akurasi_model.png)

Gambar 5.5 menunjukkan perbandingan rata-rata akurasi model pada setiap skenario pembagian data. Error bar pada grafik menunjukkan standar deviasi akurasi dari 10 kali pengujian.

## Ringkasan Narasi Revisi 5.4.1

Berdasarkan Tabel 5.19, skenario pembagian data 90:10 menghasilkan performa terbaik berdasarkan nilai rata-rata akurasi sebesar 86,60%. Skenario tersebut memperoleh nilai rata-rata precision sebesar 85,69%, recall sebesar 86,60%, dan F1-score sebesar 85,52%. Hasil tersebut diperoleh dari rata-rata 10 kali pengujian, sehingga evaluasi model tidak hanya bergantung pada satu kali pembagian data.

## Artefak

- Tabel ringkasan dan detail pengujian: `tabel_hasil_pengujian_model.xlsx`
- Detail 30 eksperimen: `detail_30_eksperimen_pengujian_model.xlsx`
- Grafik rata-rata akurasi: `grafik_akurasi_model.png`
- Grafik rata-rata seluruh metrik: `grafik_metrik_rata_rata_model.png`

## Konfigurasi Pengujian

                     Parameter                                                       Nilai
                       Dataset C:\Project\siklas\siklas-ml\model\preprocessed_dataset.xlsx
              Jumlah data awal                                                        3082
 Jumlah data setelah filtering                                                        3081
Jumlah kelas setelah filtering                                                          15
                  Filter kelas                                Kelas dengan jumlah data > 1
       Skenario pembagian data                                         70:30, 80:20, 90:10
      Pengulangan per skenario                                                          10
              Total eksperimen                                                          30
                  Random state                                                       42-51
           TF-IDF max_features                                                        7000
            TF-IDF ngram_range                                                      (1, 2)
                 TF-IDF min_df                                                           2
                    SVM kernel                                                      linear
             Calibration folds                                                           3

## Standar Deviasi Metrik

Skenario Pembagian Data  Std Akurasi (%)  Std Precision (%)  Std Recall (%)  Std F1-Score (%)  Min Akurasi (%)  Max Akurasi (%)
                  70:30             0.96               1.17            0.96              1.10            84.97            87.78
                  80:20             1.35               1.35            1.35              1.48            83.79            88.82
                  90:10             1.93               1.71            1.93              1.99            83.50            89.64
