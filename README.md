# 🧠 SIKLAS – Machine Learning Service (FastAPI)

Layanan *Machine Learning* untuk **SIKLAS (Sistem Informasi Klasifikasi Topik Skripsi)**.  
Service ini bertugas melakukan klasifikasi topik skripsi secara otomatis menggunakan algoritma **Support Vector Machine (SVM)**, dan berkomunikasi dengan backend Laravel melalui REST API.

---

## 🚀 Tech Stack
- **Language**: Python 3.10+  
- **Framework**: FastAPI  
- **Machine Learning**: scikit-learn  
- **Text Processing**: TF-IDF Vectorizer  
- **Server**: Uvicorn  

---

## ⚙️ Fitur Utama
- Endpoint `/predict` untuk klasifikasi topik berdasarkan judul/abstrak skripsi  
- Proses training model dari dataset berlabel  
- Penyimpanan model dalam format `.pkl`  
- Komunikasi REST API dengan backend Laravel  

---

## 🧩 Instalasi & Menjalankan

```bash
# Clone repository
git clone https://github.com/username/siklas-ml.git
cd siklas-ml

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Jalankan service
uvicorn app.main:app --reload
