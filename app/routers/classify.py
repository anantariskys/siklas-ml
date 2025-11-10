from fastapi import APIRouter, HTTPException
from app.models import SkripsiInput, KlasifikasiOutput
from app import svm_model

router = APIRouter()

@router.get("/")
def read_root():
    """
    Simple GET endpoint that returns a welcome message
    """
    return {"message": "Welcome to Skripsi Classification API"}

@router.post("/", response_model=KlasifikasiOutput)
def klasifikasi(data: SkripsiInput):
    """
    Endpoint untuk klasifikasi judul dan abstrak skripsi.
    - Input: judul dan abstrak
    - Output: kategori prediksi dan confidence score
    """
    try:
        kategori, confidence = svm_model.predict(data.judul, data.abstrak)
        return {"kategori": kategori, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
