from pydantic import BaseModel

class SkripsiInput(BaseModel):
    judul: str
    abstrak: str

class KlasifikasiOutput(BaseModel):
    kategori: str
    confidence: float
