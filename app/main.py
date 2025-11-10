from fastapi import FastAPI
from app.routers import classify
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SiKlas API",
    description="API backend untuk klasifikasi topik skripsi menggunakan SVM",
    version="1.0.0"
)
origins = [
    "http://localhost:3000",  # Next.js dev
    "http://127.0.0.1:3000",  # alternatif
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register router
app.include_router(classify.router, prefix="/classify", tags=["Klasifikasi"])

@app.get("/")
def root():
    return {"message": "Selamat datang di API SiKlas"}
