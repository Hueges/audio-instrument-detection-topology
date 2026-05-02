from fastapi import FastAPI, UploadFile, File
import os
import tempfile
from Klasifikuj import klasifikuj

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/klasifikuj")
async def klasifikuj_endpoint(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read()) # cita sadrzaj uploadovanog fajla i upisuje ga u privremeni fajl
        tmp_path = tmp.name 

    try:
        rezultati = klasifikuj(tmp_path)
    finally:
        os.unlink(tmp_path)  # obriši privremeni fajl

    return {"instrumenti": rezultati}