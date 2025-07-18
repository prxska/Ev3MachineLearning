from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os
import gdown

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ID del archivo de Google Drive
drive_file_id = "1kWeaiVGBssKEB2AnbLyDCAvFivLmCTkf"  # <-- Reemplaza con el tuyo real
output_path = "scaler.pkl"

# Descargar scaler.pkl desde Google Drive si no existe
if not os.path.exists(output_path):
    url = f"https://drive.google.com/uc?id={drive_file_id}"
    gdown.download(url, output_path, quiet=False)

# Cargar modelo y scaler
model = joblib.load("random_forest.pkl")
scaler = joblib.load(output_path)

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("formulario.html", {"request": request})

@app.post("/predecir", response_class=HTMLResponse)
async def form_post(request: Request,
                    travelled: float = Form(...),
                    equipment: float = Form(...)):
    X = scaler.transform([[travelled, equipment]])
    pred = model.predict(X)[0]
    resultado = "GANADOR" if pred == 1 else "PERDEDOR"
    return templates.TemplateResponse("formulario.html", {
        "request": request,
        "resultado": resultado
    })
