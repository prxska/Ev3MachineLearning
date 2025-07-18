from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os
import gdown

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Diccionario con archivos a descargar desde Google Drive
drive_files = {
    "scaler.pkl": "1kWeaiVGBssKEB2AnbLyDCAvFivLmCTkf",           # <-- tu ID real del scaler
    "random_forest.pkl": "1HQiSKqTGVTILN1U2z3nbD9a5StTCffx7"      # <-- tu ID real del modelo
}

# Descargar archivos si no existen
for filename, file_id in drive_files.items():
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)

# Cargar modelo y scaler
model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")

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
