import pickle
from typing import List
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, Query
from sklearn.preprocessing import LabelEncoder

# Load ML model
model = pickle.load(open("models/randomforest.pkl", "rb"))

# ----------------------------------- MEDICINE DATA -----------------------------------
MED_CSV = "data/med_dataset/disease2med.csv"
_med_df = pd.read_csv(MED_CSV).fillna("")
_med_df.columns = [c.strip() for c in _med_df.columns]

def _norm(s: str) -> str:
    return str(s).strip().lower()

def get_medicines_for_disease(disease: str, limit: int = 5):
    d = _norm(disease)
    df = _med_df
    mask = (df["prognosis"].str.lower() == d) | (df["disease_name"].str.lower() == d)
    subset = df.loc[mask]

    meds = (
        subset["med_name"]
        .astype(str).str.strip()
        .replace("", pd.NA).dropna().unique().tolist()
    )

    url = None
    if "disease_url" in subset.columns and not subset.empty:
        urls = subset["disease_url"].astype(str).str.strip()
        urls = urls[urls != ""]
        if not urls.empty:
            url = urls.iloc[0]

    return {
        "disease": disease,
        "medicines": meds[:limit],
        "count": min(len(meds), limit),
        "disease_url": url,
    }

# ----------------------------------- SYMPTOMS DATA -----------------------------------
train_df = pd.read_csv("data/disease_prediction_dataset/Training.csv")
train_df.columns = train_df.columns.str.replace("_", " ")
all_symptoms = train_df.drop(columns=["prognosis"]).columns.tolist()

le = LabelEncoder()
le.fit(train_df["prognosis"])


app = FastAPI()                                 # FastAPI app

# Mount static frontend
app.mount("/Frontend", StaticFiles(directory="Frontend"), name="Frontend")

@app.get("/")
def home():
    return {"message": "FastAPI is alive!"}



# Request body model
class SymptomsRequest(BaseModel):
    symptoms: List[str]

@app.get("/symptoms")
def get_symptoms():
    return {"symptoms": all_symptoms}

@app.post("/predict")
def predict_disease(data: SymptomsRequest):
    selected = data.symptoms

    if len(selected) < 4:
        raise HTTPException(status_code=400, detail="Select at least 4 symptoms")

    user_input = np.zeros((1, len(all_symptoms)), dtype=int)
    for idx, s in enumerate(all_symptoms):
        if s in selected:
            user_input[0, idx] = 1

    best_class = model.predict(user_input)[0]
    disease = le.inverse_transform([best_class])[0]
    return {"Predicted disease": disease}

@app.get("/medicines/{disease}")
def medicines_endpoint(
    disease: str,
    limit: int = Query(5, ge=1, le=10, description="Max medicines to return"),
):
    try:
        result = get_medicines_for_disease(disease, limit)
        if result["count"] == 0:
            result["message"] = "No medicines found. Check disease spelling or update CSV."
        else:
            result["message"] = "Success"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
