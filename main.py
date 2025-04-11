from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("classifier.pkl")
scaler = joblib.load("scaler.pkl")

class PatientData(BaseModel):
    motif_admission_asthenie: int
    motif_admission_alt_fonction: int
    motif_admission_hta: int
    motif_admission_oedeme: int
    motif_admission_diabete: int
    pm_hta: int
    pm_diabete1: int
    pm_diabete2: int
    pm_cardiovasculaire: int
    pm_pathologies_virales: int
    symptome_anemie: int
    symptome_nausees: int
    symptome_hta: int
    symptome_flou_visuel: int
    symptome_asthenie: int
    symptome_vomissements: int
    symptome_insomnie: int
    symptome_perte_poids: int
    symptome_omi: int
    etat_general_admission: int
    uree: float
    creatinine: float
    anemie: int

@app.post("/predict")
def predict(data: PatientData):
    values = list(data.dict().values())
    X_scaled = scaler.transform([values])
    prediction = model.predict(X_scaled)[0]

    stade_dict = {
        1: "CKD 1", 2: "CKD 2", 3: "CKD 3a",
        4: "CKD 3b", 5: "CKD 4", 6: "CKD 5"
    }
    conseils_dict = {
        1: "⚠️ Surveillance régulière et contrôle de la tension artérielle. Éviter les médicaments néphrotoxiques.",
        2: "🥦 Maintenir une bonne hygiène de vie : alimentation pauvre en sel et protéines, éviter le tabac.",
        3: "💊 Suivi spécialisé recommandé. Adapter le traitement de l’HTA, surveiller la protéinurie.",
        4: "🩺 Préparer la prise en charge néphrologique. Discussion possible sur la dialyse à moyen terme.",
        5: "🚨 Stade terminal. Planification de la dialyse ou d'une greffe. Suivi rapproché par néphrologue.",
        6: "🏥 Stade très avancé. Prise en charge en hôpital ou dialyse. Soutien psychologique et nutritionnel."
    }

    stade_label = stade_dict.get(prediction, 'inconnu')
    conseil = conseils_dict.get(prediction, "Aucun conseil disponible pour ce stade.")

    return {
        "result": f"Le patient est au stade de l'IRC {stade_label}",
        "conseil": conseil
    }
