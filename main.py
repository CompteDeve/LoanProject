from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
import json

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle
try:
    model = joblib.load('loan_approval_model.joblib')
    print("✅ Modèle chargé avec succès")
    print(f"📊 Features attendues: {model.feature_names_in_}")
except Exception as e:
    print(f"❌ Erreur de chargement du modèle: {e}")
    model = None

class LoanData(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: float
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    credit_score: int
    cb_person_cred_hist_length: float
    previous_loan_defaults_on_file: str

@app.get("/")
def read_root():
    return {"message": "API de prédiction de crédit opérationnelle"}

@app.post("/predict")
def predict(data: LoanData):
    if model is None:
        return {"error": "Modèle non chargé", "status": 0, "probability": 0.0}
    
    try:
        print("\n" + "="*50)
        print("📥 DONNÉES REÇUES DU CLIENT:")
        print(json.dumps(data.dict(), indent=2))
        
        # 1. Convertir en DataFrame
        df = pd.DataFrame([data.dict()])
        
        # 2. Calculer les features supplémentaires (comme pendant l'entraînement)
        df['loan_percent_income'] = df['loan_amnt'] / df['person_income'] if data.person_income > 0 else 0.0
        
        # 3. ORDRE DES FEATURES (EXACT comme le modèle attend)
        # NOTE: Vérifiez l'ordre exact avec: print(model.feature_names_in_)
        # Voici un exemple basé sur vos données
        expected_features = [
            'person_age', 
            'person_gender', 
            'person_education', 
            'person_income',
            'person_emp_exp', 
            'person_home_ownership', 
            'loan_amnt', 
            'loan_intent',
            'loan_int_rate', 
            'loan_percent_income', 
            'cb_person_cred_hist_length',
            'credit_score', 
            'previous_loan_defaults_on_file'
        ]
        
        # 4. Encodage des variables catégorielles
        # Convertir en codes numériques comme pendant l'entraînement
        gender_map = {'male': 1, 'female': 0}
        education_map = {
            'High School': 0,
            'Associate': 1,
            'Bachelor': 2,
            'Master': 3,
            'Doctorate': 4
        }
        home_map = {
            'RENT': 0,
            'OWN': 1,
            'MORTGAGE': 2,
            'OTHER': 3
        }
        intent_map = {
            'PERSONAL': 0,
            'EDUCATION': 1,
            'MEDICAL': 2,
            'VENTURE': 3,
            'DEBTCONSOLIDATION': 4,
            'HOMEIMPROVEMENT': 5
        }
        default_map = {'No': 0, 'Yes': 1}
        
        # Appliquer les encodages
        df['person_gender'] = df['person_gender'].map(gender_map).fillna(0)
        df['person_education'] = df['person_education'].map(education_map).fillna(0)
        df['person_home_ownership'] = df['person_home_ownership'].map(home_map).fillna(0)
        df['loan_intent'] = df['loan_intent'].map(intent_map).fillna(0)
        df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map(default_map).fillna(0)
        
        # 5. S'assurer que toutes les colonnes sont numériques
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 6. Remplir les NaN avec 0 (ou la moyenne si vous préférez)
        df = df.fillna(0)
        
        # 7. Réordonner les colonnes dans l'ordre attendu
        df = df[expected_features]
        
        print("\n📊 DONNÉES PRÉTRAITÉES POUR LE MODÈLE:")
        print(df)
        print(f"\n📋 Types de données: {df.dtypes.tolist()}")
        print(f"🔍 Valeurs NaN: {df.isna().sum().sum()}")
        
        # 8. Faire la prédiction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        probability = float(probabilities[1])  # Probabilité de classe positive
        
        print(f"\n🎯 RÉSULTAT:")
        print(f"   Prédiction: {prediction}")
        print(f"   Probabilités: {probabilities}")
        print(f"   Probabilité (classe 1): {probability:.2%}")
        
        # 9. Préparer la réponse
        result = {
            "status": int(prediction),
            "probability": probability,
            "confidence": float(probabilities.max())  # Niveau de confiance
        }
        
        print(f"📤 Réponse envoyée: {result}")
        print("="*50 + "\n")
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Erreur: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERREUR: {error_msg}")
        return {
            "error": str(e),
            "status": 0,
            "probability": 0.0,
            "traceback": traceback.format_exc()
        }