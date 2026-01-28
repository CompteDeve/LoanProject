#I'am here
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
import json
from typing import Dict, List
import datetime

app = FastAPI(title="FinScore AI API", version="1.0.0")

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
except Exception as e:
    print(f"❌ Erreur de chargement du modèle: {e}")
    model = None

# Vérifier et ajuster les noms de features
if model is not None and hasattr(model, 'feature_names_in_'):
    # Sauvegarder les noms de features originaux
    original_features = model.feature_names_in_.tolist()
    print(f"📊 Features originales: {original_features}")
    
    # Créer une copie des noms de features comme strings Python
    feature_names = [str(feat) for feat in original_features]
    print(f"📊 Features converties en str: {feature_names}")

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

class AnalysisResult(BaseModel):
    status: int
    probability: float
    confidence: float
    risk_factors: Dict[str, float]
    recommendation: str
    timestamp: str

@app.get("/")
async def read_root():
    return {
        "service": "FinScore AI - Credit Analysis API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "/predict - POST",
            "health": "/health - GET",
            "stats": "/stats - GET"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/stats")
async def get_statistics():
    """Retourne des statistiques pour les graphiques"""
    return {
        "education_distribution": {
            "High School": 15,
            "Associate": 25,
            "Bachelor": 35,
            "Master": 20,
            "Doctorate": 5
        },
        "approval_rates": {
            "PERSONAL": 62,
            "EDUCATION": 78,
            "MEDICAL": 55,
            "VENTURE": 45,
            "DEBTCONSOLIDATION": 65,
            "HOMEIMPROVEMENT": 72
        },
        "risk_factors": [
            "credit_score",
            "loan_to_income_ratio",
            "employment_experience",
            "age",
            "interest_rate",
            "credit_history"
        ]
    }

def prepare_features_for_model(data: LoanData):
    """Prépare les features pour le modèle XGBoost"""
    
    # 1. Calculer les features supplémentaires
    loan_percent_income = data.loan_amnt / data.person_income if data.person_income > 0 else 0.0
    
    # 2. Encodage CATÉGORIEL
    gender_map = {'male': 1, 'female': 0}
    education_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
    home_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    intent_map = {
        'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2,
        'VENTURE': 3, 'DEBTCONSOLIDATION': 4, 'HOMEIMPROVEMENT': 5
    }
    default_map = {'No': 0, 'Yes': 1}
    
    # Appliquer les mappings
    person_gender_encoded = gender_map.get(data.person_gender, 0)
    person_education_encoded = education_map.get(data.person_education, 0)
    person_home_ownership_encoded = home_map.get(data.person_home_ownership, 0)
    loan_intent_encoded = intent_map.get(data.loan_intent, 0)
    previous_loan_defaults_encoded = default_map.get(data.previous_loan_defaults_on_file, 0)
    
    # 3. Créer le dictionnaire de features
    features_dict = {
        'person_age': float(data.person_age),
        'person_gender': float(person_gender_encoded),
        'person_education': float(person_education_encoded),
        'person_income': float(data.person_income),
        'person_emp_exp': float(data.person_emp_exp),
        'person_home_ownership': float(person_home_ownership_encoded),
        'loan_amnt': float(data.loan_amnt),
        'loan_intent': float(loan_intent_encoded),
        'loan_int_rate': float(data.loan_int_rate),
        'loan_percent_income': float(loan_percent_income),
        'cb_person_cred_hist_length': float(data.cb_person_cred_hist_length),
        'credit_score': float(data.credit_score),
        'previous_loan_defaults_on_file': float(previous_loan_defaults_encoded)
    }
    
    return features_dict

@app.post("/predict", response_model=AnalysisResult)
async def predict(data: LoanData):
    if model is None:
        return AnalysisResult(
            status=0,
            probability=0.0,
            confidence=0.0,
            risk_factors={},
            recommendation="Modèle non disponible",
            timestamp=datetime.datetime.now().isoformat()
        )
    
    try:
        print(f"📥 Données reçues: {data.dict()}")
        
        # Préparer les features
        features_dict = prepare_features_for_model(data)
        
        # Récupérer les features attendues par le modèle
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_.tolist()
            print(f"📊 Features attendues par le modèle: {expected_features}")
            
            # Vérifier que nous avons toutes les features nécessaires
            missing_features = set(expected_features) - set(features_dict.keys())
            if missing_features:
                print(f"⚠️ Features manquantes: {missing_features}")
                # Ajouter les features manquantes avec une valeur par défaut
                for feature in missing_features:
                    features_dict[feature] = 0.0
        else:
            expected_features = list(features_dict.keys())
        
        # Créer le DataFrame dans le bon ordre
        df = pd.DataFrame([features_dict])
        
        # Réorganiser les colonnes selon l'ordre attendu par le modèle
        if hasattr(model, 'feature_names_in_'):
            # Assurer que toutes les colonnes attendues sont présentes
            for col in model.feature_names_in_:
                if str(col) not in df.columns:
                    df[str(col)] = 0.0
            
            # Réorganiser les colonnes
            df = df[model.feature_names_in_]
        
        print(f"📈 DataFrame créé - Shape: {df.shape}")
        print(f"📈 Colonnes du DataFrame: {df.columns.tolist()}")
        print(f"📈 Types de données: {df.dtypes.to_dict()}")
        
        # S'assurer que toutes les colonnes sont numériques
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.fillna(0)
        
        # PRÉDICTION
        print("🤖 Lancement de la prédiction...")
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        probability = float(probabilities[1])
        
        print(f"✅ Prédiction terminée - Résultat: {prediction}, Probabilité: {probability}")
        
        # Calcul des facteurs de risque
        risk_factors = {
            "credit_score": min(100, (data.credit_score / 850) * 100),
            "loan_to_income_ratio": min(100, ((data.loan_amnt / data.person_income) * 100) * 3) if data.person_income > 0 else 100,
            "employment_experience": min(100, (data.person_emp_exp / 30) * 100),
            "age_risk": 100 - abs(data.person_age - 35) if 25 <= data.person_age <= 55 else 50,
            "interest_rate": min(100, (data.loan_int_rate / 25) * 100),
            "credit_history": min(100, (data.cb_person_cred_hist_length / 20) * 100)
        }
        
        # Recommandation
        if probability > 0.7:
            recommendation = "Demande fortement recommandée"
        elif probability > 0.5:
            recommendation = "Demande recommandée avec conditions"
        elif probability > 0.3:
            recommendation = "Demande à examiner manuellement"
        else:
            recommendation = "Demande non recommandée"
        
        result = AnalysisResult(
            status=int(prediction),
            probability=probability,
            confidence=float(probabilities.max()),
            risk_factors=risk_factors,
            recommendation=recommendation,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        print(f"📤 Résultat envoyé: {result.dict()}")
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ ERREUR DÉTAILLÉE: {str(e)}")
        print(f"🔍 Stack trace:\n{error_details}")
        
        return AnalysisResult(
            status=0,
            probability=0.0,
            confidence=0.0,
            risk_factors={},
            recommendation=f"Erreur de traitement: {str(e)[:100]}",
            timestamp=datetime.datetime.now().isoformat()
        )

@app.post("/batch_predict")
async def batch_predict(data: List[LoanData]):
    """Prédiction par lot pour l'historique"""
    results = []
    for item in data:
        result = await predict(item)
        results.append(result.dict())
    return {"results": results}

# Endpoint pour debug
@app.post("/debug_features")
async def debug_features(data: LoanData):
    """Endpoint pour debugger les features envoyées au modèle"""
    try:
        features_dict = prepare_features_for_model(data)
        
        return {
            "input_data": data.dict(),
            "encoded_features": features_dict,
            "model_features": model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else [],
            "missing_features": list(set(model.feature_names_in_) - set(features_dict.keys())) if hasattr(model, 'feature_names_in_') else []
        }
    except Exception as e:
        return {"error": str(e)}