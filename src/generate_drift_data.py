import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def load_training_data():
    """Charge les données d'entraînement"""
    data_path = PROJECT_ROOT / "data" / "init" / "application_train.csv"
    df = pd.read_csv(data_path)
    return df

def add_engineered_features(df):
    """Ajoute les features engineered (comme dans le training)"""
    df = df.copy()
    
    # Credit to income ratio
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Annuity to income ratio
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    # Credit term
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # Employment to age ratio
    if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    return df

def df_to_api_format(df):
    """Convertit un DataFrame en format API (liste d'objets avec clé 'features')"""
    records = []
    for _, row in df.iterrows():
        # Convertir les valeurs NaN en None pour JSON
        features = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        records.append({"features": features})
    return records

def create_scenario_1_income_inflation(df, sample_size=100):
    """Scénario 1: Inflation des revenus"""
    df_drift = df.sample(n=sample_size, random_state=42).copy()
    df_drift['AMT_INCOME_TOTAL'] = df_drift['AMT_INCOME_TOTAL'] * 1.5
    df_drift['AMT_CREDIT'] = df_drift['AMT_CREDIT'] * 1.2
    return add_engineered_features(df_drift)

def create_scenario_2_demographic_shift(df, sample_size=100):
    """Scénario 2: Changement démographique"""
    df_drift = df.sample(n=sample_size, random_state=43).copy()
    df_drift['DAYS_BIRTH'] = df_drift['DAYS_BIRTH'] + 3650
    df_drift['CNT_CHILDREN'] = np.clip(df_drift['CNT_CHILDREN'] + 1, 0, 10)
    return add_engineered_features(df_drift)

def create_scenario_3_economic_crisis(df, sample_size=100):
    """Scénario 3: Crise économique"""
    df_drift = df.sample(n=sample_size, random_state=44).copy()
    df_drift['AMT_INCOME_TOTAL'] = df_drift['AMT_INCOME_TOTAL'] * 0.7
    df_drift['DAYS_EMPLOYED'] = df_drift['DAYS_EMPLOYED'] * 0.5
    return add_engineered_features(df_drift)

def create_scenario_4_data_quality_issue(df, sample_size=100):
    """Scénario 4: Problème de qualité des données"""
    df_drift = df.sample(n=sample_size, random_state=45).copy()
    
    # Introduire des valeurs manquantes
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        if col in df_drift.columns:
            mask = np.random.random(len(df_drift)) < 0.3
            df_drift.loc[mask, col] = np.nan
    
    return add_engineered_features(df_drift)

def save_scenarios():
    """Génère et sauvegarde tous les scénarios"""
    print("📊 Chargement des données d'entraînement...")
    df = load_training_data()
    
    output_dir = PROJECT_ROOT / "data" / "drift_scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = {
        "scenario_1_income_inflation": create_scenario_1_income_inflation(df),
        "scenario_2_demographic_shift": create_scenario_2_demographic_shift(df),
        "scenario_3_economic_crisis": create_scenario_3_economic_crisis(df),
        "scenario_4_data_quality": create_scenario_4_data_quality_issue(df)
    }
    
    for name, df_scenario in scenarios.items():
        # Sauvegarder en CSV
        csv_path = output_dir / f"{name}.csv"
        df_scenario.to_csv(csv_path, index=False)
        print(f"✅ Sauvegardé: {csv_path}")
        
        # Sauvegarder en format API (JSON avec structure correcte)
        api_data = df_to_api_format(df_scenario)
        json_path = output_dir / f"{name}_api_format.json"
        with open(json_path, 'w') as f:
            json.dump(api_data, f, indent=2)
        print(f"✅ Sauvegardé (format API): {json_path}")
    
    print(f"\n🎉 {len(scenarios)} scénarios générés avec succès!")
    return scenarios

if __name__ == "__main__":
    save_scenarios()