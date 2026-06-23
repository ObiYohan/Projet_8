import pandas as pd
import numpy as np
import mlflow
import joblib
import tempfile
import os
from pathlib import Path
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset
import json
import sys

# Ajouter le répertoire src au PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from mlflow_config import setup_mlflow_auto

def load_model_and_preprocessors(experiment_name="home_credit_risk_training"):
    """
    Charge le modèle et les preprocessors depuis MLflow
    """
    print("⏳ Chargement du modèle depuis MLflow...")
    
    setup_mlflow_auto(experiment_name)
    
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"Expérience {experiment_name} non trouvée")
    
    # Récupérer le dernier run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("Aucun run trouvé dans l'expérience")
    
    run_id = runs[0].info.run_id
    print(f"📦 Chargement depuis le run: {run_id}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Charger les preprocessors
        preprocessors_dir = client.download_artifacts(
            run_id, 
            "preprocessors",
            tmpdir
        )
        
        imputer = joblib.load(Path(preprocessors_dir) / "imputer.pkl")
        scaler = joblib.load(Path(preprocessors_dir) / "scaler.pkl")
        feature_names = joblib.load(Path(preprocessors_dir) / "feature_names.pkl")
        
        print("✅ Preprocessors chargés")
        
        # Charger les données de référence
        data_dir = client.download_artifacts(
            run_id, 
            "data",
            tmpdir
        )
        reference_data = pd.read_csv(Path(data_dir) / "reference_data.csv")
        print(f"✅ Données de référence chargées: {reference_data.shape}")
    
    # Charger le modèle
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Récupérer le threshold
    threshold = float(runs[0].data.params.get('threshold_value', '0.5'))
    
    print(f"✅ Modèle chargé (threshold: {threshold})")
    
    return model, imputer, scaler, feature_names, reference_data, threshold, model_uri


def encode_categorical_features(data):
    """
    Encode les features catégorielles avec pd.get_dummies
    (même logique que dans le preprocessing d'entraînement)
    """
    # Identifier les colonnes catégorielles
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        return data
    
    print(f"🔤 Encodage de {len(categorical_cols)} colonnes catégorielles")
    
    # Appliquer get_dummies
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    return data_encoded


def preprocess_data(data, imputer, scaler, feature_names):
    """
    Applique le preprocessing aux données
    """
    # Encoder les features catégorielles AVANT tout le reste
    data = encode_categorical_features(data)
    
    # Ajouter les features engineered
    if 'AMT_CREDIT' in data.columns and 'AMT_INCOME_TOTAL' in data.columns:
        data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    
    if 'AMT_ANNUITY' in data.columns and 'AMT_INCOME_TOTAL' in data.columns:
        data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    
    if 'AMT_ANNUITY' in data.columns and 'AMT_CREDIT' in data.columns:
        data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
    
    if 'DAYS_EMPLOYED' in data.columns and 'DAYS_BIRTH' in data.columns:
        data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    
    # Ajouter les features manquantes (avec valeur 0 pour les dummies manquants)
    missing_features = [f for f in feature_names if f not in data.columns]
    if missing_features:
        print(f"⚠️ Ajout de {len(missing_features)} features manquantes (valeur par défaut: 0)")
        for feat in missing_features:
            data[feat] = 0
    
    # Supprimer les features en trop (qui n'étaient pas dans l'entraînement)
    extra_features = [f for f in data.columns if f not in feature_names]
    if extra_features:
        print(f"⚠️ Suppression de {len(extra_features)} features non utilisées")
        data = data.drop(columns=extra_features)
    
    # Réordonner les colonnes
    data = data[feature_names]
    
    # Vérifier qu'il n'y a plus de colonnes non-numériques
    non_numeric = data.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        raise ValueError(f"Colonnes non-numériques détectées après encodage: {non_numeric}")
    
    # Appliquer le preprocessing
    data_array = imputer.transform(data)
    data_array = scaler.transform(data_array)
    
    return data_array, data


def make_predictions(model, data_array, threshold):
    """
    Fait des prédictions avec le modèle
    """
    predictions_proba = model.predict_proba(data_array)[:, 1]
    predictions = (predictions_proba >= threshold).astype(int)
    
    return predictions, predictions_proba


def analyze_drift(reference_data, current_data):
    """
    Analyse le drift entre les données de référence et actuelles
    Retourne: (drift_report, drift_metrics_dict)
    """
    print("📊 Analyse du drift avec Evidently...")
    
    # Filtrer les colonnes non vides dans current_data
    non_empty_cols = current_data.columns[current_data.notna().any()].tolist()
    
    # Garder seulement les colonnes communes
    common_cols = [col for col in reference_data.columns if col in non_empty_cols]
    
    if len(common_cols) == 0:
        print("⚠️ Aucune colonne commune pour l'analyse de drift")
        return None, None
    
    print(f"📈 Analyse sur {len(common_cols)} features")
    
    # ✅ Créer et exécuter le rapport de drift (v0.7+)
    drift_report = Report(metrics=[DataDriftPreset()])
    
    drift_report.run(
        reference_data=reference_data[common_cols], 
        current_data=current_data[common_cols]
    )
    
    # ✅ Pour v0.7+, utiliser json() au lieu de as_dict()
    drift_metrics = {}
    
    try:
        results = drift_report.json()
        results_dict = json.loads(results)
        
        if 'metrics' in results_dict:
            for metric in results_dict['metrics']:
                if 'result' in metric:
                    result = metric['result']
                    if 'dataset_drift' in result:
                        dataset_drift = result['dataset_drift']
                        drift_share = result.get('drift_share', 0)
                        n_drifted = result.get('number_of_drifted_columns', 0)
                        n_total = result.get('number_of_columns', len(common_cols))
                        
                        drift_metrics = {
                            'dataset_drift': dataset_drift,
                            'drift_share': drift_share,
                            'n_drifted_features': n_drifted,
                            'n_total_features': n_total
                        }
                        
                        print(f"\n📊 Résultats du drift:")
                        print(f"  - Dataset drift détecté: {'⚠️ OUI' if dataset_drift else '✅ NON'}")
                        print(f"  - Features en drift: {n_drifted}/{n_total} ({drift_share*100:.1f}%)")
                        
                        if 'drift_by_columns' in result:
                            drifted_features = [
                                col for col, info in result['drift_by_columns'].items() 
                                if isinstance(info, dict) and info.get('drift_detected', False)
                            ]
                            if drifted_features:
                                print(f"\n  📋 Top features en drift:")
                                for feat in sorted(drifted_features)[:10]:
                                    print(f"    - {feat}")
                                if len(drifted_features) > 10:
                                    print(f"    ... et {len(drifted_features) - 10} autres")
    except Exception as e:
        print(f"⚠️ Impossible d'extraire les métriques détaillées: {e}")
    
    return drift_report, drift_metrics


def main(test_data_path):
    """
    Fonction principale
    """
    print("=" * 60)
    print("🚀 Prédiction et Analyse de Drift")
    print("=" * 60)
    
    # Charger le modèle et les preprocessors
    model, imputer, scaler, feature_names, reference_data, threshold, model_uri = \
        load_model_and_preprocessors()
    
    # Charger les données de test
    print(f"\n📂 Chargement des données: {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    print(f"✅ Données chargées: {test_data.shape}")
    
    # Preprocessing
    print("\n🔧 Preprocessing des données...")
    test_array, test_processed = preprocess_data(
        test_data.copy(), 
        imputer, 
        scaler, 
        feature_names
    )
    
    # Prédictions
    print("\n🎯 Génération des prédictions...")
    predictions, predictions_proba = make_predictions(model, test_array, threshold)
    
    # Afficher les résultats
    print("\n📊 Résultats des prédictions:")
    print(f"  - Nombre de prédictions: {len(predictions)}")
    print(f"  - Défauts prédits: {np.sum(predictions)} ({np.mean(predictions)*100:.2f}%)")
    print(f"  - Probabilité moyenne: {np.mean(predictions_proba):.4f}")
    print(f"  - Threshold utilisé: {threshold}")
    
    # Logger dans MLflow
    print("\n📝 Logging dans MLflow...")
    setup_mlflow_auto("home_credit_risk_inference")
    
    run_name = f"batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Lien vers le modèle de base
        mlflow.set_tag("baseline_model_uri", model_uri)
        mlflow.set_tag("test_data_path", str(test_data_path))
        
        # Métriques de prédiction
        mlflow.log_metric("num_predictions", len(predictions))
        mlflow.log_metric("num_defaults_predicted", int(np.sum(predictions)))
        mlflow.log_metric("default_rate", float(np.mean(predictions)))
        mlflow.log_metric("avg_probability", float(np.mean(predictions_proba)))
        mlflow.log_metric("threshold", threshold)
        
        # Sauvegarder les prédictions
        results_df = pd.DataFrame({
            'prediction': predictions,
            'probability': predictions_proba
        })
        
        results_path = "predictions_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path, "predictions")
        os.remove(results_path)

        if reference_data is not None:
            # Filter out columns that are empty in current data
            non_empty_cols = test_processed.columns[test_processed.notna().any()].tolist()
            
            # Keep only common columns between reference and current that are non-empty
            common_cols = [col for col in reference_data.columns if col in non_empty_cols]

            if len(common_cols) > 0:
                drift_report = Report(metrics=[DataDriftPreset()])
                mon_evaluation = drift_report.run(
                    reference_data=reference_data[common_cols], 
                    current_data=test_processed[common_cols]
                )
                    
                # Save report as HTML
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
                    temp_path = tmp.name

                mon_evaluation.save_html(temp_path)
                
                # Log drift report to MLflow
                mlflow.log_artifact(temp_path, "drift_reports")
                os.unlink(temp_path)
    
    print("\n" + "=" * 60)
    print("✅ Analyse terminée avec succès!")
    print("=" * 60)
    
    return results_df


def load_raw_reference_data():
    """
    Charge les données de référence BRUTES (avant preprocessing)
    """
    data_path = PROJECT_ROOT / "data" / "init" / "application_train.csv"
    df = pd.read_csv(data_path)
    
    # Prendre un échantillon représentatif (même taille que les scénarios)
    df_sample = df.sample(n=1000, random_state=42)
    
    return df_sample


if __name__ == "__main__":
    # Chemin par défaut vers les données de test
    DATA_DIR = PROJECT_ROOT / "data"
    test_file = DATA_DIR / "test_data.csv"
    
    # Vérifier si un chemin est fourni en argument
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
    
    if not test_file.exists():
        print(f"❌ Fichier non trouvé: {test_file}")
        print(f"Usage: python {Path(__file__).name} [chemin_vers_test_data.csv]")
        sys.exit(1)
    
    # Exécuter l'analyse
    results = main(test_file)
    
    print(f"\n📊 Aperçu des résultats:")
    print(results.head(10))