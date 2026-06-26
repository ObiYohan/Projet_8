# Credit Risk API - Guide de Démarrage

## Contexte Métier

### 🎯 Objectif Principal

Prédire le risque de défaut de paiement des clients pour une institution de crédit, afin d'optimiser les décisions d'octroi de prêts.

### 📊 Problématique Métier

#### Déséquilibre des Classes
- 92% de clients solvables (pas de défaut)
- 8% de clients en défaut
- Ce déséquilibre rend l'accuracy non pertinente comme métrique

#### Asymétrie Financière
Le coût des erreurs n'est pas symétrique :
- Faux Négatif (FN) : Accorder un prêt à un client qui fera défaut → Perte de capital (coût x10)
- Faux Positif (FP) : Refuser un prêt à un bon client → Manque à gagner (coût x1)

#### 🎯 Classes du Modèle
Le modèle prédit une classification binaire :
##### Classe 0 : Pas de Défaut ✅
- Client solvable qui remboursera son prêt
- Décision métier : Accorder le crédit
- Représente ~92% des cas dans les données d'entraînement
##### Classe 1 : Défaut de Paiement ⚠️
- Client à risque qui risque de ne pas rembourser
- Décision métier : Refuser le crédit (ou demander des garanties supplémentaires)
- Représente ~8% des cas dans les données d'entraînement


#### 💡 Solution Adoptée
##### Métrique Personnalisée : Business Cost
- Optimisation d'un coût métier qui pénalise 10x plus les FN que les FP
- Permet de définir un seuil de décision optimal aligné sur les objectifs financiers
- Minimise l'impact financier total plutôt que de maximiser l'accuracy
##### Architecture Technique
- Modèle XGBoost avec scale_pos_weight pour gérer le déséquilibre
- Monitoring du drift avec Evidently pour détecter les changements de distribution

## 🎯 Objetif du projet

Préparer la mise en production du modèle développé précédemment :
- Architecture Git
- Api fonctionnel et accessible via une interface
- Tests unitaires automatisés
- Conteneurisation du code avec Docker
- Analyse du Data Drift
- Pipeline CI/CD

## 🚀 Lancement de l'Application

### Option 1: Avec Docker
```bash
# Construire l'image Docker
docker build -t credit-risk-app .

# Lancer le conteneur
docker run --rm --name credit-risk-container -p 8000:8000 -p 7860:7860 credit-risk-app
```

### Option 2: Sans Docker

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API FastAPI
uvicorn api.api:app --host 0.0.0.0 --port 8000

# Lancer l'interface Gradio (dans un autre terminal)
python gradio_app/app.py
```

### Version hébergé

- Application : https://huggingface.co/spaces/0biyohan/Projet_8
- Stockage : https://huggingface.co/buckets/0biyohan/Projet_8-storage
- MlFlow : https://dagshub.com/ObiYohan/my-ml-mlflow.mlflow/#/experiments

## 🌐 Accès aux Interfaces

- API FastAPI: http://localhost:8000
- Documentation API: http://localhost:8000/docs
- Interface Gradio: http://localhost:7860

## 📊 Monitoring et Drift Detection

### Génération de Scénarios de Drift

```bash
# Créer des données avec drift simulé
python src/generate_drift_data.py
```
### Analyse du Drift

```bash
# Analyser un scénario spécifique
python src/predict_and_analyze_drift.py data/drift_scenarios/scenario_3_economic_crisis.csv
```

## Interprétation des Résultats

Le système détecte automatiquement le drift lors des prédictions:
- ✅ Pas de drift: Les données actuelles sont similaires aux données d'entraînement
- ⚠️ Drift détecté: Les données ont changé significativement

### Métriques clés:
- dataset_drift: Drift global détecté (True/False)
- drift_share: Pourcentage de features en drift
- n_drifted_features: Nombre de features affectées

### Rapports:
- Les rapports HTML sont générés dans MLflow sous drift_reports/
- Consultez MLflow UI pour visualiser l'évolution du drift

## 🧪 Tests

```bash
bash
# Lancer les tests avec couverture
pytest tests/ --cov=api --cov-report=html --cov-report=term-missing
```

## 📈 MLflow Tracking

Les prédictions et le drift sont automatiquement trackés dans MLflow:
- Probabilités de prédiction
- Rapports de drift
- Lien vers le modèle baseline