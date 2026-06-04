
import pandas as pd
from mlflow_config import setup_mlflow
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import mlflow
import shap

# Configuration MLflow
setup_mlflow()

import mlflow_call
import model_functions as mf


# Définir les chemins
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"

# Charger les données
app_train = pd.read_csv(DATA_DIR / 'application_train_processed.csv')
# app_test = pd.read_csv(DATA_DIR / 'application_test_processed.csv')


# add features engeneering

app_train['CREDIT_INCOME_PERCENT'] = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
app_train['ANNUITY_INCOME_PERCENT'] = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
app_train['CREDIT_TERM'] = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']
app_train['DAYS_EMPLOYED_PERCENT'] = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']


train_labels = app_train['TARGET']

# Drop the target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()
    
# Feature names
features = list(train.columns)


# Median imputation of missing values
imputer = SimpleImputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
# test = imputer.transform(app_test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
# test = scaler.transform(test)

print('Training data shape: ', train.shape)
# print('Testing data shape: ', test.shape)


random_state = 42

X = train
y = train_labels.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=random_state, stratify=y
)

print(f"Taille entraînement : {X_train.shape[0]} lignes")
print(f"Taille test : {X_test.shape[0]} lignes")

# ## Rappel du contexte métier
# 
# Dans le projet Home Credit Default Risk, le fort déséquilibre des classes (92% / 8%) rend l'accuracy non pertinente. Bien que l'AUC aide à évaluer la puissance de classement du modèle, elle ne reflète pas l'asymétrie financière du métier.
# 
# Pour aligner le modèle sur les objectifs de la banque, nous optimisons un Business Cost personnalisé qui pénalise 10 fois plus un Faux Négatif (perte de capital) qu'un Faux Positif (manque à gagner). Cette métrique nous permet de définir le seuil de décision optimal pour minimiser l'impact financier total.


# ## XGBClassifier


from xgboost import XGBClassifier

import mlflow

mlflow.start_run(run_name="XGBClassifier_run", 
                description="Test on run duration")

## Analyse du déséquilibre
n_positives = np.sum(y == 1)
n_negatives = np.sum(y == 0)
imbalance_ratio = n_negatives / n_positives

print(f"Classes positives: {n_positives:,}")
print(f"Classes négatives: {n_negatives:,}")
print(f"Ratio de déséquilibre: {imbalance_ratio:.2f}:1")
print(f"scale_pos_weight recommandé: {imbalance_ratio:.2f}")

scale_pos_weight_optimal = n_negatives / n_positives

# Créer et entraîner le modèle XGBoost
xgb_model = XGBClassifier(
    random_state=random_state,
    scale_pos_weight=scale_pos_weight_optimal,
    n_estimators=300,
    max_depth=4,
    min_child_weight=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    eval_metric='auc',
    tree_method='hist',
    n_jobs=-1
)

# Cross-validation
classification_scoring_metrics = ["recall", "precision", "f1", "roc_auc"]

scores, scores_dict, xgb_model = mf.perform_cross_validation(
    X_train,
    y_train,
    model=xgb_model,
    cross_val_type=StratifiedKFold(),
    scoring_metrics=classification_scoring_metrics,
)

# Évaluation avec la fonction réutilisable
scores_dict_xgb, predictions_xgb = mf.evaluate_and_update_scores(
    model=xgb_model,
    X=X_test,
    y=y_test,
    model_name="xgboost Classifier",
    cost_fn=10,
    cost_fp=1,
    cv_scores_dict=scores_dict
)

# Extract feature importances
# Exclure explicitement TARGET de app_train
domain_features_names = [col for col in app_train.columns if col != 'TARGET']

features_importances_values = xgb_model.feature_importances_
feature_importances = pd.DataFrame({'feature': domain_features_names, 'importance': features_importances_values})

# Show the feature importances for the default features
feature_importances_sorted = mf.plot_feature_importances(feature_importances)

# SHAP
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values on test set (use a sample if dataset is too large)
sample_size = min(1000, X_test.shape[0])
X_test_sample = X_test[:sample_size]
shap_values = explainer.shap_values(X_test_sample)

# Summary plot (bar chart of mean absolute SHAP values)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=domain_features_names, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - XGBoost")
plt.tight_layout()
mlflow.log_figure(plt.gcf(), "shap_feature_importance_bar.png")

# Summary plot (beeswarm plot showing feature impact distribution)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=domain_features_names, show=False)
plt.title("SHAP Feature Impact Distribution - XGBoost")
plt.tight_layout()
mlflow.log_figure(plt.gcf(), "shap_summary_plot.png")

# Create SHAP feature importance dataframe
shap_importance = pd.DataFrame({
    'feature': domain_features_names,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

print("\nTop 15 features by SHAP importance:")
print(shap_importance.head(15))

# Log SHAP importance as artifact
shap_importance.to_csv("shap_feature_importance.csv", index=False)
mlflow.log_artifact("shap_feature_importance.csv")
# Delete local csv
os.remove("shap_feature_importance.csv")

# Log dans MLflow
mlflow_call.call_mlflow_start_run(
    app_train,
    scores_dict_xgb, 
    xgb_model,
    predictions_xgb['confusion_matrix'],
    model_name="xgboost_Classifier",
    description="xgboost baseline model"
)

mlflow.end_run() 