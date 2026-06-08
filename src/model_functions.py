import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt



def perform_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model,
    cross_val_type,
    scoring_metrics: tuple,
    groups=None,
):
    scores = cross_validate(
        model,
        X,
        y,
        cv=cross_val_type,
        return_train_score=True,
        return_estimator=True,
        scoring=scoring_metrics,
        groups=groups,
        n_jobs=-1,
    )

    scores_dict = {}
    for metric in scoring_metrics:
        scores_dict["average_train_" + metric] = np.mean(scores["train_" + metric])
        scores_dict["train_" + metric + "_std"] = np.std(scores["train_" + metric])
        scores_dict["average_test_" + metric] = np.mean(scores["test_" + metric])
        scores_dict["test_" + metric + "_std"] = np.std(scores["test_" + metric])

    model.fit(X, y)

    return scores, scores_dict, model

def calculate_business_cost(y_true, y_pred, cost_fn=10, cost_fp=1):
    """
    Calcule le coût métier basé sur les faux négatifs et faux positifs.
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions (classes 0 ou 1)
        cost_fn: Coût relatif d'un faux négatif (défaut: 10)
        cost_fp: Coût relatif d'un faux positif (défaut: 1)
    
    Returns:
        Coût total métier (score sans unité)
    """
    # Récupération des valeurs de la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calcul de la pénalité totale
    penalite_totale = (fp * cost_fp) + (fn * cost_fn)

    # Normalisation : pénalité moyenne par dossier
    total_dossiers = tn + fp + fn + tp
    score_proportionnel = penalite_totale / total_dossiers
    
    return score_proportionnel

def evaluate_and_update_scores(
    model,
    X,
    y,
    model_name="Classification Model",
    cost_fn=10,
    cost_fp=1,
    cv_scores_dict=None,
    optimize_threshold=False,
    custom_threshold=None
):
    """
    Évalue le modèle et retourne les scores.
    
    Args:
        custom_threshold: Seuil de décision personnalisé (si None, utilise 0.5 ou optimise)
        optimize_threshold: Si True, optimise le seuil (ignoré si custom_threshold est fourni)
    
    Returns:
        tuple: (results_dict, predictions_dict)
    """
    
    scores_dict = cv_scores_dict.copy() if cv_scores_dict else {}
    
    # Generate probability predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Déterminer le seuil à utiliser
    if custom_threshold is not None:
        # Utiliser le seuil fourni (priorité maximale)
        threshold = custom_threshold
        scores_dict['threshold_value'] = threshold  # Renommé pour clarté
        scores_dict['threshold_source'] = 'custom'  # Reste en string dans le dict
        scores_dict['is_custom_threshold'] = 1  # Indicateur numérique pour MLflow
        y_pred = (y_pred_proba >= threshold).astype(int)
        
    elif optimize_threshold:
        # Optimiser le seuil
        optimization_results = find_optimal_threshold(
            y, y_pred_proba, cost_fn, cost_fp
        )
        threshold = optimization_results['optimal_threshold']
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        scores_dict['threshold_value'] = threshold
        scores_dict['threshold_source'] = 'optimized'
        scores_dict['is_custom_threshold'] = 1
        scores_dict['threshold_optimization_performed'] = 1
        
    else:
        # Utiliser le seuil par défaut (0.5)
        threshold = 0.5
        y_pred = model.predict(X)
        scores_dict['threshold_value'] = threshold
        scores_dict['threshold_source'] = 'default'
        scores_dict['is_custom_threshold'] = 0
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Update scores dictionary
    scores_dict.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "manual_auc_roc": roc_auc_score(y, y_pred_proba),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
        "business_cost": calculate_business_cost(y, y_pred, cost_fn, cost_fp)
    })

    # for key, value in scores_dict.items():
    #     print(f"{key}: {value} {type(value)}")
    
    predictions = {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'threshold': threshold
    }
    
    return scores_dict, predictions


def find_optimal_threshold(y_true, y_pred_proba, cost_fn=10, cost_fp=1, thresholds=None):
    """
    Trouve le seuil optimal qui minimise le coût métier.
    
    Args:
        y_true: Vraies étiquettes
        y_pred_proba: Probabilités prédites pour la classe positive
        cost_fn: Coût relatif d'un faux négatif
        cost_fp: Coût relatif d'un faux positif
        thresholds: Liste de seuils à tester (si None, teste de 0.01 à 0.99)
    
    Returns:
        dict avec le seuil optimal et les métriques associées
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    
    costs = []
    best_threshold = 0.5
    best_cost = float('inf')
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cost = calculate_business_cost(y_true, y_pred, cost_fn, cost_fp)
        costs.append(cost)
        
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    
    return {
        'optimal_threshold': best_threshold,
        'optimal_cost': best_cost,
        'all_thresholds': thresholds,
        'all_costs': costs
    }

def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    
    return df