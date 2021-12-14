"""
LeTech - Predicting Course Dropouts Early Might be Easier

This module is responsible for making the prediction tasks.

"""
###########
# Imports #
###########

import math

import numpy as np
import pandas as pd

from sklearn.metrics import (
    auc,
    precision_recall_curve,
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    make_scorer,
    fbeta_score,
    get_scorer,
    precision_score,
    recall_score,
    mean_squared_error,
    balanced_accuracy_score
)

from sklearn.model_selection import (
    KFold, StratifiedKFold, GridSearchCV, cross_validate
)
from sklearn.inspection import permutation_importance

# Custom
import util
from .ModelCreator import ModelCreator

import warnings
warnings.filterwarnings("ignore")

#############
# Functions #
#############


def classification_task(dataset, main_metric, feature_names):
    """ Classify dropouts and non dropouts.

    dataset: tuple
        Dataset of input-output X, y
    """

    # Metrics for evaluation
    metrics = get_metrics()
    # Models used to predict
    models = ModelCreator().get_all_models()
    # Training and evaluation procedure
    scores, details = evaluate(dataset, models, metrics, main_metric, feature_names)

    return scores, details


def get_metrics():
    """ Obtain a dictionary of metrics that can be used by sklearn grid-search. """

    metrics = {}

    def macro_f1(y_true, y_pred):
        return (f1_score(y_true, y_pred) + f1_score(1 - y_true, 1 - y_pred)) / 2

    def pr_auc(y_true, y_pred):
        fpr, tpr, thresholds = precision_recall_curve(y_true, y_pred, pos_label=1)
        return auc(tpr, fpr)

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    metrics.update({
        'pr_auc': make_scorer(pr_auc, needs_proba=True),
        'roc_auc': get_scorer('roc_auc'),
        'f1': make_scorer(fbeta_score, beta=1),
        'f2': make_scorer(fbeta_score, beta=2),
        'macro_f1': make_scorer(macro_f1),
        'recall': make_scorer(recall_score, zero_division=1),
        'precision': make_scorer(precision_score, zero_division=1),
        # 'balanced_accuracy': make_scorer(balanced_accuracy_score),
        # 'mcc': make_scorer(matthews_corrcoef),
        'acc': make_scorer(accuracy_score),
        'rmse': make_scorer(rmse, needs_proba=True)
    })

    other_metrics = []
    metrics.update({m: get_scorer(m) for m in other_metrics})

    return metrics


# @util.Time('evaluating models')
def evaluate(dataset, models, scoring, main_metric, feature_names):
    """ Train and evaluate the performance of the models using
    a nested cross validation strategy. """

    table, all_results = [], []
    for model, sp in models:

        model_name = type(model[-1]).__name__

        # print("Evaluating model", model_name)
        results = nested_cross_validation(dataset, model, sp, scoring, main_metric)

        scores = [np.mean(results[f'test_{score}']) for score in scoring]
        scores = [round(score, 3) for score in scores]
        table.append(scores)
        scores = {score: value for score, value in zip(scoring, scores)}

        best_params, best_estimator = find_final_model(dataset, model, sp, scoring, main_metric)
        if model_name == "RandomForestClassifier":
            feature_importance = random_forest_feature_importance(best_estimator, feature_names)
        elif model_name == "LogisticRegression":
            feature_importance = logistic_regression_feature_importance(best_estimator, feature_names)
        else:
            feature_importance = None

        all_results.append({
            "model": model_name,
            "search_space": sp,
            "scores": scores,
            "feature_importance": feature_importance,
            "best_params": best_params,
            "details": results})

    return table, all_results


def nested_cross_validation(dataset, model, search_space, scoring, main_metric):
    """ Evaluates the generalisation performance of a model on a dataset.

    Parameters
    ----------
    dataset:  tuple
        X, y (input - output)
    model:
        scikit-learn model (or pipeline)
    search_space: dict
        the model search space
    scoring: dict
        Metrics on which to evaluate the model
    """

    # Sources
    # https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/

    X, y = dataset
    inner_n_splits, outer_n_splits, n_dropouts  = 3, 5, y.sum()
    # if n_dropouts > (inner_n_splits * outer_n_splits):
    cv_inner = StratifiedKFold(n_splits=inner_n_splits)
    search = GridSearchCV(model, search_space, scoring=scoring,
                        cv=cv_inner, refit=main_metric,
                        n_jobs=-1, error_score=0)
    cv_outer = StratifiedKFold(n_splits=outer_n_splits)
    results = cross_validate(search, X, y, scoring=scoring,
                            cv=cv_outer, n_jobs=-1,
                            error_score=0, verbose=0)
    # else:
    #     cv_outer = StratifiedKFold(n_splits=n_dropouts)
    #     results = cross_validate(model, X, y, scoring=scoring,
    #                             cv=cv_outer, n_jobs=-1,
    #                             error_score=0, verbose=0)

    return results


def find_final_model(dataset, model, search_space, scoring, main_metric):
    """ Find the configuration of the final model. """

    X, y = dataset
    n_dropouts = y.sum()
    #cv = StratifiedKFold(n_splits=min(5, n_dropouts))
    cv = StratifiedKFold(n_splits=5)
    search = GridSearchCV(model, search_space, scoring=scoring,
                          cv=cv, refit=main_metric,
                          n_jobs=-1, error_score='raise')
    search = search.fit(X, y)

    return search.best_params_, search.best_estimator_


def random_forest_feature_importance(pipeline, feature_names):
    """ """

    feature_names = filter_non_selected_features(pipeline, feature_names)
    forest = pipeline.named_steps['cls']
    feature_importance = forest.feature_importances_
    data = {"importance": feature_importance,
            "name"      : feature_names}

    return pd.DataFrame(data).sort_values(by="importance")


def logistic_regression_feature_importance(pipeline, feature_names):
    """ """

    # Source: https://sefiks.com/2021/01/06/feature-importance-in-logistic-regression/

    feature_names = filter_non_selected_features(pipeline, feature_names)
    logistic = pipeline.named_steps['cls']
    weights = logistic.coef_[0]
    feature_importance = pow(math.e, weights)
    data = {"importance": feature_importance,
            "name"      : feature_names}

    return pd.DataFrame(data).sort_values(by="importance")


def filter_non_selected_features(pipeline, feature_names):

    feature_names = np.array(feature_names)
    feature_names = feature_names[pipeline.named_steps['vts'].get_support()]
    feature_names = feature_names[pipeline.named_steps['fsl'].get_support()]

    return feature_names
