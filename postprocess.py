import pandas as pd
from util import get_vars_as_dict


class Cols:
    dataset = 'Dataset'
    features_up_to_round = 'Features up to round'
    round_to_predict = 'Round to predict'
    model = 'Model'
    metric = 'Metric'
    score = 'Score'
    approach = 'Approach'
    n_dropouts = 'Number of dropouts'


class Models:
    nb = 'Naive Bayes'
    mv = 'Majority Vote'
    rf = 'Random Forest'
    lr = 'Logistic Regression'


class Datasets:
    wd = 'WD'
    cs1_submissions = 'CS1-S'
    cs1_browsing = 'CS1-B'
    cs0_web = 'CS0-Web'


dataset_map = {
    'wd': Datasets.wd,
    'cs1-s': Datasets.cs1_submissions,
    'cs1-b': Datasets.cs1_browsing,
    'cs0_web': Datasets.cs0_web,
}

model_map = {
    'DummyClassifier': Models.mv,
    'LogisticRegression': Models.lr,
    'RandomForestClassifier': Models.rf,
    'GaussianNB': Models.nb
}

metric_map = {
    'acc': 'Accuracy',
    'rmse': 'RMSE',
    'f1': 'F1',
    'roc_auc': 'ROC-AUC',
    'precision': 'Precision',
    'recall': 'Recall',
    'pr_auc': 'PR-AUC'
}


def update_result_labels(df: pd.DataFrame):
    df = df.rename(columns=get_vars_as_dict(Cols))

    if Cols.dataset in df.columns:
        df[Cols.dataset] = df[Cols.dataset].apply(
            lambda x: dataset_map[x] if x in dataset_map else x)

    if Cols.metric in df.columns:
        df[Cols.metric] = df[Cols.metric].apply(
            lambda x: metric_map[x] if x in metric_map else x)

    if Cols.model in df.columns:
        df[Cols.model] = df[Cols.model].apply(
            lambda x: model_map[x] if x in model_map else x)
    return df
