import numpy as np

from sklearn.base import (
    clone,
    BaseEstimator, TransformerMixin
)

from sklearn.feature_selection import (
    SelectFromModel, SelectKBest, SelectPercentile,
    RFE, RFECV, VarianceThreshold,
    chi2, f_classif, mutual_info_classif
)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """ Base feature selector that only removes the constant
    features from the dataset. """

    def fit(self, X, y=None):
        self.max_features = ideal_number_of_features(X)
        self.selector = VarianceThreshold().fit(X)

        return self

    def transform(self, X, y=None):
        return self.selector.transform(X)


    @classmethod
    def get_full_search_space(cls):
        sp = {}
        return sp


class UnivariateFeatureSelector(FeatureSelector):

    def __init__(self, method="anova", k="max"):
        self.method = method
        self.k = k

    def fit(self, X, y=None):
        super().fit(X, y)

        if self.method == "anova":
            method = f_classif
        elif self.method == "minfo":
            method = mutual_info_classif
        elif self.method == "chi":
            method = chi2
        else:
            raise ValueError("Non valid method", self.method)

        if type(self.k) == int:
            k = self.k
        elif type(self.k) == float:
            k = int(self.k * n_features(X))  # Percentage of features
        elif self.k == "max":
            k = self.max_features  # Maximum recommended number of features
        elif self.k == "all":
            k = "all"  # All features selected (so no feature selection)
        else:
            raise ValueError("Unknwon argument for parameter k")

        self.selector = SelectKBest(method, k=k)
        self.selector = self.selector.fit(X, y)

        return self

    def transform(self, X, y=None):
        return self.selector.transform(X)

    def get_support(self):
        return self.selector.get_support()

    @classmethod
    def get_full_search_space(cls):
        sp = {}
        sp['method'] = ['anova']
        sp['k'] = ['max', 'all']

        return sp


def n_features(X):
    return X.shape[1]

def n_data_points(X):
    return X.shape[0]

def ideal_number_of_features(X):
    return min(int(np.sqrt(X.shape[0])), X.shape[1])