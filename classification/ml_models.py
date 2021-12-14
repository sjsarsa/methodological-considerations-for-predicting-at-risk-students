"""


"""

##########
# Import #
##########

# Scikit-learn
from sklearn.linear_model import (
    RidgeClassifier,
    LogisticRegression,
    Perceptron
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


###########
# Methods #
###########
def majority_vote(self):
    return DummyClassifier(strategy="prior"), {}


def probabilistic_classifier(self):
    """ Get a probabilistic classifier. """

    return GaussianNB(), {}


def linear_model(self, model):
    """ Obtain a linear model and its search space. """

    lr_sp = {}
    if model == "logistic":
        lr = LogisticRegression(
            solver='liblinear', class_weight='balanced') # Needed to avoid error
        lr_sp['C'] = [0.001, 0.01, 0.1, 1]
    elif model == "ridge":
        lr = RidgeClassifier()

    return lr, lr_sp


def tree(self, model):
    """ Obtain a decision tree model and its search space. """

    dtc_sp = {}
    dtc_sp['splitter'] = ['best', 'random']
    dtc_sp['min_weight_fraction_leaf'] = [0.0, 0.05, 0.10]

    return DecisionTreeClassifier(class_weight='balanced'), dtc_sp


def nearest_neighbors(self):
    """ Get a nearest neighbors classifier. """

    knn = KNeighborsClassifier()
    knn_sp = {}
    knn_sp['n_neighbors'] = list(range(1, 10, 2))
    knn_sp['weights'] = ['distance']
    knn_sp['p'] = [1, 2, 3]

    return knn, knn_sp


def svc(self, model):
    """ Obtain the model and the search space of the SVC
    classifier. """

    svc_sp = {}
    if model == "linear":
        svc = LinearSVC(dual=False, class_weight='balanced')
    else:
        svc = SVC(cache_size=1000, class_weight='balanced')
        svc_sp['kernel'] = ['linear', 'poly', 'rbf']
        svc_sp['degree'] = [2, 3, 4]
        svc_sp['gamma'] = ["auto", "scale"]

    svc_sp['C'] = [0.001, 0.01, 0.1, 1.0]

    return svc, svc_sp


def neural_network(self, model):
    """ Obtain a neural network and its search space. """

    if model == "mlp":
        classifier = MLPClassifier()
        mlpc_sp = {}
        mlpc_sp['hidden_layer_sizes'] = [(100,), (100, 100)]
    elif model == "perceptron":
        classifier = Perceptron(class_weight='balanced')
        mlpc_sp = {}
        mlpc_sp["penalty"] = ["l1", "l2"]
    else:
        raise ValueError("Uknown model", model)

    return classifier, mlpc_sp


def ensemble_trees(self, model):
    """ Obtain the model and the search space of the random
    forest classifier. """

    if model == "forest":
        classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    elif model == "extra":
        classifier = ExtraTreesClassifier(n_estimators=100, class_weight='balanced')
    else:
        raise ValueError("Unkown parameter", model)

    rf_sp = {}
    rf_sp['max_features'] = ["auto", None]
    rf_sp['max_depth'] = [10, 20, None]
    rf_sp['min_samples_split'] = [2, 3, 4]
    rf_sp['min_weight_fraction_leaf'] = [0.0, 0.05]

    return classifier, rf_sp


def adaboost(self):
    """ Get an adaboost classifier. """

    ada_sp = {}
    ada_sp["n_estimators"] = [50, 100]
    ada_sp["base_estimator"] = [DecisionTreeClassifier(i) for i in [1, 5, 10]]

    return AdaBoostClassifier(), ada_sp


def voting_classifier(self):
    """ Votting classifier that combines the best model obtained. """

    logistic, logistic_sp = self.linear_model("logistic")
    svc, svc_sp = self.svc("svc")
    svc_sp['probability'] = [True]
    forest, forest_sp = self.ensemble_trees("forest")

    estimators = [("lr", logistic), ("svc", svc), ("rdf", forest)]
    classifier = VotingClassifier(estimators, voting='soft')

    cls_sp = {}
    cls_sp.update(embbed_search_space(logistic_sp, 'lr'))
    cls_sp.update(embbed_search_space(svc_sp, 'svc'))
    cls_sp.update(embbed_search_space(forest_sp, 'rdf'))

    return classifier, cls_sp


def embbed_search_space(sp, prefix):
    """ Add a prefix to each key of search space. """
    if type(sp) == dict:
        return {f'{prefix}__{k}': v for k, v in sp.items()}
    else:
        return [embbed_search_space(d, prefix) for d in sp]
