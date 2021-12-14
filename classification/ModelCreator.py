"""
Allows the retrieval and the creation of classification models

"""

###########
# Imports #
###########

# Sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from .ml_models import embbed_search_space
from .feature_selection import UnivariateFeatureSelector


#########
# Class #
#########

class ModelCreator():
    """ Allows the creation and the retrieveal of
    Scikit learn classifiers and their search spaces. """

    from .ml_models import (
        linear_model, ensemble_trees, probabilistic_classifier, majority_vote
    )

    def get_all_models(self):
        pairs = [
            self.linear_model("logistic"),
            self.ensemble_trees("forest"),
            self.probabilistic_classifier(),
            self.majority_vote()
        ]

        return [self.prepare_output(model, model_sp)
                for model, model_sp in pairs]

    def prepare_output(self, classifier, classifier_sp):
        """ Obtain a Pipeline of transformers to apply
        for a classification problem.

        Parameters
        -----------
        classifier: Sklearn classifier
            The model to fit on the data.
        classifier_sp: dict
            Search space of the classifier.

        Returns
        -------
        pip: Sklearn Pipeline
            Pipeline with several transformers (depending
            on the class arguments, and the given final
            classifier.
        search_space: dict
            The search space of all transformers in the
            output pipeline.

        """

        steps, search_space = [], {}

        # Standardization
        steps.append(("std", StandardScaler()))
        # Feature selection
        steps.append(('vts', VarianceThreshold()))
        steps.append(('fsl', UnivariateFeatureSelector()))
        # Classification step
        steps.append(('cls', classifier))
        
        uni_fsl_sp = UnivariateFeatureSelector.get_full_search_space()
        search_space.update(embbed_search_space(classifier_sp, 'cls'))
        search_space.update(embbed_search_space(uni_fsl_sp, 'fsl'))

        return Pipeline(steps), search_space
