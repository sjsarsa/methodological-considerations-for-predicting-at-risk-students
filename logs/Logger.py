"""
Class used to save the results of the classification pipeline.

"""

###########
# Imports #
###########

from datetime import datetime
from pathlib import Path
import os
import pandas as pd

#########
# Class #
#########

class Logger():
    """ Class used to save the results of the machine learning pipelines.

    Instances of this class can be used to save in a dataframe format, in a csv file
    for each classification task, the results obtained by the models.

    """

    def __init__(self, prefix='log'):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
        self.savepath = f"logs/results/{prefix}_{dt_string}"
        self.logs = []

    def save_results(self, results, keep_dropped, dataset, features_up_to_round, round_to_predict):
        """ Save the results obtained by models trained on a given dataset,
        using given features. """
        Path(os.path.dirname(self.savepath)).mkdir(parents=True, exist_ok=True)

        for result in results:
            for metric, value in result["scores"].items():
                log = { "dataset"                   : dataset,
                        "keep_dropped"              : keep_dropped,
                        "features_up_to_round"      : features_up_to_round,
                        "round_to_predict"          : round_to_predict,
                        "model"                     : result["model"],
                        "metric"                    : metric,
                        "score"                     : value,
                        "feature_importance"        : result["feature_importance"],
                        "best_parameters"           : result["best_params"],
                        "details"                   : result["details"]}

                self.logs.append(log)

        # Save for each task (i.e. round predicted) the result
        # Since the logs are appended, the results are conserved
        self.commit()

    def commit(self):
        dataframe = pd.DataFrame(self.logs)
        # dataframe.to_csv(self.savepath)
        dataframe.to_pickle(self.savepath)

        return dataframe