import argparse
import numpy as np
import random

"""
Predicting Course Dropouts Early Might be Easier.

LeTech

"""

# Custom
import util
from instances.BrowsingData import BrowsingData
from instances.SubmissionData import SubmissionData
from classification.prediction import classification_task, get_metrics
from preprocess import PreprocessedDataPath
from logs.Logger import Logger


def get_datasets(selected=[]):
    instances = {
        'cs1-s': SubmissionData(PreprocessedDataPath.cs1_s, year=2020),
        'cs1-b': BrowsingData(),
        'wd': SubmissionData(PreprocessedDataPath.wd),
        'cs0_web': SubmissionData(PreprocessedDataPath.cs0_web),
    }

    if len(selected) > 0:
        instances = {i: v for i, v in instances.items() if i in selected}

    return instances


@util.Time(f'Solving for dataset')
def solve_dataset_task(name, instance, main_metric: str, logger: Logger):

    print(f'Dataset {name}')
    for filter_previous in [True, False]:
        print("Using previous dropouts: ", filter_previous)
        rounds_to_predict = range(2, min(8, instance.n_rounds + 1))

        feature_round_limits_and_target_rounds = [(features_up_to_round, round_to_predict)
                                                  for round_to_predict in rounds_to_predict
                                                  for features_up_to_round in range(1, round_to_predict)]

        @util.ForEachETA(feature_round_limits_and_target_rounds, f'Solving for each set of rounds and targets')
        def solve_rounds_to_predict(params):
            features_up_to_round, round_to_predict = params
            try:
                results = classify(instance,
                                   features_up_to_round,
                                   round_to_predict,
                                   filter_previous=filter_previous,
                                   main_metric=main_metric)
                logger.save_results(results, not filter_previous,
                                    name, features_up_to_round, round_to_predict)

            except ValueError as e:
                print('Encountered ValueError', e)

        solve_rounds_to_predict()


def solve_tasks(main_metric: str, selected, logger):
    """ Solves each task of the paper. """

    instances = get_datasets(selected)

    for name, instance in instances.items():
        solve_dataset_task(
            name, instance, main_metric=main_metric, logger=logger)


def classify(instance, features_up_to_round, round_to_predict, main_metric, filter_previous=False,):
    """ Classify which students dropped out at a given week given data subsequent data
    from previous weeks.
    Parameters
    ----------
    instances: Instance objects
        Handlers to each course instance collected data
    rounds_to_use: list (int)
        List of which rounds of the course to use
    round_to_predict: int
        Which round to predict students who dropped out
    filter_previous: bool, optional, default = False
        Wether or not to remove students who dropped out in a previous round
        for each subtask.
    """
    # print(f'classification started for feature rounds 1 to {features_up_to_round} and target round {round_to_predict}')
    # print(f'filter previous = {filter_previous}')

    features = instance.get_features(
        features_up_to_round, round_to_predict, keep_dropped=not filter_previous)
    dropouts = instance.get_dropouts(
        features_up_to_round, round_to_predict, keep_dropped=not filter_previous)

    (X, feature_names), y = features, dropouts

    # print('Features shape', X.shape, 'Targets shape', 'Percentage of dropouts', y.mean())

    _, results = classification_task((X, y), main_metric, feature_names)

    return results


def parse_args():
    instances = list(get_datasets().keys())
    parser = argparse.ArgumentParser(description='Methodological Considerations for Predicting At-risk Students',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--main-metric',
                        default='pr_auc',
                        choices=list(get_metrics().keys()),
                        help='Metric to use for selecting the best model in nested cross-validation')
    parser.add_argument('--datasets',
                        default=instances,
                        choices=instances,
                        nargs='+',
                        help="Which datasets to run")
    parser.add_argument('--results-prefix', default="logs",
                        help="optional prefix of results file name, saved in logs/results/")

    return parser.parse_args()


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    args = parse_args()
    logger = Logger(args.results_prefix)

    solve_tasks(args.main_metric, args.datasets, logger=logger)
