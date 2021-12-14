import pandas as pd
import numpy as np
from preprocess import DataCols

"""
"""


class SubmissionData(object):

    def __init__(self, datapath, year=None, ignore_late_submissions=True):
        """
        Parameters
        ----------
        datapath: string
        """
        self.data = pd.read_csv(datapath)  # Preprocessed csv with one row per submission
        self.has_late_submissions = DataCols.late in self.data.columns
        self.ignore_late_submissions = ignore_late_submissions

        if ignore_late_submissions and self.has_late_submissions:
            self.data = self.data[self.data[DataCols.late] == False]

        if year:
            self.data = self.data.query(f'course_year == {year}')

        # Main information
        self.n_rounds = self.data[DataCols.round].max()
        self.total_student_count = len(self.data[DataCols.subject_id].unique())
        self.n_assignments = len(self.data[DataCols.assignment_id].unique())

    def _get_aggregate_features_per_student(self, features):
        feature_cols = [DataCols.subject_id, DataCols.assignment_id, DataCols.solved, DataCols.submission_time]
        aggregations = {
            DataCols.submission_time: "count",
            DataCols.solved: "max"
        }

        if not self.ignore_late_submissions and self.has_late_submissions:
            feature_cols.append(DataCols.late)
            aggregations[DataCols.late] = "count"

        feature_rounds = features[feature_cols]

        student_assignment_groups = feature_rounds.groupby([DataCols.subject_id, DataCols.assignment_id])
        agg_over_assignments = student_assignment_groups.agg(aggregations)
        return agg_over_assignments.unstack().fillna(0).astype('int32')

    def get_features(self, last_feature_round: int, round_to_predict: int, keep_dropped=True):
        """ Obtain a dataframe of features for all students
        for all given rounds.
        Parameters
        ----------
        last_round: int
            round up to which features will be selected from
        Returns
        -------
        features: 2d numpy array
            row per student
            shape: (n_students, n_features)
            ordered by student id
        """
        feature_rounds: pd.DataFrame = self.data.query(f'round <= {last_feature_round}')
        if not keep_dropped:
            active_on_last_round_or_future = self.data.query(f'round >= {last_feature_round}')[
                DataCols.subject_id].unique()
            mask = feature_rounds[DataCols.subject_id].isin(
                active_on_last_round_or_future)
            feature_rounds = feature_rounds[mask]

        feature_rounds = feature_rounds.sort_values([DataCols.subject_id, DataCols.submission_time])
        agg_features_per_student = self._get_aggregate_features_per_student(feature_rounds)
        features = agg_features_per_student.to_numpy()

        return features, agg_features_per_student.columns

    def get_dropouts(self, last_feature_round: int, round_to_predict: int, keep_dropped=True):
        """ Obtain the identifiers of which students dropped out at/by
        a given week.
        Parameters
        ----------
        last_feature_round: int
            The last round included in training features
        round_to_predict: int
            Which round we want to predict the dropout.
        keep_dropped: boolean
            Whether to include dropouts prior to target round in dropout targets
        Returns
        -------
        dropouts: 1d numpy array
                    boolean per student where true indicates dropping out
                    shape: (n_students)
                    ordered by student id
            """
        assert round_to_predict <= self.n_rounds, \
            f"The data has {self.n_rounds} rounds, cannot get dropouts for round {round_to_predict}"
        assert last_feature_round < round_to_predict, \
            f"last feature round {last_feature_round} should be smaller than prediction round {round_to_predict}"

        remaining_rounds = self.data.query(f'round >= {round_to_predict}')
        feature_rounds = self.data.query(f'round <= {last_feature_round}')

        potential_dropouts = feature_rounds[DataCols.subject_id].unique()
        if not keep_dropped:
            active_on_last_round_or_future = self.data.query(
                f'round >= {last_feature_round}')[DataCols.subject_id].unique()
            potential_dropouts = np.intersect1d(potential_dropouts, active_on_last_round_or_future)

        remaining_students = remaining_rounds[DataCols.subject_id].unique()
        dropouts = np.setdiff1d(potential_dropouts, remaining_students)
        potential_dropouts.sort()  # Ensure student order is same as in features
        return np.isin(potential_dropouts, dropouts)

    # Easy accessors

    def get_active_on_previous(self, round):
        assert round in list(range(1, self.n_rounds + 1))

        df = self.data.query(f'{DataCols.round} < {round}')
        return df[DataCols.subject_id].unique()

    def get_active_on_current(self, round):
        assert round in list(range(1, self.n_rounds + 1))

        df = self.data.query(f'{DataCols.round} == {round}')
        return df[DataCols.subject_id].unique()

    def get_active_on_current_or_future(self, round):
        assert round in list(range(1, self.n_rounds + 1))

        df = self.data.query(f'round >= {round}')
        return df[DataCols.subject_id].unique()

    def get_active_on_immediate_previous(self, round):
        assert round in list(range(1, self.n_rounds + 1))

        df = self.data.query(f'{DataCols.round} == {round - 1}')
        return df[DataCols.subject_id].unique()

    def get_active_on_previous_but_not_immediate_previous(self, round):
        assert round in list(range(1, self.n_rounds + 1))

        active_on_previous = self.get_active_on_previous(round)
        active_on_immediate_previous = self.get_active_on_immediate_previous(round)
        return active_on_previous[~np.isin(active_on_previous, active_on_immediate_previous)]

    def get_features_of_each_round(self):
        d = self.data.groupby([DataCols.round])[DataCols.assignment_id].unique()
        return {feature: r for r, features in d.items() for feature in features}