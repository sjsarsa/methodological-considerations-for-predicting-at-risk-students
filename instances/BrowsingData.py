"""


"""

import os 
import pickle
import numpy as np
import pandas as pd 

from instances.SubmissionData import SubmissionData




class BrowsingData():

    def __init__(self, aggregated=False):
        self.aggregated = aggregated
        self.activity = pd.read_pickle("data/browsing/activity.pkl")
        self.week_to_dropouts = get_week_to_dropouts(self.activity)
        self.round_to_dataframe = {}
        
        self.n_rounds = 8
        self.total_student_count = len(self.activity.index)

    def load_dataframe(self, last_feature_round):
        """ Load the dataframe containing features information until
        the given round. 

        Parameters
        ----------
        last_feature_round: int
            Round until which to obtain features.
        
        Returns
        -------
        dataframe: pandas Dataframe
            Dataframe containing features until the given round.
            The features are either aggregated for all rounds
            or computed individually for each round,
            depending on the instance object "aggregated" variable.
        """

        weeks = list(range(1, last_feature_round + 1))
        
        if self.aggregated:
            dataframe = get_dataframe(get_filename(weeks, self.aggregated))
        else:
            dataframe = []
            for week in weeks:
                df = get_dataframe(get_filename([week], self.aggregated))
                df.columns = [f"{week}_{c}" for c in df.columns]
                dataframe.append(df)
            dataframe = pd.concat(dataframe, axis=1)

        return dataframe
    

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
            shape: n_students, n_features
        """

        df = self.load_dataframe(last_feature_round)
        
        if not keep_dropped:

            previous_dropouts = []
            for week in list(range(1, last_feature_round + 1)):
                previous_dropouts.extend(self.week_to_dropouts[week])
            
            df = df.drop(list(set(previous_dropouts).intersection(df.index)))

        self.round_to_dataframe[last_feature_round] = df

        return np.nan_to_num(df.to_numpy(), posinf=0, neginf=0), df.columns


    def get_dropouts(self, last_feature_round: int, round_to_predict: int, keep_dropped=True):
        """ Obtain the identifiers of which students dropped out at/by
        a given week.
        Parameters
        ----------
        round_to_predict: int
            Which round we want to predict the dropout.
        round_to_predict: int
            Which round we want to predict the dropout.
        Returns
        -------
        dropouts: list
            List of the identifiers of the dropouts """

        # We are using the data from week 1 to last_feature_round (included) 
        # to detect which students are AT RISK up till round 
        # round_to_predict; The difference is that in the traditional approach
        # the population is composed of all students, while in the proposed
        # approach the population is composed of only students who were
        # active (did not dropout) during the data week. 

        start = last_feature_round + 1 if not keep_dropped else 1
        weeks = list(range(start, round_to_predict + 1))
        dropouts = [self.week_to_dropouts[week] for week in weeks]
        dropouts = [i for l in dropouts for i in l]

        # We need the features ndarray to know the indices of the
        # students who are part of the dataset
        if last_feature_round not in self.round_to_dataframe:
            self.get_features(last_feature_round, 
                              keep_dropped=keep_dropped)
        df = self.round_to_dataframe[last_feature_round]

        return np.isin(df.index, list(set(dropouts)))
    
    ###########
    # Getters #
    ###########

    def get_active_on_previous(self, round):
        assert round in list(range(1, self.n_rounds + 1)) 

        rounds = list(range(1, round))
        mask = self.activity[rounds].any(axis=1)
        return self.activity[mask].index
        
    def get_active_on_current(self, round):
        assert round in list(range(1, self.n_rounds + 1)) 

        mask = self.activity[round]
        return self.activity[mask].index

    def get_active_on_current_or_future(self, round):
        assert round in list(range(1, self.n_rounds + 1))  

        rounds = list(range(round, self.n_rounds + 1))
        mask = self.activity[rounds].any(axis=1)
        return self.activity[mask].index 

    def get_active_on_immediate_previous(self, round):
        assert round in list(range(1, self.n_rounds + 1)) 

        if round <= 1:
            return []
        mask = self.activity[round - 1]
        return self.activity[mask].index 
    
    def get_active_on_previous_but_not_immediate_previous(self, round):
        assert round in list(range(1, self.n_rounds + 1)) 

        active_on_previous = self.get_active_on_previous(round)
        active_on_immediate_previous = self.get_active_on_immediate_previous(round)
        return active_on_previous[~np.isin(active_on_previous, active_on_immediate_previous)]

    def get_features_of_each_round(self):
        """ Only logic if individual metrics. """
        features = self.load_dataframe(self.n_rounds).columns
        extract_round = lambda feature: int(feature[:feature.index('_')])
        return {f:extract_round(f) for f in features}

#####################
# Utility functions # 
#####################

def get_filename(weeks, aggregated):
    """ Obtain the name of the file containing data for the given weeks. """

    w = '_'.join([str(week) for week in weeks])
    prefix = "aggregated" if aggregated else "individual"
    suffix = f"weeks_{w}" if aggregated else f"week_{w}"
    return f"{prefix}_metrics_{suffix}"

def get_dataframe(filename):
    """ Obtain the dataframe stored under the given name. """

    filepath = os.path.join("data/browsing/", filename)
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
    else: # Fetch the dataframe from Charles's triton workspace
        path = "/scratch/work/koutchc1/thesis_new/SIGCSE"
        df = pd.read_pickle(os.path.join(path, filename))
        df.to_pickle(filepath)

    return df 

def when_did_student_drop(activity, weeks):
    """ Determine when did a student drop out of the course
    given the student activity. """

    if activity[-1] == True:
        return -1
    else:
        activity = list(activity)
        activity.reverse()
        return weeks[len(activity) - activity.index(True)]
    
def get_week_to_dropouts(activity):
    """ Obtain a dictionary mapping each student to when the student
    dropped out. """

    d = {uid: when_did_student_drop(a, activity.columns) 
            for uid, a in zip(activity.index, list(activity.to_numpy()))}
    return {w: [uid for uid in d if d[uid] == w]
            for w in activity.columns}    
    

def flatten(ll):
    if ll:
        return [i for l in ll for i in l]
    return []