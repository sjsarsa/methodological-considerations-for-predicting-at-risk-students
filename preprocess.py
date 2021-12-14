import pandas as pd
import os
from pathlib import Path
from util import get_var_values


class DataPath:
    wd = 'data/wd.csv'
    cs1_s = 'data/cs1_s.csv'
    cs0_web = 'data/cs0_web.csv'


class PreprocessedDataPath:
    _preprocessed_data_dir = 'data/preprocessed'
    wd = f'{_preprocessed_data_dir}/wd.csv'
    cs1_s = f'{_preprocessed_data_dir}/cs1-s.csv'
    cs0_web = f'{_preprocessed_data_dir}/cs0_web_2021.csv'

    @classmethod
    def init(cls):
        Path(cls._preprocessed_data_dir).mkdir(parents=True, exist_ok=True)


class DataCols():
    # preprocess everything to match these columns
    round = 'round'
    subject_id = 'subject_id'
    submission_time = 'submission_time'
    course_year = 'course_year'
    assignment_id = 'assignment_id'
    solved = 'solved'
    late = 'late'


def preprocess_cs1(cs1_df):
    # rename columns
    cs1_df.rename(columns={
        'student': DataCols.subject_id,
        'week': DataCols.round,
        'submission_time': DataCols.submission_time,
        'course_year': DataCols.course_year,
        'exercise': DataCols.assignment_id,
        'points': DataCols.solved,
        'late_penalty_applied': DataCols.late,
    }, inplace=True)

    # Add round info from deadlines
    cs1_course_dfs = [x for _, x in cs1_df.groupby([DataCols.course_year])]

    def deadlines_to_rounds(df):
        deadlines = df.deadline
        deadline_cats = pd.Categorical(deadlines).codes + 1
        df[DataCols.round] = deadline_cats
        return df

    cs1_with_rounds_df = pd.concat([deadlines_to_rounds(df)
                                   for df in cs1_course_dfs])
    Path(os.path.dirname(PreprocessedDataPath.cs1_s))
    cs1_with_rounds_df.to_csv(PreprocessedDataPath.cs1_s, index=False)
    print(f'wrote {PreprocessedDataPath.cs1_s}')


def preprocess_wd(wd_df: pd.DataFrame):
    # rename columns
    wd_df.rename(columns={
        'user_id': DataCols.subject_id,
        'week': DataCols.round,
        'created_at': DataCols.submission_time,
        'assignment_id': DataCols.assignment_id,
        'correct': DataCols.solved,
    }, inplace=True)
    wd_df.to_csv(PreprocessedDataPath.wd, index=False)
    print(f'wrote {PreprocessedDataPath.wd}')


def preprocess_cs0_web(cs0_web_df: pd.DataFrame):
    # rename columns
    cs0_web_df.rename(columns={
        'user': DataCols.subject_id,
        ' round': DataCols.round,
        ' received': DataCols.submission_time,
        ' exerciseId': DataCols.assignment_id,
        ' late': DataCols.late,
    }, inplace=True)

    cs0_web_df[DataCols.solved] = cs0_web_df[' points'] / cs0_web_df[' max_points']
    cs0_web_df.to_csv(PreprocessedDataPath.cs0_web, index=False)
    print(f'wrote {PreprocessedDataPath.cs0_web}')


preprocessing_funcs = {
    DataPath.cs1_s: preprocess_cs1,
    DataPath.wd: preprocess_wd,
    DataPath.cs0_web: preprocess_cs0_web
}


def preprocess_data(datapath, data):
    PreprocessedDataPath.init()
    if datapath not in preprocessing_funcs:
        raise NotImplementedError(
            f"Preprocessing for data in {datapath} has not been implemented")
    preprocessing_funcs[datapath](data)


if __name__ == "__main__":
    datapaths = get_var_values(DataPath)

    dfs = [pd.read_csv(datapath) for datapath in datapaths]

    for datapath, df in zip(datapaths, dfs):
        preprocess_data(datapath, df)
