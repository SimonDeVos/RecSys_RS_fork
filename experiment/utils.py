import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def df_to_piv(df, df_train_val, df_train, df_val, df_test):
    """
    :param df:
    :param df_train_val:
    :param df_train:
    :param df_val:
    :param df_test:
    :return: piv, piv_train_val,piv_train,piv_val,piv_test
    """
    # get selected indices from subselections
    index_arr_train_val = df_train_val.index.values.tolist()
    index_arr_train = df_train.index.values.tolist()
    index_arr_val = df_val.index.values.tolist()
    index_arr_test = df_test.index.values.tolist()

    # initialize subsamples as df/df_train_val, then y-value of non-selected observations to NaN
    # 1. train_val is df without test
    df_train_val = df.copy()
    df_train_val.loc[df_train_val.index.isin(index_arr_test), 'objective'] = np.nan
    # 2. test is df without train_val
    df_test = df.copy()
    df_test.loc[df_test.index.isin(index_arr_train_val), 'objective'] = np.nan
    # 3. train is train_val without val
    df_train = df_train_val.copy()
    df_train.loc[df_train.index.isin(index_arr_val), 'objective'] = np.nan
    # 4. val is train_val without train
    df_val = df_train_val.copy()
    df_val.loc[df_val.index.isin(index_arr_train), 'objective'] = np.nan

    values_log = 'objective'
    index_log = ['case:concept:name']
    columns_log = ['concept:name']

    # convert dataframes into pivot tables
    piv = pd.pivot_table(df, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_train_val = pd.pivot_table(df_train_val, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_train = pd.pivot_table(df_train, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_val = pd.pivot_table(df_val, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_test = pd.pivot_table(df_test, values=values_log, index=index_log, columns=columns_log, dropna=False)

    return piv, piv_train_val, piv_train, piv_val, piv_test


def func_one_in_train_val(df, df_train_val, df_test):
    """
    :param df: dataframe that is used as input
    :param df_train_val: subset of df that contains train+validation samples
    :param df_test: subset of df that contains test samples
    :return:
    """
    # make sure all instances have at least one observation in train_val (otherwise no info to make prediction)
    df = df.sort_values(by=['time_start'])
    empl_train_val = df_train_val['case:concept:name'].unique()
    to_add = []
    rows_to_drop = []

    # run through elements of test set
    for i in range(len(df_test)):

        # if this employee has no observations in train_val:
        if df_test.iloc[i]['case:concept:name'] not in empl_train_val:

            # save row to be added to train_val as np array
            to_add.append(df_test.iloc[i].values)
            # this person is now present in train_val set
            empl_train_val = np.append(empl_train_val, df_test.iloc[i]['case:concept:name'])
            # this observation should be dropped from the test set
            rows_to_drop.append(df_test.iloc[i].name)
            #print('employee ' + str(
            #    df_test.iloc[i]['case:concept:name']) + ' has an observation moved from test to train_val')

    # the values of train_val are the values of the old dataframe, plus the values that should be added
    if to_add != []:
        arr_train_val = np.concatenate((df_train_val.values, to_add), axis=0)
        # re-initialize df_train_val with observations from test set
        df_train_val = pd.DataFrame(arr_train_val, columns=df.columns)
        # remove rows from test set that moved to df_train_val
        df_test = df_test.drop(rows_to_drop).copy()

    return df, df_train_val, df_test


def piv_to_R(piv, piv_train_val, piv_train, piv_val, piv_test):
    """
    :param piv:
    :param piv_train_val:
    :param piv_train:
    :param piv_val:
    :param piv_test:
    :return:
    """
    R = piv.to_numpy()
    R_train = piv_train.to_numpy()
    R_test = piv_test.to_numpy()
    R_train_val = piv_train_val.to_numpy()
    R_val = piv_val.to_numpy()
    return R, R_train_val, R_train, R_val, R_test


def prep_data(df, test_size, oot, oitv):
    """
    :param df: dataframe that is used as input
    :param test_size: test size as fraction in [0,1]
    :param oot: 1 if train val test split should be made out of time. 0 otherwise
    :param oitv: 1 if train_val needs at least one observation per employee. 0 otherwise
    :return: df_train_val,df_train,df_val,df_test
    """
    if oitv == 0:
        if oot == 0:
            df_train_val, df_test = train_test_split(df, test_size=test_size)
            df_train, df_val = train_test_split(df_train_val, test_size=test_size / (1 - test_size))

        if oot == 1:
            # Sort df based on 'time_start'
            df = df.sort_values(by=['time_start'])

            # Calculate the indices for splitting
            n = len(df.index)
            train_end_idx = int(n * (1-2*test_size))
            val_end_idx = int(n * (1-test_size))

            # Split df into train, val, and test
            df_train = df.iloc[:train_end_idx].copy()  # First 50%
            df_val = df.iloc[train_end_idx:val_end_idx].copy()  # 25% in the middle
            df_test = df.iloc[val_end_idx:].copy()  # Last 25%

            df_train_val = pd.concat([df_train, df_val], ignore_index=True)

    if oitv == 1:
        if oot == 0:
            df_train_val, df_test = train_test_split(df, test_size=test_size)
            df_train, df_val = train_test_split(df_train_val, test_size=test_size / (1 - test_size))

            # make sure train_val has at least one observation per employee
            df_train, df_val, df_test, df_train_val, df = move_to_train(df_train, df_val, df_test)


        if oot == 1:

            # Sort df based on 'time_start'
            df = df.sort_values(by=['time_start'])

            # Calculate the indices for splitting
            n = len(df.index)
            train_end_idx = int(n * (1-2*test_size))
            val_end_idx = int(n * (1-test_size))

            # Split df into train, val, and test
            df_train = df.iloc[:train_end_idx].copy()  # First 50%
            df_val = df.iloc[train_end_idx:val_end_idx].copy()  # 25% in the middle
            df_test = df.iloc[val_end_idx:].copy()  # Last 25%

            # make sure train_val has at least one observation per employee
            df_train, df_val, df_test, df_train_val, df = move_to_train(df_train, df_val, df_test)


    return df, df_train_val, df_train, df_val, df_test



def move_to_train(df_train, df_val, df_test):
    """
    Move observations from df_val or df_test to df_train if unique id ('case:concept:name')
    is not present in df_train.

    :param df_train: DataFrame containing training samples
    :param df_val: DataFrame containing validation samples
    :param df_test: DataFrame containing test samples
    :return: Updated dataframes df_train, df_val, df_test, df_train_val, df
    """

    # Combine df_train, df_val, and df_test into a single DataFrame df
    df = pd.concat([df_train, df_val, df_test])
    df.sort_values(by=['time_start'], inplace=True)

    # Get unique employees (case:concept:name) from df_train
    train_employees = set(df_train['case:concept:name'])

    # Find employees in df_val or df_test that are not in df_train
    new_employees = set(df_val['case:concept:name']).union(set(df_test['case:concept:name'])) - train_employees

    # Update df_train with the first observation for each new employee
    for employee in new_employees:
        # Check if the employee is in df_val
        if employee in set(df_val['case:concept:name']):
            # Move the first observation to df_train
            idx = df_val[df_val['case:concept:name'] == employee].index[0]
            df_train = df_train.append(df_val.loc[idx])
            # Remove the observation from df_val
            df_val = df_val.drop(idx)
        else:
            # Move the first observation to df_train from df_test
            idx = df_test[df_test['case:concept:name'] == employee].index[0]
            df_train = df_train.append(df_test.loc[idx])
            # Remove the observation from df_test
            df_test = df_test.drop(idx)

    # Update df_train_val with updated df_train and df_val
    df_train_val = pd.concat([df_train, df_val])

    # Update df with updated df_train, df_val, and df_test
    df = pd.concat([df_train, df_val, df_test])

    return df_train, df_val, df_test, df_train_val, df

def func_oitv(df, df_train_val, df_train, df_val, df_test):
    """
    :param df: dataframe that is used as input
    :param df_train_val: subset of df that contains train+validation samples
    :param df_test: subset of df that contains test samples
    :return:
    """
    # make sure all instances have at least one observation in train (otherwise no info to make prediction)
    df = df.sort_values(by=['time_start'])
    empl_train_val = df_train_val['case:concept:name'].unique()
    to_add = []
    rows_to_drop = []

    # run through elements of test set
    for i in range(len(df_test)):

        # if this employee has no observations in train_val:
        if df_test.iloc[i]['case:concept:name'] not in empl_train_val:
            to_add.append(df_test.iloc[i].values)
            empl_train_val = np.append(empl_train_val, df_test.iloc[i]['case:concept:name'])
            rows_to_drop.append(df_test.iloc[i].name)

    if to_add != []:
        arr_train_val = np.concatenate((df_train_val.values, to_add), axis=0)
        df_train_val = pd.DataFrame(arr_train_val, columns=df.columns)
        df_test = df_test.drop(rows_to_drop).copy()

    return df, df_train_val, df_test

def read_data(dataset,dir):
    """
    Read dataset and return dataframe along with column arrays.

    Parameters:
    dataset (int): Dataset number (0, 1, 2, 3)

    Returns:
    df (pd.DataFrame): DataFrame containing the dataset
    cov_array_personal (list): List of personal covariate column names
    cov_array_personal_num (list): List of numerical personal covariate column names
    cov_array_personal_cat (list): List of categorical personal covariate column names
    """

    if dataset == 0:
        path = dir+r'\Data\toy_example.csv'
        separator = ';'
        cov_array_personal = ['X1', 'X2', 'X3', 'X4']
        cov_array_personal_num = ['X1', 'X2', 'X3']
        cov_array_personal_cat = ['X4']
        df = pd.read_csv(path, sep=str(separator))

    elif dataset == 1:
        path = dir+r'\Data/dataset_1.csv'
        separator = ';'
        df = pd.read_csv(path, sep=str(separator))
        cov_array_personal = df.columns[[22, 23]].tolist()
        cov_array_personal_num = []
        cov_array_personal_cat = df.columns[[22, 23]].tolist()

        raise TypeError("This dataset is not publicly available. Please select \"dataset_3\" in Settings")

    elif dataset == 2:
        path = dir+r"\Data/dataset_2.csv"
        separator = ";"
        df = pd.read_csv(path, sep=str(separator))
        cov_array_personal = df.columns[[6, 61]].tolist()
        cov_array_personal_num = []
        cov_array_personal_cat = df.columns[[6, 61]].tolist()
        raise TypeError("This dataset is not publicly available. Please select \"dataset_3\" in Settings")

    elif dataset == 3:
        path = dir+r"\Data/dataset3.csv"
        separator = ';'
        cov_array_personal = ['V06', 'V08']     #These features have been selected on separate validation set
        cov_array_personal_num = ['V08']        #These features have been selected on separate validation set
        cov_array_personal_cat = ['V06']
        df = pd.read_csv(path, sep=str(separator))
        df = df.rename(columns={'id': 'case:concept:name', 'act': 'concept:name'})

    else:
        raise ValueError("Invalid dataset number. Please select 0 (toy), 1, 2, or 3.")

    df = df.drop_duplicates(subset=['case:concept:name', 'concept:name'], keep='last')

    return df, cov_array_personal, cov_array_personal_num, cov_array_personal_cat


def df_to_piv(df, df_train_val, df_train, df_val, df_test):
    """
    :param df:
    :param df_train_val:
    :param df_train:
    :param df_val:
    :param df_test:
    :return: piv, piv_train_val,piv_train,piv_val,piv_test
    """
    # get selected indices from subselections
    indices_train_val = df_train_val.index.values.tolist()
    indices_train = df_train.index.values.tolist()
    indices_val = df_val.index.values.tolist()
    indices_test = df_test.index.values.tolist()

    def check_overlap(arr1, arr2, arr3):
        set1 = set(arr1)
        set2 = set(arr2)
        set3 = set(arr3)

        if set1.intersection(set2, set3):
            print('overlap exists')
            return True
        else:
            print('no overlap')
            return False

    # check if overlap exists (for data cleaning and debugging purposes)
    overlap_exists = check_overlap(indices_train, indices_val, indices_test)

    # initialize subsamples as df/df_train_val, then y-value of non-selected observations to NaN
    # 1. train_val is df without test
    df_train_val = df.copy()
    df_train_val.loc[df_train_val.index.isin(indices_test), 'objective'] = np.nan
    # 2. test is df without train_val
    df_test = df.copy()
    df_test.loc[df_test.index.isin(indices_train_val), 'objective'] = np.nan

    # 3. train is df without val and test
    df_train = df.copy()
    df_train.loc[df_train.index.isin(indices_val), 'objective'] = np.nan
    df_train.loc[df_train.index.isin(indices_test), 'objective'] = np.nan

    # 4. val is df without train and test
    df_val = df.copy()
    df_val.loc[df_val.index.isin(indices_train), 'objective'] = np.nan
    df_val.loc[df_val.index.isin(indices_test), 'objective'] = np.nan

    values_log = 'objective'
    index_log = ['case:concept:name']
    columns_log = ['concept:name']

    # convert dataframes into pivot tables
    piv = pd.pivot_table(df, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_train_val = pd.pivot_table(df_train_val, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_train = pd.pivot_table(df_train, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_val = pd.pivot_table(df_val, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_test = pd.pivot_table(df_test, values=values_log, index=index_log, columns=columns_log, dropna=False)

    return piv, piv_train_val, piv_train, piv_val, piv_test

def create_initial_matrix(N, M, L):
    np.random.seed(0)

    return np.random.rand(N, L), np.random.rand(M, L)


def calculate_similarity_matrix(df, cov_array_personal, cov_array_personal_cat, cov_array_personal_num):
    """
    :param df:
    :param N:
    :param cov_array_personal:
    :param cov_array_personal_cat:
    :param cov_array_personal_num:
    :return:
    """
    df_sim = df.drop_duplicates(subset=["case:concept:name"])[cov_array_personal].copy(deep=True).reset_index(
        drop=True)

    N = len(df_sim)

    # Initialize similarity_matrix
    similarity_matrix = np.zeros((N, N))

    nr_features = len(cov_array_personal_cat) + len(cov_array_personal_num)
    fract_num = len(cov_array_personal_num) / nr_features

    if len(cov_array_personal_num) != 0:  # Rescale numerical values to domain [0,1]
        x = df_sim[cov_array_personal_num].values
        x_scaled = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
        df_sim_num = pd.DataFrame(x_scaled)

    for i in tqdm(range(N), leave=False):
        for j in range(i, N):  # Only compute upper triangle
            similarity_num = 0
            if len(cov_array_personal_num) != 0:
                dist_num = np.abs(df_sim_num.iloc[i].to_numpy() - df_sim_num.iloc[j].to_numpy()).sum()
                similarity_num = 1 - dist_num / len(cov_array_personal_num)

            similarity_cat = 0
            if len(cov_array_personal_cat) != 0:
                similarity_cat = np.sum(
                    df_sim[cov_array_personal_cat].iloc[i] == df_sim[cov_array_personal_cat].iloc[j]) / len(
                    cov_array_personal_cat)

            similarity = fract_num * similarity_num + (1 - fract_num) * similarity_cat

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Since it's symmetric

    return similarity_matrix

def load_similarity_matrix_mfsr(dataset, dir) -> np.ndarray:

    if dataset == 0:
        SM = np.load(dir+'\similarity_matrix\mfsr\SM_toy_data.npy')
    if dataset == 1:
        SM = np.load(dir+'\similarity_matrix\mfsr\similarity_matrix_mfsr_dataset_1.npy')
    if dataset == 2:
        SM = np.load(dir+'\similarity_matrix\mfsr\similarity_matrix_mfsr_dataset_2.npy')
    if dataset == 3:
        SM = np.load(dir+'\similarity_matrix\mfsr\similarity_matrix_mfsr_dataset_3__.npy')
    return SM

def rename_to_surprise_notation(df):
    df.rename(
        columns={
            'case:concept:name': 'userId',
            'concept:name':'movieId',
            'objective':'rating',
            'time_start':'timestamp'},
        inplace=True)
    return df

def upscale_to_surprise(df):
    df['rating'] = df['rating']*5
    return df

def downscale_from_surprise(df):
    df['rating'] = df['rating'].apply(lambda x: x / 5)
    return df

def get_saved_optimal_hyperparameters(dataset,method, dir):
    with open(dir+r'\experiment\optimal_hyperparameters.yaml', 'r') as file:
        optimal_hyperparameters_dict = yaml.safe_load(file)

    return optimal_hyperparameters_dict[dataset][method]

