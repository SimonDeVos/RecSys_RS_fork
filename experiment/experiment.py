import itertools
import warnings

import numpy as np
import pandas as pd
import yaml

from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

from methods import mf, svd, slopeone, knnc, knnp, calculate_similarity_matrix_smd, get_neighbor_rating_vectors, \
    calculate_r_a_j, calculate_similarity_matrix_ta, mfsr
from utils import read_data, df_to_piv, piv_to_R, get_saved_optimal_hyperparameters, calculate_similarity_matrix, \
     prep_data, load_similarity_matrix_mfsr


class Experiment:
    def __init__(self, settings, dir) -> None:
        """
        :param settings: a dictionary with the following keys
        :param dir: the directory where the project is stored
        """

        self.dataset = settings['dataset']
        self.method = settings['method']
        self.test_size = settings['test_size']
        self.oot = settings['oot']                          # out of time split
        self.oitv = settings['oitv']                        # one in trainval split
        self.hyperpara_tune = settings['hyperpara_tune']
        self.load_similarity = settings['load_similarity']

        self.dir = dir
        print(f'directory: {self.dir}')

        with open(self.dir + "\experiment\hyperparameter_grid.yaml", 'r') as file:
            self.hyperparameter_grid = yaml.safe_load(file)

        self.df, self.cov_array_personal, self.cov_array_personal_num, self.cov_array_personal_cat = read_data(settings['dataset'], self.dir)
        self.df, self.df_train_val, self.df_train, self.df_val, self.df_test = prep_data(self.df, test_size=settings['test_size'], oot=settings['oot'], oitv=settings['oitv'])
        self.piv, self.piv_train_val, self.piv_train, self.piv_val, self.piv_test = self.generate_piv()
        self.R, self.R_train_val, self.R_train, self.R_val, self.R_test = piv_to_R(self.piv, self.piv_train_val, self.piv_train, self.piv_val, self.piv_test)

        self.similarity_matrix = np.zeros((len(self.piv), len(self.piv))) # to be filled in later, dependent on method

        self.optimal_hyperparameters = None # to be filled in later, dependent on hyperpara_tune
        self.r_hat_val = None   #to be filled in later, dependent on method
        self.r_hat_test = None  #to be filled in later, dependent on method

        self.results = None      #to be filled in later, dependent on method
        self.test_results = None #to be filled in later, dependent on method

        self.print_details()

    def run(self) -> dict:
        self.tune_optimal_hyperparameters()
        self.pred_test()
        self.get_test_results()

        print('optimal hyperparameters: ', self.optimal_hyperparameters)

        return self.results

    def print_details(self) -> None:
        # Details on dataset:
        print(f'dataset: {self.dataset}')
        print(f'method: {self.method}')
        #print(f'the size of the dataframe is: {self.df.size}')
        print(f'nr of employees: {len(self.piv.index.values.tolist())}')
        print(f'nr of jobs: {len(self.piv.columns.values.tolist())}')


    def set_hyperparameters(self) -> None:
        """
        Set optimal hyperparameters for the experiment
        """
        # if hyperpara tuning is necessary:
        if self.hyperpara_tune:
            self.optimal_hyperparameters = self.tune_optimal_hyperparameters()
        else:
            # Get optimal hyperparameters from saved file (result from previous hyperparameter tuning)
            self.optimal_hyperparameters = self.get_optimal_hyperparameters


    def tune_optimal_hyperparameters(self) -> dict:
        """
        notation:
        hyperarameter_grid = a dictionary of dictionaries of lists
        hyperpara_grid_method = hyperparameter_grid['method'] = a dictionary of lists
        hyperparas_method = a dictionary of single hyperparameters
        optimal_hyperparameters_method = a dictionary of single hyperparameters
        """

        # if hyperpara tuning is necessary:
        if self.hyperpara_tune:
            best_rmse = float('inf')

            hyperpara_grid_method = self.hyperparameter_grid[self.method]

            # Get all keys and values from the hyperparameter grid
            keys = list(hyperpara_grid_method.keys())
            values = list(hyperpara_grid_method.values())

            # Initialize optimal_hyperparameters dictionary
            optimal_hyperparameters_method = {key: None for key in keys}

            # Create a tqdm iterator to wrap itertools.product
            for combination in tqdm(itertools.product(*values), total=len(list(itertools.product(*values)))):
                # Create a dictionary with the hyperparameter setup
                hyperparas_method = dict(zip(keys, combination))

                # Run through validation set
                # return results on validation set
                pred_val = self.pred_val(hyperpara_grid=hyperparas_method)

                # Evaluate results on validation set
                mae, rmse, mse, spcorr, kendalcorr = self.evaluate(pred_val, self.R_val, save=False)

                # Check if the current combination of hyperparameters gives a lower RMSE
                if rmse < best_rmse:
                    best_rmse = rmse
                    optimal_hyperparameters_method = hyperparas_method

            self.optimal_hyperparameters = optimal_hyperparameters_method

            return optimal_hyperparameters_method

        # if hyperparameter tuning is not necessary:
        elif not self.hyperpara_tune:
            print("Getting optimal hyperparameters with get_optimal_hyperparameters()")
            optimal_hyperparameters_method = self.get_optimal_hyperparameters()
            self.optimal_hyperparameters = optimal_hyperparameters_method

            return optimal_hyperparameters_method


    def generate_piv(self):

        # get selected indices from subsets
        indices_train_val = self.df_train_val.index.values.tolist()
        indices_train = self.df_train.index.values.tolist()
        indices_val = self.df_val.index.values.tolist()
        indices_test = self.df_test.index.values.tolist()

        # 1. train_val is df without test
        df_train_val = self.df.copy()
        df_train_val.loc[df_train_val.index.isin(indices_test), 'objective'] = np.nan

        # 2. test is df without train and val
        df_test = self.df.copy()
        df_test.loc[df_test.index.isin(indices_train), 'objective'] = np.nan
        df_test.loc[df_test.index.isin(indices_val), 'objective'] = np.nan

        # 3. train is df without val and test
        df_train = self.df.copy()
        df_train.loc[df_train.index.isin(indices_val), 'objective'] = np.nan
        df_train.loc[df_train.index.isin(indices_test), 'objective'] = np.nan

        # 4. val is df without train and test
        df_val = self.df.copy()
        df_val.loc[df_val.index.isin(indices_train), 'objective'] = np.nan
        df_val.loc[df_val.index.isin(indices_test), 'objective'] = np.nan

        values_log = 'objective'
        index_log = ['case:concept:name']
        columns_log = ['concept:name']

        # convert dataframes into pivot tables
        piv = pd.pivot_table(self.df, values=values_log, index=index_log, columns=columns_log, dropna=False)
        piv_train_val = pd.pivot_table(df_train_val, values=values_log, index=index_log, columns=columns_log,
                                       dropna=False)
        piv_train = pd.pivot_table(df_train, values=values_log, index=index_log, columns=columns_log, dropna=False)
        piv_val = pd.pivot_table(df_val, values=values_log, index=index_log, columns=columns_log, dropna=False)
        piv_test = pd.pivot_table(df_test, values=values_log, index=index_log, columns=columns_log, dropna=False)

        return piv, piv_train_val, piv_train, piv_val, piv_test

    def get_optimal_hyperparameters(self):

        optimal_hyperparameters_method = get_saved_optimal_hyperparameters(dataset=self.dataset, method=self.method, dir=self.dir)
        print('the stored optimal hyperparameters for dataset {} and method {} are: {}'.format(self.dataset, self.method, optimal_hyperparameters_method))

        return optimal_hyperparameters_method


    def pred_val(self, hyperpara_grid):

        if self.method == 'mf':
            r_hat = mf(self.R_train, hyperpara_grid)
            pass


        elif self.method == 'mfsr':

            # return results on validation set
            if np.all(self.similarity_matrix == 0):
                self.get_similarity_matrix()

            r_hat = mfsr(experiment=self,
                         R=self.R_train,
                         R_test=self.R_val,
                         similarity_matrix=self.similarity_matrix,
                         hyperpara_grid=hyperpara_grid)
            pass

        elif self.method == 'svd':

            r_hat = svd(df=self.df,
                        piv=self.piv,
                        df_train=self.df_train,
                        df_test=self.df_val,
                        hyperpara_grid=hyperpara_grid)
            pass

        elif self.method == 'slopeone':
            r_hat = slopeone(piv=self.piv,
                             df_train=self.df_train,
                             df_test=self.df_val)
            pass

        elif self.method == 'knnc':
            r_hat = knnc(piv=self.piv,
                         df_train=self.df_train,
                         df_test=self.df_val,
                         hyperpara_grid=hyperpara_grid)

        elif self.method == 'knnp':
            r_hat = knnp(piv=self.piv,
                         df_train=self.df_train,
                         df_test=self.df_val,
                         hyperpara_grid=hyperpara_grid)

        elif self.method == 'knnsmd':
            r_hat = self.knnsmd(#experiment=self,
                                df_train=self.df_train,
                                df_test=self.df_val,
                                piv_train=self.piv_train,
                                piv_test=self.piv_val,
                                hyperpara_grid=hyperpara_grid,
                                validation_phase=True)
            pass

        elif self.method == 'knnta':
            r_hat = self.knnta(#experiment=self,
                                df_train=self.df_train,
                                df_test=self.df_val,
                                piv_train=self.piv_train,
                                piv_test=self.piv_val,
                                hyperpara_grid=hyperpara_grid,
                                validation_phase=True)
            pass

        return r_hat

    def pred_test(self):

        # check if optimal hyperparameters are given to the experiment
        if self.optimal_hyperparameters == None:
            if self.hyperpara_tune:
                warnings.warn('Hyperparameter tuning is necessary, but no optimal hyperparameters are given. '
                              'You should first execute experiment.optimal_hyperparameters = experiment.tune_optimal_hyperparameters()', category=Warning)
            else:
                self.optimal_hyperparameters = self.get_optimal_hyperparameters()

        if self.method == 'mf':
            r_hat = mf(self.R_train_val, self.optimal_hyperparameters) # optimal_hyperparameters is a dictionary of single hyperparameters for the method 'mf'

        elif self.method == 'mfsr':
            # return results on test set
            if np.all(self.similarity_matrix == 0):
                self.get_similarity_matrix()

            r_hat = mfsr(experiment=self,
                                  R=self.R_train,
                                  R_test=self.R_test,
                                  similarity_matrix=self.similarity_matrix,
                                  hyperpara_grid=self.optimal_hyperparameters)

        elif self.method == 'svd':

            r_hat = svd(df=self.df,
                        piv=self.piv,
                        df_train=self.df_train_val,
                        df_test=self.df_test,
                        hyperpara_grid=self.optimal_hyperparameters)

        elif self.method == 'slopeone':

            r_hat = slopeone(piv=self.piv,
                             df_train=self.df_train_val,
                             df_test=self.df_test)

        elif self.method == 'knnc':

            r_hat = knnc(piv=self.piv,
                     df_train=self.df_train_val,
                     df_test=self.df_test,
                     hyperpara_grid=self.optimal_hyperparameters)

        elif self.method == 'knnp':

            r_hat = knnp(piv=self.piv,
                     df_train=self.df_train_val,
                     df_test=self.df_test,
                     hyperpara_grid=self.optimal_hyperparameters)

        elif self.method == 'knnsmd':
            r_hat = self.knnsmd(#experiment=self,
                                df_train=self.df_train_val,
                                df_test=self.df_test,
                                piv_train=self.piv_train_val,
                                piv_test=self.piv_test,
                                hyperpara_grid=self.optimal_hyperparameters,
                                validation_phase=False)

        elif self.method == 'knnta':
            r_hat = self.knnta(#experiment=self,
                                df_train=self.df_train_val,
                                df_test=self.df_test,
                                piv_train=self.piv_train_val,
                                piv_test=self.piv_test,
                                hyperpara_grid=self.optimal_hyperparameters,
                                validation_phase=False)
        else:
            raise ValueError('method {} is not available'.format(self.method))

        self.r_hat_test = r_hat
        self.evaluate(self.r_hat_test, self.R_test, save=True)

        return r_hat


    def evaluate(self, matrix_pred, matrix_actual, save):
        """
        :param matrix_pred: Takes the predictions as input
        :param matrix_actual: Takes the actual values as input
        :return: Returns MAE, RMSE, MSE, Spearman Rank Correlation and Kendall Tau Correlation
        """

        #Calculate the mean absolute error, root mean squared error and mean squared error for the predictions
        mae = 0
        rmse = 0
        mse = 0

        y_pred = matrix_pred.flatten()
        y_actual = matrix_actual.flatten()

        # Convert arrays to float64
        y_pred = y_pred.astype(np.float64)
        y_actual = y_actual.astype(np.float64)

        nan_mask = ~np.isnan(y_pred) & ~np.isnan(y_actual)

        # Calculate mean absolute error for non-NaN elements
        mae = np.mean(np.abs(y_pred[nan_mask] - y_actual[nan_mask]))
        rmse = np.sqrt(np.mean(np.square(y_pred[nan_mask] - y_actual[nan_mask])))
        mse = np.mean(np.square(y_pred[nan_mask] - y_actual[nan_mask]))

        #Calculate the Spearman rank correlation and Kendall Tau correlation for the predictions
        def check_constant(arr):
            unique_values = np.unique(arr)
            return len(unique_values) == 1

        matrix_pred = matrix_pred.astype(np.float64)
        matrix_actual = matrix_actual.astype(np.float64)

        matrix_pred = matrix_pred.T
        matrix_actual = matrix_actual.T

        arr_spcorr = []
        arr_kendcorr = []

        for i in range(matrix_pred.shape[0]):
            non_nan_mask = ~np.isnan(matrix_pred[i]) & ~np.isnan(matrix_actual[i])

            if np.sum(non_nan_mask) > 2:
                if check_constant(matrix_pred[i][non_nan_mask]) or check_constant(matrix_actual[i][non_nan_mask]):
                    pass
                else:
                    spcorr, _ = spearmanr(matrix_pred[i][non_nan_mask], matrix_actual[i][non_nan_mask], nan_policy='omit')
                    arr_spcorr.append(spcorr)

                    kendallcorr = kendalltau(matrix_pred[i][non_nan_mask], matrix_actual[i][non_nan_mask], nan_policy='omit')
                    arr_kendcorr.append(kendallcorr)

        spcorr = np.mean(arr_spcorr)
        kendallcorr = np.mean(arr_kendcorr)

        if save:
            self.results = {'mae': mae.round(4),
                            'rmse': rmse.round(4),
                            'mse': mse.round(4),
                            'spcorr': spcorr.round(4),
                            'kendallcorr': kendallcorr.round(4)}

        return mae, rmse, mse, spcorr, kendallcorr


    def get_test_results(self):
        if self.results == None:
            warnings.warn('No results are saved. You should first execute experiment.pred_test()', category=Warning)
        else:
            self.test_results = self.evaluate(self.r_hat_test,
                                              self.R_test,
                                              save=True)
            return self.results

    def get_similarity_matrix(self):

        if np.all(self.similarity_matrix == 0):
            if self.load_similarity:
                self.similarity_matrix = load_similarity_matrix_mfsr(self.dataset, self.dir)

            else:
                self.similarity_matrix = calculate_similarity_matrix(df=self.df,
                                                                     cov_array_personal=self.cov_array_personal,
                                                                     cov_array_personal_cat = self.cov_array_personal_cat,
                                                                     cov_array_personal_num = self.cov_array_personal_num)
        pass

    def knnsmd(self, df_train, df_test, piv_train, piv_test, hyperpara_grid, validation_phase):

        k = hyperpara_grid['k']
        piv_test_hat = pd.DataFrame(index=piv_test.index, columns=piv_test.columns)

        # Load or calculate similarity matrix
        if self.load_similarity:
            if validation_phase:
                self.similarity_matrix = pd.read_csv(self.dir+fr'\similarity_matrix/smd/similarity_matrix_smd_train_dataset_{self.dataset}_.csv', index_col=0)

            elif not validation_phase:
                self.similarity_matrix = pd.read_csv(self.dir+fr'\similarity_matrix/smd/similarity_matrix_smd_trainval_dataset_{self.dataset}_.csv', index_col=0)

            # Convert index and columns to integers
            self.similarity_matrix.index = self.similarity_matrix.index.astype(int)
            self.similarity_matrix.columns = self.similarity_matrix.columns.astype(int)
        else:
            #calculate similarity matrix
            self.similarity_matrix = calculate_similarity_matrix_smd(piv_train)

        similarity_matrix = self.similarity_matrix.copy()

        for index in piv_test.index:
            # get k neighbor rating vector around the index
            neighbor_vectors, neighbor_indices = get_neighbor_rating_vectors(piv_train, similarity_matrix, index, k)
            piv_test_hat_a = pd.DataFrame(index=[index], columns=piv_test.columns)

            for col in piv_test.columns:
                if not pd.isna(piv_test.loc[index,col]):
                    r_a_j = calculate_r_a_j(piv_train, similarity_matrix, neighbor_indices, index, col)
                    piv_test_hat_a.loc[index, col] = r_a_j

            piv_test_hat.loc[index] = piv_test_hat_a.loc[index].values
            r_test_hat = piv_test_hat.to_numpy()

        return r_test_hat


    def knnta(self, df_train, df_test, piv_train, piv_test, hyperpara_grid, validation_phase):

        k = hyperpara_grid['k']
        piv_test_hat = pd.DataFrame(index=piv_test.index, columns=piv_test.columns)

        # Load or calculate similarity matrix
        if self.load_similarity:
            if validation_phase:
                self.similarity_matrix = pd.read_csv(self.dir+fr'\similarity_matrix\ta\similarity_matrix_ta_train_dataset_{self.dataset}.csv', index_col=0)
                #print('similarity_matrix_train ta is loaded')

            elif not validation_phase:
                self.similarity_matrix = pd.read_csv(self.dir+fr'\similarity_matrix\ta\similarity_matrix_ta_trainval_dataset_{self.dataset}.csv', index_col=0)
                #print('similarity_matrix_trainval ta is loaded')

            # Convert index and columns to integers
            self.similarity_matrix.index = self.similarity_matrix.index.astype(int)
            self.similarity_matrix.columns = self.similarity_matrix.columns.astype(int)
        else:
            #calculate similarity matrix
            self.similarity_matrix = calculate_similarity_matrix_ta(piv_train)

        similarity_matrix = self.similarity_matrix.copy()

        for index in piv_test.index:
            # get k neighbor rating vector around the index
            neighbor_vectors, neighbor_indices = get_neighbor_rating_vectors(piv_train, similarity_matrix, index, k)
            piv_test_hat_a = pd.DataFrame(index=[index], columns=piv_test.columns)

            for col in piv_test.columns:
                if not pd.isna(piv_test.loc[index,col]):
                    r_a_j = calculate_r_a_j(piv_train, similarity_matrix, neighbor_indices, index, col)
                    piv_test_hat_a.loc[index, col] = r_a_j

            piv_test_hat.loc[index] = piv_test_hat_a.loc[index].values
            r_test_hat = piv_test_hat.to_numpy()

        return r_test_hat


