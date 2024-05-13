import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import rename_to_surprise_notation, create_initial_matrix

from surprise import Reader, Dataset, SVD, KNNBasic, SlopeOne, NMF, accuracy


def mf(R, hyperpara_grid):
    """
    :param R:
    :param hyperpara_grid:
    :return: r_hat
    """
    # Get hyperparameters from grid
    lambda1 = hyperpara_grid['lambda']
    alpha = hyperpara_grid['alpha']
    L = hyperpara_grid['L']
    steps = hyperpara_grid['steps']

    N = len(R)  # N: num of employees
    M = len(R[0])  # M: num of jobs

    # Initialize P and Q matrices
    P, Q = create_initial_matrix(N, M, L)

    Q = Q.T

    for step in range(steps):#tqdm(range(steps), leave=False):
    #for step in range(steps):
        # Update P and Q matrices
        for i, j in zip(*np.where(~np.isnan(R) & (R > 0))):  # Iterate over non-NaN and positive values in R
            # calculate error
            eij = R[i, j] - np.dot(P[i, :], Q[:, j])

            # Update P and Q using gradient descent
            P[i, :] += alpha * (2 * eij * Q[:, j] - lambda1 * P[i, :])
            Q[:, j] += alpha * (2 * eij * P[i, :] - lambda1 * Q[:, j])

        # Update matrices P and Q using the adjusted errors
        e = R - np.dot(P, Q)
        mask = ~np.isnan(R) & (R > 0)  # Mask for non-NaN and positive elements of R
        e_masked = np.nan_to_num(e * mask)  # Set NaN values to 0
        P += alpha * (2 * np.dot(e_masked, Q.T) - lambda1 * P)
        Q += alpha * (2 * np.dot(P.T, e_masked) - lambda1 * Q)

        # Calculate total error
        error = np.sum(np.square(e_masked))
        error += (lambda1 / 2) * (np.sum(np.square(P)) + np.sum(np.square(Q)))

        # Terminate if error is below threshold
        if error < 0.001:
            break

    r_hat = np.dot(P, Q)

    return r_hat

def mfsr(experiment, R, R_test, hyperpara_grid, similarity_matrix):
    # Get hyperparameters from grid
    lambda1 = hyperpara_grid['lambda']
    alpha = hyperpara_grid['alpha']
    beta = hyperpara_grid['beta']
    K = L = hyperpara_grid['L']
    steps = hyperpara_grid['steps']

    N = len(R)  # N: num of employees
    M = len(R[0])  # M: num of jobs

    # Initialize P and Q matrices
    P, Q = create_initial_matrix(N, M, L)

    Q = Q.T

    for step in tqdm(range(steps), leave=False):  # tqdm inserts nice loading bar
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        # calculate error
                        eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                        for k in range(K):
                            soc_reg = 0
                            for f in range(len(R[i])):
                                soc_reg = soc_reg + similarity_matrix[i][f] * (P[i][k] - P[f][k])
                            # calculate gradient with a and beta parameter
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - lambda1 * P[i][k] - beta * soc_reg)
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - lambda1 * Q[k][j])

            eR = np.dot(P, Q)
            e = 0

            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                        for k in range(K):
                            e = e + (lambda1 / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
            # 0.001: local minimum
            if e < 0.001:
                break

    r_hat = np.dot(P, Q)

    return r_hat


# svd is implemented using surprise library
def svd(df, piv, df_train, df_test, hyperpara_grid):

    df_train = rename_to_surprise_notation(df_train)
    df_test = rename_to_surprise_notation(df_test)

    df_train['rating'] = df_train['rating'].apply(lambda x: x * 5)
    df_test['rating'] = df_test['rating'].apply(lambda x: x * 5)

    columns_to_convert = {'userId': 'string', 'movieId': 'string', 'rating': 'float', 'timestamp': 'string'}
    df_train = df_train.astype(columns_to_convert)
    df_test = df_test.astype(columns_to_convert)

    df_train, df_test= [df.astype({'userId': 'string', 'movieId': 'string', 'rating': 'float', 'timestamp': 'string'}) for df in [df_train, df_test]]

    # Create reader object
    reader = Reader()

    # Load the train_val, train, val, and test dataset.
    data_train = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
    data_test = Dataset.load_from_df(df_test[['userId', 'movieId', 'rating']], reader)

    raw_ratings_train = data_train.raw_ratings
    raw_ratings_test = data_test.raw_ratings

    data_train.raw_ratings = raw_ratings_train
    data_test.raw_ratings = raw_ratings_test

    trainset = data_train.build_full_trainset()
    testset = data_test.construct_testset(raw_ratings_test)

    algo = SVD(n_factors=hyperpara_grid['n_factors'],
               n_epochs=hyperpara_grid['n_epochs'],
               reg_pu=hyperpara_grid['reg_pu'],
               reg_qi=hyperpara_grid['reg_qi'],
               random_state=42,
               verbose=False)

    # Train the algorithm on the training set
    algo.fit(trainset)

    # Predict ratings for the test set
    test_predictions = algo.test(testset)

    # Extract attributes from the Prediction objects
    results_test = [(int(pred.uid), pred.iid, pred.est) for pred in test_predictions]

    # Convert the list to a DataFrame
    df_test_hat = pd.DataFrame(results_test, columns=['uid', 'iid', 'est'])

    # Create the empty pivot table with the same dimensions as the original
    empty_pivot_table_hat = pd.DataFrame(index=piv.index, columns=piv.columns)

    # Fill the empty pivot table with the values from df_test
    for index, row in df_test_hat.iterrows():
        empty_pivot_table_hat.loc[row['uid'], row['iid']] = row['est']

    pivot_table_test_hat = empty_pivot_table_hat.copy()

    r_test_hat = pivot_table_test_hat.to_numpy()/5

    return r_test_hat

# slopeone is implemented using surprise library
def slopeone(piv, df_train, df_test):

    df_train = rename_to_surprise_notation(df_train)
    df_test = rename_to_surprise_notation(df_test)

    df_train['rating'] = df_train['rating'].apply(lambda x: x * 5)
    df_test['rating'] = df_test['rating'].apply(lambda x: x * 5)

    columns_to_convert = {'userId': 'string', 'movieId': 'string', 'rating': 'float', 'timestamp': 'string'}
    df_train = df_train.astype(columns_to_convert)
    df_test = df_test.astype(columns_to_convert)

    df_train, df_test= [df.astype({'userId': 'string', 'movieId': 'string', 'rating': 'float', 'timestamp': 'string'}) for df in [df_train, df_test]]

    # Create reader object
    reader = Reader()

    # Load the train_val, train, val, and test dataset.
    data_train = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
    data_test = Dataset.load_from_df(df_test[['userId', 'movieId', 'rating']], reader)

    raw_ratings_train = data_train.raw_ratings
    raw_ratings_test = data_test.raw_ratings

    data_train.raw_ratings = raw_ratings_train
    data_test.raw_ratings = raw_ratings_test

    trainset = data_train.build_full_trainset()
    testset = data_test.construct_testset(raw_ratings_test)

    algo = SlopeOne()

    # Train the algorithm on the training set
    algo.fit(trainset)

    # Predict ratings for the test set
    test_predictions = algo.test(testset)

    # Extract attributes from the Prediction objects
    results_test = [(int(pred.uid), pred.iid, pred.est) for pred in test_predictions]


    # Convert the list to a DataFrame
    df_test_hat = pd.DataFrame(results_test, columns=['uid', 'iid', 'est'])

    # Create the empty pivot table with the same dimensions as the original
    empty_pivot_table_hat = pd.DataFrame(index=piv.index, columns=piv.columns)

    # Fill the empty pivot table with the values from df_test
    for index, row in df_test_hat.iterrows():
        empty_pivot_table_hat.loc[row['uid'], row['iid']] = row['est']

    pivot_table_test_hat = empty_pivot_table_hat.copy()

    r_test_hat = pivot_table_test_hat.to_numpy()/5

    return r_test_hat

# knnc is implemented using surprise library
def knnc(piv, df_train, df_test, hyperpara_grid):

    df_train = rename_to_surprise_notation(df_train)
    df_test = rename_to_surprise_notation(df_test)

    df_train['rating'] = df_train['rating'].apply(lambda x: x * 5)
    df_test['rating'] = df_test['rating'].apply(lambda x: x * 5)

    columns_to_convert = {'userId': 'string', 'movieId': 'string', 'rating': 'float', 'timestamp': 'string'}
    df_train = df_train.astype(columns_to_convert)
    df_test = df_test.astype(columns_to_convert)

    df_train, df_test= [df.astype({'userId': 'string', 'movieId': 'string', 'rating': 'float', 'timestamp': 'string'}) for df in [df_train, df_test]]

    # Create reader object
    reader = Reader()

    # Load the train_val, train, val, and test dataset.
    data_train = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
    data_test = Dataset.load_from_df(df_test[['userId', 'movieId', 'rating']], reader)

    raw_ratings_train = data_train.raw_ratings
    raw_ratings_test = data_test.raw_ratings

    data_train.raw_ratings = raw_ratings_train
    data_test.raw_ratings = raw_ratings_test

    trainset = data_train.build_full_trainset()
    testset = data_test.construct_testset(raw_ratings_test)

    algo = KNNBasic(k=hyperpara_grid['k'],
                    min_k=hyperpara_grid['min_k'],
                    sim_options={'name':'cosine'},
                    verbose=False)

    # Train the algorithm on the training set
    algo.fit(trainset)

    # Predict ratings for the test set
    test_predictions = algo.test(testset)

    # Extract attributes from the Prediction objects
    results_test = [(int(pred.uid), pred.iid, pred.est) for pred in test_predictions]

    # Convert the list to a DataFrame
    df_test_hat = pd.DataFrame(results_test, columns=['uid', 'iid', 'est'])

    # Create the empty pivot table with the same dimensions as the original
    empty_pivot_table_hat = pd.DataFrame(index=piv.index, columns=piv.columns)

    # Fill the empty pivot table with the values from df_test
    for index, row in df_test_hat.iterrows():
        empty_pivot_table_hat.loc[row['uid'], row['iid']] = row['est']

    pivot_table_test_hat = empty_pivot_table_hat.copy()

    r_test_hat = pivot_table_test_hat.to_numpy()/5

    return r_test_hat

# knnp is implemented using surprise library
def knnp(piv, df_train, df_test, hyperpara_grid):
    df_train = rename_to_surprise_notation(df_train)
    df_test = rename_to_surprise_notation(df_test)

    df_train['rating'] = df_train['rating'].apply(lambda x: x * 5)
    df_test['rating'] = df_test['rating'].apply(lambda x: x * 5)

    columns_to_convert = {'userId': 'string', 'movieId': 'string', 'rating': 'float', 'timestamp': 'string'}
    df_train = df_train.astype(columns_to_convert)
    df_test = df_test.astype(columns_to_convert)

    df_train, df_test = [df.astype({'userId': 'string', 'movieId': 'string', 'rating': 'float', 'timestamp': 'string'})
                         for df in [df_train, df_test]]

    # Create reader object
    reader = Reader()

    # Load the train_val, train, val, and test dataset.
    data_train = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
    data_test = Dataset.load_from_df(df_test[['userId', 'movieId', 'rating']], reader)

    raw_ratings_train = data_train.raw_ratings
    raw_ratings_test = data_test.raw_ratings

    data_train.raw_ratings = raw_ratings_train
    data_test.raw_ratings = raw_ratings_test

    trainset = data_train.build_full_trainset()
    testset = data_test.construct_testset(raw_ratings_test)

    algo = KNNBasic(k=hyperpara_grid['k'],
                    min_k=hyperpara_grid['min_k'],
                    sim_options={'name':'pearson'},
                    verbose=False)

    # Train the algorithm on the training set
    algo.fit(trainset)

    # Predict ratings for the test set
    test_predictions = algo.test(testset)

    # Extract attributes from the Prediction objects
    results_test = [(int(pred.uid), pred.iid, pred.est) for pred in test_predictions]

    # Convert the list to a DataFrame
    df_test_hat = pd.DataFrame(results_test, columns=['uid', 'iid', 'est'])

    # Create the empty pivot table with the same dimensions as the original
    empty_pivot_table_hat = pd.DataFrame(index=piv.index, columns=piv.columns)

    # Fill the empty pivot table with the values from df_test
    for index, row in df_test_hat.iterrows():
        empty_pivot_table_hat.loc[row['uid'], row['iid']] = row['est']

    pivot_table_test_hat = empty_pivot_table_hat.copy()

    r_test_hat = pivot_table_test_hat.to_numpy() / 5

    return r_test_hat

def get_neighbor_rating_vectors(piv, sim_matrix, index, k):
    '''
    :param piv:
    :param sim_matrix:
    :param index: the entity of which neighbors are to be found
    :param k: number of neighbors
    :return:
    '''

    # Get similarity values for the given user
    user_similarities = sim_matrix.loc[index]

    # Get the indices of the k highest values in the specified column
    top_k_indices = user_similarities.nlargest(k).index
    top_k_indices = top_k_indices.values.tolist()
    top_k_indices = [int(x) for x in top_k_indices]

    neighbor_vectors = piv.loc[top_k_indices]

    return neighbor_vectors, top_k_indices


def calculate_r_a_j(piv_train, sim, k_neighbor_indices, a, j):
    # Calculate average(r[a])
    piv_a_avg = piv_train.loc[a].mean(skipna=True)

    # Initialize numerator and denominator for the summation
    numerator_sum = 0
    denominator_sum = 0

    # Loop through indices_k and calculate the numerator and denominator
    for i in k_neighbor_indices:
        #print(type(i))
        piv_i_j = piv_train.loc[i, j]

        if not np.isnan(piv_i_j):  # Ignore NaN values in r
            piv_i_avg = piv_train.loc[i].mean(skipna=True)
            sim_a_i = sim.at[a, i]
            numerator_sum += ((piv_i_j - piv_i_avg) * sim_a_i)
            denominator_sum += abs((sim_a_i))

    # if rating vector is too sparse, denominator could be 0.
    if denominator_sum == 0:
        ratio = 0
    else:
        ratio = numerator_sum / denominator_sum

    # Return r[a, j]
    return piv_a_avg + ratio


def smd(u1, u2):
    # N_1_2 is the number of common values between u_1 and u_2
    N_1_2 = np.sum(u1 * u2)

    # N is the length of the vector
    N = len(u1)

    # F is the sum of the number of differences between u_1 and u_2
    F = np.sum(np.abs(u1 - u2))

    # N_1 and N_2 are the number of non-NaN values in u_1 and u_2 respectively
    N_1 = np.sum(u1 == 1)
    N_2 = np.sum(u2 == 1)

    # Calculate smd
    smd = (1 - (F / N) + (2 * N_1_2 / (N_1 + N_2))) / 2

    return smd

def calculate_similarity_matrix_smd(piv_train):
    # Initialize a square matrix to store similarity values
    num_users = len(piv_train.index)
    similarity_matrix = np.zeros((num_users, num_users))

    # Convert all non-NaN values to the specific value '1'
    df_binary = piv_train.isna().replace({True: 0, False: 1})

    # Calculate similarity between each pair of users
    for i in tqdm(range(len(piv_train.index)), desc='calculating similarities (SMD)', unit='user'):
        # for i in range(len(df.index)):
        for j in range(i+1, len(piv_train.index)):
            similarity_matrix[i, j] = smd(df_binary.iloc[i, :], df_binary.iloc[j, :])

    # set upper triangle to the corresponding lower triangle values
    similarity_matrix += similarity_matrix.T

    # set diagonal to 0 (exclude itself from nearest neighbors)
    np.fill_diagonal(similarity_matrix, 0)

    # Convert the similarity matrix to a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=piv_train.index, columns=piv_train.index)

    return similarity_df


def ta(u1, u2):
    # Create boolean masks for NaN values
    mask_u1 = ~np.isnan(u1)
    mask_u2 = ~np.isnan(u2)

    # Find common indices (intersect non NaN values u1 and u2)
    indices = mask_u1 & mask_u2

    # Calculate the sum of the product of common elements
    sum_prod = np.sum(u1[indices] * u2[indices])

    # Calculate the squared sum of the common elements
    sum_squared_u1 = np.sum(u1[indices] ** 2)
    sum_squared_u2 = np.sum(u2[indices] ** 2)

    # Calculate the square roots of the squared sums
    u1_dist = np.sqrt(sum_squared_u1)
    u2_dist = np.sqrt(sum_squared_u2)

    # Determine the cases and calculate ta accordingly
    if sum_prod >= 0:
        if u1_dist <= u2_dist:
            ta = sum_prod ** 2 / (u1_dist * (u2_dist ** 3))
        else:
            ta = sum_prod ** 2 / ((u1_dist ** 3) * u2_dist)
    else:
        if u1_dist <= u2_dist:
            ta = sum_prod / (u2_dist ** 2)
        else:
            ta = sum_prod / (u1_dist ** 2)

    return ta

def calculate_similarity_matrix_ta(piv_train):
    # Initialize a square matrix to store similarity values
    num_users = len(piv_train.index)
    similarity_matrix = np.zeros((num_users, num_users))

    # Calculate similarity between each pair of users
    for i in tqdm(range(len(piv_train.index)), desc='calculating similarities (TA)', unit='user'):
        # for i in range(len(df.index)):
        for j in range(i+1, len(piv_train.index)):
            similarity_matrix[i, j] = ta(piv_train.iloc[i, :], piv_train.iloc[j, :])

    # set upper triangle to the corresponding lower triangle values
    similarity_matrix += similarity_matrix.T

    # set diagonal to 0 (exclude itself from nearest neighbors)
    np.fill_diagonal(similarity_matrix, 0)

    # Convert the similarity matrix to a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=piv_train.index, columns=piv_train.index)

    return similarity_df