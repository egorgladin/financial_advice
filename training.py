"""
Recommendation of merchants to bank's customers via latent semantic search.

GitHub: https://github.com/egorgladin/financial_advice
"""

import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplin

def train_test_split(data_matrix, nb_nonzero, test_sz):
    nonzero_rows = np.argwhere(data_matrix.getnnz(axis=1) >= nb_nonzero)[:, 0]
    test_mat = data_matrix[nonzero_rows[:test_sz]]
    train_mat1 = data_matrix[data_matrix.getnnz(axis=1) < nb_nonzero]
    train_mat2 = data_matrix[nonzero_rows[test_sz:]]
    train_mat = spsp.vstack([train_mat1, train_mat2])
    return train_mat, test_mat


def train(rank, nb_favorite, train_mat, test_mat, top_k):
    _, S, Vt = spsplin.svds(train_mat, k=rank, return_singular_vectors='vh')
    top_k_precisions = []
    for client in test_mat:
        client_spendings = client.toarray()[0]
        top_spendings = np.argsort(-client_spendings)[:nb_favorite]
        correlation = Vt.T @ Vt[:, top_spendings]
        score = correlation.sum(axis=1)
        top_merchants = np.argsort(-score)
        
        precision = 0.
        recommendations = 0
        for rec in top_merchants:
            if rec not in top_spendings:
                recommendations += 1
                if client_spendings[rec] > 0:
                    precision += 1./top_k
            if recommendations == top_k:
                break
        top_k_precisions.append(precision)
    return sum(top_k_precisions) / len(top_k_precisions)


data_matrix = spsp.load_npz("data_matrix.npz")
nb_nonzero = 20
test_sz = 200
train_mat, test_mat = train_test_split(data_matrix, nb_nonzero, test_sz)

rank = 3
nb_favorite = 5
top_k = 3
precision = train(rank, nb_favorite, train_mat, test_mat, top_k)

print(f"Test precision @top{top_k} is {round(precision, 3)}")

