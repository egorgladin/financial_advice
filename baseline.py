"""
Baseline for recommendation of merchants to bank's customers.

Unpersonalized recommendations based on popularity.
GitHub: https://github.com/egorgladin/financial_advice
"""

import numpy as np
import scipy.sparse as spsp


def train_test_split(data_matrix, nb_nonzero, test_sz):
    nonzero_rows = np.argwhere(data_matrix.getnnz(axis=1) >= nb_nonzero)[:, 0]
    test_mat = data_matrix[nonzero_rows[:test_sz]]
    train_mat1 = data_matrix[data_matrix.getnnz(axis=1) < nb_nonzero]
    train_mat2 = data_matrix[nonzero_rows[test_sz:]]
    train_mat = spsp.vstack([train_mat1, train_mat2])
    return train_mat, test_mat


def baseline_precision(top_merchants, test_mat):
    top_k_precisions = []
    for client in test_mat:
        client_spendings = client.toarray()[0]
        precision = 0.
        recommendations = 0
        for rec in top_merchants:
            if client_spendings[rec] > 0:
                recommendations += 1
                precision += 1./top_k
            if recommendations == top_k:
                break
        top_k_precisions.append(precision)
    return sum(top_k_precisions) / len(top_k_precisions)


data_matrix = spsp.load_npz("data_matrix.npz")
nb_nonzero = 20
test_sz = 200
train_mat, test_mat = train_test_split(data_matrix, nb_nonzero, test_sz)

top_k = 3

# Most popular merchants
merchants_scores = np.squeeze(np.asarray(train_mat.sum(axis=0)))
top_merchants = np.argsort(-merchants_scores)[:top_k]
precision = baseline_precision(top_merchants, test_mat)

print(f"Test precision @top{top_k} is {round(precision, 3)}")

