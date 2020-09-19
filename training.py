import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplin
import random

data_matrix = spsp.load_npz("data_matrix.npz")

nb_nonzero = 20
nonzero_rows = np.argwhere(data_matrix.getnnz(axis=1) >= nb_nonzero)[:, 0]

test_sz = 200
test_mat = data_matrix[nonzero_rows[:test_sz]]

train_mat1 = data_matrix[data_matrix.getnnz(axis=1) < nb_nonzero]
train_mat2 = data_matrix[nonzero_rows[test_sz:]]
train_mat = spsp.vstack([train_mat1, train_mat2])
print(f"Number of clients in train set: {train_mat.shape[0]}", \
      f"\nNumber of clients in train set: {test_mat.shape[0]}")

def train(rank, nb_favorite, train_mat, test_mat, mode, top_k):
    _, S, Vt = spsplin.svds(train_mat, k=rank, return_singular_vectors='vh')
    top_k_precisions = []
    for client in test_mat:
        client_spendings = client.toarray()[0]
        top_spendings = np.argsort(-client_spendings)[:nb_favorite]
        correlation = Vt.T @ Vt[:, top_spendings]
        if mode == 'sum':
            score = correlation.sum(axis=1)
        elif mode == 'dot':
            score = correlation @ client_spendings[top_spendings]
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

#rank = 3
#nb_favorite = 5
#mode = 'sum'
#top_k = 3

ranks = [i for i in range(3, 15, 2)]
nb_favorites = [i for i in range(3, 6)]
modes = ['sum', 'dot']
top_ks = [3, 5]

it = 1
best = [0, None]
for rank in ranks:
    for nb_favorite in nb_favorites:
        for mode in modes:
            for top_k in top_ks:
                if it % 10 == 0:
                    print(f"Iteration {it}/{len(ranks) * len(nb_favorites) * len(modes) * len(top_ks)}")
                precision = train(rank, nb_favorite, train_mat, test_mat, mode, top_k)
                if precision > best[0]:
                    print("Best precision:", precision)
                    best[0] = precision
                    best[1] = (rank, nb_favorite, mode, top_k)
                it += 1
print(best)

