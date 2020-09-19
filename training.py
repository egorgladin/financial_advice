import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplin
import random

data_matrix = spsp.load_npz("data_matrix.npz")

rank = 100
_, S, Vt = spsplin.svds(data_matrix, k=rank, return_singular_vectors='vh')

nb_favourites = 5
nb_nonzero = 20
client_id = 1507

if False:
    client_id = random.randint(0, data_matrix.shape[0] - 1)
    client_spendings = data_matrix[client_id].toarray()[0]
    while (client_spendings > 0).sum() < nb_nonzero:
        client_id = random.randint(0, data_matrix.shape[0] - 1)
        client_spendings = data_matrix[client_id].toarray()[0]
client_spendings = data_matrix[client_id].toarray()[0]
top_spendings = np.argsort(-client_spendings)[:nb_favourites]


MODE = 'dot' # 'sum'

nb_recommends = 5
correlation = Vt.T @ Vt[:, top_spendings]
if MODE == 'sum':
    score = correlation.sum(axis=1)
elif MODE == 'dot':
    score = correlation @ client_spendings[top_spendings]

top_merchants = np.argsort(-score)
recommendations = []
for rec in top_merchants:
    if rec not in top_spendings:
        recommendations.append(rec)
    if len(recommendations) == nb_recommends:
        break
print("How many are successfull?", (client_spendings[recommendations] > 0).sum())
