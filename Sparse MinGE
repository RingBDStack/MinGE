import numpy as np
from scipy import sparse
import pandas as pd


spm = (
    lambda data: (
        lambda nodes:
            sparse.csr_matrix(
                (np.ones((data.shape[0])), 
                (data[:, 0].flatten(), data[:, 1].flatten())), shape=(nodes, nodes)
            )
    ) (np.max(data)+1))(pd.read_csv("./data/citeseer/citeseer.edges", header=None).to_numpy())

second_order = spm.multiply(spm)
for i in range(second_order.shape[0]):
    print(second_order[i,i])
nnz = second_order.nonzero()
for i, j in zip(nnz[0], nnz[1]):
    if i == j: continue
    second_order[i,j] = second_order[i,j] / (second_order[i,i] if i < j else second_order[j,j])
for i in range(second_order.shape[0]):
    second_order[i,i] = 1.0

N = spm.shape[0]
mtrx_1xn = sparse.csr_matrix(np.array([[1] for i in range(N)]))
degree = spm * mtrx_1xn

v_degree_n = np.dot(second_order, degree)
z = np.sum(v_degree_n)
p = v_degree_n/z
print(np.max(p))
logp = np.log(p)
H = -np.dot(p, logp)
print(H)
n=(np.log(N * N)+H)/0.24
print(n)
