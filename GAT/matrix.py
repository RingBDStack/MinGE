import numpy as np


with open('./data/cora/cora.cites', 'r') as f:
    adj_list = {}
    v = set()
    for line in f.readlines():
        cited, citing = line.split('\t')
        cited, citing = eval(cited), eval(citing)
        if cited in adj_list.keys():
            adj_list[cited].append(citing)
        else:
            adj_list[cited] = [citing]
        v.update([cited, citing])

# 顶点集合
v = list(v)

# 1阶邻接矩阵
adj_mtrx = np.eye(len(v), dtype='int32')
# 2阶邻接矩阵
adj_mtrx_2 = np.eye(len(v), dtype='float32')
# 度矩阵
degree_mtrx = np.empty((len(v), len(v)), dtype='int32')

# 计算一阶邻接矩阵
for cited in adj_list.keys():
    for citing in adj_list[cited]:
        adj_mtrx[v.index(cited), v.index(citing)] = 1
        adj_mtrx[v.index(citing), v.index(cited)] = 1

# 计算每个顶点的度：所有相连的顶点个数 + 1（考虑自环）
v_degree = np.sum(adj_mtrx, axis=0) + 1

# 计算度矩阵：Dij = Degree(i) + Degree(j)
for i, cited in enumerate(v):
    for j, citing in enumerate(v):
        degree_mtrx[i, j] = v_degree[i] + v_degree[j]



"""
with open('./data/cora/cora.adj_mtrx', 'w') as f:
    for index, cite in enumerate(v):
        f.write('\t'.join([str(cite)] + [str(item) for item in adj_mtrx[index]]) + '\n')
with open('./data/cora/cora.adj_mtrx_2', 'w') as f:
    for index, cite in enumerate(v):
        f.write('\t'.join([str(cite)] + [str(item) for item in adj_mtrx_2[index]]) + '\n')
with open('./data/cora/cora.degree_mtrx', 'w') as f:
    for index, cite in enumerate(v):
        f.write('\t'.join([str(cite)] + [str(item) for item in degree_mtrx[index]]) + '\n')
with open('./data/cora/cora.mult_mtrx', 'w') as f:
    for index, cite in enumerate(v):
        f.write('\t'.join([str(cite)] + [str(item) for item in mult_mtrx[index]]) + '\n')
"""
