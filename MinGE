import numpy as np
import math
import time

start = time.time()
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


v = list(v)
N = len(v)
adj_mtrx = np.eye(len(v), dtype='int32')

adj_mtrx_2 = np.eye(len(v), dtype='float32')
mult_mtrx = np.eye(len(v), dtype='float32')

degree_mtrx = np.empty((len(v), len(v)), dtype='int32')


for cited in adj_list.keys():
    for citing in adj_list[cited]:
        adj_mtrx[v.index(cited), v.index(citing)] = 1
        adj_mtrx[v.index(citing), v.index(cited)] = 1

v_degree1 = np.sum(adj_mtrx, axis=0) + 1

#z = np.sum(v_degree1)
#p = v_degree1/z
#print(np.max(p))
#logp = np.log(p)
#H = -np.dot(p, logp)
#print(H)

#print(np.average(v_degree1))
#print(np.average(v_degree1 * np.log(v_degree1)))
#print(np.log(np.average(v_degree1))-np.average(v_degree1 * np.log(v_degree1)/np.average(v_degree1)))
v_degree = np.sum(adj_mtrx, axis=0) + 1

adj_mtrx_2 = adj_mtrx.astype('float32').dot(adj_mtrx.astype('float32'))
#for i in range(len(v)):
 #   adj_mtrx_2[i + 1:, i] /= adj_mtrx_2[i, i]
  #  adj_mtrx_2[i, i + 1:] /= adj_mtrx_2[i, i]
   # adj_mtrx_2[i, i] = 1.0

t = np.sum(adj_mtrx_2, axis=1)
print(t)
for i in range(N):
    mult_mtrx[i,:] = adj_mtrx_2[i,:]/t[i]

v_degree_n = np.dot(mult_mtrx, v_degree1)

print(v_degree_n)

z = np.sum(v_degree_n)
p = v_degree_n/z
print(np.max(p))
logp = np.log(p)
H = -np.dot(p, logp)
print(H)


#mult_mtrx = adj_mtrx_2 * degree_mtrx

#Z = np.sum(mult_mtrx)
#P = mult_mtrx/Z
#Logp = np.log(P)
#print(np.sum(-np.dot(P, Logp)))

n=(np.log(N * N)+H)/0.24
end = time.time()
print(n)
print('runing time: %s Second'%(end-start))

#with open('./data/cora/cora.adj_mtrx', 'w') as f:
#    for index, cite in enumerate(v):
#        f.write('\t'.join([str(cite)] + [str(item) for item in adj_mtrx[index]]) + '\n')
#with open('./data/cora/cora.adj_mtrx_2', 'w') as f:
 #   for index, cite in enumerate(v):
  #      f.write('\t'.join([str(cite)] + [str(item) for item in adj_mtrx_2[index]]) + '\n')
#with open('./data/cora/cora.degree_mtrx', 'w') as f:
 #   for index, cite in enumerate(v):
  #      f.write('\t'.join([str(cite)] + [str(item) for item in degree_mtrx[index]]) + '\n')
#with open('./data/cora/cora.mult_mtrx', 'w') as f:
 #   for index, cite in enumerate(v):
  #      f.write('\t'.join([str(cite)] + [str(item) for item in mult_mtrx[index]]) + '\n')
