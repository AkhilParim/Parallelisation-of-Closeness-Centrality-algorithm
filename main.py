import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import time
import timeit
import pymp
from multiprocessing import Pool
import numba
from numba import types
from numba.typed import Dict
from numba.typed import List



import facebook


print("imported facebook data")

startt = time.time()

#load the network
facebook.load_network()

network = facebook.network

print("graph loaded")

A = nx.adjacency_matrix(network).tolil()
D = scipy.sparse.csgraph.floyd_warshall( \
             A, directed=False, unweighted=True)

print("matrix formed")

# Number of users: 4039
# Number of connections: 88234

serialt = time.time()
n = D.shape[0]
closeness_centrality_serial = {}


for r in range(0, n):
    
    cc = 0.0
    
    possible_paths = list(enumerate(D[r, :]))
    shortest_paths = dict(filter( \
        lambda x: not x[1] == np.inf, possible_paths))
    
    total = sum(shortest_paths.values())
    n_shortest_paths = len(shortest_paths) - 1.0
    if total > 0.0 and n > 1:
        s = n_shortest_paths / (n - 1)
        cc = (n_shortest_paths / total) * s
    closeness_centrality_serial[r] = cc
    
serialend = time.time()
print("serial:  ", serialend - serialt)

# *******************************************************

pympt2 = time.time()
n = D.shape[0]
closeness_centrality_pymp = pymp.shared.dict()

with pymp.Parallel(num_threads=2) as p:
    for r in p.range(0, n):
        cc = 0.0
    
        possible_paths = list(enumerate(D[r, :]))
        shortest_paths = dict(filter( \
            lambda x: not x[1] == np.inf, possible_paths))
    
        total = sum(shortest_paths.values())
        n_shortest_paths = len(shortest_paths) - 1.0
        if total > 0.0 and n > 1:
            s = n_shortest_paths / (n - 1)
            cc = (n_shortest_paths / total) * s
        closeness_centrality_pymp[r] = cc
    
pympend = time.time()
print("pymp for 2 threads:  ", pympend - pympt2)

# *******************************************************

pympt4 = time.time()
n = D.shape[0]
closeness_centrality_pymp = pymp.shared.dict()

with pymp.Parallel(num_threads=4) as p:
    for r in p.range(0, n):
        cc = 0.0
    
        possible_paths = list(enumerate(D[r, :]))
        shortest_paths = dict(filter( \
            lambda x: not x[1] == np.inf, possible_paths))
    
        total = sum(shortest_paths.values())
        n_shortest_paths = len(shortest_paths) - 1.0
        if total > 0.0 and n > 1:
            s = n_shortest_paths / (n - 1)
            cc = (n_shortest_paths / total) * s
        closeness_centrality_pymp[r] = cc
    
pympend = time.time()
print("pymp for 4 threads:  ", pympend - pympt4)

# *******************************************************

pympt8 = time.time()
n = D.shape[0]
closeness_centrality_pymp = pymp.shared.dict()

with pymp.Parallel(num_threads=8) as p:
    for r in p.range(0, n):
        cc = 0.0
    
        possible_paths = list(enumerate(D[r, :]))
        shortest_paths = dict(filter( \
            lambda x: not x[1] == np.inf, possible_paths))
    
        total = sum(shortest_paths.values())
        n_shortest_paths = len(shortest_paths) - 1.0
        if total > 0.0 and n > 1:
            s = n_shortest_paths / (n - 1)
            cc = (n_shortest_paths / total) * s
        closeness_centrality_pymp[r] = cc
    
pympend = time.time()
print("pymp for 8 threads:  ", pympend - pympt8)

# *******************************************************

closeness_centrality_pool = {}
def close(D):
  n = D.shape[0]
  global closeness_centrality
  for r in range(0, n):
    
    cc = 0.0
    
    possible_paths = list(enumerate(D[r, :]))
    shortest_paths = dict(filter( \
        lambda x: not x[1] == np.inf, possible_paths))
    
    total = sum(shortest_paths.values())
    n_shortest_paths = len(shortest_paths) - 1.0
    if total > 0.0 and n > 1:
        s = n_shortest_paths / (n - 1)
        cc = (n_shortest_paths / total) * s
    closeness_centrality_pool[r] = cc
  return closeness_centrality_pool

# *******************************************************

poolt2 = time.time()
pool = Pool(processes=4)
mat = [D[(D.shape[0]*i)//2:(D.shape[0] * (i+1))//2, :] for i in range(2)]
res = pool.map(close, mat)
# print(np.array(res))
poolend = time.time()
print("pool2:  ", poolend - poolt2)

# *******************************************************

poolt4 = time.time()
pool = Pool(processes=4)
mat = [D[(D.shape[0]*i)//2:(D.shape[0] * (i+1))//2, :] for i in range(4)]
res = pool.map(close, mat)
# print(np.array(res))
poolend = time.time()
print("pool4:  ", poolend - poolt4)

# *******************************************************

poolt8 = time.time()
pool = Pool(processes=8)
mat = [D[(D.shape[0]*i)//2:(D.shape[0] * (i+1))//2, :] for i in range(8)]
res = pool.map(close, mat)
# print(np.array(res))
poolend = time.time()
print("pool8:  ", poolend - poolt8)

# *******************************************************


print("Total time:", poolend - startt)
