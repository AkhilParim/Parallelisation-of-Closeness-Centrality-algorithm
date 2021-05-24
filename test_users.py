import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import time

n_users = 4000
n_relationships = 88000

# G = nx.barabasi_albert_graph(n_users, \
#                              n_relationships, \
#                              seed=11)

G = nx.dense_gnm_random_graph(n_users, \
                             n_relationships, \
                             seed=11)

G.add_node(11)

print("graph formed")

# %%timeit
A = nx.adjacency_matrix(G).tolil()
D = scipy.sparse.csgraph.floyd_warshall( \
             A, directed=False, unweighted=True)
