import os
import networkx as nx
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np

dirpath = os.path.dirname(os.path.realpath(__file__))
links = {}
fnames = ["angelinajolie.html", "bradpitt.html", "jenniferaniston.html",
          "jonvoight.html","martinscorcese.html", "robertdeniro.html"]
for filename in fnames:
    filepath = os.path.join(dirpath,"pages",filename)
    links[filename] = []
    f = open(filepath)
    for line in f.readlines():
        #print line
        #print "------------------------"
        while True:
            p = line.partition('<a href="http://')[2]
            if p == '':
                break;
            url,_,line =  p.partition('\">')
            links[filename].append(url)
    f.close()

DG = nx.DiGraph()
DG.add_nodes_from(fnames)

edges = []
#print links
for key,values in links.iteritems():
    edgeweight = {}
    for v in values:
        if v in edgeweight:
            edgeweight[v] += 1
        else:
            edgeweight[v] = 1
    for succ,weight in edgeweight.iteritems():
        edges.append([key, succ, {'weight': weight}])
DG.add_edges_from(edges)

#plt.figure(figsize=(9,9))
#pos = nx.spring_layout(DG, iterations=10)
#nx.draw(DG, pos, node_size=0, alpha=0.4, edge_color='r', font_size=16)
#plt.savefig("link_graph.png")
#plt.show()

#write out the graph using pickle
pickle.dump(DG, open('DG.pkl', 'w'))

#reload the graph using pickle
DG = pickle.load(open('DG.pkl'))

N = len(fnames)
T = np.zeros((N, N))

#print T

f2i = dict((fn, i) for i, fn in enumerate(fnames))

#print f2i

for pred, succ in DG.adj.iteritems():
    for s, edata in succ.iteritems():
        T[f2i[pred], f2i[s]] =  edata['weight']

#print T
epsilon = 0.01

E = np.ones(T.shape)
E = E/N
E = epsilon * E

L = T+E
G = L.copy()
S = np.sum(L, axis=1)
S = S.reshape(-1,1)
G = G/S

R = np.random.random(N)
R = R/np.sum(R)
PI = R.copy()
print R
for x in xrange(1000):
    R = np.dot(R,G)

evolution = []
for i in xrange(20):
    PI = np.dot(PI, G)
    evolution.append(PI)
#print evolution
plt.figure()
for i in xrange(N):
    plt.plot([step[i] for step in evolution], label=fnames[i], lw=2)
plt.draw()
plt.xlabel('iterations')
plt.ylabel('rank')
plt.legend()
plt.show()
print R
