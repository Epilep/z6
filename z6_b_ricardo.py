
import networkx as nx

def z6(z, b, soma, w):
    a = -1
    #b = 2
    c = -0.9
    #w = 0.3
    beta = 0.07
    return (a * abs(z) ** 4 + b * abs(z) ** 2 + c + w*1j) * z + beta*soma

def eqb(b, z):
    
    a = -1
    bo = 5
    c = -0.9
    w = 0.3
    bth = 2*np.sqrt(a*c)
    
    tau = 1000#2 * np.pi * 100 / 0.3
 #   zo = -0.5 * bo * (1 - np.sqrt(1 - 4 * a * c / bo ** 2)) / a
    zo = -0.5 * bth/ a
    
    return (bo * (1 - abs(z) /zo) - b) / tau
    #return 0

def rk_z6(g,dt,ac):
    
    w = [0, .5, .5, 1]
    w2 = [0,1,2,2,1]
    
    #db/dt = (b* (1 - rho / rho *) - b )/tau
    
    for i in range(4): 
        for (node) in nx.nodes(g):
            z_no = g.node[node]['z'][0] + w[i] * g.node[node]['z'][i]
            b_no = g.node[node]['b'][0] + w[i] * g.node[node]['b'][i]
            soma = 0
            for (u) in nx.all_neighbors(g, node):
                soma += g.node[u]['z'][0] + w[i] * g.node[u]['z'][i]
            
            #g.node[node]['z'][i+1] = dt * z6(z_no, g.node[node]['b'][0], ac[node][u] * soma, g.node[node]['w'])
            #g.node[node]['z'][i+1] = dt * z6(z_no, g.node[node]['b'][0], ac * soma, g.node[node]['w'])
            g.node[node]['z'][i+1] = dt * z6(z_no, g.node[node]['b'][0], soma, g.node[node]['w'])
            g.node[node]['b'][i+1] = dt * eqb(b_no, g.node[node]['z'][0])
            

    for (node) in nx.nodes(g):
        zt = g.node[node]['z']
        bt = g.node[node]['b']
        for i in range(1,5):
            zt[i] = w2[i] * zt[i] / 6
            bt[i] = w2[i] * bt[i] / 6
            
        #
        # Descomentar a linha abaixo para incluir o ruído
        #
        g.node[node]['z'][0] = sum(zt) + np.sqrt(dt) * (random.gauss(0,.1) + random.gauss(0,.1)*1j)
        g.node[node]['b'][0] = sum(bt)
    
    return g
    
import random
import numpy as np

n = 19
#g = nx.complete_graph(n)
g = nx.Graph()
g.add_node(0)
g.add_node(1)
while not nx.is_connected(g):
    g = nx.erdos_renyi_graph(n,0.5)


a = -1
bo = 5
c = -0.9
w = 0.3

tau = 100#2 * np.pi * 100 / 0.3
zo = -0.5 * bo * (1 + np.sqrt(1 - 4 * a * c / bo ** 2)) / a
print(zo)    


    
x = []
y = []
b = []
z = []
for node in nx.nodes(g):
    #fase = random.uniform(0,2*np.pi)
    #g.node[node]['z'] = [1.5* np.cos(fase) + 1.5 *np.sin(fase) * 1j, 0, 0, 0, 0]
    g.node[node]['z'] = [0 + 0*1j, 0, 0, 0, 0]
    g.node[node]['b'] = [1.9, 0, 0, 0, 0]
    g.node[node]['w'] = random.uniform(-0.9, 1.1)
    x += [[g.node[node]['z'][0].real]]
    y += [[g.node[node]['z'][0].imag]]
    b += [[g.node[node]['b'][0]]]
    z += [[abs(g.node[node]['z'][0])]]
    
dt = 0.1


# ac = [[0, 1, -1, 1], 
#       [1, 0, 1, -1],
#       [-1, 1, 0, -1],
#       [1, -1, -1, 0]]https://github.com/Epilep/z6
# ac = n * [n * [1j]]
ac = 1

for i in range(int(1000/dt)):
    g = rk_z6(g, dt, ac)
    #print(i * dt)
    for node in nx.nodes(g):
        x[node] += [g.node[node]['z'][0].real]
        y[node] += [g.node[node]['z'][0].imag]
        b[node] += [g.node[node]['b'][0]]
        z[node] += [abs(g.node[node]['z'][0])]
    

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


plt.clf()
# plt.plot(x[0],y[0],'b:')
# plt.plot(x[1],y[1],'r:')
# plt.plot(x[2],y[2],'g:')
# plt.plot(x[3],y[3],'m:')
# plt.plot(x[0],'b:')
# plt.plot(x[1],'r:')
# plt.plot(x[2],'g:')
# plt.plot(x[3],'m:')

plt.subplot(3,1,1)
plt.plot(b[0],'b:')
plt.plot(b[1],'r:')
plt.plot(b[2],'g:')
plt.plot(b[3],'m:')
#plt.ylim(-2,2)

plt.subplot(3,1,2)
plt.plot(y[0],'b:')
plt.plot(y[1],'r:')
plt.plot(y[2],'g:')
plt.plot(y[3],'m:')

plt.subplot(3,1,3)
plt.plot(z[0],'b:')
plt.plot(z[1],'r:')
plt.plot(z[2],'g:')
plt.plot(z[3],'m:')
plt.show()

