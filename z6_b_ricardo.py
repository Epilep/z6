
# coding: utf-8

# In[1]:


import networkx as nx


# In[2]:


def z6(z, b, soma, w):
    a = -1
    #b = 2
    c = -0.9
    #w = 0.3
    beta = 0.07
    return (a * abs(z) ** 2 + b * abs(z) + c + w*1j) * z + beta*soma


# In[71]:


def eqb(b, z):
    
    a = -1
    bo = 2
    c = -0.9
    w = 0.3
    
    tau = 100#2 * np.pi * 100 / 0.3
    zo = -0.5 * bo * (1 + np.sqrt(1 - 4 * a * c / bo ** 2)) / a
    
    return bo * ((1 - abs(z) /zo) - b) / tau
    #return 0


# In[72]:


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
        g.node[node]['z'][0] = sum(zt) #+ np.sqrt(dt) * (random.gauss(0,1) + random.gauss(0,1)*1j)
        g.node[node]['b'][0] = sum(bt)
    
    return g
    


# In[73]:


import random
import numpy as np

n = 19
#g = nx.complete_graph(n)
g = nx.Graph()
g.add_node(0)
g.add_node(1)
while not nx.is_connected(g):
    g = nx.erdos_renyi_graph(n,0.5)


x = []
y = []
for node in nx.nodes(g):
    fase = random.uniform(0,2*np.pi)
    g.node[node]['z'] = [1.5* np.cos(fase) + 1.5 *np.sin(fase) * 1j, 0, 0, 0, 0]
    #g.node[node]['z'] = [0 + 0*1j, 0, 0, 0, 0]
    g.node[node]['b'] = [2, 0, 0, 0, 0]
    g.node[node]['w'] = random.uniform(-0.2, 0.2)
    x += [[g.node[node]['z'][0].real]]
    y += [[g.node[node]['z'][0].imag]]
    
dt = 0.01

# ac = [[0, 1, -1, 1], 
#       [1, 0, 1, -1],
#       [-1, 1, 0, -1],
#       [1, -1, -1, 0]]https://github.com/Epilep/z6
# ac = n * [n * [1j]]
ac = 1

for i in range(int(500/dt)):
    g = rk_z6(g, dt, ac)
    for node in nx.nodes(g):
        x[node] += [g.node[node]['z'][0].real]
        y[node] += [g.node[node]['z'][0].imag]
    
#         y[node] += [g.node[node]['zn'].imag]
#     for (node) in nx.nodes(g):
#         z_no = g.node[node]['z']
#         soma = 0
#         for (u) in nx.all_neighbors(g, node):
#             soma += g.node[u]['z']
#         g.node[node]['zn'] = rk_z6(z_no, soma, dt)
#         x[node] += [g.node[node]['zn'].real]
#         y[node] += [g.node[node]['zn'].imag]
        
#     for (node) in nx.nodes(g):
#         g.node[node]['z'] = g.node[node]['zn']


# In[74]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

get_ipython().run_line_magic('matplotlib', 'notebook')

fig, ax = plt.subplots()
points = []

#points, = ax.plot(n*[],n*[], 'bo')

for i in range(n):
    point, = ax.plot([],[], 'bo')
    points += [ point ]

# point0, = ax.plot([],[], 'bo')
# point1, = ax.plot([],[], 'ro')
# point2, = ax.plot([],[], 'go')
# point3, = ax.plot([],[], 'mo')

plt.xlim(-3,3)
plt.ylim(-3,3)

def init():
    for i in range(n):
        points[i].set_data([], [])
#     point0.set_data([], [])
#     point1.set_data([], [])
#     point2.set_data([], [])
#     point3.set_data([], [])
    #return points,

def animate(t):
    t = t * 1000
    for i in range(n):
        points[i].set_data([x[i][t]], [y[i][t]])
#     point0.set_data([x[0][i]], [y[0][i]])
#     point1.set_data([x[1][i]], [y[1][i]])
#     point2.set_data([x[2][i]], [y[2][i]])
#     point3.set_data([x[3][i]], [y[3][i]])
    
    
anim = animation.FuncAnimation(fig, animate, frames=int(len(x[0]) / 1000), init_func=init, interval=1, blit=True)
#anim = animation.FuncAnimation(fig, animate, frames=10, init_func=init, interval=1, blit=True)


# ani = animation.FuncAnimation(fig, update, generate_points, interval=300)
#ani.save('animation.gif', writer='imagemagick', fps=24);

# mywriter = animation.FFMpegWriter()
anim.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")

plt.show()


# plt.plot(x[0],y[0])
# plt.plot(x[1],y[1],'r-.')
# plt.plot(x[2],y[2],'g:',)
# plt.show()


# In[75]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.clf()
# plt.plot(x[0],y[0],'b:')
# plt.plot(x[1],y[1],'r:')
# plt.plot(x[2],y[2],'g:')
# plt.plot(x[3],y[3],'m:')
# plt.plot(x[0],'b:')
# plt.plot(x[1],'r:')
# plt.plot(x[2],'g:')
# plt.plot(x[3],'m:')
plt.plot(y[0],'b:')
plt.plot(y[1],'r:')
plt.plot(y[2],'g:')
plt.plot(y[3],'m:')
plt.ylim(-2,2)
plt.show()


# In[76]:


g.node[0]['b']


# In[18]:


x = []
y = []
for i in range(10000):
    x += [random.gauss(0,1)]
    y += [random.gauss(0,1)]
    
plt.plot(x,y,'b.', alpha=0.1)
plt.show()

