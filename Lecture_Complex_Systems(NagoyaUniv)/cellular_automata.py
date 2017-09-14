import matplotlib.pyplot as plt
import numpy as np
import random as rnd
def ca_1d(l,t,rule,cell_i):
    cell=cell_i
    data=[cell]
    for i in range(t):
        cell_next=[0 for i in range(l)]
        for j in range(l):
          neighboringstate=cell[(j-1+l)%l]*4+cell[j]*2+cell[(j+1)%l]
          cell_next[j]=rule[neighboringstate]
        cell=cell_next
        data.append(cell)
    return(data)

L=101
T=100
SEED=100
rnd.seed(SEED)
RNO=90
RULE=[(RNO>>i)&1for i in range(8)]
#[0,0,...,0,1,0,...,0,0]
cell_init=[0 for i in range(L)]
cell_init[L//2]=1
#random
#cell_init=[rnd.randint(0,1) for i in range(L)]

dataXY=ca_1d(L,T,RULE,cell_init)
fig=plt.figure(figsize=(5,6))
ax=fig.add_subplot(1,1,1)
ax.pcolor(np.array(dataXY),vmin=0,vmax=1,cmap=plt.cm.binary)
ax.set_xlim(0,L)
ax.set_ylim(T-1,0)
ax.set_xlabel("cellnumber")
ax.set_ylabel("step")
ax.set_title("rule"+str(RNO))
plt.show()
