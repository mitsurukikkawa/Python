import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rnd

# set parameters
N=100
T=100
W=30
SEED=101
agents=[]

## define functions
def clip(x):
    if x<0:
        return(x+W)
    elif x>=W:
        return(x-W)
    else:
        return(x)

## define classes
class Agent(object):

 def __init__(self):
     self.x=rnd.randint(0,W-1)
     self.y=rnd.randint(0,W-1)

 def randomwalk(self):
     self.x+=rnd.randint(-1,1)
     self.y+=rnd.randint(-1,1)
     self.x=clip(self.x)
     self.y=clip(self.y)

# initialize variables
rnd.seed(SEED)
agents=[Agent()for i in range(N)]
fig = plt.figure()

#main loop (call back function for animation)
def main_loop(t):
    step()
    update(t)

# events in a step
def step():
    rnd.shuffle(agents)
    for a in agents:
        a.randomwalk()
# update function for graph
def update(t):
    fig.clear()
    ax1=fig.add_subplot(2,2,1)
    x=[a.x for a in agents]
    y=[a.y for a in agents]
    ax1.scatter(x,y,color='brown')
    ax1.axis([-1,W,-1,W])
    ax1.set_title('t='+str(t))

    ax2=fig.add_subplot(2,2,3)
    ax2.hist(x,W)
    ax2.axis([-1,W,-1,W])
    
    ax3=fig.add_subplot(2,2,4)
    ax3.hist(y,W)
    ax3.axis([-1,W,-1,W])

ani=animation.FuncAnimation(fig,main_loop,np.arange(0,T),interval=25,repeat=False)
plt.show()



