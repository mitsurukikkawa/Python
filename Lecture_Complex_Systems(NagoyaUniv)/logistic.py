import matplotlib.pyplot as plt
Nt=1.0
r=1.5
K=100
T=30
dataX=[]
dataY=[]
print("0"+"\t"+str(Nt))
for i in range(T):
    Nt=Nt+r*Nt*(1.0-Nt/K)
    dataX.append(i+1)
    dataY.append(Nt)
    #print(str(i+1)+"\t"+str(Nt))
plt.plot(dataX,dataY)
plt.xlabel("time")
plt.ylabel("population size")
plt.show()
