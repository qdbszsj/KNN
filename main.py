import numpy as np
import pandas as pd
dataset=pd.read_csv('/home/parker/watermelonData/watermelon3_0a.csv', delimiter=",")
data=dataset.values
X=data[:,1:3]
y=data[:,-1]
# print(y)

# def KNN(iX,iY,K):
#     m,n=np.shape(iX)
#     for i in range(m):
#         dist=(iX-iX[i])**2
#         for j in range(m):
#             dist[j,0]=sum(dist[j])
#             dist[j,1]=j
#         dist=sorted(dist, key=lambda dist: dist[0])
#         # for j in range(K):
#         print(dist)

import random
def KNNone(x,iX,iY,K):
    m,n=np.shape(iX)
    dist = (iX - x) ** 2
    for j in range(m):
        dist[j, 0] = sum(dist[j])
        dist[j, 1] = j
    dist = sorted(dist, key=lambda dist: dist[0])
    cnt=[0,0]
    for j in range(K):
        if iY[int(dist[j][1])]==0:cnt[0]+=1
        else:cnt[1]+=1
    if cnt[0]==cnt[1]:return random.randint(0,1)
    if cnt[0]>cnt[1]:return 0
    else:return 1

def showConfusionMatrix(trueY,myY):
    confusionMatrix = np.zeros((2, 2))
    for i in range(len(trueY)):
        if myY[i] == trueY[i]:
            if trueY[i] == 0:
                confusionMatrix[0, 0] += 1
            else:
                confusionMatrix[1, 1] += 1
        else:
            if trueY[i] == 0:
                confusionMatrix[0, 1] += 1
            else:
                confusionMatrix[1, 0] += 1
    print(trueY)
    print(myY)
    print(confusionMatrix)

m,n=np.shape(X)
predictY=np.zeros(m)
for i in range(m):
    curlist=[x for x in range(0,i)]
    # print(curlist)
    curlist.extend([x for x in range(i+1,m)])
    predictY[i]=KNNone(X[i],X[curlist],y[curlist],3)


showConfusionMatrix(y,predictY)