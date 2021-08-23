import torch
import numpy as np

name="myTest"
TEST_ACCURACY_DATA_PATH='../datasets/testAccuracy/'
'''
a=np.array([[1,-2],[1,2],[4,5],[5,5]])
c=np.array([[a[0][1],a[0][0]]])
b=c.copy()
for i in range(a.shape[0]):
    print("k")
    b=(np.concatenate((c,b),axis=0)).copy()
print(b)
'''




#b=-100*np.ones((a.shape[0],1))
#rint(b)
#c=np.concatenate((a,b),axis=1)
#print(c.shape)

# 得到表面点坐标及其法向
testData=[]
f = open(f'{TEST_ACCURACY_DATA_PATH}{name}.txt', 'w')
line = f.readline()
while line:
    x, y, sdfx, sdfy = map(lambda n: float(n), line.strip('\n').split(' '))
    testData.append([x, y, sdfx, sdfy])
    line = f.readline()
f.close()

xy=[]
sdfxy=[]



# 得到