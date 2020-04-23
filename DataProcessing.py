import numpy as npy  
import pandas as pds 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataFrame_1 = pds.read_csv('Dataset_1.csv')  

grouped_sIPs = dataFrame_1.groupby('sourceIP').count()
grouped_dIPs = dataFrame_1.groupby('destIP').count()
grouped_clfs = dataFrame_1.groupby('classification').count()

#print (grouped_sIPs.shape)

distict_sIPs_count = grouped_sIPs.shape[0]
distict_dIPs_count = grouped_dIPs.shape[0]
distict_clfs_count = grouped_clfs.shape[0]

print(distict_sIPs_count)
print(distict_dIPs_count)
print(distict_clfs_count)

#############################################




def getCount(a):
    return a.size
 
x = map(lambda x: x.size, gs.values())
y = gs.keys()
print(len(y))

#y = map(getCount, gs.values())
#z = map(getCount, gs.values())

print(list(gs.keys()))

grouped_sIPs = dataFrame_1.groupby('sourceIP').count()
distict_sIPs_count = grouped_sIPs.shape[0]
gs = dataFrame_1.groupby('sourceIP').groups

frequency = list(x)

sIPs = list(gs.keys())
plt.bar(sIPs, frequency)
plt.xticks(sIPs)
plt.yticks(frequency) #This may be included or excluded as per need
plt.xlabel('sIPs')
plt.ylabel('frequency')
plt.show()



''' for value in gs.values():
    print(value)   '''


 

