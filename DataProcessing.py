import numpy as npy  
import pandas as pds 

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


gs = dataFrame_1.groupby('classification').groups




for value in gs.values():
    print(value.size)
    


 

