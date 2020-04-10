import numpy as npy  
import pandas as pds 

dataFrame_1 = pds.read_csv('Dataset_1.csv')  
 
print (dataFrame_1.groupby('sourceIP').groups)