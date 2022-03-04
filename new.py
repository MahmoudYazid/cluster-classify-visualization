import sklearn as sk
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mat
from sklearn.cluster import KMeans
# discover best one by plotting observation

recent=[]
saved=[]
def save(x):
    for x in recent:
        saved.append(x)
        recent.clear()
  
    return 0

csv= pd.read_csv("C:\\Users\\ahmed\\Desktop\\pyprojects\\CC GENERAL.csv")

for first in range(1,17):
    for second in range(1, 17):
        y = pd.DataFrame(csv["{}".format(csv.keys()[first])]).to_numpy()
        x = pd.DataFrame(csv["{}".format(csv.keys()[second])]).to_numpy()
        plt.scatter(x,y)
        plt.xlabel(csv.keys()[first])
        plt.ylabel(csv.keys()[second])
        recent.append([csv.keys()[first], csv.keys()[second]])
        
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        wd = mat.widgets.Button(axprev, "save")
        wd.on_clicked(
            
            save
            
            )
        plt.show()
        
print(saved)

"""
# clustering
results=[]
pred = 0.545455
selected_arrays=[
    ['BALANCE_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY'], 
    ['BALANCE_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY'], 
    ['BALANCE_FREQUENCY', 'CASH_ADVANCE_FREQUENCY'], 
    
  
 
    ['PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY'], 
    ['PURCHASES_FREQUENCY', 'CASH_ADVANCE_FREQUENCY'], 
  

    ['ONEOFF_PURCHASES_FREQUENCY', 'CASH_ADVANCE_FREQUENCY'], 
   
  
    ['PURCHASES_INSTALLMENTS_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY'], 
   

    ['CASH_ADVANCE_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY'], 
    ['CASH_ADVANCE_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY'], 
    

    ['PRC_FULL_PAYMENT', 'ONEOFF_PURCHASES_FREQUENCY'], 
    ['PRC_FULL_PAYMENT', 'PURCHASES_INSTALLMENTS_FREQUENCY'], 
    ['PRC_FULL_PAYMENT', 'CASH_ADVANCE_FREQUENCY']]
csv = pd.read_csv("C:\\Users\\ahmed\\Desktop\\pyprojects\\CC GENERAL.csv")
test_file = pd.read_csv("C:\\Users\\ahmed\\Desktop\\pyprojects\\test_final.csv")
test = [152.225975	,0.545455	,1281.6	,1281.6	,0	,0	,0.166667	,0.166667	,0	,0	,0	,3	,11000	,1164.770591	,100.302262	,0	,12]

for itr in range(0,len(selected_arrays)):
    data = pd.DataFrame(
        csv[["{}".format(selected_arrays[itr][0]), "{}".format(selected_arrays[itr][1])]]).to_numpy()




    kmeans = KMeans(n_clusters=2).fit(data)
    kmeans.fit(data)
    lkk = kmeans.predict(
        [[int(test_file['{}'.format(selected_arrays[itr][0])]), int(test_file['{}'.format(selected_arrays[itr][1])]) ]])
    print(lkk)
    plt.scatter(csv["{}".format(selected_arrays[itr][0])], csv["{}".format(
        selected_arrays[itr][1])])
    
    plt.xlabel("{}".format(selected_arrays[itr][0]))
    plt.ylabel("{}".format(selected_arrays[itr][1]))
    
    plt.show()
    results.append(lkk)
"""

# i tried to separate the bulk info. and find a squances by itration process for all factors with each others that led to shapes [window or half of window] that is good distribution for information to classify