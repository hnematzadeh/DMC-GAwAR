import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, StandardScaler
from sklearn import tree,svm
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from collections import Counter, defaultdict
import random
import math
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, auc,balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
from time import time
from sklearn.metrics.cluster import normalized_mutual_info_score
import random
from sklearn import metrics 
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")



# Colon; labels are 0 or 1
# df = pd.read_excel ('...\Colon.xlsx', header=None)
# df.iloc[:,df.shape[1]-1].replace({'Normal':1, 'Tumor':2},inplace=True)

# CNS; labels are 1 or 2
# df = pd.read_excel ('...\CNS.xlsx', header=None)
# df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# Leukemia-2c; labels are 1 or 2
# df = pd.read_excel ('...\Leukemia.xlsx', header=None)
#
#SMK
# df = pd.read_csv ('...\SMK.csv', header=None)
# df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# GLI
# df = pd.read_csv ('...\GLI.csv', header=None)
# df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# Covid-2c
# df = pd.read_csv ('...\Covid.csv', header=None)
# df.iloc[:,df.shape[1]-1].replace({'no virus':1, 'other virus':1, 'SC2':2},inplace=True)

# Covid-3c
# df = pd.read_csv ('...\Covid.csv', header=None)
# df.iloc[:,df.shape[1]-1].replace({'no virus':1, 'other virus':2, 'SC2':3},inplace=True)


#Leukemia-3c
# df = pd.read_excel ('...\Leukemia_3c.xlsx', header=None)

#MLL-3c
# df = pd.read_excel ('...\MLL.xlsx', header=None)

#SRBCT-4c
# df = pd.read_excel ('...\SRBCT.xlsx', header=None)


X=df.iloc[:,0:df.shape[1]-1]
# X=pd.DataFrame(scale(X))
y=df.iloc[:,df.shape[1]-1]

##### Calculating  quantity of each label in a
labels=np.unique(y)
a = {}

c=1
for i in range (len(labels)):
    # dynamically create key
    key = c
    # calculate value
    value = sum(y==labels[i])
    a[key] = value 
    
    c +=1

#####################################################
# ============= Just to use Mutual Congestion and DMC ===============

ones=sum(df[df.shape[1]-1]==1)
twos=sum(df[df.shape[1]-1]==2)



#============= Without FS==================
limit=5
zz=alpha.argsort()
X_selected = zz[:int(limit)]
X_selected=X[X_selected]

precision=np.zeros(100)
recall=np.zeros(100)
f1=np.zeros(100)
s=np.zeros(100)
mcc=np.zeros(100)

for i in range(100):
  X_train, X_test, y_train, y_test = train_test_split(X_selected,y, stratify=y, test_size=0.2)

  dectree = tree.DecisionTreeClassifier()
  dectree.fit(X_train,y_train)
  s[i]=dectree.score(X_test,y_test)
  
  
  # cm=confusion_matrix(y_test,dectree.predict(X_test))
  precision[i] = metrics.precision_score(y_test, dectree.predict(X_test))
  recall[i] = metrics.recall_score(y_test, dectree.predict(X_test))
  f1[i] = metrics.f1_score(y_test, dectree.predict(X_test))
  mcc[i] = metrics.matthews_corrcoef(y_test, dectree.predict(X_test)) 
  
 
print('acc   ',  np.mean(s))
print('pre   ',np.mean(precision))
print('rec   ',np.mean(recall)) 
print('fscore   ',np.mean(f1) )
print('mcc   ',np.mean(mcc))



###################Distance-based Mutual Congestion#########################

alpha=np.zeros(df.shape[1]-1)
sorted_alpha=np.zeros(df.shape[1]-1)
for i in range(df.shape[1]-1):
    print(i) 
    newdf=df.sort_values(i)
    # if labels start with 1, find the location of the first place!='1'
    if newdf.iloc[0,df.shape[1]-1]==1:
      ymin=df.iloc[newdf[newdf.shape[1]-1].ne(1).idxmax(),i]
      first=newdf.index.get_loc(newdf[newdf.shape[1]-1].ne(1).idxmax())
#      co=1
      # co=first
      s1=0
      for j in range(first+1,newdf.shape[0]):

        if newdf.iloc[j,newdf.shape[1]-1]==1:
          s1=(abs(df.iloc[newdf.index[j],i]-ymin))+s1
      s2=0    
      for j in range(first):

          s2=(abs(df.iloc[newdf.index[j],i]-ymin))+s2   
      if s2==0:
          s2=1
      A=s1/s2    
      
      co=0
      for j in range(newdf.shape[0]):
        if newdf.iloc[j,newdf.shape[1]-1]==1:
          co=co+1
        if co==ones:
          last=j
          break
      xmax=df.iloc[newdf.index[last],i]
      s1=0 
      for j in range(first+1,last+1):
        
         if newdf.iloc[j,newdf.shape[1]-1]==2:
            s1=(abs(df.iloc[newdf.index[j],i]-xmax))+s1
      s2=0  
      for j in range(last+1, newdf.shape[0]):

          s2=(abs(df.iloc[newdf.index[j],i]-xmax))+s2   
          
      if s2==0:
          s2=1    
      B=s1/s2
      alpha[i]=A+B
      
    else:
        
      ymin=df.iloc[newdf[newdf.shape[1]-1].ne(2).idxmax(),i]
      first=newdf.index.get_loc(newdf[newdf.shape[1]-1].ne(2).idxmax())
#      co=1
      # co=first
      s1=0
      for j in range(first+1,newdf.shape[0]):

        if newdf.iloc[j,newdf.shape[1]-1]==2:
          s1=(abs(df.iloc[newdf.index[j],i]-ymin))+s1
      s2=0    
      for j in range(first):

          s2=(abs(df.iloc[newdf.index[j],i]-ymin))+s2   
          
      if s2==0:
          s2=1    
      A=s1/s2    
      
      co=0
      for j in range(newdf.shape[0]):
        if newdf.iloc[j,newdf.shape[1]-1]==2:
          co=co+1
        if co==twos:
          last=j
          break
      xmax=df.iloc[newdf.index[last],i]
      s1=0 
      for j in range(first+1,last+1):
        
         if newdf.iloc[j,newdf.shape[1]-1]==1:
            s1=(abs(df.iloc[newdf.index[j],i]-xmax))+s1
      s2=0  
      for j in range(last+1, newdf.shape[0]):

          s2=(abs(df.iloc[newdf.index[j],i]-xmax))+s2  
          
      if s2==0:
          s2=1    
      B=s1/s2
      alpha[i]=A+B
        
      
# limit=10    

######################################################
############# Mutual Congestion (MC) #########################
alpha=np.zeros(df.shape[1]-1)
sorted_alpha=np.zeros(df.shape[1]-1)
for i in range(df.shape[1]-1):
    print(i) 
    newdf=df.sort_values(i)
    # if labels start with 1, find the location of the first place!='1'
    if newdf.iloc[0,df.shape[1]-1]==1:
      first=newdf.index.get_loc(newdf[newdf.shape[1]-1].ne(1).idxmax())
#      co=1
      co=first
      for j in range(first,newdf.shape[0]):
      
        if newdf.iloc[j,newdf.shape[1]-1]==1:
          co=co+1
        if co==ones:
          last=j
          break
      alpha[i]=(last-first)/(df.shape[0])
    else:
        first=newdf.index.get_loc(newdf[newdf.shape[1]-1].ne(2).idxmax())
#        co=1
        co=first
        for j in range(first,newdf.shape[0]):
      
          if newdf.iloc[j,newdf.shape[1]-1]==2:
            co=co+1
          if co==twos:
            last=j
            break
        alpha[i]=(last-first)/(df.shape[0])


##############  Genetic Algorithm #####################

import copy
def fitness(Xn, yn):
    global NFE
    s=np.zeros(10)
    precision=np.zeros(10)
    recall=np.zeros(10)
    f1=np.zeros(10)
    mcc=np.zeros(10)
    balanced_acc=np.zeros(10)
    for i in range(10):
      X_train, X_test, y_train, y_test = train_test_split(Xn,yn, stratify=y, test_size=0.2)

      dectree = tree.DecisionTreeClassifier()
      dectree.fit(X_train,y_train)
      s[i]=dectree.score(X_test,y_test)
      y_pred = dectree.predict(X_test)
      precision[i] = metrics.precision_score(y_test, y_pred)
      recall[i] = metrics.recall_score(y_test, y_pred)
      f1[i] = metrics.f1_score(y_test, y_pred)
      mcc[i] = metrics.matthews_corrcoef(y_test, dectree.predict(X_test)) 
      balanced_acc[i] = balanced_accuracy_score(y_test, dectree.predict(X_test))
      
      
        
     
    NFE=NFE+1  

    return np.mean(s), np.mean(precision), np.mean(recall), np.mean(f1), np.mean(mcc)




def SinglePointCrossover(x1,x2):
    import random
    import numpy as np
    nVar=len(x1)
    C=random.randint(1,nVar-1)
    y1=(x1[0:C]).tolist() + (x2[C:]).tolist()
    y2=(x2[0:C]).tolist() + (x1[C:]).tolist()
    return y1,y2

def Mutate(x,seq):
    import random
    import numpy as np
    random_number = random.choice(seq)   
    nVar=len(x)
    J=random.randint(0,nVar-1)
    y=copy.deepcopy(x)
    y[J]=random_number
    return y



def Mutate2(x,seq):
    import random
    import numpy as np
    nVar=len(x)
    
    random_number = random.choice(seq)   
    J=random.randint(0,nVar-1)
    J1=random.randint(0,nVar-1)
    J2=random.randint(0,nVar-1)

    y=copy.deepcopy(x)
    y[J]=random_number
    random_number = random.choice(seq)   
    y[J1]=random_number
    random_number = random.choice(seq)   
    y[J2]=random_number
    return y



def RouletteWheelSelection(P):
    r=random.uniform(0,1)
    c=np.cumsum(P)
    i=np.where(r<np.array(c))[0][0]
    return i


#####################################

limit=0.05*(df.shape[1]-1)
zz=alpha.argsort()
search_space = zz[:int(limit)]
# search_space = zz[:50]







transposed_data = X[search_space].T  # T stands for transpose

# Number of clusters
num_clusters = 100

# Apply K-means clustering to the transposed dataset
kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # Adjust random_state as needed
kmeans.fit(transposed_data)

# Get the cluster labels for each feature
feature_clusters = kmeans.labels_

# Create a DataFrame to store feature names and their corresponding cluster labels
feature_cluster_df = pd.DataFrame({'Feature': transposed_data.index, 'Cluster': feature_clusters})

# Randomly select one feature from each cluster
selected_features = []
for cluster_id in range(num_clusters):
    # Get features belonging to the current cluster
    cluster_features = feature_cluster_df[feature_cluster_df['Cluster'] == cluster_id]['Feature'].values
    if len(cluster_features) > 0:
        # Randomly select one feature from the current cluster
        selected_feature = np.random.choice(cluster_features, size=1, replace=False)[0]
        selected_features.append(selected_feature)

# Print or use the selected features
print("Selected features from each cluster:", selected_features)
search_space=selected_features


## random selection from entire dataset
# search_space =  list(range(X.shape[1])) # Total number of features in your dataset
# 


pcc=0.9
pmm=0.4
Adaptive=0
sp=12
nPop=20
nVar=10
pc=0.9;
nc=2*round(pc*nPop/2)
pm=0.4;
nm=round(pm*nPop)
NFE=0
MaxIt=150
MutRate=np.zeros(MaxIt)
MutRate[0]=pm

CrossRate=np.zeros(MaxIt)
CrossRate[0]=pc
BestPosition=0
from ypstruct import struct
empty_individual=struct(position=None,  List=None, fit=None, precision=None, recall=None, fmeasure=None, mcc=None)
pop=empty_individual.repeat(nPop)

Fits=np.zeros(nPop)
it=0
for i in range (nPop):
       pop[i].List= np.random.choice(search_space, size=nVar, replace=False)
       pop[i].position=X[pop[i].List]
       pop[i].fit,pop[i].precision, pop[i].recall, pop[i].fmeasure,pop[i].mcc= fitness(pop[i].position,y)  
       Fits[i]=pop[i].fit
       
       
    
P=np.zeros(nPop)     
for j in range (nPop):
     P[j]=Fits[j]/sum(Fits)        




##### Sort population
    
import operator
pop=sorted(pop,key=operator.attrgetter('fit'), reverse=True)
    
for i in range (nPop):
   print(pop[i].position, "  ",pop[i].List, "  ",pop[i].fit)
   
   
   
### store best solutions in each iteration
BestSol=pop[0]
BestFits=np.zeros(MaxIt)
BestFits[it]=BestSol.fit
BestAcc=BestSol.fit
##store worst fit
WorstFit=pop[nPop-1].fit


### array to hold best values in all iterations


#### array to hold NFEs
nfe=np.zeros(MaxIt)   
nfe[it]=nPop
print("Iteration ", str(it) ,": Best fit = ",  BestAcc,  "NFE =  ", nfe[it])

### Main Loop
import random
import math
# for it in range (MaxIt):
it=it+1
Tag=1
TagCheck=30
AdaptCheck=6
ATag=1     
while (it<MaxIt and  Tag!=TagCheck):
        
    
    popc1=empty_individual.repeat(int(nc/2))
    popc2=empty_individual.repeat(int(nc/2))
    Xover=list(zip(popc1,popc2))
    for k in range (int(nc/2)):
             
             
             # Select First Parent
             i1=RouletteWheelSelection(P)
             # i1=random.randint(0,nPop-1)
             p1=pop[i1].List
             # Select Second Parent
             i2=RouletteWheelSelection(P)
             # i2=random.randint(0,nPop-1)
             p2=pop[i2].List
             #Apply Crossover
             Xover[k][0].List,Xover[k][1].List=np.array(SinglePointCrossover(p1,p2))
             Xover[k][0].position=X[Xover[k][0].List]
             Xover[k][1].position=X[Xover[k][1].List]
             #Evaluate Offspring
             Xover[k][0].fit,Xover[k][0].precision, Xover[k][0].recall, Xover[k][0].fmeasure,Xover[k][0].mcc=fitness(Xover[k][0].position,y)
             Xover[k][1].fit,Xover[k][1].precision, Xover[k][1].recall, Xover[k][1].fmeasure,Xover[k][1].mcc=fitness(Xover[k][1].position,y)
             
    popc=empty_individual.repeat(nc)
    i=0
    for s in range (len(Xover)):
        for j in range(2):
             popc[i]=Xover[s][j]
             i=i+1
    # Mutation
    popm=empty_individual.repeat(nm)    
    for k in range(nm):
       # Select Parent
         i=random.randint(0,nPop-1)
         p=pop[i].List
         available_numbers = list(set(search_space) - set(p))
         
         popm[k].List=Mutate(p,available_numbers)
         popm[k].position=X[popm[k].List]
         popm[k].fit, popm[k].precision, popm[k].recall, popm[k].fmeasure,popm[k].mcc=fitness(popm[k].position,y)
   
       
            
               
  # Evaluate mutatnt
             
              
# Distructor
   
    
   
     # Distructor
    
             
             
     # merge population        
    pop= pop+popc+popm    
    pop=sorted(pop,key=operator.attrgetter('fit'), reverse=True)
    
      #truncate
    pop=pop[0:nPop]
    
    # d=random.randint(1,MaxIt)
    # if (d<=it):
    #    actives=np.where(np.array(pop[0].position)==1)[0]
    #    for active in actives:
    #      if (random.randint(0,1)) == 1: 
    #          pop[0].position[active] = 0
             
             
    
    # Update WorstFit
    # WorstFit=min(pop[nPop-1].fit,WorstFit)
    
    # # Calculate selection probabilities
    # Fits=np.zeros(nPop)
    # for jj in range (nPop):
    #       Fits[jj]=pop[jj].fit
    # Fits=np.sort(Fits)[::-1]
    # P=np.zeros(nPop)
    # for j in range (nPop):
    #       P[j]=math.exp(-sp*(1/Fits[j])/(1/WorstFit))

    # P=P/sum(P)
    
    
    for j in range (nPop):
       Fits[i]=pop[i].fit
       
    for j in range (nPop):
        P[j]=Fits[j]/sum(Fits)        

    # store best solution ever found
    BestSol=pop[0]
    BestFits[it]=BestSol.fit
    if BestSol.fit > BestAcc:
       BestAcc=BestSol.fit
       BestList=BestSol.List
       BestPosition=BestSol.position
       BestPre=BestSol.precision
       BestRec=BestSol.recall
       BestFmeasure=BestSol.fmeasure
       BestMCC=BestSol.mcc
       # pcc=0.9;
       # pmm=0.4;
       nc=2*round(pcc*nPop/2)
       nm=round(pmm*nPop) 
       MutRate[it]=pmm
       CrossRate[it]=pcc
    ### store NFE
    
    nfe[it]=NFE
    
    
    if (BestFits[it]==BestFits[it-1]):
         Tag=Tag+1 
         ATag=ATag+1
         MutRate[it]=pmm
         CrossRate[it]=pcc
    else:
        Tag=1
        ATag=1
        pcc=0.9;
        pmm=0.4;
    if ATag==6:
      ATag=1
      pcc=pcc-0.2;
      if pcc<0.3:
          pcc=0.3
      pmm=pmm+0.2;
      if pmm<=1:
             
         nm=round(pmm*nPop)
         nc=2*round(pcc*nPop/2)
         Adaptive=Adaptive+1
         MutRate[it]=pmm
         CrossRate[it]=pcc

      else:
          # pmm=1.3
          pcc=0  
          nm=26
          nc=0
          Adaptive=Adaptive+1
          MutRate[it]=pmm
          CrossRate[it]=pcc      
    
    
    print("Iteration ", str(it) ,": Best fit = ", BestAcc,  "NFE =  ", nfe[it], "mutation rate =   ", pmm, "cross-over rate =   ", pcc)
    print("sum  P is  ",sum(P))
    it=it+1   
       
print ('PRE  ', BestPre, "REC  ",BestRec, "BestFmeasure  ", BestFmeasure, "BestMCC  ", BestMCC)





############Performance measure################

fits = BestFits[BestFits != 0]


plt.plot(fits, color='dimgrey')
# plt.plot(fitsW, color='black')
plt.title('Colon')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.legend(['DMC-Kmeans-GA', 'GA'],loc='lower right')
plt.show()
             



############Mutrate CrossRate################

last_non_zero_index = len(MutRate) - 1
while last_non_zero_index >= 0 and MutRate[last_non_zero_index] == 0:
    last_non_zero_index -= 1

# Trim the array to remove trailing zeros
MutRate = MutRate[:last_non_zero_index + 1]
CrossRate = CrossRate[:len(MutRate)]

M = MutRate.copy()
M[np.where(M == 1.2)] = 1
M[np.where(M == 1.4)] = 1

# Trim the array to remove trailing zeros


# Plot the mutation rate in blue
plt.plot(M, color='blue')

# Identify positions where CrossRate is 0
zero_positions = np.where(CrossRate == 0)[0]

# Overlay downward-pointing triangles at positions where CrossRate is 0
plt.scatter(zero_positions, M[zero_positions], marker='s', color='red', label='CrossRate = 0', zorder=5)

plt.title('Colon')
plt.xlabel('Iteration')
plt.ylabel('Mutation rate')
plt.grid(True)
plt.show()

###################  MuRate on Fits
# Your existing code for change_indices
change_indices = [i for i in range(1, len(MutRate)) if MutRate[i] != MutRate[i - 1]]

# Plot Fits
plt.plot(fits, color='dimgrey', label='DMC-Kmeans-GA')
# Uncomment the line below if fitsW is available
# plt.plot(fitsW, color='black', label='GA')

# Highlight specified iterations
for iteration in change_indices:
    marker = 'o'  # default marker
    if CrossRate[iteration] == 0:
        marker = 'v'  # Change to triangle if CrossRate is 0
        size = 70  # Set the size of the triangle
    else:
        size = 40  # Default size for circles

    color = 'red' if marker == 'v' else 'blue'  # Set color to grey for triangles, blue for circles
    plt.scatter(iteration, fits[iteration], color=color, marker=marker, s=size, label=f'Iteration {iteration}', zorder=5)

plt.title('Colon')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.legend(loc='lower right')
plt.show()




#################################
crate = [0.9] * 5 + [0.7] * 5 + [0.5] * 5 + [0.3] * 5 + [0] * 10
mrate = [0.4] * 5 + [0.6] * 5 + [0.8] * 5 + [1] * 5 + [1] * 10

iterations = list(range(1, 31))

# Plot the values
plt.plot(iterations, crate, color='blue', marker='o')
plt.plot(iterations, mrate, color='red',marker='*')

plt.xlabel('Iteration')


# plt.plot(crate, color='dimgrey')
# plt.plot(mrate, color='green')
# plt.title('Available rates')
plt.xlabel('Iteration')
plt.ylabel('Rates')
plt.grid(True)
plt.legend(['Pc', 'Pm'],loc='lower left')
plt.show()













# last_non_zero_index = len(CrossRate) - 1
# while last_non_zero_index >= 0 and CrossRate[last_non_zero_index] == 0:
#     last_non_zero_index -= 1
change_indices = [i for i in range(1, len(MutRate)) if MutRate[i] != MutRate[i - 1]]
# change_indices = [i + 1 for i in range(1, len(MutRate)) if MutRate[i] != MutRate[i - 1]]

# Plot Fits
plt.plot(fits, color='red', label='DMC-Kmeans-GA')
# Uncomment the line below if fitsW is available
# plt.plot(fitsW, color='black', label='GA')

# Highlight specified iterations
# for iteration in change_indices:
#     plt.scatter(iteration, fits[iteration], color='blue', marker='o', label=f'Iteration {iteration}', zorder=5)
    
for iteration in change_indices:
    marker = 'o'  # default marker
    if CrossRate[iteration] == 0:
        marker = '^'  # Change to triangle if CrossRate is 0
    color = 'grey' if marker == '^' else 'blue'  # Set color to green for triangles, blue for circles
    plt.scatter(iteration, fits[iteration], color=color, marker=marker, label=f'Iteration {iteration}', zorder=5)

plt.title('Colon')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.legend(loc='lower right')
plt.show()
# # Trim the array to remove trailing zeros
# CrossRate = CrossRate[:last_non_zero_index + 1]

# print(CrossRate)

############# Pre, Rec, Fm

###############Colon

# set height of bar
Be = [0.62, 0.64, 0.63,0.48 ]
Af = [0.97, 0.95, 0.95 , 0.93]

barWidth = 0.3
# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='black', width = barWidth,
         label ='Before applying DMC-GAwAR')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying DMC-GAwAR')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'F-score', 'MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('Colon')
plt.show()

#######CNS

Be = [0.51, 0.53, 0.52,0.39]
Af = [0.84, 0.80, 0.80, 0.73]

 
# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='black', width = barWidth,
         label ='Before applying DMC-GAwAR')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying DMC-GAwAR')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'F-score','MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('CNS')
plt.show()


#######Leu2

Be = [0.93, 0.88, 0.91, 0.66]
Af = [1, 1, 1, 1]

 
# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='black', width = barWidth,
         label ='Before applying DMC-GAwAR')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying DMC-GAwAR')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'F-score','MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('Leukemia')
plt.show()

#########SMK


Be = [0.58, 0.59, 0.58,0.19]
Af = [0.78, 0.75, 0.76,0.55]

 
# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='black', width = barWidth,
         label ='Before applying DMC-GAwAR')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying DMC-GAwAR')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'F-score','MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('SMK')
plt.show()

#########GLI

Be = [0.68, 0.65, 0.67,0.52]
Af = [0.9, 0.88, 0.89,0.84]

 
# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='black', width = barWidth,
         label ='Before applying DMC-GAwAR')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying DMC-GAwAR')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'F-score','MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('GLI')
plt.show()




########Acc Comparison##########
Be = [0.71, 0.56, 0.75,0.61,0.91]
Af = [0.97, 0.87, 0.93,0.78, 1]
values = [0.74, 0.58, 0.79, 0.59, 0.84] 
# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
barWidth=0.35
# Make the plot
plt.bar(br1, Be, color ='black', width = barWidth,
         label ='DMC-Random Selection')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='DMC-GAwAR')
 
# Adding Xticks
plt.ylabel('Accuracy percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Colon', 'CNS', 'GLI','SMK','Leukemia'])

for i, value in enumerate(values):
    plt.hlines(value, br1[i]-0.3, br2[i] + barWidth, color='blue', linestyle='-', linewidth=2)
    
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
# plt.title('GLI')
plt.show()


#############################################
############################################














### COLON####################
xpoints=[1,2,3,4,5,6,7,8,9,10]
MC= [0.68,	0.69,	0.70,	0.74,	0.73,	0.72,	0.73,	0.73,	0.71,	0.76]
EMC = [0.76,	0.67,	0.69,	0.71,	0.69,	0.74,	0.74,	0.74,	0.72,	0.72]
SLI_g= [0.68, 0.69, 0.69,  0.70, 0.70, 0.69,  0.69,  0.69, 0.71, 0.71]
SLI= [0.70, 0.81, 0.75, 0.76, 0.77,   0.81,  0.80, 0.80, 0.79, 0.76]
MPR= [0.71,	0.74,	0.79,	0.80,	0.81,	0.80,	0.81,	0.81,	0.81,	0.81]
DMC=[0.77,	0.82,	0.83,	0.83,	0.83,	0.83,	0.82,	0.81,	0.79,	0.78]
Comprehensive=[0.72,	0.79,	0.77,	0.79,	0.82,	0.80,	0.80,	0.80,	0.79,	0.79]
line1=plt.plot(xpoints, MC, 'go-') 
line2=plt.plot(xpoints, SLI, 'co-')   
line3=plt.plot(xpoints, SLI_g, 'mo-')   
# line4=plt.plot(xpoints, EMC, 'bo-')
# line5=plt.plot(xpoints, MPR, 'yo-')         
line6=plt.plot(xpoints, DMC, 'ro-')         
# line7=plt.plot(xpoints, Comprehensive, 'ko-')         

plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1)
plt.legend( ['MC', 'SLI', 'SLI-\u03B3','DMC'],ncol=5, loc='lower center')
plt.title('Colon')
plt.grid()

print('MC   =', np.mean(MC))
print('EMC   =', np.mean(EMC))
print('SLI_g   =', np.mean(SLI_g))
print('SLI   =', np.mean(SLI))
print('MPR   =', np.mean(MPR))
print('DMC   =', np.mean(DMC))
print('Comprehensive   =', np.mean(Comprehensive))


### CNS####################
xpoints=[1,2,3,4,5,6,7,8,9,10]
MC= [0.66,0.64,0.71,0.69, 0.72,0.68, 0.69,0.69,0.71, 0.71]
EMC = [0.63, 0.66, 0.66, 0.67, 0.66, 0.75, 0.76, 0.75, 0.72, 0.75]
SLI_g= [0.72, 0.72, 0.68, 0.66, 0.68, 0.65, 0.66, 0.68, 0.66,0.65]
SLI= [0.65, 0.65,0.71, 0.68,0.69,0.68, 0.68, 0.67,0.68,0.69]
PRFS= [0.69, 0.73, 0.72, 0.73, 0.73, 0.72, 0.71, 0.73, 0.75, 0.75]
DMC=[0.66,	0.68,	0.64,	0.67,	0.67,	0.69,	0.67,	0.67,	0.67,	0.67]
Comprehensive=[0.60,	0.73,	0.73,	0.69,	0.76,	0.76,	0.76,	0.75,	0.75,	0.75]
line1=plt.plot(xpoints, MC, 'go-') 
line2=plt.plot(xpoints, SLI, 'co-') 
line3=plt.plot(xpoints, SLI_g, 'mo-') 
# line4=plt.plot(xpoints, EMC, 'bo-')
# line5=plt.plot(xpoints, MPR, 'yo-')
line6=plt.plot(xpoints, DMC, 'ro-')         
# line7=plt.plot(xpoints, Comprehensive, 'ko-')         

plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1)
plt.legend( ['MC', 'SLI', 'SLI-\u03B3','DMC'],ncol=5, loc='lower center')
plt.title('CNS')
plt.grid()
print('MC   =', np.mean(MC))
print('EMC   =', np.mean(EMC))
print('SLI_g   =', np.mean(SLI_g))
print('SLI   =', np.mean(SLI))
print('MPR   =', np.mean(MPR))
print('DMC   =', np.mean(DMC))
print('Comprehensive   =', np.mean(Comprehensive))


### Leukemia####################
xpoints=[1,2,3,4,5,6,7,8,9,10]

MC= [0.94,	0.92,	0.93,	0.92,	0.91,	0.91,	0.90,	0.91,	0.90,	0.90]
EMC = [0.94,	0.93,	0.92,	0.92,	0.91,	0.91,	0.91,	0.90,	0.90,	0.90]
SLI_g= [0.94,0.93,0.93,0.93,0.92,0.92,0.92,0.92,0.91,0.91]
SLI= [0.94, 0.92, 0.92, 0.90, 0.91, 0.90, 0.91,0.90,0.90, 0.90]
MPR= [0.94,	0.93,	0.93,	0.91,	0.91,	0.91,	0.91,	0.91,	0.91,	0.90]
DMC=[0.94,	0.92,	0.92,	0.92,	0.92,	0.91,	0.91,	0.91,	0.91,	0.91]
Comprehensive=[0.95,	0.92,	0.91,	0.92,	0.92,	0.91,	0.91,	0.91,	0.91,	0.91]
line1=plt.plot(xpoints, MC, 'go-') 
line2=plt.plot(xpoints, SLI, 'co-') 
line3=plt.plot(xpoints, SLI_g, 'mo-') 
# line4=plt.plot(xpoints, EMC, 'bo-')
# line5=plt.plot(xpoints, MPR, 'yo-')
line6=plt.plot(xpoints, DMC, 'ro-') 
# line7=plt.plot(xpoints, Comprehensive, 'ko-')         
   
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1)
plt.legend( ['MC', 'SLI', 'SLI-\u03B3','DMC'],ncol=5, loc='lower center')
plt.title('Leukemia')
plt.grid()
print('MC   =', np.mean(MC))
print('EMC   =', np.mean(EMC))
print('SLI_g   =', np.mean(SLI_g))
print('SLI   =', np.mean(SLI))
print('MPR   =', np.mean(MPR))
print('DMC   =', np.mean(DMC))
print('Comprehensive   =', np.mean(Comprehensive))


### SMK####################
xpoints=[1,2,3,4,5,6,7,8,9,10]

MC= [0.62,	0.63,	0.62,	0.64,	0.62,	0.64,	0.63,	0.63,	0.63,	0.64]
EMC = [0.57,	0.62,	0.65,	0.66,	0.66,	0.67,	0.67,	0.67,	0.66,	0.67]
SLI_g= [0.53,0.60,0.62,0.65,0.67,0.67,0.65,0.69,0.68,0.69]
SLI= [0.62, 0.62, 0.63, 0.63, 0.61, 0.64, 0.64,0.63,0.63, 0.64]
MPR= [0.62,	0.67,	0.67,	0.68,	0.67,	0.69,	0.68,	0.68,	0.68,	0.67]
DMC=[0.47,	0.50,	0.57,	0.58,	0.58,	0.59,	0.60,	0.64,	0.64,	0.62]
Comprehensive=[0.63,	0.63,	0.65,	0.66,	0.65,	0.65,	0.65,	0.63,	0.63,	0.63]
line1=plt.plot(xpoints, MC, 'go-') 
line2=plt.plot(xpoints, SLI, 'co-') 
line3=plt.plot(xpoints, SLI_g, 'mo-') 
# line4=plt.plot(xpoints, EMC, 'bo-')
# line5=plt.plot(xpoints, MPR, 'yo-')
line6=plt.plot(xpoints, DMC, 'ro-')
# line7=plt.plot(xpoints, Comprehensive, 'ko-')         
     
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.ylim(0.4, 1)
plt.legend( ['MC', 'SLI', 'SLI-\u03B3','DMC'],ncol=5, loc='lower center')
plt.title('SMK')
plt.grid()

print('MC   =', np.mean(MC))
print('EMC   =', np.mean(EMC))
print('SLI_g   =', np.mean(SLI_g))
print('SLI   =', np.mean(SLI))
print('MPR   =', np.mean(MPR))
print('DMC   =', np.mean(DMC))
print('Comprehensive   =', np.mean(Comprehensive))






### GLI####################
xpoints=[1,2,3,4,5,6,7,8,9,10]

MC= [0.79,	0.82,	0.85,	0.85,	0.84,	0.85,	0.87,	0.86,	0.85,	0.84]
EMC = [0.80,	0.82,	0.83,	0.84,	0.84,	0.84,	0.85,	0.85,	0.84,	0.85]
SLI_g= [0.71,0.75,0.78, 0.77,0.77,0.76 ,0.76,0.75,0.77, 0.77]
SLI= [0.79,0.82,0.85, 0.85, 0.83, 0.83, 0.86, 0.86,0.87, 0.87]
MPR= [0.84,	0.86,	0.86,	0.85,	0.85,	0.85,	0.85,	0.85,	0.85,	0.84]
DMC=[0.77,	0.79,	0.83,	0.81,	0.82,	0.86,	0.85,	0.86,	0.86,	0.86]
# Comprehensive=[0.75,	0.82,	0.82,	0.81,	0.83,	0.84,	0.84,	0.84,	0.84,	0.85]
line1=plt.plot(xpoints, MC, 'go-') 
line2=plt.plot(xpoints, SLI, 'co-') 
line3=plt.plot(xpoints, SLI_g, 'mo-') 
# line4=plt.plot(xpoints, EMC, 'bo-')
# line5=plt.plot(xpoints, MPR, 'yo-')
line6=plt.plot(xpoints, DMC, 'ro-')   
# line7=plt.plot(xpoints, Comprehensive, 'ko-')         
 
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1)
plt.legend( ['MC', 'SLI', 'SLI-\u03B3','DMC'],ncol=5, loc='lower center')
plt.title('GLI')
plt.grid()

print('MC   =', np.mean(MC))
print('EMC   =', np.mean(EMC))
print('SLI_g   =', np.mean(SLI_g))
print('SLI   =', np.mean(SLI))
print('MPR   =', np.mean(MPR))
print('DMC   =', np.mean(DMC))
print('Comprehensive   =', np.mean(Comprehensive))


#####





### Covid ####################
xpoints=[1,2,3,4,5,6,7,8,9,10]

EMC = [0.36,	0.49,	0.58,	0.58,	0.57,	0.58,	0.60,	0.64,	0.65,	0.65]

PRFS= [0.54,	0.53,	0.57,	0.54,	0.54,	0.58,	0.67,	0.66,	0.65,	0.64]


plt.plot(xpoints, EMC, 'bo-')
plt.plot(xpoints, PRFS, 'ro-') 
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.ylim(0.3, 1)
plt.legend( [ 'EMC', 'MPR'],ncol=5, loc='lower left')
plt.title('Covid-19')
plt.grid()

### Multi-label classification ####################
xpoints=[1,2,3,4,5,6,7,8,9,10]

EMC = [0.80,	0.86,	0.84,	0.83,	0.83,	0.83,	0.83,	0.82,	0.84,	0.83]

PRFS= [0.77,	0.86,	0.87,	0.85,	0.84,	0.84,	0.83,	0.83,	0.83,	0.84]


plt.plot(xpoints, EMC, 'bo-')
plt.plot(xpoints, PRFS, 'ro-') 
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.ylim(0.3, 1)
plt.legend( [ 'EMC', 'MPR'],ncol=5, loc='lower left')
plt.title('Leukemia - Multi-label')
plt.grid()

### MLL ####################
xpoints=[1,2,3,4,5,6,7,8,9,10]

EMC = [0.75,	0.88,	0.89,	0.87,	0.90,	0.89,	0.88,	0.94,	0.94,	0.92]

PRFS= [0.85,	0.84,	0.85,	0.88,	0.91,	0.89,	0.89,	0.88,	0.88,	0.92]


plt.plot(xpoints, EMC, 'bo-')
plt.plot(xpoints, PRFS, 'ro-') 
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.ylim(0.3, 1)
plt.legend( [ 'EMC', 'MPR'],ncol=5, loc='lower left')
plt.title('MLL')
plt.grid()




### SRBCT ####################
xpoints=[1,2,3,4,5,6,7,8,9,10]

EMC = [0.49,	0.76,  0.75,	0.86,	0.85,	0.84,	0.84,	0.85,	0.87,	0.87]

PRFS= [0.60,	0.72,	0.75,	0.78,	0.78,	0.76,	0.84,	0.83,	0.84,	0.84]


plt.plot(xpoints, EMC, 'bo-')
plt.plot(xpoints, PRFS, 'ro-') 
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.ylim(0.3, 1)
plt.legend( [ 'EMC', 'MPR'],ncol=5, loc='lower left')
plt.title('SRBCT')
plt.grid()













data = [2173, 1926, 1221, 1788, 1199]
labels = ['Colon', 'CNS', 'GLI', 'SMK', 'Leukemia']
# Creating the bar chart
plt.bar(labels, data, color='blue')
plt.xlabel('Dataset')
plt.ylabel('Average NFE')
plt.show()















