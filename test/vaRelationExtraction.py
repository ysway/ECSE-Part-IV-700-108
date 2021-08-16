import pandas as pd
from pandas import DataFrame
from pandas import Series
import seaborn as sns
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

# Relationship extraction Arousal - Valence
# VA Extraction Begin
# a = []
# v = []

# # RECOLA
# inputPath = '../inputFile/emotional_behaviour/'
# print('RECOLA Reading starts')
# for i in range(16, 66):
#     try:
#         # arousal extraction
#         with open(inputPath+'arousal/P'+str(i)+'.csv', 'r', newline='') as csv_file:
#             reader = csv.reader(line.replace(';', ',') for line in csv_file)
#             tVSa = list(reader)
#         csv_file.close()
#         tVSa = tVSa[1:]

#         for index, data in enumerate(tVSa):
#             sumA = 0
#             for j in range(1, 7):
#                 sumA += float(data[j])

#             a.append(sumA / 6.0)

#         # valence extraction
#         with open(inputPath+'valence/P'+str(i)+'.csv', 'r', newline='') as csv_file:
#             reader = csv.reader(line.replace(';', ',') for line in csv_file)
#             tVSv = list(reader)
#         csv_file.close()

#         tVSv = tVSv[1:]

#         for index, data in enumerate(tVSv):
#             sumA = 0
#             for j in range(1, 7):
#                 sumA += float(data[j])

#             v.append(sumA / 6.0)
#     except:
#         print('P'+str(i)+'.csv is missing, skipping...')


# #Load data
# dc = np.stack((v,a), axis=0)
# a = np.array(a)
# v = np.array(v)
# d = {'a' : a, 'v' : v}
# data = pd.DataFrame(data = d)

# #Calculate and plot for correletion/covariance value
# correlationV = data.corr(method='pearson')
# print(correlationV)
# print()

# print(np.cov(dc))

# df = data[['v','a']]
# sns.pairplot(df, kind="scatter")
# plt.show()

# plt.scatter(data['v'], data['a'])
# plt.show()

# Relationship extraction Arousal - RMS
# SEMAINE
dataset = pd.DataFrame()

dataset = pd.read_csv(
    "D:/Documents/PartIV/ECSE-Part-IV-700-108/inputFile/modelInput/allFileCombineU.csv")

RMS = dataset.RMS
F0 = dataset.F0Log10
MFCC1 = dataset.MFCC1
MFCC2 = dataset.MFCC2
MFCC3 = dataset.MFCC3
MFCC4 = dataset.MFCC4
MFCC5 = dataset.MFCC5
a = dataset.Arousal
v = dataset.Valence

dVA = np.stack(((v.tolist()), (a.tolist())), axis=0)
dRMS = np.stack(((RMS.tolist()), (a.tolist())), axis=0)
dvRMS = np.stack(((RMS.tolist()), (v.tolist())), axis=0)
dF0 = np.stack(((F0.tolist()), (a.tolist())), axis=0)
dvF0 = np.stack(((F0.tolist()), (v.tolist())), axis=0)
dMFCC1 = np.stack(((MFCC1.tolist()), (a.tolist())), axis=0)
dvMFCC1 = np.stack(((MFCC1.tolist()), (v.tolist())), axis=0)
dMFCC2 = np.stack(((MFCC2.tolist()), (a.tolist())), axis=0)
dvMFCC2 = np.stack(((MFCC2.tolist()), (v.tolist())), axis=0)
dMFCC3 = np.stack(((MFCC3.tolist()), (a.tolist())), axis=0)
dvMFCC3 = np.stack(((MFCC3.tolist()), (v.tolist())), axis=0)
dMFCC4 = np.stack(((MFCC4.tolist()), (a.tolist())), axis=0)
dvMFCC4 = np.stack(((MFCC4.tolist()), (v.tolist())), axis=0)
dMFCC5 = np.stack(((MFCC5.tolist()), (a.tolist())), axis=0)
dvMFCC5 = np.stack(((MFCC5.tolist()), (v.tolist())), axis=0)

a = a.to_numpy()
v = v.to_numpy()
RMS = RMS.to_numpy()
F0 = F0.to_numpy()
MFCC1 = MFCC1.to_numpy()
MFCC2 = MFCC2.to_numpy()
MFCC3 = MFCC3.to_numpy()
MFCC4 = MFCC4.to_numpy()
MFCC5 = MFCC5.to_numpy()

#################################
d0 = {'a': a, 'v': v}
data0 = pd.DataFrame(data=d0)

# Calculate and plot for correletion/covariance value
correlationV = data0.corr(method='pearson')
print(correlationV)
print()

print(np.cov(dVA))

df0 = data0[['v', 'a']]
sns.pairplot(df0, kind="scatter")
plt.show()

plt.scatter(data0['v'], data0['a'])
plt.show()

#################################
f1 = {'v': v, 'RMS': RMS}
dataf1 = pd.DataFrame(data=f1)

# Calculate and plot for correletion/covariance value
correlationVRMS = dataf1.corr(method='pearson')
print(correlationVRMS)
print()

print(np.cov(dvRMS))

dvf1 = dataf1[['RMS', 'v']]
sns.pairplot(dvf1, kind="scatter")
plt.show()

plt.scatter(dataf1['RMS'], dataf1['v'])
plt.show()

#################################
d1 = {'a': a, 'RMS': RMS}
data1 = pd.DataFrame(data=d1)

# Calculate and plot for correletion/covariance value
correlationRMS = data1.corr(method='pearson')
print(correlationRMS)
print()

print(np.cov(dRMS))

df1 = data1[['RMS', 'a']]
sns.pairplot(df1, kind="scatter")
plt.show()

plt.scatter(data1['RMS'], data1['a'])
plt.show()

#################################

f2 = {'v': v, 'F0': F0}
dataf2 = pd.DataFrame(data=f2)

# Calculate and plot for correletion/covariance value
correlationvF0 = dataf2.corr(method='pearson')
print(correlationvF0)
print()

print(np.cov(dvF0))

dvf2 = dataf2[['F0', 'v']]
sns.pairplot(dvf2, kind="scatter")
plt.show()

plt.scatter(dataf2['F0'], dataf2['v'])
plt.show()

#################################

d2 = {'a': a, 'F0': F0}
data2 = pd.DataFrame(data=d2)

# Calculate and plot for correletion/covariance value
correlationF0 = data2.corr(method='pearson')
print(correlationF0)
print()

print(np.cov(dF0))

df2 = data2[['F0', 'a']]
sns.pairplot(df2, kind="scatter")
plt.show()

plt.scatter(data2['F0'], data2['a'])
plt.show()

#################################

f3 = {'v': v, 'MFCC1': MFCC1}
dataf3 = pd.DataFrame(data=f3)

# Calculate and plot for correletion/covariance value
correlationvMFCC1 = dataf3.corr(method='pearson')
print(correlationvMFCC1)
print()

print(np.cov(dvMFCC1))

dvf3 = dataf3[['MFCC1', 'v']]
sns.pairplot(dvf3, kind="scatter")
plt.show()

plt.scatter(dataf3['MFCC1'], dataf3['v'])
plt.show()

#################################

d3 = {'a': a, 'MFCC1': MFCC1}
data3 = pd.DataFrame(data=d3)

# Calculate and plot for correletion/covariance value
correlationMFCC1 = data3.corr(method='pearson')
print(correlationMFCC1)
print()

print(np.cov(dMFCC1))

df3 = data3[['MFCC1', 'a']]
sns.pairplot(df3, kind="scatter")
plt.show()

plt.scatter(data3['MFCC1'], data3['a'])
plt.show()

#################################

f4 = {'v': v, 'MFCC2': MFCC2}
dataf4 = pd.DataFrame(data=f4)

# Calculate and plot for correletion/covariance value
correlationvMFCC2 = dataf4.corr(method='pearson')
print(correlationvMFCC2)
print()

print(np.cov(dvMFCC2))

dvf4 = dataf4[['MFCC2', 'v']]
sns.pairplot(dvf4, kind="scatter")
plt.show()

plt.scatter(dataf4['MFCC2'], dataf4['v'])
plt.show()

#################################

d4 = {'a': a, 'MFCC2': MFCC2}
data4 = pd.DataFrame(data=d4)

# Calculate and plot for correletion/covariance value
correlationMFCC2 = data4.corr(method='pearson')
print(correlationMFCC2)
print()

print(np.cov(dMFCC2))

df4 = data4[['MFCC2', 'a']]
sns.pairplot(df4, kind="scatter")
plt.show()

plt.scatter(data4['MFCC2'], data4['a'])
plt.show()

#################################

f5 = {'v': v, 'MFCC3': MFCC3}
dataf5 = pd.DataFrame(data=f5)

# Calculate and plot for correletion/covariance value
correlationvMFCC3 = dataf5.corr(method='pearson')
print(correlationvMFCC3)
print()

print(np.cov(dvMFCC3))

dvf5 = dataf5[['MFCC3', 'v']]
sns.pairplot(dvf5, kind="scatter")
plt.show()

plt.scatter(dataf5['MFCC3'], dataf5['v'])
plt.show()

#################################

d5 = {'a': a, 'MFCC3': MFCC3}
data5 = pd.DataFrame(data=d5)

# Calculate and plot for correletion/covariance value
correlationMFCC3 = data5.corr(method='pearson')
print(correlationMFCC3)
print()

print(np.cov(dMFCC3))

df5 = data5[['MFCC3', 'a']]
sns.pairplot(df5, kind="scatter")
plt.show()

plt.scatter(data5['MFCC3'], data5['a'])
plt.show()

#################################

f6 = {'v': v, 'MFCC4': MFCC4}
dataf6 = pd.DataFrame(data=f6)

# Calculate and plot for correletion/covariance value
correlationvMFCC4 = dataf6.corr(method='pearson')
print(correlationvMFCC4)
print()

print(np.cov(dvMFCC4))

dvf6 = dataf6[['MFCC4', 'v']]
sns.pairplot(dvf6, kind="scatter")
plt.show()

plt.scatter(dataf6['MFCC4'], dataf6['v'])
plt.show()

#################################

d6 = {'a': a, 'MFCC4': MFCC4}
data6 = pd.DataFrame(data=d6)

# Calculate and plot for correletion/covariance value
correlationMFCC4 = data6.corr(method='pearson')
print(correlationMFCC4)
print()

print(np.cov(dMFCC4))

df6 = data6[['MFCC4', 'a']]
sns.pairplot(df6, kind="scatter")
plt.show()

plt.scatter(data6['MFCC4'], data6['a'])
plt.show()

#################################

f7 = {'v': v, 'MFCC5': MFCC5}
dataf7 = pd.DataFrame(data=f7)

# Calculate and plot for correletion/covariance value
correlationvMFCC5 = dataf7.corr(method='pearson')
print(correlationvMFCC5)
print()

print(np.cov(dvMFCC5))

dvf7 = dataf7[['MFCC5', 'v']]
sns.pairplot(dvf7, kind="scatter")
plt.show()

plt.scatter(dataf7['MFCC5'], dataf7['v'])
plt.show()

#################################

d7 = {'a': a, 'MFCC5': MFCC5}
data7 = pd.DataFrame(data=d7)

# Calculate and plot for correletion/covariance value
correlationMFCC5 = data7.corr(method='pearson')
print(correlationMFCC5)
print()

print(np.cov(dMFCC5))

df7 = data7[['MFCC5', 'a']]
sns.pairplot(df7, kind="scatter")
plt.show()

plt.scatter(data7['MFCC5'], data7['a'])
plt.show()
