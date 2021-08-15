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

for dirname, _, filenames in os.walk('D:\Documents\PartIV\ECSE-Part-IV-700-108\inputFile\modelInput'):
    for filename in filenames:
        pathname = os.path.join(dirname, filename)
        if(pathname.find('tu') != -1):
            data_temp = pd.read_csv(pathname.replace("\\", "/"))
            dataset = dataset.append(data_temp, ignore_index=True)


RMS = dataset.RMS
F0 = dataset.F0Log10
MFCC1 = dataset.MFCC1
MFCC2 = dataset.MFCC2
MFCC3 = dataset.MFCC3
MFCC4 = dataset.MFCC4
MFCC5 = dataset.MFCC5
a = dataset.Arousal
v = dataset.Valence

dRMS = np.stack(((RMS.tolist()), (a.tolist())), axis=0)
dF0 = np.stack(((F0.tolist()), (a.tolist())), axis=0)
dMFCC1 = np.stack(((MFCC1.tolist()), (a.tolist())), axis=0)
dMFCC2 = np.stack(((MFCC2.tolist()), (a.tolist())), axis=0)
dMFCC3 = np.stack(((MFCC3.tolist()), (a.tolist())), axis=0)
dMFCC4 = np.stack(((MFCC4.tolist()), (a.tolist())), axis=0)
dMFCC5 = np.stack(((MFCC5.tolist()), (a.tolist())), axis=0)

a = a.to_numpy()
RMS = RMS.to_numpy()
F0 = F0.to_numpy()
MFCC1 = MFCC1.to_numpy()
MFCC2 = MFCC2.to_numpy()
MFCC3 = MFCC3.to_numpy()
MFCC4 = MFCC4.to_numpy()
MFCC5 = MFCC5.to_numpy()

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
