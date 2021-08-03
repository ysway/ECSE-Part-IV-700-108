import pandas as pd
from pandas import DataFrame
from pandas import Series
import seaborn as sns
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

# Relationship extraction Arousal - Valence
# # VA Extraction Begin
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

## Relationship extraction Arousal - 
# SEMAINE
dataset = pd.DataFrame()
datasetV = pd.DataFrame()

for dirname, _, filenames in os.walk('D:\Documents\PartIV\ECSE-Part-IV-700-108\outputFile\Semaine\TrainingInput'):
    for filename in filenames:
        pathname = os.path.join(dirname, filename)
        if(pathname.find('TU_Features') != -1):
            data_temp = pd.read_csv(pathname.replace("\\","/"))
            dataset = dataset.append(data_temp,ignore_index = True)


for dirname, _, filenames in os.walk('D:\Documents\PartIV\ECSE-Part-IV-700-108\outputFile\Semaine\TrainingInput'):
    for filename in filenames:
        pathname = os.path.join(dirname, filename)
        if(pathname.find('TU_VA') != -1):
            data_temp = pd.read_csv(pathname.replace("\\","/"))
            datasetV = datasetV.append(data_temp,ignore_index = True)

RMS = dataset.RMS
F0 = dataset.F0Log10
MFCC = dataset.MFCC
a = datasetV.Arousal
print(RMS.shape[0])
print(a.shape[0])

# dc = np.stack(((RMS.tolist()),(a.tolist())), axis = 0)
# a = a.to_numpy()
# RMS = RMS.to_numpy()
# d = {'a' : a, 'RMS' : RMS}
# data = pd.DataFrame(data = d)
# # #Load data
# # dc = np.stack((v,a), axis=0)
# # a = np.array(a)
# # v = np.array(v)
# # d = {'a' : a, 'v' : v}
# # data = pd.DataFrame(data = d)

# #Calculate and plot for correletion/covariance value
# correlationRMS = data.corr(method='pearson')
# print(correlationRMS)
# print()

# print(np.cov(dc))

# df = data[['RMS','a']]
# sns.pairplot(df, kind="scatter")
# plt.show()

# plt.scatter(data['RMS'], data['a'])
# plt.show()