import pandas as pd
from pandas import DataFrame
from pandas import Series
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt

# VA Extraction Begin
a = []
v = []

# RECOLA
inputPath = '../inputFile/emotional_behaviour/'
print('RECOLA Reading starts')
for i in range(16, 66):
    try:
        # arousal extraction
        with open(inputPath+'arousal/P'+str(i)+'.csv', 'r', newline='') as csv_file:
            reader = csv.reader(line.replace(';', ',') for line in csv_file)
            tVSa = list(reader)
        csv_file.close()
        tVSa = tVSa[1:]

        for index, data in enumerate(tVSa):
            sumA = 0
            for j in range(1, 7):
                sumA += float(data[j])

            a.append(sumA / 6.0)

        # valence extraction
        with open(inputPath+'valence/P'+str(i)+'.csv', 'r', newline='') as csv_file:
            reader = csv.reader(line.replace(';', ',') for line in csv_file)
            tVSv = list(reader)
        csv_file.close()

        tVSv = tVSv[1:]

        for index, data in enumerate(tVSv):
            sumA = 0
            for j in range(1, 7):
                sumA += float(data[j])

            v.append(sumA / 6.0)
    except:
        print('P'+str(i)+'.csv is missing, skipping...')


#Load data
dc = np.stack((v,a), axis=0)
a = np.array(a)
v = np.array(v)
d = {'a' : a, 'v' : v}
data = pd.DataFrame(data = d)

#Calculate and plot for correletion/covariance value
correlationV = data.corr(method='pearson')
print(correlationV)
print()

print(np.cov(dc))

df = data[['v','a']]
sns.pairplot(df, kind="scatter")
plt.show()

plt.scatter(data['v'], data['a'])
plt.show()