import os
import sys
import csv
import pandas as pd

def vaArrayExtractor(inputPathV, inputPathA):
    # arousal extraction
    with open(inputPathA, 'r', newline='') as csv_file:
        reader = csv.reader(line.replace(' ', ',') for line in csv_file)
        tVSa = list(reader)
    csv_file.close()

    a = []
    for data in tVSa:
        if len(data)==0:
            break
        a.append(float(data[1]))
    
    # valence extraction
    with open(inputPathV, 'r', newline='') as csv_file:
        reader = csv.reader(line.replace(' ', ',') for line in csv_file)
        tVSv = list(reader)
    csv_file.close()
    
    v = []
    t = list()
    for data in tVSv:
        if len(data)==0:
            break
        t.append(float(data[0]))
        v.append(float(data[1]))

    # TVA length handler
    if len(v)<len(a):
        a = a[:len(v)]
    elif len(v)>len(a):
        v = v[:len(a)]
        t = t[:len(a)]
    else:
        pass

    return t, v, a

def sumSession(oldV, oldA, newV, newA, vaDataCountList):
    # first interation
    if len(oldV) == 0:
        for index, data in enumerate(newV):
            vaDataCountList.append(1)
        return newV, newA, vaDataCountList
    else:
        for index, data in enumerate(newV):
            try:
                newV[index] = oldV[index]+data
                # V A has equal length, so only need to process once
                try:
                    vaDataCountList[index] += 1
                except:
                    vaDataCountList.append(1)
            except:
                vaDataCountList.append(1)
                print('\t\t\tOld V: Summing index '+str(index)+' is out of bounds, appending new V')
        for index, data in enumerate(newA):
            try:
                newA[index] = oldA[index]+data
            except:
                print('\t\t\tOld A: Summing index '+str(index)+' is out of bounds, appending new A')

            v = newV+oldV[len(newV):]
            a = newA+oldA[len(newV):]
    return v, a, vaDataCountList

def oneSecondVa(df):
    vaDf = df[0:0]
    rowCount = 0 # used for averaging, count the number of rows in a second
    currentValence = 0
    currentArousal = 0
    for row in df.itertuples(index=False):
        if (row.Time == int(row.Time) and rowCount != 0):
            vaDf.loc[vaDf.shape[0]] = [int(row.Time) - 1, currentValence / rowCount, currentArousal / rowCount]
            rowCount = 0
            currentValence = 0
            currentArousal = 0
        currentArousal = currentArousal + row.Arousal
        currentValence = currentValence + row.Valence
        rowCount = rowCount + 1

    vaDf.loc[vaDf.shape[0]] = [int(row.Time) - 1, currentValence / rowCount, currentArousal / rowCount]

    return vaDf

def saveAvgValues(inputPath, outputPath, talker, session, saveFormat, t, a, v, vaDataCountList):
    outputPath = outputPath+session+'/'
    print('\tMode: '+talker)
    tmpList = list()
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    for f_name in os.listdir(inputPath+session+'/'):
        if (f_name.find(talker) == -1):
            continue
        if f_name.endswith('V.txt'):
            print('\t\tFile ID: '+f_name[:-5]+'\t\tLength of List:'+str(len(v)))
            try:
                tmpList = vaArrayExtractor(inputPath+session+'/'+f_name, inputPath+session+'/'+f_name[:-5]+'A.txt')
                v, a, vaDataCountList = sumSession(v, a, tmpList[1], tmpList[2], vaDataCountList)
            except:
                print('\t\t\tPartial file ('+f_name[:-5]+') is missing, skipping...')
                # no talker session matched
    if len(tmpList) == 0:
        return
    if len(t) <= len(tmpList[0]):
        t = tmpList[0]
    del tmpList
    
    # save path
    outputPath = outputPath+talker+'_VA'+saveFormat

    # construct data frame
    tvaDf = pd.DataFrame(columns=list(['Time', 'Valence', 'Arousal']))
    for index, timeStamp in enumerate(t):
        try:
            tvaDf.loc[tvaDf.shape[0]] = [timeStamp, round(v[index]/vaDataCountList[index], 3), round(a[index]/vaDataCountList[index], 3)]
        except:
            print('\t\tCSV writing index '+str(index)+' is out of bounds')
            break

    tvaDf1s = oneSecondVa(tvaDf)
    tvaDf1s.to_csv(outputPath, index=False)

def sessionIterator():
    validSessionsText = open('validSessions.txt', 'r').readlines()
    validSessions = [session[:-1] for session in validSessionsText[:-1]]
    validSessions.append(validSessionsText[-1])
    del validSessionsText

    inputPath = '../../inputFile/Sessions/'
    outputPath = '../../outputFile/Semaine/TrainingInput/oneSecond/'
    saveFormat = '.csv'

    logging = False

    #logging output
    if logging:
        sys.stdout = open(outputPath+'0utputLog.txt', 'w')

    print('Reading starts\r\n')
    for session in validSessions:
        print('Session: '+session)
        # init TAV for each session
        t = list()
        a = list()
        v = list()
        vaDataCountList = list()
        saveAvgValues(inputPath, outputPath, 'TU', session, saveFormat, t, a, v, vaDataCountList) # User
        saveAvgValues(inputPath, outputPath, 'TO', session, saveFormat, t, a, v, vaDataCountList) # Operator
        print('Saving averaged session: '+session+' to output path\r\n')
    print('Averaging task has finished\r\n')
    
    if logging:
        sys.stdout.close()

if __name__ == '__main__':
    sessionIterator()
