import os
import sys
import csv

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
                print('\t\tOld V: Summing index '+str(index)+' is out of bounds, appending new V')
        for index, data in enumerate(newA):
            try:
                newA[index] = oldA[index]+data
            except:
                print('\t\tOld A: Summing index '+str(index)+' is out of bounds, appending new A')

            v = newV+oldV[len(newV):]
            a = newA+oldA[len(newV):]
    return v, a, vaDataCountList

def sessionIterator():
    validSessionsText = open('validSessions.txt', 'r').readlines()
    validSessions = [session[:-1] for session in validSessionsText[:-1]]
    validSessions.append(validSessionsText[-1])
    del validSessionsText

    inputPath = '../../inputFile/Sessions/'
    outputPath = '../../outputFile/Semaine/averageVA/'
    mode = 'averageVA'
    saveFormat = '.csv'

    #logging output
    sys.stdout = open(outputPath+'0utputLog.txt', "w")

    print('Reading starts\r\n')
    for session in validSessions:
        print('Session: '+session)
        # init TAV for each session
        t = list()
        a = list()
        v = list()
        vaDataCountList = list()
        for f_name in os.listdir(inputPath+session+'/'):
            if f_name.endswith('V.txt'):
                print('\tFile ID: '+f_name[:-5]+'\tLength of List:'+str(len(v)))
                try:
                    tmpList = vaArrayExtractor(inputPath+session+'/'+f_name, inputPath+session+'/'+f_name[:-5]+'A.txt')
                    v, a, vaDataCountList = sumSession(v, a, tmpList[1], tmpList[2], vaDataCountList)
                except:
                    print('\t\tPartial file ('+f_name[:-5]+') is missing, skipping...')
        if len(t) <= len(tmpList[0]):
            t = tmpList[0]
        del tmpList
        # save to ouput path
        file = open(outputPath+session+saveFormat, "w", newline='', encoding='utf-8')
        writer = csv.writer(file)
        writer.writerow(['Time', 'Arousal', 'Valence'])
        for index, timeStamp in enumerate(t):
            try:
                writer.writerow([timeStamp, round(a[index]/vaDataCountList[index], 3), round(v[index]/vaDataCountList[index], 3)])
            except:
                print('\t\tCSV writing index '+str(index)+' is out of bounds')
        file.close()
        print('Saving averaged session: '+session+' to output path\r\n')

    print('Averaging task has finished\r\n')
    
    sys.stdout.close()

if __name__ == '__main__':
    sessionIterator()
