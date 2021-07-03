import os
import matplotlib.pyplot as plt
import numpy as np
import csv

def vaVisCombinedPlot(inputPathV, inputPathA, outputPath, scatterMode, saveFormat):
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

    a = np.array(a)

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

    v = np.array(v)
    t = np.array(t)

    # TVA length handler
    if len(v)<len(a):
        a = a[:len(v)]
    elif len(v)>len(a):
        v = v[:len(a)]
        t = t[:len(a)]
    else:
        pass

    plt.ioff()
    if saveFormat.lower() == 'png':
        fig = plt.figure(figsize=[48, 24])
    else:
        fig = plt.figure(figsize=[24, 12])
    fig.suptitle('Time vs VA of '+inputPathV[inputPathV.rfind('/')+1:-5], fontsize=16)
    plt.xlabel('Time(s)')
    plt.ylabel('VA')

    if scatterMode:
        plt.scatter(t, v, s=2, label = 'Valence')
        plt.scatter(t, a, s=2, label = 'Arousal')
    else:  
        plt.plot(t, v, label = 'Valence')
        plt.plot(t, a, label = 'Arousal')

    plt.legend()
    plt.savefig(outputPath+'.'+saveFormat, format=saveFormat)
    plt.close(fig)

def batchPlot():
    validSessionsText = open('validSessions.txt', 'r').readlines()
    validSessions = [session[:-1] for session in validSessionsText[:-1]]
    validSessions.append(validSessionsText[-1])
    del validSessionsText

    inputPath = '../../inputFile/Sessions/'
    outputPath = '../../outputFile/Semaine/combinedVis/'
    scatterMode = False
    saveFormat = 'svg'

    print('Combined visualisation plot starts\r\n')
    for session in validSessions:
        print('Session: '+session)
        for f_name in os.listdir(inputPath+session+'/'):
            if f_name.endswith('V.txt'):
                try:
                    if not os.path.exists(outputPath+session+'/'):
                        os.mkdir(outputPath+session+'/')
                    vaVisCombinedPlot(inputPath+session+'/'+f_name, inputPath+session+'/'+f_name[:-5]+'A.txt', outputPath+session+'/'+f_name[:-5], scatterMode, saveFormat)
                except:
                    print('\tPartial file ('+f_name[:-5]+') is missing, skipping...')
    print('Combined visualisation plot tasks are finished\r\n')

if __name__ == '__main__':
    batchPlot()
